import os

import corner
import emcee
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from spectres import spectres
from WDPhotTools import theoretical_lf


def log_prior(theta):
    if (theta < 0.0).any():
        return -np.inf
    else:
        return 0.0


def log_probability(rel_norm, obs_normed, err_normed, model_list):
    rel_norm /= np.sum(rel_norm)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = rel_norm[:, None].T @ model_list
    model /= np.nansum(model)
    log_likelihood = -np.nansum(((model - obs_normed) / err_normed) ** 2.0)
    return log_likelihood


# Load the GCNS data
gcns_wdlf = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]

# n_bin_optimal = 32

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.load("pwdlf_bin_optimal_mapping.npy")
mag_obs_optimal, resolution_optimal = np.load("mbol_resolution.npy").T

mag_obs_optimal_bin_edges = np.append(
    mag_obs_optimal - resolution_optimal * 0.5,
    mag_obs_optimal[-1] + resolution_optimal[-1] * 0.5,
)

h_gen_optimal, b_optimal = np.histogram(
    gcns_wdlf["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf["Vgen"],
)

e_gen_optimal, _ = np.histogram(
    gcns_wdlf["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf["Vgen"] ** 2.0,
)

obs_wdlf_optimal = h_gen_optimal / resolution_optimal
obs_wdlf_err_optimal = e_gen_optimal**0.5 / resolution_optimal


# Load the pwdlfs
data = []
age_list_1 = np.arange(0.049, 0.100, 0.001)
age_list_2 = np.arange(0.100, 0.350, 0.005)
age_list_3 = np.arange(0.35, 15.01, 0.01)
age_list_3dp = np.concatenate((age_list_1, age_list_2))
age_list_2dp = age_list_3

age = np.concatenate((age_list_3dp, age_list_2dp))

for i in age_list_3dp:
    data.append(
        np.loadtxt(
            os.path.join(
                "output",
                f"montreal_co_da_20_K01_PARSECz0014_C08_{i:.3f}_Mbol.csv",
            ),
            delimiter=",",
        )
    )

for i in age_list_2dp:
    data.append(
        np.loadtxt(
            os.path.join(
                "output",
                f"montreal_co_da_20_K01_PARSECz0014_C08_{i:.2f}_Mbol.csv",
            ),
            delimiter=",",
        )
    )


mag_pwdlf = data[0][:, 0]


# Stack up the pwdlfs to the desired resolution

partial_wdlf_optimal = []
partial_age_optimal = []
for idx in np.sort(list(set(pwdlf_mapping_bin_optimal))):
    pwdlf_temp = np.zeros_like(mag_obs_optimal)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_optimal == idx)[0]:
        pwdlf_temp += spectres(
            mag_obs_optimal, mag_pwdlf, data[i][:, 1], fill=0.0
        )
        age_temp += age[i]
        age_count += 1
    partial_wdlf_optimal.append(pwdlf_temp)
    partial_age_optimal.append(age_temp / age_count)


pwdlf_model_optimal = np.vstack(partial_wdlf_optimal)[
    :, obs_wdlf_optimal > 0.0
]

nwalkers_optimal = 1000

ndim_optimal = len(partial_wdlf_optimal)

obs_normed = obs_wdlf_optimal[obs_wdlf_optimal > 0.0]
obs_err_normed = obs_wdlf_err_optimal[obs_wdlf_optimal > 0.0]
obs_err_normed /= np.sum(obs_normed)
obs_normed /= np.sum(obs_normed)


initial_weights = np.array(
    [
        0.00743825,
        0.00973495,
        0.00833878,
        0.02137616,
        0.04841087,
        0.04618622,
        0.02068454,
        0.02967219,
        0.05353962,
        0.0213752,
        0.04235483,
        0.03109259,
        0.04260385,
        0.03949018,
        0.02721844,
        0.03188883,
        0.03322586,
        0.03396735,
        0.0337657,
        0.04128011,
        0.05309941,
        0.04316444,
        0.03748048,
        0.05341513,
        0.02272911,
        0.03329748,
        0.01653212,
        0.00450217,
        0.01473943,
        0.01988275,
        0.01779972,
        0.0038156,
        0.02219265,
        0.00457476,
        0.00045124,
        0.00016956,
        0.00016031,
        0.00026658,
        0.00038806,
        0.00030394,
        0.00047451,
        0.00071062,
        0.00059391,
        0.00049896,
        0.00068063,
        0.00094117,
        0.00113014,
        0.00125013,
        0.00151658,
        0.00160442,
        0.00163118,
        0.00440231,
    ]
)

for i in range(1):
    print(i)
    rel_norm_optimal = np.vstack(
        [
            np.random.normal(initial_weights, initial_weights * 0.1)
            for i in range(nwalkers_optimal)
        ]
    )

    sampler_optimal = emcee.EnsembleSampler(
        nwalkers_optimal,
        ndim_optimal,
        log_probability,
        args=(
            obs_normed,
            obs_err_normed,
            pwdlf_model_optimal,
        ),
    )
    sampler_optimal.run_mcmc(rel_norm_optimal, 10000, progress=True)

    flat_samples_optimal = sampler_optimal.get_chain(discard=1000, flat=True)

    solution_optimal = np.zeros(ndim_optimal)
    for i in range(ndim_optimal):
        solution_optimal[i] = np.percentile(flat_samples_optimal[:, i], [50.0])

    initial_weights = solution_optimal

    solution_optimal_normed = solution_optimal / np.nansum(solution_optimal)

    np.save(
        "gcns_sfh_optimal_resolution_bin_optimal.npy",
        np.column_stack((partial_age_optimal, solution_optimal)),
    )
    np.save(
        "gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.npy",
        np.column_stack(
            (mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal)
        ),
    )

    np.save("mcmc_flattened_chain", flat_samples_optimal)


# Finally refining with a minimizer
lsq_res = least_squares(
    log_probability,
    solution_optimal,
    args=(
        obs_normed,
        obs_err_normed,
        pwdlf_model_optimal,
    ),
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-12,
)

np.save("gcns_sfh_optimal_resolution_lsq_solution", lsq_res)
