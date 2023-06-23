import os

import emcee
from matplotlib import pyplot as plt
import numpy as np
from spectres import spectres
from WDPhotTools import theoretical_lf


def log_prior(theta):
    if (theta < 0.0).any():
        return -np.inf
    else:
        return 0.0


def log_probability(rel_norm, obs, err, model_list):
    rel_norm /= np.sum(rel_norm)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = np.nansum(rel_norm[:, None] * model_list, axis=0)
    model /= np.nansum(model)
    obs /= np.nansum(obs)
    log_likelihood = -np.nansum(((model - obs) / err) ** 2.0)
    return log_likelihood


# Load the GCNS data
gcns_wdlf = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]

n_bin_optimal = 32

h_gen_optimal, b_optimal = np.histogram(
    gcns_wdlf["Mbol"],
    bins=n_bin_optimal,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf["Vgen"],
)

e_gen_optimal, _ = np.histogram(
    gcns_wdlf["Mbol"],
    bins=n_bin_optimal,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf["Vgen"] ** 2.0,
)

bin_size_optimal = b_optimal[1] - b_optimal[0]

mag_obs_optimal = np.around(b_optimal[1:] - 0.5 * bin_size_optimal, 2)
obs_wdlf_optimal = h_gen_optimal / bin_size_optimal
obs_wdlf_err_optimal = e_gen_optimal**0.5 / bin_size_optimal

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

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.insert(np.load("pwdlf_bin_optimal_mapping.npy"), 0, 0)

# Stack up the pwdlfs to the desired resolution

partial_wdlf_optimal = []
partial_age_optimal = []
for idx in np.sort(list(set(pwdlf_mapping_bin_optimal))):
    pwdlf_temp = np.zeros_like(mag_obs_optimal)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_optimal == idx)[0]:
        pwdlf_temp += spectres(mag_obs_optimal, mag_pwdlf, data[i][:, 1], fill=0.0)
        age_temp += age[i]
        age_count += 1
    partial_wdlf_optimal.append(pwdlf_temp)
    partial_age_optimal.append(age_temp / age_count)


pwdlf_model_optimal = np.vstack(partial_wdlf_optimal)[:, obs_wdlf_optimal > 0.0]

nwalkers_optimal = 1000

ndim_optimal = len(partial_wdlf_optimal)

rel_norm_optimal = np.random.rand(nwalkers_optimal, ndim_optimal)

sampler_optimal = emcee.EnsembleSampler(
    nwalkers_optimal,
    ndim_optimal,
    log_probability,
    args=(
        obs_wdlf_optimal[obs_wdlf_optimal > 0.0],
        obs_wdlf_err_optimal[obs_wdlf_optimal > 0.0],
        pwdlf_model_optimal,
    ),
)
sampler_optimal.run_mcmc(rel_norm_optimal, 200000, progress=True)

flat_samples_optimal = sampler_optimal.get_chain(discard=2000, thin=5, flat=True)

solution_optimal = np.zeros(ndim_optimal)
for i in range(ndim_optimal):
    solution_optimal[i] = np.percentile(flat_samples_optimal[:, i], [50.0])

solution_optimal /= np.nansum(solution_optimal)

np.save("gcns_sfh_optimal_resolution_bin_optimal.npy",
    np.column_stack((partial_age_optimal, solution_optimal)),
)
np.save(
    "gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.npy",
    np.column_stack((mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal)),
)
