import os
import emcee
import numpy as np
from scipy.optimize import least_squares
from spectresc import spectres


def log_prior(theta):
    if (theta < 0.0).any():
        return -np.inf
    else:
        return 0.0


def log_probability(rel, obs_normed, err_normed, model_list):
    rel_norm = rel / np.nansum(rel)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = rel_norm[:, None].T @ model_list
    model /= np.nansum(model, axis=0)
    residuals = (obs_normed - model) / err_normed
    log_likelihood = -0.5 * np.nansum(residuals**2 + np.log(2 * np.pi * err_normed**2))
    return log_likelihood


def residuals_function(rel, obs_normed, err_normed, model_list):
    rel_norm = rel / np.nansum(rel)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = rel_norm[:, None].T @ model_list
    model /= np.nansum(model, axis=0)
    residuals = np.nansum(((obs_normed - model) / err_normed)**2.0)
    return residuals


# Load the GCNS data
gcns_wdlf = np.load("pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz")["data"]
gcns_wdlf_20pc_subset = gcns_wdlf[gcns_wdlf["dpc"] <= 20.0]


# n_bin_optimal = 32

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.load("SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy")
mag_obs_optimal, resolution_optimal = np.load("SFH-WDLF-article/figure_data/mbol_resolution.npy").T

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

# 20pc sample
h_gen_optimal_20pc_subset, b_optimal_20pc_subset = np.histogram(
    gcns_wdlf_20pc_subset["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf_20pc_subset["Vgen"],
)

e_gen_optimal_20pc_subset, _ = np.histogram(
    gcns_wdlf_20pc_subset["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf_20pc_subset["Vgen"] ** 2.0,
)


obs_wdlf_optimal = h_gen_optimal / resolution_optimal
obs_wdlf_err_optimal = e_gen_optimal**0.5 / resolution_optimal

obs_wdlf_optimal_20pc_subset = h_gen_optimal_20pc_subset / resolution_optimal
obs_wdlf_err_optimal_20pc_subset = e_gen_optimal_20pc_subset**0.5 / resolution_optimal

# Load the pwdlfs
data = []
age_list_1 = np.arange(0.049, 0.100, 0.001)
age_list_2 = np.arange(0.100, 0.350, 0.005)
age_list_3 = np.arange(0.35, 14.01, 0.01)
age_list_3dp = np.concatenate((age_list_1, age_list_2))
age_list_2dp = age_list_3

age = np.concatenate((age_list_3dp, age_list_2dp))

for i in age_list_3dp:
    data.append(
        np.loadtxt(
            os.path.join(
                "output",
                f"montreal_co_da_20_C03_PARSECz0017_C08_{i:.3f}_Mbol.csv",
            ),
            delimiter=",",
        )
    )

for i in age_list_2dp:
    data.append(
        np.loadtxt(
            os.path.join(
                "output",
                f"montreal_co_da_20_C03_PARSECz0017_C08_{i:.2f}_Mbol.csv",
            ),
            delimiter=",",
        )
    )


mag_pwdlf = data[0][:, 0]


# Stack up the pwdlfs to the desired resolution
partial_wdlf_optimal = []
partial_age_optimal = []
partial_age_duration = []
for idx in np.sort(list(set(pwdlf_mapping_bin_optimal))):
    pwdlf_temp = np.zeros_like(mag_obs_optimal)
    age_temp_list = []
    for i in np.where(pwdlf_mapping_bin_optimal == idx)[0]:
        pwdlf_temp += spectres(mag_obs_optimal, mag_pwdlf, data[i][:, 1], fill=0.0)
        age_temp_list.append(age[i])
    partial_wdlf_optimal.append(pwdlf_temp)
    partial_age_optimal.append((np.max(age_temp_list) + np.min(age_temp_list)) / 2.0)
    partial_age_duration.append(np.ptp(age_temp_list))


pwdlf_model_optimal = np.vstack(partial_wdlf_optimal)[:, obs_wdlf_optimal > 0.0]
pwdlf_model_optimal_20pc_subset = np.vstack(partial_wdlf_optimal)[:, obs_wdlf_optimal_20pc_subset > 0.0]

nwalkers_optimal = 250

ndim_optimal = len(partial_wdlf_optimal)

obs_normed = obs_wdlf_optimal[obs_wdlf_optimal > 0.0]
obs_err_normed = obs_wdlf_err_optimal[obs_wdlf_optimal > 0.0]
obs_err_normed /= np.sum(obs_normed)
obs_normed /= np.sum(obs_normed)

obs_normed_20pc_subset = obs_wdlf_optimal_20pc_subset[obs_wdlf_optimal_20pc_subset > 0.0]
obs_err_normed_20pc_subset = obs_wdlf_err_optimal_20pc_subset[obs_wdlf_optimal_20pc_subset > 0.0]
obs_err_normed_20pc_subset /= np.sum(obs_normed_20pc_subset)
obs_normed_20pc_subset /= np.sum(obs_normed_20pc_subset)

initial_weights = np.ones(len(pwdlf_model_optimal)) * 1e-2
initial_errors = initial_weights * 0.01

n_step = 20000
n_burn = 0

for i in range(5):
    print(i)
    rel_norm_optimal = np.vstack([np.random.normal(initial_weights, initial_errors) for i in range(nwalkers_optimal)])

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
    sampler_optimal.run_mcmc(rel_norm_optimal, n_step, progress=True)

    flat_samples_optimal = sampler_optimal.get_chain(discard=n_burn, flat=True)

    solution_optimal = np.zeros(ndim_optimal)
    solution_lower = np.zeros(ndim_optimal)
    solution_upper = np.zeros(ndim_optimal)
    for i in range(ndim_optimal):
        (
            solution_lower[i],
            solution_optimal[i],
            solution_upper[i],
        ) = np.nanpercentile(flat_samples_optimal[:, i], [31.7310508, 50.0, 68.2689492])

    initial_weights = solution_optimal
    initial_errors = (solution_upper - solution_lower) / 2.0

    solution_optimal_normed = solution_optimal / np.nansum(solution_optimal)

    np.save(
        "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy",
        np.column_stack(
            (
                partial_age_optimal,
                partial_age_duration,
                solution_optimal,
                solution_lower,
                solution_upper,
            )
        ),
    )
    np.save(
        "SFH-WDLF-article/figure_data/gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.npy",
        np.column_stack((mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal)),
    )


n_step = 200000
n_burn = 20000

rel_norm_optimal = np.vstack(
    [np.random.normal(initial_weights, initial_errors) for i in range(nwalkers_optimal)]
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
sampler_optimal.run_mcmc(rel_norm_optimal, n_step, progress=True)

flat_samples_optimal = sampler_optimal.get_chain(discard=n_burn, flat=True)

solution_optimal = np.zeros(ndim_optimal)
solution_lower = np.zeros(ndim_optimal)
solution_upper = np.zeros(ndim_optimal)
for i in range(ndim_optimal):
    (
        solution_lower[i],
        solution_optimal[i],
        solution_upper[i],
    ) = np.nanpercentile(flat_samples_optimal[:, i], [31.7310508, 50.0, 68.2689492])

initial_weights = solution_optimal

solution_optimal_normed = solution_optimal / np.nansum(solution_optimal)

np.save(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy",
    np.column_stack(
        (
            partial_age_optimal,
            partial_age_duration,
            solution_optimal,
            solution_lower,
            solution_upper,
        )
    ),
)
np.save(
    "SFH-WDLF-article/figure_data/gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.npy",
    np.column_stack((mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal)),
)


sfh_mcmc_lower = np.zeros(ndim_optimal)
sfh_mcmc = np.zeros(ndim_optimal)
sfh_mcmc_upper = np.zeros(ndim_optimal)
for i in range(ndim_optimal):
    sfh_mcmc_lower[i], sfh_mcmc[i], sfh_mcmc_upper[i] = np.nanpercentile(
        flat_samples_optimal[:, i], [31.7310508, 50.0, 68.2689492]
    )


sfh_mcmc_lower /= np.nanmax(sfh_mcmc)
sfh_mcmc_upper /= np.nanmax(sfh_mcmc)

# Finally refining with a minimizer
lsq_res = least_squares(
    residuals_function,
    solution_optimal,
    args=(
        obs_normed,
        obs_err_normed,
        pwdlf_model_optimal,
    ),
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    jac="cs",
    tr_solver="exact",
    verbose=2,
)

np.save(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_lsq_solution",
    lsq_res,
)

initial_weights_20pc_subset = obs_wdlf_optimal
initial_errors_20pc_subset = initial_weights_20pc_subset * 0.01

del sampler_optimal
del flat_samples_optimal

n_step = 20000
n_burn = 0

for i in range(5):
    rel_norm_optimal_20pc_subset = np.vstack(
        [
            np.random.normal(initial_weights_20pc_subset, initial_errors_20pc_subset)
            for i in range(nwalkers_optimal)
        ]
    )
    sampler_optimal_20pc_subset = emcee.EnsembleSampler(
        nwalkers_optimal,
        ndim_optimal,
        log_probability,
        args=(
            obs_normed_20pc_subset,
            obs_err_normed_20pc_subset,
            pwdlf_model_optimal_20pc_subset,
        ),
    )
    sampler_optimal_20pc_subset.run_mcmc(rel_norm_optimal_20pc_subset, n_step, progress=True)
    flat_samples_optimal_20pc_subset = sampler_optimal_20pc_subset.get_chain(discard=n_burn, flat=True)
    solution_optimal_20pc_subset = np.zeros(ndim_optimal)
    solution_lower_20pc_subset = np.zeros(ndim_optimal)
    solution_upper_20pc_subset = np.zeros(ndim_optimal)
    for i in range(ndim_optimal):
        (
            solution_lower_20pc_subset[i],
            solution_optimal_20pc_subset[i],
            solution_upper_20pc_subset[i],
        ) = np.nanpercentile(
            flat_samples_optimal_20pc_subset[:, i],
            [31.7310508, 50.0, 68.2689492],
        )
    initial_weights_20pc_subset = solution_optimal_20pc_subset
    initial_errors_20pc_subset = (solution_upper_20pc_subset - solution_lower_20pc_subset) / 2.0
    solution_optimal_normed_20pc_subset = solution_optimal_20pc_subset / np.nansum(solution_optimal_20pc_subset)
    np.save(
        "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal_20pc_subset.npy",
        np.column_stack(
            (
                partial_age_optimal,
                partial_age_duration,
                solution_optimal_20pc_subset,
                solution_lower_20pc_subset,
                solution_upper_20pc_subset,
            )
        ),
    )
    np.save(
        "SFH-WDLF-article/figure_data/gcns_reconstructed_wdlf_optimal_resolution_bin_optimal_20pc_subset.npy",
        np.column_stack(
            (
                mag_obs_optimal,
                obs_wdlf_optimal_20pc_subset,
                obs_wdlf_err_optimal_20pc_subset,
            )
        ),
    )


n_step = 200000
n_burn = 20000

rel_norm_optimal_20pc_subset = np.vstack(
    [
        np.random.normal(initial_weights_20pc_subset, initial_weights_20pc_subset * 0.01)
        for i in range(nwalkers_optimal)
    ]
)
sampler_optimal_20pc_subset = emcee.EnsembleSampler(
    nwalkers_optimal,
    ndim_optimal,
    log_probability,
    args=(
        obs_normed_20pc_subset,
        obs_err_normed_20pc_subset,
        pwdlf_model_optimal_20pc_subset,
    ),
)
sampler_optimal_20pc_subset.run_mcmc(rel_norm_optimal_20pc_subset, n_step, progress=True)
flat_samples_optimal_20pc_subset = sampler_optimal_20pc_subset.get_chain(discard=n_burn, flat=True)
solution_optimal_20pc_subset = np.zeros(ndim_optimal)
solution_lower_20pc_subset = np.zeros(ndim_optimal)
solution_upper_20pc_subset = np.zeros(ndim_optimal)
for i in range(ndim_optimal):
    (
        solution_lower_20pc_subset[i],
        solution_optimal_20pc_subset[i],
        solution_upper_20pc_subset[i],
    ) = np.nanpercentile(
        flat_samples_optimal_20pc_subset[:, i],
        [31.7310508, 50.0, 68.2689492],
    )
initial_weights_20pc_subset = solution_optimal_20pc_subset
solution_optimal_normed_20pc_subset = solution_optimal_20pc_subset / np.nansum(solution_optimal_20pc_subset)
np.save(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal_20pc_subset.npy",
    np.column_stack(
        (
            partial_age_optimal,
            partial_age_duration,
            solution_optimal_20pc_subset,
            solution_lower_20pc_subset,
            solution_upper_20pc_subset,
        )
    ),
)
np.save(
    "SFH-WDLF-article/figure_data/gcns_reconstructed_wdlf_optimal_resolution_bin_optimal_20pc_subset.npy",
    np.column_stack(
        (
            mag_obs_optimal,
            obs_wdlf_optimal_20pc_subset,
            obs_wdlf_err_optimal_20pc_subset,
        )
    ),
)

sfh_mcmc_lower_20pc_subset = np.zeros(ndim_optimal)
sfh_mcmc_20pc_subset = np.zeros(ndim_optimal)
sfh_mcmc_upper_20pc_subset = np.zeros(ndim_optimal)
for i in range(ndim_optimal):
    (
        sfh_mcmc_lower_20pc_subset[i],
        sfh_mcmc_20pc_subset[i],
        sfh_mcmc_upper_20pc_subset[i],
    ) = np.nanpercentile(flat_samples_optimal_20pc_subset[:, i], [31.7310508, 50.0, 68.2689492])

sfh_mcmc_lower_20pc_subset /= np.nanmax(sfh_mcmc_20pc_subset)
sfh_mcmc_upper_20pc_subset /= np.nanmax(sfh_mcmc_20pc_subset)

lsq_res_20pc_subset = least_squares(
    residuals_function,
    solution_optimal_20pc_subset,
    args=(
        obs_normed_20pc_subset,
        obs_err_normed_20pc_subset,
        pwdlf_model_optimal_20pc_subset,
    ),
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    jac="cs",
    tr_solver="exact",
    verbose=2,
)

np.save(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_lsq_solution_20pc_subset",
    lsq_res_20pc_subset,
)
