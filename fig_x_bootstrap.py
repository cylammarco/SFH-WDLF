import os
import sys
import emcee
import numpy as np
from scipy.optimize import least_squares
from spectresc import spectres


def log_prior(theta):
    if (theta < 0.0).any():
        return -np.inf
    else:
        return 0.0


def log_probability(rel_norm, obs_normed, err_normed, model_list):
    rel_norm /= np.nansum(rel_norm)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = rel_norm[:, None].T @ model_list
    model /= np.nansum(model)
    residuals = (obs_normed - model) / err_normed
    log_likelihood = -0.5 * np.nansum(residuals**2 + np.log(2 * np.pi * err_normed**2))
    return log_likelihood


def residuals_function(rel_norm, obs_normed, err_normed, model_list):
    rel_norm /= np.nansum(rel_norm)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = rel_norm[:, None].T @ model_list
    model /= np.nansum(model)
    residuals = np.nansum(((obs_normed - model) / err_normed) ** 2.0)
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

idx = int(sys.argv[1])
print(f"Processing bootstrap_sample_folder sample {idx}...")


Mag = np.arange(5.0, 18.0, 0.1)
(
    partial_age_optimal,
    partial_age_duration,
    _,
    _,
    _,
) = np.load("SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy").T

partial_wdlf_optimal = []
for age, duration in zip(partial_age_optimal, partial_age_duration):
    print(f"Currently loading {age} Gyr population.")
    pwdlf_mag, pwdlf = np.loadtxt(
        os.path.join(
            f"bootstrap_sample_folder/sample_{idx}",
            f"montreal_co_da_20_C03_PARSECz0017_C08_{age:.4f}.csv",
        ),
        delimiter=",",
    ).T
    partial_wdlf_optimal.append(spectres(mag_obs_optimal, pwdlf_mag, pwdlf, fill=0.0))


pwdlf_model_optimal = np.column_stack(partial_wdlf_optimal)[:, obs_wdlf_optimal > 0.0]
pwdlf_model_optimal_20pc_subset = np.column_stack(partial_wdlf_optimal)[:, obs_wdlf_optimal_20pc_subset > 0.0]

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


# Load the lsq solution as starting point
lsq_res = np.load(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_lsq_solution.npy",
    allow_pickle=True,
).item()
lsq_res_20pc_subset = np.load(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_lsq_solution_20pc_subset.npy",
    allow_pickle=True,
).item()

solution_optimal_lsq = lsq_res.x
solution_optimal_jac = lsq_res.jac

solution_optimal_lsq_20pc_subset = lsq_res_20pc_subset.x
solution_optimal_jac_20pc_subset = lsq_res_20pc_subset.jac

_, _s, _vh = np.linalg.svd(solution_optimal_jac, full_matrices=False)
tol = np.finfo(float).eps * _s[0] * max(solution_optimal_jac.shape)
_w = _s > tol
cov = (_vh[_w].T / _s[_w] ** 2) @ _vh[_w]  # robust covariance matrix
stdev = np.sqrt(np.diag(cov))

_, _s_20pc_subset, _vh_20pc_subset = np.linalg.svd(solution_optimal_jac_20pc_subset, full_matrices=False)
tol_20pc_subset = np.finfo(float).eps * _s_20pc_subset[0] * max(solution_optimal_jac_20pc_subset.shape)
_w_20pc_subset = _s_20pc_subset > tol_20pc_subset
cov_20pc_subset = (_vh_20pc_subset[_w_20pc_subset].T / _s_20pc_subset[_w_20pc_subset] ** 2) @ _vh_20pc_subset[
    _w_20pc_subset
]  # robust covariance matrix
stdev_20pc_subset = np.sqrt(np.diag(cov_20pc_subset))


initial_weights = solution_optimal_lsq
initial_errors = stdev

initial_weights_20pc_subset = solution_optimal_lsq_20pc_subset
initial_errors_20pc_subset = stdev_20pc_subset

n_step = 10000
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
        ) = np.percentile(flat_samples_optimal[:, i], [31.7310508, 50.0, 68.2689492])
    initial_weights = solution_optimal
    initial_errors = (solution_upper - solution_lower) / 2.0
    solution_optimal_normed = solution_optimal / np.nansum(solution_optimal)
    np.save(
        f"bootstrap_sample_folder/sample_{idx}/gcns_sfh_sample_{idx}.npy",
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
        f"bootstrap_sample_folder/sample_{idx}/gcns_reconstructed_wdlf_sample_{idx}.npy",
        np.column_stack((mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal)),
    )


n_step = 150000
n_burn = 0

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
    ) = np.percentile(flat_samples_optimal[:, i], [31.7310508, 50.0, 68.2689492])

initial_weights = solution_optimal

solution_optimal_normed = solution_optimal / np.nansum(solution_optimal)

np.save(
    f"bootstrap_sample_folder/sample_{idx}/gcns_sfh_sample_{idx}.npy",
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
    f"bootstrap_sample_folder/sample_{idx}/gcns_reconstructed_wdlf_sample_{idx}.npy",
    np.column_stack((mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal)),
)


sfh_mcmc_lower = np.zeros(ndim_optimal)
sfh_mcmc = np.zeros(ndim_optimal)
sfh_mcmc_upper = np.zeros(ndim_optimal)
for i in range(ndim_optimal):
    sfh_mcmc_lower[i], sfh_mcmc[i], sfh_mcmc_upper[i] = np.percentile(
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
    f"bootstrap_sample_folder/sample_{idx}/gcns_sfh_lsq_solution_sample_{idx}",
    lsq_res,
)

del sampler_optimal
del flat_samples_optimal

n_step = 10000
n_burn = 0

for i in range(5):
    rel_norm_optimal_20pc_subset = np.vstack(
        [np.random.normal(initial_weights_20pc_subset, initial_errors_20pc_subset) for i in range(nwalkers_optimal)]
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
        ) = np.percentile(
            flat_samples_optimal_20pc_subset[:, i],
            [31.7310508, 50.0, 68.2689492],
        )
    initial_weights_20pc_subset = solution_optimal_20pc_subset
    initial_errors_20pc_subset = (solution_upper_20pc_subset - solution_lower_20pc_subset) / 2.0
    solution_optimal_normed_20pc_subset = solution_optimal_20pc_subset / np.nansum(solution_optimal_20pc_subset)
    np.save(
        f"bootstrap_sample_folder/sample_{idx}/gcns_sfh_20pc_subset_sample_{idx}.npy",
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
        f"bootstrap_sample_folder/sample_{idx}/gcns_reconstructed_wdlf_20pc_subset_sample_{idx}.npy",
        np.column_stack(
            (
                mag_obs_optimal,
                obs_wdlf_optimal_20pc_subset,
                obs_wdlf_err_optimal_20pc_subset,
            )
        ),
    )


n_step = 150000
n_burn = 0

rel_norm_optimal_20pc_subset = np.vstack(
    [np.random.normal(initial_weights_20pc_subset, initial_weights_20pc_subset * 0.01) for i in range(nwalkers_optimal)]
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
    ) = np.percentile(
        flat_samples_optimal_20pc_subset[:, i],
        [31.7310508, 50.0, 68.2689492],
    )
initial_weights_20pc_subset = solution_optimal_20pc_subset
solution_optimal_normed_20pc_subset = solution_optimal_20pc_subset / np.nansum(solution_optimal_20pc_subset)
np.save(
    f"bootstrap_sample_folder/sample_{idx}/gcns_sfh_20pc_subset_sample_{idx}.npy",
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
    f"bootstrap_sample_folder/sample_{idx}/gcns_reconstructed_wdlf_20pc_subset_sample_{idx}.npy",
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
    ) = np.percentile(flat_samples_optimal_20pc_subset[:, i], [31.7310508, 50.0, 68.2689492])

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
    f"bootstrap_sample_folder/sample_{idx}/gcns_sfh_lsq_solution_20pc_subset_sample_{idx}",
    lsq_res_20pc_subset,
)


recomputed_wdlf_optimal = np.nansum(solution_optimal * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_lsq = np.nansum(solution_optimal_lsq * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_20pc_subset = np.nansum(solution_optimal_20pc_subset * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_lsq_20pc_subset = np.nansum(
    solution_optimal_lsq_20pc_subset * np.array(partial_wdlf_optimal).T, axis=1
)


wdlf_err_low = np.nansum((solution_lower) * np.array(partial_wdlf_optimal).T, axis=1)
wdlf_err_high = np.nansum((solution_upper) * np.array(partial_wdlf_optimal).T, axis=1)
wdlf_err_low_20pc_subset = np.nansum((solution_lower_20pc_subset) * np.array(partial_wdlf_optimal).T, axis=1)
wdlf_err_high_20pc_subset = np.nansum((solution_upper_20pc_subset) * np.array(partial_wdlf_optimal).T, axis=1)


# append for plotting the first bin
solution_optimal_lsq = np.insert(solution_optimal_lsq, 0, 0.0)
solution_optimal = np.insert(solution_optimal, 0, 0.0)
solution_upper = np.insert(solution_upper, 0, 0.0)
solution_lower = np.insert(solution_lower, 0, 0.0)
solution_optimal_lsq_20pc_subset = np.insert(solution_optimal_lsq_20pc_subset, 0, 0.0)
solution_optimal_20pc_subset = np.insert(solution_optimal_20pc_subset, 0, 0.0)
solution_upper_20pc_subset = np.insert(solution_upper_20pc_subset, 0, 0.0)
solution_lower_20pc_subset = np.insert(solution_lower_20pc_subset, 0, 0.0)
# append for plotting the last bin
solution_optimal_lsq = np.append(solution_optimal_lsq, 0.0)
solution_optimal = np.append(solution_optimal, 0.0)
solution_upper = np.append(solution_upper, 0.0)
solution_lower = np.append(solution_lower, 0.0)
solution_optimal_lsq_20pc_subset = np.append(solution_optimal_lsq_20pc_subset, 0.0)
solution_optimal_20pc_subset = np.append(solution_optimal_20pc_subset, 0.0)
solution_upper_20pc_subset = np.append(solution_upper_20pc_subset, 0.0)
solution_lower_20pc_subset = np.append(solution_lower_20pc_subset, 0.0)


normalisation_this_work = np.sum(obs_wdlf_optimal) / np.sum(recomputed_wdlf_optimal_lsq) * 1e9
normalisation_this_work_20pc_subset = (
    np.sum(obs_wdlf_optimal_20pc_subset) / np.sum(recomputed_wdlf_optimal_lsq_20pc_subset) * 1e9
)
bin_norm_this_work = np.concatenate(
    [
        [partial_age_optimal[1] - partial_age_optimal[0]],
        (np.diff(partial_age_optimal)[:-1] + np.diff(partial_age_optimal)[1:]) / 2.0,
        [partial_age_optimal[-1] - partial_age_optimal[-2]],
    ]
)


sfh_mcmc = solution_optimal * normalisation_this_work / bin_norm_this_work
sfh_lsq = solution_optimal_lsq * normalisation_this_work / bin_norm_this_work
sfh_err_lower = solution_lower * normalisation_this_work / bin_norm_this_work
sfh_err_upper = solution_upper * normalisation_this_work / bin_norm_this_work
sfh_mcmc_20pc_subset = solution_optimal_20pc_subset * normalisation_this_work_20pc_subset / bin_norm_this_work
sfh_lsq_20pc_subset = solution_optimal_lsq_20pc_subset * normalisation_this_work_20pc_subset / bin_norm_this_work
sfh_err_lower_20pc_subset = solution_lower_20pc_subset * normalisation_this_work_20pc_subset / bin_norm_this_work
sfh_err_upper_20pc_subset = solution_upper_20pc_subset * normalisation_this_work_20pc_subset / bin_norm_this_work


sfh_csv_output = np.column_stack(
    [
        partial_age_optimal,
        sfh_mcmc,
        sfh_lsq,
        sfh_err_lower,
        sfh_err_upper,
        sfh_mcmc_20pc_subset,
        sfh_lsq_20pc_subset,
        sfh_err_lower_20pc_subset,
        sfh_err_upper_20pc_subset,
    ]
)

np.savetxt(
    f"bootstrap_sample_folder/gcns_sfh_sample_{idx}.csv",
    sfh_csv_output,
)

# Prepare to output CSV of the reconstructed WDLFs
wdlf_output = (
    recomputed_wdlf_optimal_lsq_20pc_subset
    / np.nansum(recomputed_wdlf_optimal_lsq_20pc_subset)
    * np.nansum(obs_wdlf_optimal_20pc_subset)
)

wdlf_err_output = obs_wdlf_optimal_20pc_subset
wdlf_20_output = (
    recomputed_wdlf_optimal_lsq_20pc_subset
    / np.nansum(recomputed_wdlf_optimal_lsq_20pc_subset)
    * np.nansum(obs_wdlf_optimal_20pc_subset)
)

wdlf_20_err_output = obs_wdlf_optimal_20pc_subset


wdlf_csv_output = np.column_stack(
    [
        mag_obs_optimal,
        wdlf_output,
        wdlf_err_output,
        wdlf_20_output,
        wdlf_20_err_output,
    ]
)

np.savetxt(
    f"bootstrap_sample_folder/gcns_reconstructed_wdlf_sample_{idx}.csv",
    wdlf_csv_output,
)
