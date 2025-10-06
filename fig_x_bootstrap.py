"""This script performs the following tasks:

1. **Log Probability and Residuals Calculation**:
    - Defines functions to compute the log-prior, log-probability, and residuals for a given set of relative weights,
    observed data, and model predictions.

2. **Data Loading and Preprocessing**:
    - Loads the GCNS white dwarf luminosity function (WDLF) data and subsets it for stars within 20 parsecs.
    - Loads precomputed mappings and resolutions for the WDLF and prepares histogram data for observed magnitudes and
    errors.

3. **Model Preparation**:
    - Loads partial WDLF models for different ages and interpolates them to match the observed magnitude bins.
    - Normalizes observed data and errors for further analysis.

4. **Least Squares Optimization**:
    - Loads initial least squares solutions and their Jacobians.
    - Computes robust covariance matrices and uncertainties for the solutions.
    - Refines the solutions using the `scipy.optimize.least_squares` function.

5. **Uncertainty Estimation**:
    - Computes 1-sigma uncertainties for the solutions using covariance matrices.
    - Propagates uncertainties to the reconstructed WDLF.

6. **Star Formation History (SFH) Calculation**:
    - Normalizes the solutions to compute the star formation history (SFH) in units of solar masses per year per cubic
    parsec.
    - Outputs the SFH and its uncertainties to a CSV file.

7. **Reconstructed WDLF Output**:
    - Reconstructs the WDLF using the optimized solutions and outputs it to a CSV file.

- The script takes a single command-line argument (`idx`) representing the index of the bootstrap sample to process.

Outputs:
- CSV files containing:
  1. The computed SFH and its uncertainties.
  2. The reconstructed WDLF and its uncertainties.

Dependencies
------------
- `numpy`
- `scipy`
- `spectresc`
- Precomputed data files and mappings located in the `SFH-WDLF-article` directory.

Usage
-----
Run the script with the index of the bootstrap sample as an argument:
     python fig_x_bootstrap.py <idx>
"""

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


def log_probability(rel, obs_normed, err_normed, model_list):
    """
    Computes the log-probability of a model given observed data, errors, and a prior.

    Parameters
    ----------
    rel : array-like
        Relative weights or parameters to normalize and use in the model.
    obs_normed : array-like
        Observed data that has been normalized.
    err_normed : array-like
        Normalized uncertainties associated with the observed data.
    model_list : array-like
        List or array of model components to combine using the normalized weights.

    Returns
    -------
    float
        The log-probability value, which is the sum of the log-prior and log-likelihood.
        Returns -np.inf if the prior is not finite.
    """
    if not np.isfinite(log_prior(rel)):
        return -np.inf
    model = rel[:, None].T @ model_list
    residuals = (obs_normed - model) / err_normed
    log_likelihood = -0.5 * np.nansum(residuals**2 + np.log(2 * np.pi * err_normed**2))
    return log_likelihood


def residuals_function(rel, obs_normed, err_normed, model_list):
    """
    Calculate the residuals for a given set of relative weights, observed data, and model predictions.

    Parameters
    ----------
    rel : numpy.ndarray
        Array of relative weights for the models.
    obs_normed : numpy.ndarray
        Normalized observed data.
    err_normed : numpy.ndarray
        Normalized errors associated with the observed data.
    model_list : numpy.ndarray
        List of model predictions, where each model is represented as an array.

    Returns
    -------
    float
        The computed residuals value. Returns -np.inf if the prior is not finite.
    """
    if not np.isfinite(log_prior(rel)):
        return -np.inf
    model = rel[:, None].T @ model_list
    residuals = np.nansum(((obs_normed - model) / err_normed) ** 2.0)
    return residuals


# Load the GCNS data
gcns_wdlf = np.load("pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz")["data"]
gcns_wdlf_20pc_subset = gcns_wdlf[gcns_wdlf["dpc"] <= 20.0]  # Subset for stars within 20 parsecs

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.load("SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy")
mag_obs_optimal, resolution_optimal = np.load("SFH-WDLF-article/figure_data/mbol_resolution.npy").T

# Define magnitude bin edges based on resolution
mag_obs_optimal_bin_edges = np.append(
    mag_obs_optimal - resolution_optimal * 0.5,
    mag_obs_optimal[-1] + resolution_optimal[-1] * 0.5,
)

# Generate histograms for observed magnitudes and errors
h_gen_optimal, b_optimal = np.histogram(
    gcns_wdlf["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf["Vgen"],  # Weight by inverse volume
)

e_gen_optimal, _ = np.histogram(
    gcns_wdlf["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf["Vgen"] ** 2.0,  # Weight by squared inverse volume
)

# Generate histograms for the 20pc subset
h_gen_optimal_20pc_subset, b_optimal_20pc_subset = np.histogram(
    gcns_wdlf_20pc_subset["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf_20pc_subset["Vgen"],  # Weight by inverse volume
)

e_gen_optimal_20pc_subset, _ = np.histogram(
    gcns_wdlf_20pc_subset["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf_20pc_subset["Vgen"] ** 2.0,  # Weight by squared inverse volume
)

# Normalize observed WDLF data and errors
obs_wdlf_optimal = h_gen_optimal / resolution_optimal
obs_wdlf_err_optimal = e_gen_optimal**0.5 / resolution_optimal

# Normalize observed WDLF data and errors for the 20pc subset
obs_wdlf_optimal_20pc_subset = h_gen_optimal_20pc_subset / resolution_optimal
obs_wdlf_err_optimal_20pc_subset = e_gen_optimal_20pc_subset**0.5 / resolution_optimal

# Get the bootstrap sample index from command-line arguments
idx = int(sys.argv[1])
print(f"Processing bootstrap_sample_folder sample {idx}...")

# Define magnitude range for interpolation
Mag = np.arange(5.0, 18.0, 0.1)

# Load partial WDLF models and their corresponding ages and durations
(
    partial_age_optimal,
    partial_age_duration,
    _,
    _,
    _,
) = np.load("SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy").T

# Interpolate partial WDLF models to match observed magnitude bins
partial_wdlf_optimal = []
for age, duration in zip(partial_age_optimal, partial_age_duration):
    print(f"Currently loading {age} Gyr population.")
    pwdlf_mag, pwdlf = np.loadtxt(
        os.path.join(
            f"SFH-WDLF-article/bootstrap_sample_folder/sample_{idx}",
            f"montreal_co_da_20_C03_PARSECz0017_C08_{age:.4f}.csv",
        ),
        delimiter=",",
    ).T
    partial_wdlf_optimal.append(spectres(mag_obs_optimal, pwdlf_mag, pwdlf, fill=0.0))  # Interpolate to match bins

# Stack partial WDLF models and filter for valid observed data
pwdlf_model_optimal = np.vstack(partial_wdlf_optimal)
pwdlf_model_optimal_20pc_subset = np.vstack(partial_wdlf_optimal)
pwdlf_model_optimal /= np.nansum(pwdlf_model_optimal)
pwdlf_model_optimal_20pc_subset /= np.nansum(pwdlf_model_optimal_20pc_subset)
pwdlf_model_optimal = pwdlf_model_optimal[obs_wdlf_optimal > 0.0][:, obs_wdlf_optimal > 0.0]
mask_20pc = obs_wdlf_optimal_20pc_subset > 0.0
pwdlf_model_optimal_20pc_subset = pwdlf_model_optimal_20pc_subset[mask_20pc][:, mask_20pc]

# Normalize observed data and errors for optimization
obs_normed = obs_wdlf_optimal[obs_wdlf_optimal > 0.0]
obs_err_normed = obs_wdlf_err_optimal[obs_wdlf_optimal > 0.0]

# Normalize observed data and errors for the 20pc subset
obs_normed_20pc_subset = obs_wdlf_optimal_20pc_subset[obs_wdlf_optimal_20pc_subset > 0.0]
obs_err_normed_20pc_subset = obs_wdlf_err_optimal_20pc_subset[obs_wdlf_optimal_20pc_subset > 0.0]

# Load least squares solutions as starting points for optimization
input_lsq_res = np.load(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_lsq_solution.npy",
    allow_pickle=True,
).item()
input_lsq_res_20pc_subset = np.load(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_lsq_solution_20pc_subset.npy",
    allow_pickle=True,
).item()

# Extract solutions and Jacobians from least squares results
input_solution_optimal_lsq = input_lsq_res.x
input_solution_optimal_jac = input_lsq_res.jac

input_solution_optimal_lsq_20pc_subset = input_lsq_res_20pc_subset.x
input_solution_optimal_jac_20pc_subset = input_lsq_res_20pc_subset.jac

# Compute robust covariance matrices and uncertainties for the solutions
_, _s, _vh = np.linalg.svd(input_solution_optimal_jac, full_matrices=False)
tol = np.finfo(float).eps * _s[0] * max(input_solution_optimal_jac.shape)
_w = _s > tol
cov = (_vh[_w].T / _s[_w] ** 2) @ _vh[_w]  # Robust covariance matrix
stdev = np.sqrt(np.diag(cov))  # Standard deviations

_, _s_20pc_subset, _vh_20pc_subset = np.linalg.svd(input_solution_optimal_jac_20pc_subset, full_matrices=False)
tol_20pc_subset = np.finfo(float).eps * _s_20pc_subset[0] * max(input_solution_optimal_jac_20pc_subset.shape)
_w_20pc_subset = _s_20pc_subset > tol_20pc_subset
cov_20pc_subset = (_vh_20pc_subset[_w_20pc_subset].T / _s_20pc_subset[_w_20pc_subset] ** 2) @ _vh_20pc_subset[
    _w_20pc_subset
]  # Robust covariance matrix
stdev_20pc_subset = np.sqrt(np.diag(cov_20pc_subset))  # Standard deviations
stdev_20pc_subset[stdev_20pc_subset <= 1e-10] = np.sqrt(
    input_solution_optimal_lsq_20pc_subset[stdev_20pc_subset <= 1e-10]
)

initial_weights = input_solution_optimal_lsq
initial_weights_20pc_subset = input_solution_optimal_lsq_20pc_subset
initial_errors = stdev
initial_errors_20pc_subset = stdev_20pc_subset

n_step = 100000
n_burn = 10000
nwalkers_optimal = 250

ndim_optimal = len(partial_wdlf_optimal)
ndim_optimal_20pc_subset = len(pwdlf_model_optimal_20pc_subset)

rel_norm_optimal = np.vstack([np.random.normal(initial_weights, initial_errors) for _ in range(nwalkers_optimal)])

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


del sampler_optimal
del flat_samples_optimal

# Refine solutions using a minimizer
lsq_res = least_squares(
    residuals_function,
    solution_optimal,
    args=(
        obs_normed,
        obs_err_normed,
        pwdlf_model_optimal,
    ),
    ftol=3e-30,
    xtol=3e-10,
    gtol=3e-30,
    jac="cs",
    tr_solver="exact",
    verbose=2,
)

# Save refined solution for the full sample
np.save(
    f"SFH-WDLF-article/bootstrap_sample_folder/sample_{idx}/gcns_sfh_lsq_solution_sample_{idx}",
    lsq_res,
)

rel_norm_optimal_20pc_subset = np.vstack(
    [np.random.normal(initial_weights_20pc_subset, initial_errors_20pc_subset) for _ in range(nwalkers_optimal)]
)

sampler_optimal_20pc_subset = emcee.EnsembleSampler(
    nwalkers_optimal,
    ndim_optimal_20pc_subset,
    log_probability,
    args=(
        obs_normed_20pc_subset,
        obs_err_normed_20pc_subset,
        pwdlf_model_optimal_20pc_subset,
    ),
)
sampler_optimal_20pc_subset.run_mcmc(rel_norm_optimal_20pc_subset, n_step, progress=True)

flat_samples_optimal_20pc_subset = sampler_optimal_20pc_subset.get_chain(discard=n_burn, flat=True)

solution_optimal_20pc_subset = np.zeros(ndim_optimal_20pc_subset)
solution_lower_20pc_subset = np.zeros(ndim_optimal_20pc_subset)
solution_upper_20pc_subset = np.zeros(ndim_optimal_20pc_subset)
for i in range(ndim_optimal_20pc_subset):
    (
        solution_lower_20pc_subset[i],
        solution_optimal_20pc_subset[i],
        solution_upper_20pc_subset[i],
    ) = np.nanpercentile(flat_samples_optimal_20pc_subset[:, i], [31.7310508, 50.0, 68.2689492])

del sampler_optimal_20pc_subset
del flat_samples_optimal_20pc_subset

# Refine solutions for the 20pc subset using a minimizer
lsq_res_20pc_subset = least_squares(
    residuals_function,
    solution_optimal_20pc_subset,
    args=(
        obs_normed_20pc_subset,
        obs_err_normed_20pc_subset,
        pwdlf_model_optimal_20pc_subset,
    ),
    ftol=3e-30,
    xtol=3e-10,
    gtol=3e-30,
    jac="cs",
    tr_solver="exact",
    verbose=2,
)

# Save refined solution for the 20pc subset
np.save(
    f"SFH-WDLF-article/bootstrap_sample_folder/sample_{idx}/gcns_sfh_lsq_solution_20pc_subset_sample_{idx}.npy",
    lsq_res_20pc_subset,
)

# Extract refined solutions and compute uncertainties
solution_optimal = lsq_res.x
solution_optimal_jac = lsq_res.jac
solution_optimal_20pc_subset = lsq_res_20pc_subset.x
solution_optimal_jac_20pc_subset = lsq_res_20pc_subset.jac

# Compute robust covariance matrices and uncertainties for refined solutions
_, _s, _vh = np.linalg.svd(solution_optimal_jac, full_matrices=False)
tol = np.finfo(float).eps * _s[0] * max(solution_optimal_jac.shape)
_w = _s > tol
cov = (_vh[_w].T / _s[_w] ** 2) @ _vh[_w]  # Robust covariance matrix
stdev = np.sqrt(np.diag(cov))  # Standard deviations

_, _s_20pc_subset, _vh_20pc_subset = np.linalg.svd(solution_optimal_jac_20pc_subset, full_matrices=False)
tol_20pc_subset = np.finfo(float).eps * _s_20pc_subset[0] * max(solution_optimal_jac_20pc_subset.shape)
_w_20pc_subset = _s_20pc_subset > tol_20pc_subset
cov_20pc_subset = (_vh_20pc_subset[_w_20pc_subset].T / _s_20pc_subset[_w_20pc_subset] ** 2) @ _vh_20pc_subset[
    _w_20pc_subset
]  # Robust covariance matrix
stdev_20pc_subset = np.sqrt(np.diag(cov_20pc_subset))  # Standard deviations

# Translate solutions to 1-sigma uncertainties
solution_lower = solution_optimal - stdev
solution_upper = solution_optimal + stdev
solution_lower_20pc_subset = solution_optimal_20pc_subset - stdev_20pc_subset
solution_upper_20pc_subset = solution_optimal_20pc_subset + stdev_20pc_subset

# Reconstruct WDLF using refined solutions
recomputed_wdlf_optimal_lsq = np.nansum(solution_optimal * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_lsq_20pc_subset = np.nansum(
    solution_optimal_20pc_subset * np.array(pwdlf_model_optimal_20pc_subset).T, axis=1
)

# Compute WDLF uncertainties
wdlf_err_low = np.nansum((solution_lower) * np.array(partial_wdlf_optimal).T, axis=1)
wdlf_err_high = np.nansum((solution_upper) * np.array(partial_wdlf_optimal).T, axis=1)
wdlf_err_low_20pc_subset = np.nansum((solution_lower_20pc_subset) * np.array(pwdlf_model_optimal_20pc_subset).T, axis=1)
wdlf_err_high_20pc_subset = np.nansum(
    (solution_upper_20pc_subset) * np.array(pwdlf_model_optimal_20pc_subset).T, axis=1
)

# Normalize solutions to compute star formation history (SFH)
normalisation_this_work = np.sum(obs_wdlf_optimal) / np.sum(recomputed_wdlf_optimal_lsq)
normalisation_this_work_20pc_subset = np.sum(obs_wdlf_optimal_20pc_subset) / np.sum(
    recomputed_wdlf_optimal_lsq_20pc_subset
)
bin_norm_this_work = np.concatenate(
    [
        [partial_age_optimal[1] - partial_age_optimal[0]],
        (np.diff(partial_age_optimal)[:-1] + np.diff(partial_age_optimal)[1:]) / 2.0,
        [partial_age_optimal[-1] - partial_age_optimal[-2]],
    ]
)

# Compute SFH and its uncertainties
sfh_lsq = solution_optimal
sfh_err_lower = solution_lower
sfh_err_upper = solution_upper
# Full-length arrays with zeros
sfh_lsq_20pc_full = np.zeros_like(partial_age_optimal)
sfh_err_lower_20pc_full = np.zeros_like(partial_age_optimal)
sfh_err_upper_20pc_full = np.zeros_like(partial_age_optimal)

# Fill only the bins where data was nonzero
sfh_lsq_20pc_full[mask_20pc] = solution_optimal_20pc_subset
sfh_err_lower_20pc_full[mask_20pc] = solution_lower_20pc_subset
sfh_err_upper_20pc_full[mask_20pc] = solution_upper_20pc_subset

# Save SFH results to a CSV file
sfh_csv_output = np.column_stack(
    [
        partial_age_optimal,
        sfh_lsq,
        sfh_err_lower,
        sfh_err_upper,
        sfh_lsq_20pc_full,
        sfh_err_lower_20pc_full,
        sfh_err_upper_20pc_full,
    ]
)

np.savetxt(
    f"SFH-WDLF-article/bootstrap_sample_folder/sample_{idx}/gcns_sfh_sample_{idx}.csv",
    sfh_csv_output,
)

# Prepare to output CSV of the reconstructed WDLFs
wdlf_output = recomputed_wdlf_optimal_lsq
wdlf_err_output = obs_wdlf_optimal_20pc_subset


# Full-length arrays with zeros, matching the magnitude bins
wdlf_20_full = np.zeros_like(mag_obs_optimal)
wdlf_20_err_full = np.zeros_like(mag_obs_optimal)

# Fill only bins where you had nonzero data
mask_mag = obs_wdlf_optimal_20pc_subset > 0.0
wdlf_20_full[mask_mag] = recomputed_wdlf_optimal_lsq_20pc_subset
wdlf_20_err_full[mask_mag] = obs_wdlf_optimal_20pc_subset[mask_mag]

# Save reconstructed WDLF results to a CSV file
wdlf_csv_output = np.column_stack(
    [
        mag_obs_optimal,
        wdlf_output,
        wdlf_err_output,
        wdlf_20_full,
        wdlf_20_err_full,
    ]
)

np.savetxt(
    f"SFH-WDLF-article/bootstrap_sample_folder/sample_{idx}/gcns_reconstructed_wdlf_sample_{idx}.csv",
    wdlf_csv_output,
)
