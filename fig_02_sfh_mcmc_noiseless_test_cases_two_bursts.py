import os

import emcee
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import least_squares
from spectres import spectres
from WDPhotTools import theoretical_lf


figure_folder = "SFH-WDLF-article/figures"


def log_prior(theta):
    if (theta <= 0.0).any():
        return -np.inf
    else:
        return 0.0


def log_probability(rel_norm, obs, err, model_list):
    rel_norm /= np.sum(rel_norm)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = rel_norm[:, None].T @ model_list
    model /= np.nansum(model)
    err /= np.nansum(obs)
    obs /= np.nansum(obs)
    log_likelihood = -np.nansum(((model - obs) / err) ** 2.0)
    return log_likelihood


# Load the pwdlfs
data = []
age_list_1 = np.arange(0.049, 0.100, 0.001)
age_list_2 = np.arange(0.100, 0.350, 0.005)
age_list_3 = np.arange(0.35, 15.01, 0.01)
age_list_3dp = np.concatenate((age_list_1, age_list_2))
age_list_2dp = age_list_3

age = np.concatenate((age_list_3dp, age_list_2dp))

for i in age_list_3dp:
    print(i)
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
    print(i)
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


###############################################################################
#
# Two Bursts - an intensity burst at 2 Gyr, a broader one one at 8 Gyr
#
###############################################################################


def gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function.
    Parameters
    ----------
    x: float or 1-d numpy array
        The data to evaluate the Gaussian over
    a: float
        the amplitude
    b: float
        the constant offset
    x0: float
        the center of the Gaussian
    sigma: float
        the width of the Gaussian
    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + b


sfh_1 = gaus(age * 1e9, 1.0, 0.0, 2.0e9, 0.25e9)
sfh_2 = gaus(age * 1e9, 0.3, 0.0, 8.0e9, 1.5e9)
sfh_total_1 = sfh_1 + sfh_2
sfh_total_2 = np.roll(max(sfh_total_1) - sfh_total_1, 200)

wdlf_reconstructed_1 = np.array(data)[:, :, 1].T @ sfh_total_1
wdlf_reconstructed_2 = np.array(data)[:, :, 1].T @ sfh_total_2

mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag_pwdlf[np.argmax(d[:, 1])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=50, k=5
)

# Load the "resolution" of the Mbol solution
mbol_err_upper_bound = interpolate.UnivariateSpline(
    *np.load("mbol_err_upper_bound.npy").T, s=0
)

mag_bin = []
mag_bin_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
mag_bin_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
bin_number = 0
stop = False
mag_bin.append(mag_resolution_itp(age[0]))
for i, a in enumerate(age):
    if i < j:
        continue
    start = float(mag_resolution_itp(a))
    end = start
    j = i
    carry_on = True
    while carry_on:
        tmp = mag_resolution_itp(age[j + 1]) - mag_resolution_itp(age[j])
        print(j, end + tmp - start)
        if end + tmp - start < mbol_err_upper_bound(end):
            end = end + tmp
            mag_bin_pwdlf[j] = bin_number
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    mag_bin.append(end)
    mag_bin_idx.append(i)
    if stop:
        break


# these the the bin edges
mag_bin = np.array(mag_bin)
# get the bin centre
mag_bin = (mag_bin[1:] + mag_bin[:-1]) / 2.0
pwdlf_mapping_mag_bin = np.insert(mag_bin_pwdlf, 0, 0)

wdlf_number_density_1 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_1)
wdlf_number_density_1 /= np.sum(wdlf_number_density_1)

wdlf_number_density_2 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_2)
wdlf_number_density_2 /= np.sum(wdlf_number_density_2)

partial_wdlf = []
partial_age = []
for idx in np.sort(list(set(pwdlf_mapping_mag_bin))):
    pwdlf_temp = np.zeros_like(mag_bin)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_mag_bin == idx)[0]:
        pwdlf_temp += spectres(mag_bin, data[i][:, 0], data[i][:, 1], fill=0.0)
        age_temp += age[i]
        age_count += 1
    partial_wdlf.append(pwdlf_temp)
    partial_age.append(age_temp / age_count)


partial_wdlf = np.array(partial_wdlf)
pwdlf_model_1 = np.vstack(partial_wdlf)[:, wdlf_number_density_1 > 0.0]
pwdlf_model_1 /= np.nansum(pwdlf_model_1) * np.nanmax(wdlf_number_density_1)

pwdlf_model_2 = np.vstack(partial_wdlf)[:, wdlf_number_density_2 > 0.0]
pwdlf_model_2 /= np.nansum(pwdlf_model_2) * np.nanmax(wdlf_number_density_2)

nwalkers = 250
ndim = len(partial_wdlf)
init_guess_1 = np.array(
    [
        9.81417526e-03,
        1.27969221e-02,
        1.41141975e-02,
        1.48105413e-02,
        1.46395567e-02,
        2.16384356e-02,
        9.99894236e-03,
        2.73840308e-02,
        3.31477890e-02,
        4.09304067e-02,
        8.25748037e-02,
        2.21109058e-01,
        1.00000000e00,
        2.73090378e-01,
        1.21694619e-02,
        4.63203854e-03,
        4.88537981e-03,
        5.04835745e-03,
        5.79464416e-03,
        6.04115033e-03,
        8.78760602e-03,
        1.19281205e-02,
        1.33335421e-02,
        5.10642775e-03,
        1.46924653e-02,
        1.01862149e-02,
        1.81843569e-02,
        2.94361142e-01,
        1.00293679e-01,
        2.45284441e-01,
        4.16589324e-02,
        1.20664300e-01,
        2.77868818e-02,
        1.79736194e-02,
        4.57930336e-03,
        3.41267151e-03,
        6.15811884e-04,
        6.36172284e-04,
        6.52652135e-04,
        7.08722919e-04,
        4.20770290e-12,
        5.83204368e-04,
        9.00507224e-04,
        5.55788804e-04,
        1.15122275e-03,
        8.51692291e-04,
        1.04879422e-03,
        1.22956217e-03,
        7.56464406e-04,
        3.13729771e-03,
        3.13729771e-03,
    ]
)
init_guess_2 = np.array(
    [
        9.81417526e-03,
        1.27969221e-02,
        1.41141975e-02,
        1.48105413e-02,
        1.46395567e-02,
        2.16384356e-02,
        9.99894236e-03,
        2.73840308e-02,
        3.31477890e-02,
        4.09304067e-02,
        8.25748037e-02,
        2.21109058e-01,
        1.00000000e00,
        2.73090378e-01,
        1.21694619e-02,
        4.63203854e-03,
        4.88537981e-03,
        5.04835745e-03,
        5.79464416e-03,
        6.04115033e-03,
        8.78760602e-03,
        1.19281205e-02,
        1.33335421e-02,
        5.10642775e-03,
        1.46924653e-02,
        1.01862149e-02,
        1.81843569e-02,
        2.94361142e-01,
        1.00293679e-01,
        2.45284441e-01,
        4.16589324e-02,
        1.20664300e-01,
        2.77868818e-02,
        1.79736194e-02,
        4.57930336e-03,
        3.41267151e-03,
        6.15811884e-04,
        6.36172284e-04,
        6.52652135e-04,
        7.08722919e-04,
        4.20770290e-12,
        5.83204368e-04,
        9.00507224e-04,
        5.55788804e-04,
        1.15122275e-03,
        8.51692291e-04,
        1.04879422e-03,
        1.22956217e-03,
        7.56464406e-04,
        3.13729771e-03,
        3.13729771e-03,
    ]
)

rel_norm_1 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_1
rel_norm_2 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_2

n_wd_1 = wdlf_number_density_1[wdlf_number_density_1 > 0.0]
n_wd_2 = wdlf_number_density_2[wdlf_number_density_2 > 0.0]

n_wd_err_1 = n_wd_1 * 0.1
n_wd_err_2 = n_wd_2 * 0.1

for i in range(10):
    sampler_1 = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            n_wd_1,
            n_wd_err_1,
            pwdlf_model_1,
        ),
    )
    sampler_2 = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            n_wd_2,
            n_wd_err_2,
            pwdlf_model_2,
        ),
    )
    #
    sampler_1.run_mcmc(rel_norm_1, 10000, progress=True)
    sampler_2.run_mcmc(rel_norm_2, 10000, progress=True)
    #
    flat_samples_1 = sampler_1.get_chain(discard=1000, flat=True)
    flat_samples_2 = sampler_2.get_chain(discard=1000, flat=True)
    #
    solution_1 = np.zeros(ndim)
    solution_2 = np.zeros(ndim)
    for i in range(ndim):
        solution_1[i] = np.percentile(flat_samples_1[:, i], [50.0])
        solution_2[i] = np.percentile(flat_samples_2[:, i], [50.0])
    solution_1 /= np.nanmax(solution_1)
    solution_2 /= np.nanmax(solution_2)
    #
    rel_norm_1 = (
        (np.random.rand(nwalkers, ndim) - 0.5) * 0.01 + 1
    ) * solution_1
    rel_norm_2 = (
        (np.random.rand(nwalkers, ndim) - 0.5) * 0.01 + 1
    ) * solution_2
    #
    recomputed_wdlf_1 = np.nansum(
        solution_1 * np.array(partial_wdlf).T, axis=1
    )
    recomputed_wdlf_2 = np.nansum(
        solution_2 * np.array(partial_wdlf).T, axis=1
    )
    #
    fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
        nrows=3, ncols=1, figsize=(8, 10), height_ratios=(5, 1, 8)
    )
    #
    ax1.plot(
        age,
        sfh_total_1 / np.nanmax(sfh_total_1),
        ls=":",
        color="C0",
    )
    ax1.plot(
        age,
        sfh_total_2 / np.nanmax(sfh_total_2),
        ls=":",
        color="C1",
    )
    #
    ax1.step(partial_age, solution_1, where="post", color="C0")
    ax1.step(partial_age, solution_2, where="post", color="C1")
    #
    ax1.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
    ax1.plot([-1, -1], [-1, -2], ls="dashed", color="black", label="MCMC")
    #
    ax1.grid()
    ax1.set_xticks(np.arange(0, 15, 2))
    ax1.set_xlim(0, 15)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Lookback time / Gyr")
    ax1.set_ylabel("Relative Star Formation Rate")
    ax1.legend()
    #
    ax_dummy1.axis("off")
    #
    ax2.plot(mag_bin, n_wd_1, ls=":", color="C0")
    ax2.plot(mag_bin, n_wd_2, ls=":", color="C1")
    #
    ax2.plot(
        mag_bin,
        recomputed_wdlf_1 / np.nansum(recomputed_wdlf_1) * np.nansum(n_wd_1),
        color="C0",
    )
    ax2.plot(
        mag_bin,
        recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2) * np.nansum(n_wd_2),
        color="C1",
    )
    #
    ax2.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
    ax2.plot([-1, -1], [-1, -2], ls="-", color="black", label="Reconstructed")
    #
    ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
    ax2.set_ylabel("log(arbitrary number density)")
    ax2.set_xlim(6.75, 18.25)
    ax2.set_ylim(1e-5, 5e-1)
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid()
    #
    plt.subplots_adjust(
        top=0.975, bottom=0.075, left=0.125, right=0.975, hspace=0.075
    )
    plt.savefig(os.path.join(figure_folder, "fig_02_two_bursts_wdlf.png"))


sfh_mcmc_lower_1 = np.zeros(ndim)
sfh_mcmc_lower_2 = np.zeros(ndim)
#
sfh_mcmc_1 = np.zeros(ndim)
sfh_mcmc_2 = np.zeros(ndim)
#
sfh_mcmc_upper_1 = np.zeros(ndim)
sfh_mcmc_upper_2 = np.zeros(ndim)
for i in range(ndim):
    sfh_mcmc_lower_1[i], sfh_mcmc_1[i], sfh_mcmc_upper_1[i] = np.percentile(
        flat_samples_1[:, i], [31.7310508, 50.0, 68.2689492]
    )
    sfh_mcmc_lower_2[i], sfh_mcmc_2[i], sfh_mcmc_upper_2[i] = np.percentile(
        flat_samples_2[:, i], [31.7310508, 50.0, 68.2689492]
    )


sfh_mcmc_1 /= np.nanmax(sfh_mcmc_1)
sfh_mcmc_2 /= np.nanmedian(sfh_mcmc_2)

sfh_mcmc_lower_1 /= np.nanmax(sfh_mcmc_1)
sfh_mcmc_lower_2 /= np.nanmedian(sfh_mcmc_2)

sfh_mcmc_upper_1 /= np.nanmax(sfh_mcmc_1)
sfh_mcmc_upper_2 /= np.nanmedian(sfh_mcmc_2)


# Finally refining with a minimizer
solution_lsq_1 = least_squares(
    log_probability,
    solution_1,
    args=(
        n_wd_1,
        n_wd_err_1,
        pwdlf_model_1,
    ),
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
).x
solution_lsq_2 = least_squares(
    log_probability,
    solution_2,
    args=(
        n_wd_2,
        n_wd_err_2,
        pwdlf_model_2,
    ),
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
).x
solution_lsq_1 /= np.nanmax(solution_lsq_1)
solution_lsq_2 /= np.nanmedian(solution_lsq_2)

recomputed_wdlf_lsq_1 = np.nansum(
    solution_lsq_1 * np.array(partial_wdlf).T, axis=1
)
recomputed_wdlf_lsq_2 = np.nansum(
    solution_lsq_2 * np.array(partial_wdlf).T, axis=1
)


np.save("fig_02_partial_wdlf", partial_wdlf)

np.save("fig_02_sfh_total_1", sfh_1)
np.save("fig_02_sfh_total_2", sfh_2)

np.save("fig_02_input_wdlf_1", wdlf_reconstructed_1)
np.save("fig_02_input_wdlf_2", wdlf_reconstructed_2)

np.save("fig_02_solution_1", np.column_stack([partial_age, solution_1]))
np.save("fig_02_solution_2", np.column_stack([partial_age, solution_2]))

np.save(
    "fig_02_solution_lsq_1", np.column_stack([partial_age, solution_lsq_1])
)
np.save(
    "fig_02_solution_lsq_2", np.column_stack([partial_age, solution_lsq_2])
)

np.save(
    "fig_02_recomputed_wdlf_mcmc_1",
    np.column_stack([mag_bin, recomputed_wdlf_1]),
)
np.save(
    "fig_02_recomputed_wdlf_mcmc_2",
    np.column_stack([mag_bin, recomputed_wdlf_2]),
)

np.save(
    "fig_02_recomputed_wdlf_lsq_1",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_1]),
)
np.save(
    "fig_02_recomputed_wdlf_lsq_2",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_2]),
)


fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(5, 1, 8)
)
ax_dummy1.axis("off")
#
ax2.plot(mag_bin, n_wd_1, ls=":", color="C0")
ax2.plot(mag_bin, n_wd_2, ls=":", color="C1")
#
ax2.plot(
    mag_bin,
    recomputed_wdlf_1 / np.nansum(recomputed_wdlf_1) * np.nansum(n_wd_1),
    ls="dashed",
    color="C0",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2) * np.nansum(n_wd_2),
    ls="dashed",
    color="C1",
)
#
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_1
    / np.nansum(recomputed_wdlf_lsq_1)
    * np.nansum(n_wd_1),
    color="C0",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_2
    / np.nansum(recomputed_wdlf_lsq_2)
    * np.nansum(n_wd_2),
    color="C1",
)
#
ax2.plot([1, 1], [1, 2], ls=":", color="black", label="Input")
ax2.plot(
    [1, 1], [1, 2], ls="dashed", color="black", label="Reconstructed (MCMC)"
)
ax2.plot([1, 1], [1, 2], ls="-", color="black", label="Reconstructed (lsq)")
ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_xlim(6.75, 18.25)
ax2.set_ylim(1e-4, 5e-1)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
ax1.plot(age, sfh_total_1 / np.nanmax(sfh_total_1), ls=":", color="C0")
ax1.plot(age, sfh_total_2 / np.nanmedian(sfh_total_2), ls=":", color="C1")
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_1 - sfh_mcmc_1 + solution_lsq_1,
    sfh_mcmc_upper_1 - sfh_mcmc_1 + solution_lsq_1,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_2 - sfh_mcmc_2 + solution_lsq_2,
    sfh_mcmc_upper_2 - sfh_mcmc_2 + solution_lsq_2,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.step(partial_age, sfh_mcmc_1, ls="dashed", where="mid", color="C0")
ax1.step(partial_age, sfh_mcmc_2, ls="dashed", where="mid", color="C1")
ax1.step(partial_age, solution_lsq_1, where="mid", color="C0")
ax1.step(partial_age, solution_lsq_2, where="mid", color="C1")
ax1.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
ax1.plot([-1, -1], [-1, -2], ls="dashed", color="black", label="MCMC")
ax1.plot([-1, -1], [-1, -2], ls="-", color="black", label="lsq-refined")
ax1.grid()
ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 15)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Lookback time / Gyr")
ax1.set_ylabel("Relative SFR")
ax1.legend()
plt.subplots_adjust(
    top=0.975, bottom=0.075, left=0.125, right=0.975, hspace=0.075
)
plt.savefig(os.path.join(figure_folder, "fig_02_two_bursts_wdlf.png"))
