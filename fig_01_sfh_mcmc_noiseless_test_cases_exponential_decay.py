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
    model = np.nansum(rel_norm[:, None] * model_list, axis=0)
    model /= np.nansum(model)
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


mag_pwdlf = data[0][:, 0][::2]


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


time = np.arange(0.001, 15.0, 0.001) * 1e9
sfh_1 = gaus(time, 1.0, 0.0, 2.0e9, 0.25e9)
sfh_2 = gaus(time, 0.3, 0.0, 8.0e9, 1.5e9)
sfh_total = sfh_1 + sfh_2

sfh_spline = interpolate.UnivariateSpline(time, sfh_total, s=0)


mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag_pwdlf[np.argmax(d[:, 1][::2])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=50, k=5
)

# Load the "resolution" of the Mbol solution
mbol_err_upper_bound = interpolate.UnivariateSpline(
    *np.load("mbol_err_upper_bound.npy").T, s=0
)

bin_02 = []
bin_02_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
bin_02_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
bin_number = 0
stop = False
bin_02.append(mag_resolution_itp(age[0]))
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
            bin_02_pwdlf[j] = bin_number
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    bin_02.append(end)
    bin_02_idx.append(i)
    if stop:
        break


bin_02 = np.array(bin_02)
pwdlf_mapping_bin_02 = np.insert(bin_02_pwdlf, 0, 0)


wdlf = theoretical_lf.WDLF()
wdlf.set_sfr_model(mode="manual", sfr_model=sfh_spline)
wdlf.compute_density(bin_02)


partial_wdlf_02 = []
partial_age_02 = []
for idx in np.sort(list(set(pwdlf_mapping_bin_02))):
    pwdlf_temp = np.zeros_like(bin_02)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_02 == idx)[0]:
        pwdlf_temp += spectres(bin_02, data[i][:, 0], data[i][:, 1], fill=0.0)
        age_temp += age[i]
        age_count += 1
    partial_wdlf_02.append(pwdlf_temp)
    partial_age_02.append(age_temp / age_count)


partial_wdlf_02 = np.array(partial_wdlf_02)
pwdlf_model_02 = np.vstack(partial_wdlf_02)[:, wdlf.number_density > 0.0]
pwdlf_model_02 /= np.nansum(pwdlf_model_02) * np.nanmax(wdlf.number_density)

nwalkers_02 = 200
ndim_02 = len(partial_wdlf_02)
init_guess = np.array(
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
    ]
)
rel_norm_02 = (
    (np.random.rand(nwalkers_02, ndim_02) - 0.5) * 0.01 + 1
) * init_guess

for i in range(10):
    sampler_02 = emcee.EnsembleSampler(
        nwalkers_02,
        ndim_02,
        log_probability,
        args=(
            wdlf.number_density[wdlf.number_density > 0.0],
            wdlf.number_density[wdlf.number_density > 0.0] * 0.1,
            pwdlf_model_02,
        ),
    )
    sampler_02.run_mcmc(rel_norm_02, 10000, progress=True)
    flat_samples_02 = sampler_02.get_chain(discard=1000, flat=True)
    solution_02 = np.zeros(ndim_02)
    for i in range(ndim_02):
        solution_02[i] = np.percentile(flat_samples_02[:, i], [50.0])
    solution_02 /= np.nanmax(solution_02)
    rel_norm_02 = (
        (np.random.rand(nwalkers_02, ndim_02) - 0.5) * 0.01 + 1
    ) * solution_02
    recomputed_wdlf_02 = np.nansum(
        solution_02 * np.array(partial_wdlf_02).T, axis=1
    )
    fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
        nrows=3, ncols=1, figsize=(8, 10), height_ratios=(5, 1, 8)
    )
    ax_dummy1.axis("off")
    ax2.plot(
        wdlf.mag,
        wdlf.number_density,
        label="Input WDLF",
    )
    ax2.plot(
        bin_02,
        recomputed_wdlf_02
        / np.nansum(recomputed_wdlf_02)
        * np.nansum(wdlf.number_density),
        label="Reconstructed WDLF",
    )
    ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
    ax2.set_ylabel("log(arbitrary number density)")
    ax2.set_xlim(6.0, 18.0)
    ax2.set_ylim(1e-5, 5e-1)
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid()
    ax1.plot(time / 1e9, sfh_total / np.nanmax(sfh_total), label="Input")
    ax1.step(partial_age_02, solution_02, where="post", label="MCMC")
    ax1.grid()
    ax1.set_xticks(np.arange(0, 15, 2))
    ax1.set_xlim(0, 15)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Lookback time / Gyr")
    ax1.set_ylabel("Relative Star Formation Rate")
    ax1.legend()
    plt.subplots_adjust(
        top=0.975, bottom=0.075, left=0.125, right=0.975, hspace=0.075
    )
    plt.savefig(os.path.join(figure_folder, "fig_02_two_bursts_wdlf.png"))


sfh_mcmc_lower = np.zeros(ndim_02)
sfh_mcmc = np.zeros(ndim_02)
sfh_mcmc_upper = np.zeros(ndim_02)
for i in range(ndim_02):
    sfh_mcmc_lower[i], sfh_mcmc[i], sfh_mcmc_upper[i] = np.percentile(flat_samples_02[:, i], [31.7310508, 50.0, 68.2689492])


sfh_mcmc_lower /= np.nanmax(sfh_mcmc)
sfh_mcmc_upper /= np.nanmax(sfh_mcmc)


# Finally refining with a minimizer
solution_02_lsq = least_squares(
    log_probability,
    solution_02,
    args=(
        wdlf.number_density[wdlf.number_density > 0.0],
        wdlf.number_density[wdlf.number_density > 0.0] * 0.1,
        pwdlf_model_02,
    ),
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-12,
).x
solution_02_lsq /= max(solution_02_lsq)

fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(5, 1, 8)
)
ax_dummy1.axis("off")
ax2.plot(
    wdlf.mag,
    wdlf.number_density,
    label="Input WDLF",
)
ax2.plot(
    bin_02,
    recomputed_wdlf_02
    / np.nansum(recomputed_wdlf_02)
    * np.nansum(wdlf.number_density),
    label="Reconstructed WDLF (MCMC)",
)
ax2.plot(
    bin_02,
    recomputed_wdlf_02
    / np.nansum(recomputed_wdlf_02)
    * np.nansum(wdlf.number_density),
    label="Reconstructed WDLF (lsq)",
)
ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_xlim(6.0, 18.0)
ax2.set_ylim(1e-5, 5e-1)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
ax1.plot(time / 1e9, sfh_total / np.nanmax(sfh_total), label="Input")
ax1.fill_between(partial_age_02, sfh_mcmc_lower, sfh_mcmc_upper, step="mid", color='lightgrey')
ax1.step(partial_age_02, solution_02, where="mid", label="MCMC")
ax1.step(partial_age_02, solution_02_lsq, where="mid", label="lsq")
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
plt.savefig(os.path.join(figure_folder, "fig_01_exponential_decay_wdlf.png"))
