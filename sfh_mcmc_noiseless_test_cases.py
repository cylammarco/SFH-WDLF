import os

import emcee
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
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


mag_pwdlf = data[0][:, 0][::5]


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


wdlf = theoretical_lf.WDLF()
wdlf.set_sfr_model(mode="manual", sfr_model=sfh_spline)
wdlf.compute_density(mag_pwdlf)


mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag_pwdlf[np.argmax(d[:, 1][::5])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=50, k=5
)


bin_05 = []
bin_05_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
bin_05_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
bin_number = 0
stop = False
bin_05.append(mag_resolution_itp(age[0]))
for i, a in enumerate(age):
    if i < j:
        continue
    start = mag_resolution_itp(a)
    end = start
    j = i
    carry_on = True
    while carry_on:
        tmp = mag_resolution_itp(age[j + 1]) - mag_resolution_itp(age[j])
        if end + tmp - start < 0.5:
            end = end + tmp
            bin_05_pwdlf[j] = bin_number
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    if stop:
        break
    bin_05.append(end)
    bin_05_idx.append(i)


pwdlf_mapping_bin_05 = np.insert(bin_05_pwdlf, 0, 0)


partial_wdlf_05 = []
partial_age_05 = []
for idx in np.sort(list(set(pwdlf_mapping_bin_05))):
    pwdlf_temp = np.zeros_like(mag_pwdlf)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_05 == idx)[0]:
        pwdlf_temp += spectres(mag_pwdlf, data[i][:, 0], data[i][:, 1], fill=0.0)
        age_temp += age[i]
        age_count += 1
    partial_wdlf_05.append(pwdlf_temp)
    partial_age_05.append(age_temp / age_count)


pwdlf_model_05 = np.vstack(partial_wdlf_05)[:, wdlf.number_density > 0.0]
pwdlf_model_05 /= np.nansum(pwdlf_model_05) * np.nanmax(wdlf.number_density)

nwalkers_05 = 100

ndim_05 = len(partial_wdlf_05)

rel_norm_05 = np.random.rand(nwalkers_05, ndim_05)

sampler_05 = emcee.EnsembleSampler(
    nwalkers_05,
    ndim_05,
    log_probability,
    args=(
        wdlf.number_density[wdlf.number_density > 0.0],
        wdlf.number_density[wdlf.number_density > 0.0] * 0.1,
        pwdlf_model_05,
    ),
)
sampler_05.run_mcmc(rel_norm_05, 20000, progress=True)

flat_samples_05 = sampler_05.get_chain(discard=2000, thin=5, flat=True)

solution_05 = np.zeros(ndim_05)
for i in range(ndim_05):
    solution_05[i] = np.percentile(flat_samples_05[:, i], [50.0])

solution_05 /= np.nanmax(solution_05)

recomputed_wdlf_05 = np.nansum(
    solution_05 * np.array(partial_wdlf_05).T, axis=1
)



fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(8, 1, 5)
)

ax_dummy1.axis("off")
ax1.plot(
    wdlf.mag,
    wdlf.number_density,
    label="Input WDLF",
)
ax1.plot(
    mag_pwdlf,
    recomputed_wdlf_05
    / np.nansum(recomputed_wdlf_05)
    * np.nansum(wdlf.number_density),
    label="Reconstructed WDLF",
)
ax1.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
ax1.set_ylabel("log(arbitrary number density)")
ax1.set_xlim(2.0, 18.0)
ax1.set_ylim(1e-5, 5e-1)
ax1.set_yscale("log")
ax1.legend()
ax1.grid()

ax2.step(partial_age_05, solution_05, where="post", label="MCMC")
ax2.plot(time / 1e9, sfh_total/np.nanmax(sfh_total), label="Input")
ax2.grid()
ax2.set_xticks(np.arange(0, 15, 2))
ax2.set_xlim(0, 15)
ax2.set_ylim(bottom=0)
ax2.set_xlabel("Lookback time / Gyr")
ax2.set_ylabel("Relative Star Formation Rate")

plt.subplots_adjust(
    top=0.975, bottom=0.05, left=0.09, right=0.975, hspace=0.075
)


plt.savefig(
    os.path.join(figure_folder, "fig_02_two_bursts_wdlf.png")
)
