import os

from matplotlib import pyplot as plt
import numpy as np
from pynverse import inversefunc
from scipy.signal import medfilt
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader

figure_folder = "SFH-WDLF-article/figures"
plt.rcParams.update({"font.size": 12})

Mbol = np.load(
    "SFH-WDLF-article/figure_data/Mbol_recomputed.npy", allow_pickle=True
).item()
Mbol_upper_1_sigma = np.load(
    "SFH-WDLF-article/figure_data/Mbol_upper_1_sigma_recomputed.npy",
    allow_pickle=True,
).item()
Mbol_upper_2_sigma = np.load(
    "SFH-WDLF-article/figure_data/Mbol_upper_2_sigma_recomputed.npy",
    allow_pickle=True,
).item()
Mbol_upper_3_sigma = np.load(
    "SFH-WDLF-article/figure_data/Mbol_upper_3_sigma_recomputed.npy",
    allow_pickle=True,
).item()
Mbol_lower_1_sigma = np.load(
    "SFH-WDLF-article/figure_data/Mbol_lower_1_sigma_recomputed.npy",
    allow_pickle=True,
).item()
Mbol_lower_2_sigma = np.load(
    "SFH-WDLF-article/figure_data/Mbol_lower_2_sigma_recomputed.npy",
    allow_pickle=True,
).item()
Mbol_lower_3_sigma = np.load(
    "SFH-WDLF-article/figure_data/Mbol_lower_3_sigma_recomputed.npy",
    allow_pickle=True,
).item()
Mbol_err = np.load(
    "SFH-WDLF-article/figure_data/Mbol_error_recomputed.npy", allow_pickle=True
).item()


(
    _mbol_err,
    _mbol,
    _mbol_upper_1_sigma,
    _mbol_upper_2_sigma,
    _mbol_upper_3_sigma,
    _mbol_lower_1_sigma,
    _mbol_lower_2_sigma,
    _mbol_lower_3_sigma,
) = np.array(
    [
        (
            Mbol_err[k],
            v,
            Mbol_upper_1_sigma[k],
            Mbol_upper_2_sigma[k],
            Mbol_upper_3_sigma[k],
            Mbol_lower_1_sigma[k],
            Mbol_lower_2_sigma[k],
            Mbol_lower_3_sigma[k],
        )
        for k, v in Mbol.items()
        if v < 18.0
    ]
).T
_arg = np.argsort(_mbol)


# get the total error
mbol_diff = (
    (_mbol - _mbol_upper_1_sigma)
    + (_mbol - _mbol_upper_2_sigma) / 2.0
    + (_mbol - _mbol_upper_3_sigma) / 3.0
    + (_mbol_lower_1_sigma - _mbol)
    + (_mbol_lower_2_sigma - _mbol) / 2.0
    + (_mbol_lower_3_sigma - _mbol) / 3.0
) / 6.0

# mbol_diff = np.abs(_mbol_lower_1_sigma - _mbol_upper_1_sigma).astype("float") / 2.0

mbol_total_err = np.sqrt(mbol_diff**2.0 + _mbol_err.astype("float") ** 2.0)
mbol_total_err = mbol_total_err[_arg]

_mbol = _mbol[_arg]

# get the precomputed envelope
mag_mean, err_mean = np.load(
    "SFH-WDLF-article/figure_data/mbol_err_mean.npy"
).T
err_mean[mag_mean > 16.5] = err_mean[mag_mean <= 16.5][-1]


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


mag = data[0][:, 0]

mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag[np.argmax(d[:, 1])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=len(age) / 150, k=5
)
age_resolution_itp = inversefunc(mag_resolution_itp)

fig, (ax1, ax_dummy, ax3, ax_dummy_2, ax5) = plt.subplots(
    nrows=5, ncols=1, figsize=(8, 12), height_ratios=(10, 2, 10, 2, 10)
)

ax_dummy.axis("off")
ax_dummy_2.axis("off")


# plot the data
ax1.scatter(
    _mbol,
    mbol_total_err,
    s=0.1,
    label=r"Post-hoc M$_{\mathrm{bol}}$ uncertainties",
)
# plot the median
ax1.plot(
    mag_mean[(mag_mean > 2.0) & (mag_mean < 18.25)],
    err_mean[(mag_mean > 2.0) & (mag_mean < 18.25)],
    lw=1,
    label=r"Weight-mean",
    color="black",
    ls="dashed",
)


ax1.set_yscale("log")
ax1.set_xlabel(r"M$_{\mathrm{bol}}$ [mag]")
ax1.set_ylabel(r"$\Delta$M$_{\mathrm{bol}}$ [mag]")
ax1.set_xlim(6, 18.2)
ax1.set_ylim(1e-3, 10)
ax1.legend(loc="lower left")
ax1.grid()

ax3.scatter(age, mag_at_peak_density, s=2, label="Measured")
ax3.plot(
    age, mag_resolution_itp(age), color="black", ls="dashed", label="Fitted"
)
ax3.grid()
ax3.legend()
ax3.set_ylabel("magnitude at peak density [mag]")
ax3.set_xlabel("log(cooling age) [Gyr]")
ax3.set_xscale("log")

"""
ax2.plot(
    age[1:],
    mag_resolution_itp(age[1:]) - mag_resolution_itp(age[:-1]),
    label="Fitted",
)
ax2.set_ylabel("magnitude resolution at peak density")
ax2.set_xscale('log')

"""
# Load the "resolution" of the Mbol solution
mbol_err_mean_itp = interpolate.UnivariateSpline(mag_mean, err_mean, s=0)

# Nyquist sampling for resolving 2 gaussian peaks -> 2.355 sigma.
mag_bin_start = []
mag_bin_end = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
mag_bin_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
stop = False

for i, a in enumerate(age):
    start = float(mag_resolution_itp(a))
    end = start
    j = i
    carry_on = True
    if j >= len(age) - 1:
        carry_on = False
    # Truncate at 16.5 (pad in latter script)
    while carry_on:
        tmp = mag_resolution_itp(age[j + 1]) - mag_resolution_itp(age[j])
        print(j, end + tmp - start)
        # Nyquist sampling: sample 2 times for every FWHM (i.e. 2.355 sigma)
        if end + tmp - start < mbol_err_mean_itp(start) * 1.1775:
            end = end + tmp
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
    mag_bin_start.append(start)
    mag_bin_end.append(end)
    if stop:
        break


j = 0
bin_number = 0
stop = False
# the mag_bin is the bin edges
for i, a in enumerate(age):
    if i < j:
        continue
    start = float(mag_resolution_itp(a))
    end = start
    j = i
    carry_on = True
    if j >= len(age) - 1:
        carry_on = False
    # Truncate at 16.5 (pad in latter script)
    while carry_on:
        tmp = mag_resolution_itp(age[j + 1]) - mag_resolution_itp(age[j])
        print(j, end + tmp - start)
        mag_bin_pwdlf[j] = bin_number
        # Nyquist sampling: sample 2 times for every FWHM (i.e. 2.355 sigma)
        if end + tmp - start < mbol_err_mean_itp(start) * 1.1775:
            end = end + tmp
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    if stop:
        break


print(set(mag_bin_pwdlf))

# these are the bin edges
mag_bin_start = np.array(mag_bin_start)
mag_bin_end = np.array(mag_bin_end)


mag_bin_mid = (mag_bin_start + mag_bin_end) / 2.0
mag_bin_age = age_resolution_itp(mag_bin_mid)
resolution_optimal = mag_bin_end - mag_bin_start

# get the bin centre
pwdlf_mapping_mag_bin = np.insert(mag_bin_pwdlf, 0, 0)

mag_resolution_itp = interpolate.UnivariateSpline(
    mag_bin_mid, resolution_optimal, s=0.1, k=5
)
ax5.scatter(mag_bin_mid, resolution_optimal, s=2, label="Measured")
ax5.plot(
    mag_bin_mid,
    mag_resolution_itp(mag_bin_mid),
    color="black",
    ls="dashed",
    label="Fitted",
)

ax5.set_ylabel("magnitude resolution [mag]")
ax5.set_xlabel(r"M$_{\mathrm{bol}}$ [mag]")
ax5.set_xticks(np.arange(6, 19, 1))
ax5.set_xlim(6, 18.2)
ax5.set_ylim(0, 2.0)
ax5.grid()
ax5.legend(loc="lower left")

plt.subplots_adjust(
    top=0.99, bottom=0.045, left=0.1, right=0.985, hspace=0.015
)

plt.savefig(
    os.path.join(figure_folder, "fig_03_mbol_sigma_magnitude_resoltuion.png")
)


# save the pdwdlf mapping
np.save(
    "SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy", mag_bin_pwdlf
)

# save the Mbol resolution
np.save(
    "SFH-WDLF-article/figure_data/mbol_resolution.npy",
    np.column_stack((mag_bin_mid, resolution_optimal)),
)
