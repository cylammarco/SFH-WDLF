import os

from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader


figure_folder = "SFH-WDLF-article/figures"
plt.rcParams.update({"font.size": 12})

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


fig, (ax1, ax_dummy, ax3) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 9), height_ratios=(10, 3, 10)
)

ax_dummy.axis("off")

ax1.scatter(age, mag_at_peak_density, s=2, label="Measured")
ax1.plot(
    age, mag_resolution_itp(age), color="black", ls="dashed", label="Fitted"
)
ax1.grid()
ax1.legend()
ax1.set_ylabel("magnitude at peak density")
ax1.set_xlabel("log(age) (Gyr)")
ax1.set_xscale("log")

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
#mbol_err_upper_bound = interpolate.UnivariateSpline(
#    *np.load("mbol_err_upper_bound.npy").T, s=0
#)
mbol_err_upper_bound = interpolate.UnivariateSpline(
    *np.load("SFH-WDLF-article/figure_data/mbol_err_mean.npy").T, s=0
)


# Nyquist sampling for resolving 2 gaussian peaks -> 2.355 sigma.

mag_bin = []
mag_bin_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
mag_bin_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
bin_number = 0
stop = False
# the mag_bin is the bin edges
mag_bin.append(mag_at_peak_density[0])
for i, a in enumerate(age):
    if i < j:
        continue
    start = float(mag_resolution_itp(a))
    end = start
    j = i
    carry_on = True
    if j >= len(age) - 1:
        carry_on = False
    while carry_on:
        tmp = mag_resolution_itp(age[j + 1]) - mag_resolution_itp(age[j])
        print(j, end + tmp - start)
        mag_bin_pwdlf[j] = bin_number
        # Nyquist sampling: sample 2 times for every FWHM (i.e. 2.355 sigma)
        if end + tmp - start < mbol_err_upper_bound(start) * 1.1775:
            end = end + tmp
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

# if magnitudes don't reach 18, pad to 18 in 0.2 mag increment
if max(mag_bin) < 18.0:
    _n = int(np.round((18.0 - mag_bin[-1]) / 0.2))
    mag_bin = np.append(mag_bin[:-1], np.linspace(mag_bin[-1], 18.0, _n))

resolution_optimal = np.diff(mag_bin)

# get the bin centre
mag_bin = (mag_bin[1:] + mag_bin[:-1]) / 2.0
pwdlf_mapping_mag_bin = np.insert(mag_bin_pwdlf, 0, 0)

ax3.plot(
    mag_bin,
    resolution_optimal,
)
ax3.set_ylabel("magnitude resolution")
ax3.set_xlabel(r"M$_{\mathrm{bol}}$ [mag]")
ax3.set_xticks(np.arange(6, 19, 1))
ax3.set_ylim(0,3.25)
ax3.grid()

# Get the Mbol to Age relation
atm = AtmosphereModelReader()
Mbol_to_age = atm.interp_am(dependent="age")
age_ticks = Mbol_to_age(8.0, ax3.get_xticks())
age_ticklabels = [f"{i/1e9:.3f}" for i in age_ticks]

# make the top axis
ax3b = ax3.twiny()
ax3b.set_xlim(ax3.get_xlim())
ax3b.set_xticks(ax3.get_xticks())
ax3b.xaxis.set_ticklabels(age_ticklabels, rotation=90)

plt.subplots_adjust(top=0.995, bottom=0.06, left=0.075, right=0.995, hspace=0.01)

fig.savefig(os.path.join(figure_folder, "fig_04_magnitude_resoltuion.png"))


# save the pdwdlf mapping
np.save("SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy", mag_bin_pwdlf)

# save the Mbol resolution
np.save("SFH-WDLF-article/figure_data/mbol_resolution.npy", np.column_stack((mag_bin, resolution_optimal)))
