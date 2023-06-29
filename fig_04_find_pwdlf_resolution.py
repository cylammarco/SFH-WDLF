import os

from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate


figure_folder = "SFH-WDLF-article/figures"

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


mag = data[0][:, 0]

mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag[np.argmax(d[:, 1])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=len(age) / 150, k=5
)


fig, (ax1, ax_dummy, ax3) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 9), height_ratios=(5, 1, 5)
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
mbol_err_upper_bound = interpolate.UnivariateSpline(
    *np.load("mbol_err_upper_bound.npy").T, s=0
)


# Nyquist sampling for resolving 2 gaussian peaks -> 2.355 sigma.

bin_optimal = []
bin_optimal_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
bin_optimal_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
bin_number = 0
stop = False
# the bin_optimal is the bin edges
bin_optimal.append(mag_resolution_itp(age[0]))
for i, a in enumerate(age):
    if i < j:
        continue
    start = mag_resolution_itp(a)
    end = start
    j = i
    carry_on = True
    while carry_on:
        tmp = mag_resolution_itp(age[j + 1]) - mag_resolution_itp(age[j])
        print(j, end + tmp - start)
        if end + tmp - start < mbol_err_upper_bound(end):
            end = end + tmp
            bin_optimal_pwdlf[j] = bin_number
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    bin_optimal.append(end)
    bin_optimal_idx.append(i)
    if stop:
        break


resolution_optimal = np.diff(bin_optimal)
bin_center = bin_optimal[:-1] + resolution_optimal / 2.0

ax3.bar(
    bin_center,
    height=resolution_optimal,
    width=resolution_optimal,
    color="white",
    edgecolor="black",
)
ax3.set_ylabel("magnitude resolution")
ax3.set_xlabel(r"M$_{\mathrm{bol}}$ [mag]")


plt.subplots_adjust(top=0.99, bottom=0.05, left=0.075, right=0.99, hspace=0.0)

fig.savefig(os.path.join(figure_folder, "fig_04_magnitude_resoltuion.png"))


# save the pdwdlf mapping
np.save("pwdlf_bin_optimal_mapping.npy", bin_optimal_pwdlf)

# save the Mbol resolution
np.save("mbol_resolution.npy", np.column_stack((bin_center, resolution_optimal)))
