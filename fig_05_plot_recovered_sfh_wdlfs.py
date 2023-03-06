import os

from matplotlib import pyplot as plt
import numpy as np
from spectres import spectres


figure_folder = "SFH-WDLF-article/figures"

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


partial_age_optimal, solution_optimal = np.load(
    "gcns_sfh_optimal_resolution_bin_optimal.npy"
).T
mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal = np.load(
    "gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.npy"
).T
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


recomputed_wdlf_optimal = np.nansum(
    solution_optimal * np.array(partial_wdlf_optimal).T, axis=1
)


fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(8, 1, 5)
)

ax_dummy1.axis("off")
ax1.errorbar(
    mag_obs_optimal,
    obs_wdlf_optimal,
    yerr=[obs_wdlf_err_optimal, obs_wdlf_err_optimal],
    fmt="+",
    markersize=5,
    label="Input WDLF",
)
ax1.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal
    / np.nansum(recomputed_wdlf_optimal)
    * np.nansum(obs_wdlf_optimal),
    label="Reconstructed WDLF",
)
ax1.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
ax1.set_ylabel("log(arbitrary number density)")
ax1.set_xlim(2.0, 18.0)
ax1.set_ylim(1e-7, 5e-3)
ax1.set_yscale("log")
ax1.legend()
ax1.grid()

ax2.step(partial_age_optimal, solution_optimal, where="post")
ax2.grid()
ax2.set_xticks(np.arange(0, 15, 2))
ax2.set_xlim(0, 15)
ax2.set_ylim(bottom=0)
ax2.set_xlabel("Lookback time / Gyr")
ax2.set_ylabel("Relative Star Formation Rate")

plt.subplots_adjust(
    top=0.975, bottom=0.05, left=0.09, right=0.975, hspace=0.075
)

fig1.savefig(
    os.path.join(
        figure_folder,
        "fig_05_gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.png",
    )
)
