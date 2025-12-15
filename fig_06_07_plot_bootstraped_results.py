import os

import numpy as np
from matplotlib import pyplot as plt
from spectresc import spectres

plt.ion()


figure_folder = "SFH-WDLF-article/figures"
figure_data_folder = "SFH-WDLF-article/figure_data"
boostrap_folder = "SFH-WDLF-article/bootstrap_sample_folder"

# Load the GCNS data
gcns_wdlf = np.load("pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz")["data"]
gcns_wdlf_20pc_subset = gcns_wdlf[gcns_wdlf["dpc"] <= 20.0]

# Load the pwdlfs
data = []
age_list_1 = np.arange(0.049, 0.100, 0.001)
age_list_2 = np.arange(0.100, 0.350, 0.005)
age_list_3 = np.arange(0.35, 15.01, 0.01)
age_list_3dp = np.concatenate((age_list_1, age_list_2))
age_list_2dp = age_list_3

age_half_bin_size_1 = np.ones_like(age_list_1) * 0.001 / 2.0
age_half_bin_size_2 = np.ones_like(age_list_2) * 0.005 / 2.0
age_half_bin_size_3 = np.ones_like(age_list_3) * 0.01 / 2.0
age_half_bin_list = np.concatenate((age_half_bin_size_1, age_half_bin_size_2, age_half_bin_size_3))

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

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.insert(np.load("SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy"), 0, 0)

# Stack up the pwdlfs to the desired resolution
partial_wdlf_optimal = []
partial_age_optimal = []
partial_age_duration = []
for idx in np.sort(list(set(pwdlf_mapping_bin_optimal))):
    pwdlf_temp = np.zeros_like(mag_obs_optimal)
    age_temp_list = []
    age_bin_extra = []
    for i in np.where(pwdlf_mapping_bin_optimal == idx)[0]:
        pwdlf_temp += spectres(mag_obs_optimal, mag_pwdlf, data[i][:, 1], fill=0.0)
        age_temp_list.append(age[i])
        age_bin_extra.append(age_half_bin_list[i])
    partial_wdlf_optimal.append(pwdlf_temp)
    partial_age_optimal.append((np.max(age_temp_list) + np.min(age_temp_list)) / 2.0)
    partial_age_duration.append(np.ptp(age_temp_list) + age_bin_extra[0] + age_bin_extra[-1])


partial_wdlf_optimal_20pc_subset = np.vstack(partial_wdlf_optimal)
partial_wdlf_optimal = np.vstack(partial_wdlf_optimal)
partial_wdlf_optimal /= np.nansum(partial_wdlf_optimal)
partial_wdlf_optimal_20pc_subset /= np.nansum(partial_wdlf_optimal_20pc_subset)
partial_wdlf_optimal = partial_wdlf_optimal[obs_wdlf_optimal > 0.0][:, obs_wdlf_optimal > 0.0]
partial_wdlf_optimal_20pc_subset = partial_wdlf_optimal_20pc_subset[obs_wdlf_optimal_20pc_subset > 0.0][
    :, obs_wdlf_optimal_20pc_subset > 0.0
]
mag_obs_optimal_20pc_subset = mag_obs_optimal[obs_wdlf_optimal_20pc_subset > 0.0]
partial_age_optimal_20pc_subset = np.array(partial_age_optimal)[obs_wdlf_optimal_20pc_subset > 0.0]
partial_age_duration_20pc_subset = np.array(partial_age_duration)[obs_wdlf_optimal_20pc_subset > 0.0]


partial_wdlf_optimal_20pc_subset = np.array(partial_wdlf_optimal)[obs_wdlf_optimal_20pc_subset > 0.0][
    :, obs_wdlf_optimal_20pc_subset > 0.0
]

(
    partial_age_optimal,
    partial_age_duration,
    solution_optimal,
    solution_lower,
    solution_upper,
) = np.load("SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy").T
(
    partial_age_optimal_20pc_subset,
    partial_age_duration_20pc_subset,
    solution_optimal_20pc_subset,
    solution_lower_20pc_subset,
    solution_upper_20pc_subset,
) = np.load("SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal_20pc_subset.npy").T

# The lsq solution
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

recomputed_wdlf_optimal = np.nansum(solution_optimal * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_lsq = np.nansum(solution_optimal_lsq * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_20pc_subset = np.nansum(
    solution_optimal_20pc_subset * np.array(partial_wdlf_optimal_20pc_subset).T, axis=1
)
recomputed_wdlf_optimal_lsq_20pc_subset = np.nansum(
    solution_optimal_lsq_20pc_subset * np.array(partial_wdlf_optimal_20pc_subset).T, axis=1
)

normalisation_this_work = np.sum(obs_wdlf_optimal * resolution_optimal) / np.sum(recomputed_wdlf_optimal_lsq)
normalisation_this_work_20pc_subset = np.sum(obs_wdlf_optimal_20pc_subset * resolution_optimal) / np.sum(
    recomputed_wdlf_optimal_lsq
)
partial_age_optimal_padded = np.insert(partial_age_optimal, 0, 0.0)
partial_age_optimal_padded = np.append(partial_age_optimal_padded, 15.0)


sfh_list = []
wdlf_list = []
sfh_20pc_list = []
wdlf_20pc_list = []

for i in range(1000):
    sfh = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_sfh_sample_{i}.csv").T[1]
    wdlf = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_reconstructed_wdlf_sample_{i}.csv").T[1]
    sfh_20pc = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_sfh_sample_{i}.csv").T[4]
    wdlf_20pc = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_reconstructed_wdlf_sample_{i}.csv").T[3]
    sfh_list.append(sfh)
    wdlf_list.append(wdlf)
    sfh_20pc_list.append(sfh_20pc)
    wdlf_20pc_list.append(wdlf_20pc)


sfh_age = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_sfh_sample_{i}.csv").T[0]
sfh_mean = np.mean(sfh_list, axis=0)
sfh_stdev = np.std(sfh_list, axis=0)
sfh_20pc_mean = np.mean(sfh_20pc_list, axis=0)
sfh_20pc_stdev = np.std(sfh_20pc_list, axis=0)
wdlf_mag = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_reconstructed_wdlf_sample_{i}.csv").T[0]
wdlf_mean = np.mean(wdlf_list, axis=0)
wdlf_stdev = np.std(wdlf_list, axis=0)
wdlf_20pc_mean = np.mean(wdlf_20pc_list, axis=0)
wdlf_20pc_stdev = np.std(wdlf_20pc_list, axis=0)

"""
plt.figure(1)
plt.clf()
plt.plot(wdlf_mag, wdlf_mean)
plt.fill_between(wdlf_mag, wdlf_mean - wdlf_stdev, wdlf_mean + wdlf_stdev, alpha=0.5)
plt.yscale("log")
"""

# Calculate signal-to-noise ratio (SNR)
bootstrapping_snr = sfh_mean / sfh_stdev
best_fit_snr = solution_optimal / ((solution_upper - solution_lower) / 2.0)
stdev_ratio = ((solution_upper - solution_lower) / 2.0) / sfh_stdev
bootstrap_signal_to_best_fit_stdev = sfh_mean / ((solution_upper - solution_lower) / 2.0)

# Create a 2:1 vertically stacked subplot
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

# Top subplot: current figure
ax1.step(
    sfh_age,
    sfh_mean / np.nansum(solution_optimal) * normalisation_this_work,
    where="mid",
    color="C00",
    label="Mean",
)

ax1.fill_between(
    sfh_age,
    (sfh_mean - sfh_stdev) / np.nansum(solution_optimal) * normalisation_this_work,
    (sfh_mean + sfh_stdev) / np.nansum(solution_optimal) * normalisation_this_work,
    alpha=0.3,
    step="mid",
    color="C00",
    label=r"1$\sigma$",
)
ax1.fill_between(
    sfh_age,
    (sfh_mean - 2 * sfh_stdev) / np.nansum(solution_optimal) * normalisation_this_work,
    (sfh_mean + 2 * sfh_stdev) / np.nansum(solution_optimal) * normalisation_this_work,
    alpha=0.5,
    step="mid",
    color="lightgrey",
    label=r"2$\sigma$",
)
ax1.set_xlabel("Lookback time [Gyr]")
ax1.set_ylabel("Star Formation Rate [N Gyr$^{-1}$ pc$^{-3}$]")
ax1.grid()
ax1.set_xlim(0, 14)
ax1.set_ylim(0.0, 0.005)
ax1.legend()

# Calculate signal-to-noise ratio (SNR)
# sfh_20pc_snr = sfh_20pc_mean / sfh_20pc_stdev
"""
ax2.step(
    sfh_age,
    sfh_20pc_mean / np.nansum(solution_optimal_20pc_subset) * normalisation_this_work_20pc_subset,
    where="mid",
    color="C00",
    label="Mean",
)

ax2.fill_between(
    sfh_age,
    (sfh_20pc_mean - sfh_20pc_stdev) / np.nansum(solution_optimal_20pc_subset) * normalisation_this_work_20pc_subset,
    (sfh_20pc_mean + sfh_20pc_stdev) / np.nansum(solution_optimal_20pc_subset) * normalisation_this_work_20pc_subset,
    alpha=0.3,
    step="mid",
    color="C00",
    label=r"1$\sigma$",
)
ax2.fill_between(
    sfh_age,
    (sfh_20pc_mean - 2 * sfh_20pc_stdev)
    / np.nansum(solution_optimal_20pc_subset)
    * normalisation_this_work_20pc_subset,
    (sfh_20pc_mean + 2 * sfh_20pc_stdev)
    / np.nansum(solution_optimal_20pc_subset)
    * normalisation_this_work_20pc_subset,
    alpha=0.5,
    step="mid",
    color="lightgrey",
    label=r"2$\sigma$",
)
ax2.set_ylabel("Star Formation Rate [N Gyr$^{-1}$ pc$^{-3}$]")
ax2.grid()
ax2.set_xlim(0, 14)
ax2.set_ylim(0.0, 0.00011)

ax2.arrow(0.4, 0.0001075, 0.0, -0.000008, head_width=0.2, head_length=0.0000025, fc="k", ec="k")
ax2.arrow(1.61, 0.0001075, 0.0, -0.000008, head_width=0.2, head_length=0.0000025, fc="k", ec="k")
ax2.arrow(2.245, 0.0001075, 0.0, -0.000008, head_width=0.2, head_length=0.0000025, fc="k", ec="k")
ax2.arrow(3.67, 0.0001075, 0.0, -0.000008, head_width=0.2, head_length=0.0000025, fc="k", ec="k")
ax2.arrow(4.3, 0.0001075, 0.0, -0.000008, head_width=0.2, head_length=0.0000025, fc="k", ec="k")
ax2.arrow(4.75, 0.0001075, 0.0, -0.000008, head_width=0.2, head_length=0.0000025, fc="k", ec="k")

ax2.text(5.75, 0.000085, "20pc sample", fontsize=14)
"""

ax1.arrow(0.4, 0.0049, 0.0, -0.0004, head_width=0.2, head_length=0.0001, fc="k", ec="k")
ax1.arrow(1.21, 0.0049, 0.0, -0.0004, head_width=0.2, head_length=0.0001, fc="k", ec="k")
ax1.arrow(1.8, 0.0049, 0.0, -0.0004, head_width=0.2, head_length=0.0001, fc="k", ec="k")
ax1.arrow(2.245, 0.0049, 0.0, -0.0004, head_width=0.2, head_length=0.0001, fc="k", ec="k")
ax1.arrow(3.67, 0.0049, 0.0, -0.0004, head_width=0.2, head_length=0.0001, fc="k", ec="k")
ax1.arrow(8.7, 0.0049, 0.0, -0.0004, head_width=0.2, head_length=0.0001, fc="k", ec="k")

plt.subplots_adjust(top=0.98, bottom=0.08, left=0.12, right=0.98, hspace=0.0)
plt.savefig(f"{figure_folder}/fig_06_sfh_bootstrap_comparison.png", dpi=300)


print(sfh_age)

fig2, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
ax = ax.flatten()
# Tests the peaks at 0.4, 1.21, 1.8, 2.245, 3.67 and 8.7 Gyr are significant compared to their neighbours
for i, a in enumerate([0.40, 1.21, 1.80, 2.25, 3.67, 8.70]):
    # Identify 0–1 Gyr bin index (get the 2 bins closest to the peak at 0.4)
    bin_idx = np.argpartition(np.abs(sfh_age - a), 1)[0:2]
    bin_idx_0 = min(bin_idx)
    bin_idx_1 = max(bin_idx)
    # Compare this bin to the sum of its two neighbours
    neighbours = [max(0, bin_idx_0 - 1), min(len(sfh_age) - 1, bin_idx_1 + 1)]
    peak_durations = np.array(partial_age_duration)[bin_idx_0:bin_idx_1 + 1]
    peak_count = (np.array(sfh_list)[:, [bin_idx_0, bin_idx_1]] * peak_durations).sum(axis=1)
    neighbour_count = (np.array(sfh_list)[:, neighbours]).mean(axis=1) * np.sum(peak_durations)
    diff = peak_count - neighbour_count
    print(f"Fraction of bootstraps with SFH({a} Gyr) > SFH(neighbours): {np.sum(diff > 0):.1f}")
    ax[i].hist(diff, bins=50, color="C0", alpha=0.3, label="2 bins")
    ax[i].axvline(0, color="k", linestyle="--")
    ax[i].set_xlabel(f"Δ = SFH({a} Gyr) - SFH(neighbours)")
    ax[i].set_ylabel("Occurances")
    ax[i].set_title(f"Peak around {a:.2f} Gyr")
    ax[i].text(0.0095, 5, f"{np.mean(diff > 0)*100:.3f}%", color="C0")

for i, a in enumerate([0.47, 1.32, 1.70, 2.25, 3.60, 8.37]):
    # Use 1 bin insteadOccurance
    bin_idx_single = np.argmin(np.abs(sfh_age - a))
    # Compare this bin to the mean of its two neighbours
    neighbours_single = [max(0, bin_idx_single - 1), min(len(sfh_age) - 1, bin_idx_single + 1)]
    peak_duration = np.array(partial_age_duration)[bin_idx_single]
    peak_count_single = np.array(sfh_list)[:, bin_idx_single] * peak_duration
    neighbour_count_single = (np.array(sfh_list)[:, neighbours_single]).mean(axis=1) * peak_duration
    diff_single = peak_count_single - neighbour_count_single
    print(f"Fraction of bootstraps with SFH({a} Gyr) > SFH(neighbours): {np.sum(diff_single > 0):.1f}")
    ax[i].hist(diff_single, bins=50, color="C1", alpha=0.3, label="1 bin")
    ax[i].text(0.0095, 15, f"{np.mean(diff_single > 0)*100:.3f}%", color="C1")


ax[1].legend()
plt.subplots_adjust(top=0.975, bottom=0.05, left=0.08, right=0.98, hspace=0.4)
plt.savefig(f"{figure_folder}/fig_07_sfh_peak_significance.png", dpi=300)

"""
fig3, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
ax = ax.flatten()
# Tests the peaks at 0.4, 1.61, 1.8, 2.245, 3.67 and 8.7 Gyr are significant compared to their neighbours
for i, a in enumerate([0.40, 1.61, 2.25, 3.67, 4.30, 4.75]):
    # Identify 0–1 Gyr bin index (get the 2 bins closest to the peak at 0.4)
    bin_idx = np.argpartition(np.abs(sfh_age - a), 1)[0:2]
    bin_idx_0 = min(bin_idx)
    bin_idx_1 = max(bin_idx)
    # Compare this bin to the sum of its two neighbours
    neighbours = [max(0, bin_idx_0 - 1), min(len(sfh_age) - 1, bin_idx_1 + 1)]
    peak_durations = np.array(partial_age_duration)[bin_idx_0:bin_idx_1 + 1]
    peak_count = (np.array(sfh_20pc_list)[:, [bin_idx_0, bin_idx_1]] * peak_durations).sum(axis=1)
    neighbour_count = (np.array(sfh_20pc_list)[:, neighbours]).mean(axis=1) * np.sum(peak_durations)
    diff = peak_count - neighbour_count
    print(f"Fraction of bootstraps with SFH({a} Gyr) > SFH(neighbours): {np.sum(diff > 0):.1f}")
    ax[i].hist(diff, bins=50, color="C0", alpha=0.3, label="2 bins")
    ax[i].axvline(0, color="k", linestyle="--")
    ax[i].set_xlabel(f"Δ = SFH({a} Gyr) - SFH(neighbours)")
    ax[i].set_ylabel("Occurances")
    ax[i].set_title(f"Peak around {a:.2f} Gyr")
    ax[i].text(0.00005, 5, f"{np.mean(diff > 0)*100:.3f}%", color="C0")

for i, a in enumerate([0.47, 1.52, 2.25, 3.60, 4.20, 4.66]):
    # Use 1 bin instead
    bin_idx_single = np.argmin(np.abs(sfh_age - a))
    # Compare this bin to the mean of its two neighbours
    neighbours_single = [max(0, bin_idx_single - 1), min(len(sfh_age) - 1, bin_idx_single + 1)]
    peak_duration = np.array(partial_age_duration)[bin_idx_single]
    peak_count_single = np.array(sfh_20pc_list)[:, bin_idx_single] * peak_duration
    neighbour_count_single = (np.array(sfh_20pc_list)[:, neighbours_single]).mean(axis=1) * peak_duration
    diff_single = peak_count_single - neighbour_count_single
    print(f"Fraction of bootstraps with SFH({a} Gyr) > SFH(neighbours): {np.sum(diff_single > 0):.1f}")
    ax[i].hist(diff_single, bins=50, color="C1", alpha=0.3, label="1 bin")
    ax[i].text(0.00005, 15, f"{np.mean(diff_single > 0)*100:.3f}%", color="C1")


ax[1].legend()
plt.subplots_adjust(top=0.95, bottom=0.075, left=0.075, right=0.98, hspace=0.4)
plt.savefig(f"{figure_folder}/fig_08_sfh_peak_significance_20pc_subset.png", dpi=300)
"""

np.savetxt(
    f"{figure_data_folder}/fig_06_sfh_bootstrap_data.csv",
    np.column_stack(
        (
            sfh_age,
            sfh_mean / np.nansum(solution_optimal) * normalisation_this_work,
            sfh_stdev / np.nansum(solution_optimal) * normalisation_this_work,
        )
    ),
    delimiter=",",
    header="age[Gyr], mean[N/Gyr/pc^3], stdev[N/Gyr/pc^3]",
    comments="",
)
"""
np.savetxt(
    f"{figure_data_folder}/fig_06_sfh_bootstrap_20pc_subset_data.csv",
    np.column_stack((sfh_age, sfh_20pc_mean, sfh_20pc_stdev)),
    delimiter=",",
    header="age[Gyr], mean[N/Gyr/pc^3], stdev[N/Gyr/pc^3]",
    comments="",
)
"""
