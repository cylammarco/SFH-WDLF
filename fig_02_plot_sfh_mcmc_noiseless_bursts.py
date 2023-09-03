import os

from matplotlib import pyplot as plt
import numpy as np
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader

figure_folder = "SFH-WDLF-article/figures"
plt.rcParams.update({"font.size": 12})

partial_wdlf = np.load("SFH-WDLF-article/figure_data/fig_02_partial_wdlf.npy")

sfh_total_1 = np.load("SFH-WDLF-article/figure_data/fig_02_sfh_total_1.npy")
sfh_total_2 = np.load("SFH-WDLF-article/figure_data/fig_02_sfh_total_2.npy")
sfh_total_3 = np.load("SFH-WDLF-article/figure_data/fig_02_sfh_total_3.npy")
sfh_total_4 = np.load("SFH-WDLF-article/figure_data/fig_02_sfh_total_4.npy")
sfh_total_5 = np.load("SFH-WDLF-article/figure_data/fig_02_sfh_total_5.npy")
sfh_total_combined = np.load(
    "SFH-WDLF-article/figure_data/fig_02_sfh_total_combined.npy"
)

wdlf_number_density_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_input_wdlf_1.npy"
)
wdlf_number_density_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_input_wdlf_2.npy"
)
wdlf_number_density_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_input_wdlf_3.npy"
)
wdlf_number_density_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_input_wdlf_4.npy"
)
wdlf_number_density_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_input_wdlf_5.npy"
)
wdlf_number_density_combined = np.load(
    "SFH-WDLF-article/figure_data/fig_02_input_wdlf_combined.npy"
)

partial_age, solution_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_solution_1.npy"
).T
_, solution_2 = np.load("SFH-WDLF-article/figure_data/fig_02_solution_2.npy").T
_, solution_3 = np.load("SFH-WDLF-article/figure_data/fig_02_solution_3.npy").T
_, solution_4 = np.load("SFH-WDLF-article/figure_data/fig_02_solution_4.npy").T
_, solution_5 = np.load("SFH-WDLF-article/figure_data/fig_02_solution_5.npy").T
_, solution_combined = np.load(
    "SFH-WDLF-article/figure_data/fig_02_solution_combined.npy"
).T

_, solution_lsq_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_1.npy"
).T
_, solution_lsq_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_2.npy"
).T
_, solution_lsq_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_3.npy"
).T
_, solution_lsq_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_4.npy"
).T
_, solution_lsq_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_5.npy"
).T
_, solution_lsq_combined = np.load(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_combined.npy"
).T


sfh_mcmc_1, sfh_mcmc_lower_1, sfh_mcmc_upper_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_mcmc_1.npy"
).T
sfh_mcmc_2, sfh_mcmc_lower_2, sfh_mcmc_upper_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_mcmc_2.npy"
).T
sfh_mcmc_3, sfh_mcmc_lower_3, sfh_mcmc_upper_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_mcmc_3.npy"
).T
sfh_mcmc_4, sfh_mcmc_lower_4, sfh_mcmc_upper_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_mcmc_4.npy"
).T
sfh_mcmc_5, sfh_mcmc_lower_5, sfh_mcmc_upper_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_mcmc_5.npy"
).T
sfh_mcmc_combined, sfh_mcmc_lower_combined, sfh_mcmc_upper_combined = np.load(
    "SFH-WDLF-article/figure_data/fig_02_mcmc_combined.npy"
).T

mag_bin, recomputed_wdlf_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_1.npy"
).T
_, recomputed_wdlf_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_2.npy"
).T
_, recomputed_wdlf_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_3.npy"
).T
_, recomputed_wdlf_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_4.npy"
).T
_, recomputed_wdlf_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_5.npy"
).T
_, recomputed_wdlf_combined = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_combined.npy"
).T

_, recomputed_wdlf_lsq_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_1.npy",
).T
_, recomputed_wdlf_lsq_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_2.npy",
).T
_, recomputed_wdlf_lsq_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_3.npy",
).T
_, recomputed_wdlf_lsq_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_4.npy",
).T
_, recomputed_wdlf_lsq_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_5.npy",
).T
_, recomputed_wdlf_lsq_combined = np.load(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_combined.npy",
).T

age_list_1 = np.arange(0.049, 0.100, 0.001)
age_list_2 = np.arange(0.100, 0.350, 0.005)
age_list_3 = np.arange(0.35, 14.01, 0.01)
age_list_3dp = np.concatenate((age_list_1, age_list_2))
age_list_2dp = age_list_3

age = np.concatenate((age_list_3dp, age_list_2dp))

n_wd_1 = wdlf_number_density_1[wdlf_number_density_1 > 0.0]
n_wd_2 = wdlf_number_density_2[wdlf_number_density_2 > 0.0]
n_wd_3 = wdlf_number_density_3[wdlf_number_density_3 > 0.0]
n_wd_4 = wdlf_number_density_4[wdlf_number_density_4 > 0.0]
n_wd_5 = wdlf_number_density_5[wdlf_number_density_5 > 0.0]
n_wd_combined = wdlf_number_density_combined[
    wdlf_number_density_combined > 0.0
]

# Figure here
fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(15, 2, 10)
)
ax_dummy1.axis("off")
#
ax1.plot(age, sfh_total_1 / np.nanmax(sfh_total_1) + 5.0, ls=":", color="C0")
ax1.plot(age, sfh_total_2 / np.nanmax(sfh_total_2) + 4.0, ls=":", color="C1")
ax1.plot(age, sfh_total_3 / np.nanmax(sfh_total_3) + 3.0, ls=":", color="C2")
ax1.plot(age, sfh_total_4 / np.nanmax(sfh_total_4) + 2.0, ls=":", color="C3")
ax1.plot(age, sfh_total_5 / np.nanmax(sfh_total_5) + 1.0, ls=":", color="C4")
ax1.plot(
    age, sfh_total_combined / np.nanmax(sfh_total_combined), ls=":", color="C5"
)
ax1.fill_between(
    partial_age,
    solution_lsq_1 - sfh_mcmc_lower_1 + 5.0,
    solution_lsq_1 + sfh_mcmc_upper_1 + 5.0,
    step="mid",
    color="C0",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_2 - sfh_mcmc_lower_2 + 4.0,
    solution_lsq_2 + sfh_mcmc_upper_2 + 4.0,
    step="mid",
    color="C1",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_3 - sfh_mcmc_lower_3 + 3.0,
    solution_lsq_3 + sfh_mcmc_upper_3 + 3.0,
    step="mid",
    color="C2",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_4 - sfh_mcmc_lower_4 + 2.0,
    solution_lsq_4 + sfh_mcmc_upper_4 + 2.0,
    step="mid",
    color="C3",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_5 - sfh_mcmc_lower_5 + 1.0,
    solution_lsq_5 + sfh_mcmc_upper_5 + 1.0,
    step="mid",
    color="C4",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_combined - sfh_mcmc_lower_combined,
    solution_lsq_combined + sfh_mcmc_upper_combined,
    step="mid",
    color="C5",
    alpha=0.3,
)
ax1.step(partial_age, sfh_mcmc_1 + 5.0, ls="dashed", where="mid", color="C0")
ax1.step(partial_age, sfh_mcmc_2 + 4.0, ls="dashed", where="mid", color="C1")
ax1.step(partial_age, sfh_mcmc_3 + 3.0, ls="dashed", where="mid", color="C2")
ax1.step(partial_age, sfh_mcmc_4 + 2.0, ls="dashed", where="mid", color="C3")
ax1.step(partial_age, sfh_mcmc_5 + 1.0, ls="dashed", where="mid", color="C4")
ax1.step(partial_age, sfh_mcmc_combined, ls="dashed", where="mid", color="C5")
ax1.step(partial_age, solution_lsq_1 + 5.0, where="mid", color="C0")
ax1.step(partial_age, solution_lsq_2 + 4.0, where="mid", color="C1")
ax1.step(partial_age, solution_lsq_3 + 3.0, where="mid", color="C2")
ax1.step(partial_age, solution_lsq_4 + 2.0, where="mid", color="C3")
ax1.step(partial_age, solution_lsq_5 + 1.0, where="mid", color="C4")
ax1.step(partial_age, solution_lsq_combined, where="mid", color="C5")
ax1.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
ax1.plot([-1, -1], [-1, -2], ls="dashed", color="black", label="MCMC")
ax1.plot([-1, -1], [-1, -2], ls="-", color="black", label="lsq-refined")
ax1.grid()
ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 6.25)
ax1.set_xlabel("Lookback time [Gyr]")
ax1.set_ylabel("Relative SFR")
ax1.legend()
#
ax2.plot(mag_bin, n_wd_1, ls=":", color="C0")
ax2.plot(mag_bin, n_wd_2 * 0.2, ls=":", color="C1")
ax2.plot(mag_bin, n_wd_3 * 0.2**2, ls=":", color="C2")
ax2.plot(mag_bin, n_wd_4 * 0.2**3, ls=":", color="C3")
ax2.plot(mag_bin, n_wd_5 * 0.2**4, ls=":", color="C4")
ax2.plot(mag_bin, n_wd_combined * 0.2**5, ls=":", color="C5")
#
ax2.plot(
    mag_bin,
    recomputed_wdlf_1 / np.nansum(recomputed_wdlf_1) * np.nansum(n_wd_1),
    ls="dashed",
    color="C0",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2) * np.nansum(n_wd_2) * 0.2,
    ls="dashed",
    color="C1",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_3
    / np.nansum(recomputed_wdlf_3)
    * np.nansum(n_wd_3)
    * 0.2**2,
    ls="dashed",
    color="C2",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_4
    / np.nansum(recomputed_wdlf_4)
    * np.nansum(n_wd_4)
    * 0.2**3,
    ls="dashed",
    color="C3",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_5
    / np.nansum(recomputed_wdlf_5)
    * np.nansum(n_wd_5)
    * 0.2**4,
    ls="dashed",
    color="C4",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_combined
    / np.nansum(recomputed_wdlf_combined)
    * np.nansum(n_wd_combined)
    * 0.2**5,
    ls="dashed",
    color="C5",
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
    * np.nansum(n_wd_2)
    * 0.2,
    color="C1",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_3
    / np.nansum(recomputed_wdlf_lsq_3)
    * np.nansum(n_wd_3)
    * 0.2**2,
    color="C2",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_4
    / np.nansum(recomputed_wdlf_lsq_4)
    * np.nansum(n_wd_4)
    * 0.2**3,
    color="C3",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_5
    / np.nansum(recomputed_wdlf_lsq_5)
    * np.nansum(n_wd_5)
    * 0.2**4,
    color="C4",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_combined
    / np.nansum(recomputed_wdlf_lsq_combined)
    * np.nansum(n_wd_combined)
    * 0.2**5,
    color="C5",
)
#
ax2.plot([1, 1], [1, 2], ls=":", color="black", label="Input")
ax2.plot(
    [1, 1], [1, 2], ls="dashed", color="black", label="Reconstructed (MCMC)"
)
ax2.plot([1, 1], [1, 2], ls="-", color="black", label="Reconstructed (lsq)")
ax2.set_xlabel(r"M${_\mathrm{bol}}$ [mag]")
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_xlim(6, 18)
ax2.set_ylim(1e-8, 2e-1)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
plt.subplots_adjust(
    top=0.975, bottom=0.075, left=0.125, right=0.975, hspace=0.075
)

# Get the Mbol to Age relation
atm = AtmosphereModelReader()
Mbol_to_age = atm.interp_am(dependent="age")
age_ticks = Mbol_to_age(8.0, ax2.get_xticks())
age_ticklabels = [f"{i/1e9:.3f}" for i in age_ticks]

"""
# make the top axis
ax2b = ax2.twiny()
ax2b.set_xlim(ax2.get_xlim())
ax2b.set_xticks(ax2.get_xticks())
ax2b.xaxis.set_ticklabels(age_ticklabels, rotation=90)
"""

plt.subplots_adjust(top=0.99, bottom=0.06, left=0.1, right=0.98, hspace=0.01)

plt.savefig(os.path.join(figure_folder, "fig_02_two_bursts_wdlf.png"))
