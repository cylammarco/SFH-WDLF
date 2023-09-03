import os

from matplotlib import pyplot as plt
import numpy as np
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader

figure_folder = "SFH-WDLF-article/figures"
plt.rcParams.update({"font.size": 12})

partial_wdlf = np.load("SFH-WDLF-article/figure_data/fig_01_partial_wdlf.npy")

sfh_1 = np.load("SFH-WDLF-article/figure_data/fig_01_sfh_1.npy")
sfh_2 = np.load("SFH-WDLF-article/figure_data/fig_01_sfh_2.npy")
sfh_3 = np.load("SFH-WDLF-article/figure_data/fig_01_sfh_3.npy")
sfh_4 = np.load("SFH-WDLF-article/figure_data/fig_01_sfh_4.npy")
sfh_5 = np.load("SFH-WDLF-article/figure_data/fig_01_sfh_5.npy")

wdlf_number_density_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_input_wdlf_1.npy"
)
wdlf_number_density_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_input_wdlf_2.npy"
)
wdlf_number_density_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_input_wdlf_3.npy"
)
wdlf_number_density_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_input_wdlf_4.npy"
)
wdlf_number_density_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_input_wdlf_5.npy"
)

partial_age, solution_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_solution_1.npy"
).T
_, solution_2 = np.load("SFH-WDLF-article/figure_data/fig_01_solution_2.npy").T
_, solution_3 = np.load("SFH-WDLF-article/figure_data/fig_01_solution_3.npy").T
_, solution_4 = np.load("SFH-WDLF-article/figure_data/fig_01_solution_4.npy").T
_, solution_5 = np.load("SFH-WDLF-article/figure_data/fig_01_solution_5.npy").T

_, solution_lsq_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_1.npy"
).T
_, solution_lsq_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_2.npy"
).T
_, solution_lsq_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_3.npy"
).T
_, solution_lsq_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_4.npy"
).T
_, solution_lsq_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_5.npy"
).T


sfh_mcmc_1, sfh_mcmc_lower_1, sfh_mcmc_upper_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_mcmc_1.npy"
).T
sfh_mcmc_2, sfh_mcmc_lower_2, sfh_mcmc_upper_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_mcmc_2.npy"
).T
sfh_mcmc_3, sfh_mcmc_lower_3, sfh_mcmc_upper_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_mcmc_3.npy"
).T
sfh_mcmc_4, sfh_mcmc_lower_4, sfh_mcmc_upper_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_mcmc_4.npy"
).T
sfh_mcmc_5, sfh_mcmc_lower_5, sfh_mcmc_upper_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_mcmc_5.npy"
).T

mag_bin, recomputed_wdlf_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_1.npy"
).T
_, recomputed_wdlf_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_2.npy"
).T
_, recomputed_wdlf_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_3.npy"
).T
_, recomputed_wdlf_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_4.npy"
).T
_, recomputed_wdlf_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_5.npy"
).T

_, recomputed_wdlf_lsq_1 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_1.npy",
).T
_, recomputed_wdlf_lsq_2 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_2.npy",
).T
_, recomputed_wdlf_lsq_3 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_3.npy",
).T
_, recomputed_wdlf_lsq_4 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_4.npy",
).T
_, recomputed_wdlf_lsq_5 = np.load(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_5.npy",
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


fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(10, 3, 15)
)
#
ax1.plot(age, sfh_1 / np.nanmax(sfh_1), ls=":", color="C0")
ax1.plot(age, sfh_2 / np.nanmax(sfh_2), ls=":", color="C1")
ax1.plot(age, sfh_3 / np.nanmax(sfh_3), ls=":", color="C2")
ax1.plot(age, sfh_4 / np.nanmax(sfh_4), ls=":", color="C3")
ax1.plot(age, sfh_5 / np.nanmax(sfh_5), ls=":", color="C4")
#
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_1,
    sfh_mcmc_upper_1,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_2,
    sfh_mcmc_upper_2,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_3,
    sfh_mcmc_upper_3,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_4,
    sfh_mcmc_upper_4,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_5,
    sfh_mcmc_upper_5,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
#
ax1.step(partial_age, solution_1, ls="dashed", where="mid", color="C0")
ax1.step(partial_age, solution_2, ls="dashed", where="mid", color="C1")
ax1.step(partial_age, solution_3, ls="dashed", where="mid", color="C2")
ax1.step(partial_age, solution_4, ls="dashed", where="mid", color="C3")
ax1.step(partial_age, solution_5, ls="dashed", where="mid", color="C4")
#
ax1.step(partial_age, solution_lsq_1, where="mid", color="C0")
ax1.step(partial_age, solution_lsq_2, where="mid", color="C1")
ax1.step(partial_age, solution_lsq_3, where="mid", color="C2")
ax1.step(partial_age, solution_lsq_4, where="mid", color="C3")
ax1.step(partial_age, solution_lsq_5, where="mid", color="C4")
#
ax1.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
ax1.plot([-1, -1], [-1, -2], ls="dashed", color="black", label="MCMC")
ax1.plot([-1, -1], [-1, -2], ls="-", color="black", label="lsq-refined")
#
ax1.grid()
ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 1.2)
ax1.set_xlabel("Lookback time [Gyr]")
ax1.set_ylabel("Relative SFR")
ax1.legend()
#
ax_dummy1.axis("off")
#
ax2.plot(
    mag_bin,
    wdlf_number_density_1 / np.nansum(wdlf_number_density_1),
    ls=":",
    color="C0",
)
ax2.plot(
    mag_bin,
    wdlf_number_density_2 / np.nansum(wdlf_number_density_2) * 0.2,
    ls=":",
    color="C1",
)
ax2.plot(
    mag_bin,
    wdlf_number_density_3 / np.nansum(wdlf_number_density_3) * 0.2**2,
    ls=":",
    color="C2",
)
ax2.plot(
    mag_bin,
    wdlf_number_density_4 / np.nansum(wdlf_number_density_4) * 0.2**3,
    ls=":",
    color="C3",
)
ax2.plot(
    mag_bin,
    wdlf_number_density_5 / np.nansum(wdlf_number_density_5) * 0.2**4,
    ls=":",
    color="C4",
)
#
ax2.plot(
    mag_bin,
    recomputed_wdlf_1 / np.nansum(recomputed_wdlf_1),
    ls="dashed",
    color="C0",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2) * 0.2,
    ls="dashed",
    color="C1",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_3 / np.nansum(recomputed_wdlf_3) * 0.2**2,
    ls="dashed",
    color="C2",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_4 / np.nansum(recomputed_wdlf_4) * 0.2**3,
    ls="dashed",
    color="C3",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_5 / np.nansum(recomputed_wdlf_5) * 0.2**4,
    ls="dashed",
    color="C4",
)
#
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_1 / np.nansum(recomputed_wdlf_lsq_1),
    color="C0",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_2 / np.nansum(recomputed_wdlf_lsq_2) * 0.2,
    color="C1",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_3 / np.nansum(recomputed_wdlf_lsq_3) * 0.2**2,
    color="C2",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_4 / np.nansum(recomputed_wdlf_lsq_4) * 0.2**3,
    color="C3",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_5 / np.nansum(recomputed_wdlf_lsq_5) * 0.2**4,
    color="C4",
)
#
ax2.plot([1, 1], [1, 2], ls=":", color="black", label="Input")
ax2.plot(
    [1, 1], [1, 2], ls="dashed", color="black", label="Reconstructed (MCMC)"
)
ax2.plot([1, 1], [1, 2], ls="-", color="black", label="Reconstructed (lsq)")
#
ax2.set_xlabel(r"M${_\mathrm{bol}}$ [mag]")
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_xlim(6.00, 18.00)
ax2.set_ylim(1e-8, 2e-1)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
#
# Get the Mbol to Age relation
atm = AtmosphereModelReader()
Mbol_to_age = atm.interp_am(dependent="age")
age_ticks = Mbol_to_age(8.0, ax2.get_xticks())
age_ticklabels = [f"{i/1e9:.3f}" for i in age_ticks]

# make the top axis
ax2b = ax2.twiny()
ax2b.set_xlim(ax2.get_xlim())
ax2b.set_xticks(ax2.get_xticks())
ax2b.xaxis.set_ticklabels(age_ticklabels, rotation=90)

plt.subplots_adjust(top=0.99, bottom=0.06, left=0.1, right=0.985, hspace=0.01)

plt.savefig(os.path.join(figure_folder, "fig_01_exponential_decay_wdlf.png"))
