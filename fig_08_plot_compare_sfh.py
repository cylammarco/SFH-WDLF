import os

from matplotlib import pyplot as plt
import numpy as np
from spectresc import spectres
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader
from WDPhotTools.theoretical_lf import WDLF
from pynverse import inversefunc


plt.rcParams.update({"font.size": 12})

figure_folder = "SFH-WDLF-article/figures"

# Load the GCNS data
gcns_wdlf = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]

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


(
    partial_age_optimal,
    solution_optimal,
    solution_lower,
    solution_upper,
) = np.load(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy"
).T
(
    partial_age_optimal,
    solution_optimal_20pc_subset,
    solution_lower_20pc_subset,
    solution_upper_20pc_subset,
) = np.load(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal_20pc_subset.npy"
).T

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

_, _s, _vh = np.linalg.svd(solution_optimal_jac, full_matrices=False)
tol = np.finfo(float).eps * _s[0] * max(solution_optimal_jac.shape)
_w = _s > tol
cov = (_vh[_w].T / _s[_w] ** 2) @ _vh[_w]  # robust covariance matrix
stdev = np.sqrt(np.diag(cov))

# from running sfh_mcmc_gcns_wdlf_optimal_resolution.py
mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal = np.load(
    "SFH-WDLF-article/figure_data/gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.npy"
).T
(
    mag_obs_optimal,
    obs_wdlf_optimal_20pc_subset,
    obs_wdlf_err_optimal_20pc_subset,
) = np.load(
    "SFH-WDLF-article/figure_data/gcns_reconstructed_wdlf_optimal_resolution_bin_optimal_20pc_subset.npy"
).T

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.insert(
    np.load("SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy"), 0, 0
)

# Stack up the pwdlfs to the desired resolution
partial_wdlf_optimal = []
partial_age_optimal = []
partial_age_optimal.append(0.0)
for idx in np.sort(list(set(pwdlf_mapping_bin_optimal))):
    pwdlf_temp = np.zeros_like(mag_obs_optimal)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_optimal == idx)[0]:
        pwdlf_temp += spectres(
            mag_obs_optimal, mag_pwdlf, data[i][:, 1], fill=0.0
        )
        age_temp = age[i]
    partial_wdlf_optimal.append(pwdlf_temp)
    partial_age_optimal.append(age_temp)

partial_age_optimal.append(15.0)

# only compute for the wdlf integrated number density
recomputed_wdlf_optimal = np.nansum(
    solution_optimal * np.array(partial_wdlf_optimal).T, axis=1
)
recomputed_wdlf_optimal_lsq = np.nansum(
    solution_optimal_lsq * np.array(partial_wdlf_optimal).T, axis=1
)
recomputed_wdlf_optimal_20pc_subset = np.nansum(
    solution_optimal_20pc_subset * np.array(partial_wdlf_optimal).T, axis=1
)
recomputed_wdlf_optimal_lsq_20pc_subset = np.nansum(
    solution_optimal_lsq_20pc_subset * np.array(partial_wdlf_optimal).T, axis=1
)

# append for plotting the first bin
solution_optimal_lsq = np.insert(solution_optimal_lsq, 0, 0.0)
solution_optimal = np.insert(solution_optimal, 0, 0.0)
solution_upper = np.insert(solution_upper, 0, 0.0)
solution_lower = np.insert(solution_lower, 0, 0.0)
solution_optimal_lsq_20pc_subset = np.insert(
    solution_optimal_lsq_20pc_subset, 0, 0.0
)
solution_optimal_20pc_subset = np.insert(solution_optimal_20pc_subset, 0, 0.0)
solution_upper_20pc_subset = np.insert(solution_upper_20pc_subset, 0, 0.0)
solution_lower_20pc_subset = np.insert(solution_lower_20pc_subset, 0, 0.0)

# append for plotting the last bin
solution_optimal_lsq = np.append(solution_optimal_lsq, 0.0)
solution_optimal = np.append(solution_optimal, 0.0)
solution_upper = np.append(solution_upper, 0.0)
solution_lower = np.append(solution_lower, 0.0)
solution_optimal_lsq_20pc_subset = np.append(
    solution_optimal_lsq_20pc_subset, 0.0
)
solution_optimal_20pc_subset = np.append(solution_optimal_20pc_subset, 0.0)
solution_upper_20pc_subset = np.append(solution_upper_20pc_subset, 0.0)
solution_lower_20pc_subset = np.append(solution_lower_20pc_subset, 0.0)

"""
# construct the inverse ifmr, aka fimr
inv_ifmr = inversefunc(WDLF()._ifmr)

# Get the (Mbol, Age) to WD mass relation
atm = AtmosphereModelReader()
Mbol_age_to_mass = atm.interp_am(dependent=["mass"], independent=["Mbol", "age"])


mass_ms_min = optimize.fminbound(
                WDLF._find_mass_ms_min,
                0.5,
                8.0,
                args=[mag_i],
                xtol=1e-8,
                maxfun=10000,
            )

# Get the pWDLF
for pwdlf, age in zip(partial_wdlf_optimal, partial_age_optimal):
    
    # Get the WD mass distribution
    wd_mass_distribution = Mbol_age_to_mass(mag_obs_optimal[pwdlf>0], age * 1e9)

    # Get the MS mass distribution
    ms_mass_distribution = inv_ifmr(wd_mass_distribution)

    # Get the mean mass from the MS mass distribution

"""

# Cignoni+ 2006 (only relative SFH)
cignoni_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_cignoni_sfh.csv", delimiter=","
)
cignoni_time = cignoni_data[:, 0]
cignoni_sfh = cignoni_data[:, 1]
cignoni_sigma_up = cignoni_data[:, 2] - cignoni_data[:, 1]
cignoni_sigma_low = cignoni_data[:, 1] - cignoni_data[:, 3]
# append for plotting the first bin
cignoni_time = np.insert(
    cignoni_time, 0, 2.0 * cignoni_time[0] - cignoni_time[1]
)
cignoni_sfh = np.insert(cignoni_sfh, 0, 0.0)
cignoni_sigma_up = np.insert(cignoni_sigma_up, 0, 0.0)
cignoni_sigma_low = np.insert(cignoni_sigma_low, 0, 0.0)
# append for plotting the last bin
cignoni_time = np.append(
    cignoni_time, 2.0 * cignoni_time[-1] - cignoni_time[-2]
)
cignoni_sfh = np.append(cignoni_sfh, 0.0)
cignoni_sigma_up = np.append(cignoni_sigma_up, 0.0)
cignoni_sigma_low = np.append(cignoni_sigma_low, 0.0)

# Isern 2019 (In mass per Gyr)
isern_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_isern_2019_sfh.csv", delimiter=","
)
isern_time = isern_data[:, 0]
isern_sfh = 10.0 ** (isern_data[:, 1])
isern_sigma_up = 10.0 ** isern_data[:, 2] - 10.0 ** isern_data[:, 1]
isern_sigma_low = 10.0 ** isern_data[:, 1] - 10.0 ** isern_data[:, 3]
# append for plotting the first bin
isern_time = np.insert(isern_time, 0, 2.0 * isern_time[0] - isern_time[1])
isern_sfh = np.insert(isern_sfh, 0, 0.0)
isern_sigma_up = np.insert(isern_sigma_up, 0, 0.0)
isern_sigma_low = np.insert(isern_sigma_low, 0, 0.0)
# append for plotting the last bin
isern_time = np.append(isern_time, 2.0 * isern_time[-1] - isern_time[-2])
isern_sfh = np.append(isern_sfh, 0.0)
isern_sigma_up = np.append(isern_sigma_up, 0.0)
isern_sigma_low = np.append(isern_sigma_low, 0.0)

# Mor+ 2019 (only relative SFH, in pc^-2...)
mor_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_mor_2019_sfh.csv", delimiter=","
)
mor_time = mor_data[:, 0]
mor_sfh = mor_data[:, 1]
mor_sigma_up = mor_data[:, 2]
mor_sigma_low = mor_data[:, 3]
# append for plotting the first bin
mor_time = np.insert(mor_time, 0, 2.0 * mor_time[0] - mor_time[1])
mor_sfh = np.insert(mor_sfh, 0, 0.0)
mor_sigma_up = np.insert(mor_sigma_up, 0, 0.0)
mor_sigma_low = np.insert(mor_sigma_low, 0, 0.0)
# append for plotting the last bin
mor_time = np.append(mor_time, 2.0 * mor_time[-1] - mor_time[-2])
mor_sfh = np.append(mor_sfh, 0.0)
mor_sigma_up = np.append(mor_sigma_up, 0.0)
mor_sigma_low = np.append(mor_sigma_low, 0.0)

# Tremblay+ 2014 (only relative SFH)
tremblay_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_tremblay_2014_sfh.csv", delimiter=","
)
tremblay_time = tremblay_data[:, 0]
tremblay_sfh = tremblay_data[:, 1]
tremblay_sigma_up = tremblay_data[:, 2]
tremblay_sigma_low = tremblay_data[:, 3]
# append for plotting the first bin
tremblay_time = np.insert(
    tremblay_time, 0, 2.0 * tremblay_time[0] - tremblay_time[1]
)
tremblay_sfh = np.insert(tremblay_sfh, 0, 0.0)
tremblay_sigma_up = np.insert(tremblay_sigma_up, 0, 0.0)
tremblay_sigma_low = np.insert(tremblay_sigma_low, 0, 0.0)
# append for plotting the last bin
tremblay_time = np.append(
    tremblay_time, 2.0 * tremblay_time[-1] - tremblay_time[-2]
)
tremblay_sfh = np.append(tremblay_sfh, 0.0)
tremblay_sigma_up = np.append(tremblay_sigma_up, 0.0)
tremblay_sigma_low = np.append(tremblay_sigma_low, 0.0)

# Reid+ 2007 (only relative SFH)
reid_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_reid_2007_sfh.csv", delimiter=","
)
reid_time = reid_data[:, 0]
reid_sfh = reid_data[:, 1]
# append for plotting the first bin
reid_time = np.insert(reid_time, 0, 2.0 * reid_time[0] - reid_time[1])
reid_sfh = np.insert(reid_sfh, 0, 0.0)
# append for plotting the last bin
reid_time = np.append(reid_time, 2.0 * reid_time[-1] - reid_time[-2])
reid_sfh = np.append(reid_sfh, 0.0)

# Bernard+ 2018
bernard_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_bernard_2018_sfh.csv", delimiter=","
)
bernard_time = bernard_data[:, 0]
bernard_sfh = bernard_data[:, 1]
# append for plotting the first bin
bernard_time = np.insert(
    bernard_time, 0, 2.0 * bernard_time[0] - bernard_time[1]
)
bernard_sfh = np.insert(bernard_sfh, 0, 0.0)
# append for plotting the last bin
bernard_time = np.append(
    bernard_time, 2.0 * bernard_time[-1] - bernard_time[-2]
)
bernard_sfh = np.append(bernard_sfh, 0.0)


# Torres+ 2021
torres_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_torres_2021_sfh.csv", delimiter=","
)
torres_time = torres_data[:, 0]
torres_sfh = torres_data[:, 1]
# append for plotting the first bin
torres_time = np.insert(torres_time, 0, 2.0 * torres_time[0] - torres_time[1])
torres_sfh = np.insert(torres_sfh, 0, 0.0)
# append for plotting the last bin
torres_time = np.append(torres_time, 2.0 * torres_time[-1] - torres_time[-2])
torres_sfh = np.append(torres_sfh, 0.0)

# Xiang & Rix 2022
xiang_data = np.load("41586_2022_4496_MOESM3_ESM.npz")["arr_0"]
xiang_age = xiang_data[:, 3].astype("float")
xiang_sfh, xiang_time_bin_edges = np.histogram(
    xiang_age, bins=75, range=(0, 14)
)
xiang_time = np.diff(xiang_time_bin_edges) * 0.5 + xiang_time_bin_edges[:-1]


# Rowell 2013
rowell_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_rowell_2013_sfh.txt"
)
rowell_time = rowell_data[:, 0] / 1e9
rowell_sfh = rowell_data[:, 1] * 1e9

# Rowell 2023
rowell_2023_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_rowell_2023_sfh.txt"
)
rowell_2023_time = rowell_2023_data[:, 0] / 1e9
rowell_2023_sfh = rowell_2023_data[:, 2] * 1e9

"""
# Fantin+ 2019
fantin_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_08_fantin_2019_sfh.csv", delimiter=","
)
fantin_thin = fantin_data[0]
fantin_thick = fantin_data[1]
fantin_halo = fantin_data[2]

from scipy.stats import skewnorm
# epsilon, sfr, sigma_t, alpha, f_He
skewed_gaussian_thin = skewnorm.pdf(
    age, -fantin_thin[3]*2, loc=fantin_thin[0], scale=(fantin_thin[2])
)
skewed_gaussian_thick = skewnorm.pdf(
    age, -fantin_thick[3]*2, loc=fantin_thick[0], scale=(fantin_thick[2])
)
skewed_gaussian_halo = skewnorm.pdf(
    age, -fantin_halo[3]*2, loc=fantin_halo[0], scale=(fantin_halo[2])
)

skewed_gaussian_thin /= max(skewed_gaussian_thin)
skewed_gaussian_thin *= fantin_thin[1]

skewed_gaussian_thick /= max(skewed_gaussian_thick)
skewed_gaussian_thick *= fantin_thick[1]

skewed_gaussian_halo /= max(skewed_gaussian_halo)
skewed_gaussian_halo *= fantin_halo[1]

clf()
plot(age, skewed_gaussian_thin)
plot(age, skewed_gaussian_thick)
plot(age, skewed_gaussian_halo)
plot(age, skewed_gaussian_thin + skewed_gaussian_thick + skewed_gaussian_halo)
"""


# This adjusts the SFH to per Gyr
age_bin_norm_this_work = np.concatenate(
    [
        [partial_age_optimal[1] - partial_age_optimal[0]],
        (np.diff(partial_age_optimal)[:-1] + np.diff(partial_age_optimal)[1:])
        / 2.0,
        [partial_age_optimal[-1] - partial_age_optimal[-2]],
    ]
)

# This allows for "integrateing" the WDLF
mag_bin_norm_this_work = np.concatenate(
    [
        [mag_obs_optimal[1] - mag_obs_optimal[0]],
        (np.diff(mag_obs_optimal)[:-1] + np.diff(mag_obs_optimal)[1:]) / 2.0,
        [mag_obs_optimal[-1] - mag_obs_optimal[-2]],
    ]
)
# This gives the total number
bin_norm_this_work = np.concatenate(
    [
        [partial_age_optimal[1] - partial_age_optimal[0]],
        (np.diff(partial_age_optimal)[:-1] + np.diff(partial_age_optimal)[1:])
        / 2.0,
        [partial_age_optimal[-1] - partial_age_optimal[-2]],
    ]
)

normalisation_this_work = (
    np.sum(obs_wdlf_optimal) / np.sum(recomputed_wdlf_optimal_lsq) * 1e9
)
normalisation_this_work_20pc_subset = (
    np.sum(obs_wdlf_optimal_20pc_subset)
    / np.sum(recomputed_wdlf_optimal_lsq_20pc_subset)
    * 1e9
)

# These are to normalise to match the GCNS WDLF integrated number density
normalisation_cignoni = (
    np.sum(obs_wdlf_optimal) / (cignoni_sfh @ cignoni_time) / 0.6
)
normalisation_mor = 1.0
normalisation_tremblay = np.sum(obs_wdlf_optimal) / np.sum(
    tremblay_sfh @ tremblay_time
)
normalisation_reid = np.sum(solution_optimal_lsq) / np.sum(reid_sfh) * 0.1

bin_norm_this_work = np.concatenate(
    [
        [partial_age_optimal[1] - partial_age_optimal[0]],
        (np.diff(partial_age_optimal)[:-1] + np.diff(partial_age_optimal)[1:])
        / 2.0,
        [partial_age_optimal[-1] - partial_age_optimal[-2]],
    ]
)

fig6, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 12), height_ratios=(10, 10, 10), sharex=True
)
# ax1 for SFH in the unit of N / Gyr / pc^3
# ax2 for SFH in the unit of M_sun / Gyr / pc^3
# ax3 for SFH in the unit of number count

# plot data from this work
ax1.step(
    partial_age_optimal,
    solution_optimal_lsq * normalisation_this_work / bin_norm_this_work,
    where="mid",
    color="black",
    label="pWDLF (this work)",
)

ax1.step(
    partial_age_optimal,
    solution_optimal_lsq_20pc_subset
    * 100
    * normalisation_this_work_20pc_subset / bin_norm_this_work,
    where="mid",
    label="20pc subset [x100]",
    color="black",
    ls="dashed",
    alpha=0.8,
)

ax1.step(
    rowell_time,
    rowell_sfh,
    where="mid",
    label="Rowell 2013",
    color="black",
    alpha=0.6,
)

ax1.step(
    rowell_2023_time,
    rowell_2023_sfh,
    where="mid",
    label="Rowell (this work)",
    color="black",
    alpha=0.4,
)


# Unit of M_sun / Gyr / pc^3


# plot Cignoni+ data
ax2.step(
    cignoni_time,
    cignoni_sfh * normalisation_cignoni,
    where="mid",
    color="C0",
    linestyle="-",
)
ax2.vlines(
    cignoni_time,
    (cignoni_sfh - cignoni_sigma_low) * normalisation_cignoni,
    (cignoni_sfh + cignoni_sigma_up) * normalisation_cignoni,
    color="C0",
    linestyle="-",
    label="Cignoni+ 2006",
)

# plot Bernard data (M / Gyr / pc^3)
ax2.step(
    bernard_time,
    bernard_sfh / np.max(bernard_sfh) * 0.004,
    where="mid",
    color="C1",
    label="Bernard 2018",
)


# plot Isern data (M / Gyr / pc^3)
ax2.step(
    isern_time,
    isern_sfh,
    where="mid",
    color="C2",
)
ax2.vlines(
    isern_time,
    isern_sfh - isern_sigma_low,
    isern_sfh + isern_sigma_up,
    color="C2",
    label="Isern 2019",
)

# plot Mor+ data
ax2.step(
    mor_time,
    mor_sfh
    / 25
    * np.max(
        solution_optimal_lsq * normalisation_this_work / age_bin_norm_this_work
    ),
    where="mid",
    color="C3",
    label="Mor+ 2019",
    linestyle="dashed",
)
ax2.vlines(
    mor_time,
    (mor_sfh - mor_sigma_low)
    / 25
    * np.max(
        solution_optimal_lsq * normalisation_this_work / age_bin_norm_this_work
    ),
    (mor_sfh + mor_sigma_up)
    / 25
    * np.max(
        solution_optimal_lsq * normalisation_this_work / age_bin_norm_this_work
    ),
    linestyle="dashed",
    color="C3",
)


# Number count

# plot Reid+ data
ax3.step(
    reid_time,
    reid_sfh / np.nanmax(reid_sfh),
    where="mid",
    color="C4",
    label="Reid+ 2007",
)


# plot Tremblay+ data
ax3.step(
    tremblay_time,
    tremblay_sfh / np.nanmax(tremblay_sfh),
    where="mid",
    color="C5",
    label="Tremblay+ 2014",
)


# plot Torres+ data (N)
ax3.step(
    torres_time,
    torres_sfh / np.nanmax(torres_sfh),
    where="mid",
    color="C6",
    label="Torres+ 2021",
)


# plot Xiang+ data (N)
ax3.step(
    xiang_time,
    xiang_sfh / np.nanmax(xiang_sfh),
    where="mid",
    color="C7",
    label="Xiang+ 2022",
)


ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 14)

ax1.set_ylim(0, 0.0065)
ax2.set_ylim(0, 0.00425)
ax3.set_ylim(0, 1.5)

ax3.set_xlabel("Lookback time [Gyr]")

ax1.set_ylabel(r"Star Formation Rate [N Gyr$^{-1}$ pc$^{-3}$]")
ax2.set_ylabel(r"Star Formation Rate [M$_{\odot}$ Gyr$^{-1}$ pc$^{-3}$]")
ax3.set_ylabel(r"Star Formation Rate [N / arbitrary unit]")


ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")

plt.tight_layout()
plt.subplots_adjust(hspace=0)


fig6.savefig(
    os.path.join(
        figure_folder,
        "fig_08_compare_sfh.png",
    )
)
