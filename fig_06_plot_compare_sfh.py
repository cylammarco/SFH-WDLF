import os

from matplotlib import pyplot as plt
import numpy as np
from spectres import spectres
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
# The lsq solution
lsq_res = np.load(
    "SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_lsq_solution.npy",
    allow_pickle=True,
).item()
solution_optimal_lsq = lsq_res.x
solution_optimal_jac = lsq_res.jac

_, _s, _vh = np.linalg.svd(solution_optimal_jac, full_matrices=False)
tol = np.finfo(float).eps * _s[0] * max(solution_optimal_jac.shape)
_w = _s > tol
cov = (_vh[_w].T / _s[_w] ** 2) @ _vh[_w]  # robust covariance matrix
stdev = np.sqrt(np.diag(cov))

# from running sfh_mcmc_gcns_wdlf_optimal_resolution.py
mag_obs_optimal, obs_wdlf_optimal, obs_wdlf_err_optimal = np.load(
    "SFH-WDLF-article/figure_data/gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.npy"
).T

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.insert(
    np.load("SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy"), 0, 0
)

# Stack up the pwdlfs to the desired resolution
partial_wdlf_optimal = []
partial_age_optimal = []
for idx in np.sort(list(set(pwdlf_mapping_bin_optimal))):
    pwdlf_temp = np.zeros_like(mag_obs_optimal)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_optimal == idx)[0]:
        pwdlf_temp += spectres(
            mag_obs_optimal, mag_pwdlf, data[i][:, 1], fill=0.0
        )
        age_temp += age[i]
        age_count += 1
    partial_wdlf_optimal.append(pwdlf_temp)
    partial_age_optimal.append(age_temp / age_count)


# only compute for the wdlf integrated number density
recomputed_wdlf_optimal_lsq = np.nansum(
    solution_optimal_lsq * np.array(partial_wdlf_optimal).T, axis=1
)

# append for plotting the first bin
partial_age_optimal = np.insert(
    partial_age_optimal,
    0,
    2.0 * partial_age_optimal[0] - partial_age_optimal[1],
)
solution_optimal_lsq = np.insert(solution_optimal_lsq, 0, 0.0)
solution_optimal = np.insert(solution_optimal, 0, 0.0)
solution_upper = np.insert(solution_upper, 0, 0.0)
solution_lower = np.insert(solution_lower, 0, 0.0)
# append for plotting the last bin
partial_age_optimal = np.append(
    partial_age_optimal,
    2.0 * partial_age_optimal[-1] - partial_age_optimal[-2],
)
solution_optimal_lsq = np.append(solution_optimal_lsq, 0.0)
solution_optimal = np.append(solution_optimal, 0.0)
solution_upper = np.append(solution_upper, 0.0)
solution_lower = np.append(solution_lower, 0.0)

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
    r"SFH-WDLF-article\figure_data\fig_06_cignoni_sfh.csv", delimiter=","
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
    r"SFH-WDLF-article\figure_data\fig_06_isern_2019_sfh.csv", delimiter=","
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
    r"SFH-WDLF-article\figure_data\fig_06_mor_2019_sfh.csv", delimiter=","
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
    r"SFH-WDLF-article\figure_data\fig_06_tremblay_2014_sfh.csv", delimiter=","
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
    r"SFH-WDLF-article\figure_data\fig_06_reid_2007_sfh.csv", delimiter=","
)
reid_time = reid_data[:, 0]
reid_sfh = reid_data[:, 1]
# append for plotting the first bin
reid_time = np.insert(reid_time, 0, 2.0 * reid_time[0] - reid_time[1])
reid_sfh = np.insert(reid_sfh, 0, 0.0)
# append for plotting the last bin
reid_time = np.append(reid_time, 2.0 * reid_time[-1] - reid_time[-2])
reid_sfh = np.append(reid_sfh, 0.0)

# Torres+ 2021
torres_data = np.loadtxt(
    r"SFH-WDLF-article\figure_data\fig_06_torres_2021_sfh.csv", delimiter=","
)
torres_time = torres_data[:, 0]
torres_sfh = torres_data[:, 1]
# append for plotting the first bin
torres_time = np.insert(torres_time, 0, 2.0 * torres_time[0] - torres_time[1])
torres_sfh = np.insert(torres_sfh, 0, 0.0)
# append for plotting the last bin
torres_time = np.append(torres_time, 2.0 * torres_time[-1] - torres_time[-2])
torres_sfh = np.append(torres_sfh, 0.0)

"""
# Fantin+ 2019
fantin_data = np.loadtxt(
    r"SFH-WDLF-article\figure_data\fig_06_fantin_2019_sfh.csv", delimiter=","
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
bin_norm_this_work = np.concatenate(
    [
        [mag_obs_optimal[1] - mag_obs_optimal[0]],
        (np.diff(mag_obs_optimal)[:-1] + np.diff(mag_obs_optimal)[1:]) / 2.0,
        [mag_obs_optimal[-1] - mag_obs_optimal[-2]],
    ]
)
# This gives the total number
normalisation_this_work = (
    np.sum(obs_wdlf_optimal * bin_norm_this_work)
    / np.sum(recomputed_wdlf_optimal_lsq * bin_norm_this_work)
    * 1e9
)

normalisation_cignoni = (
    np.sum(solution_optimal_lsq)
    * normalisation_this_work
    / np.sum(cignoni_sfh)
)
normalisation_mor = (
    np.sum(solution_optimal_lsq) * normalisation_this_work / np.sum(mor_sfh)
)
normalisation_tremblay = (
    np.sum(solution_optimal_lsq)
    * normalisation_this_work
    / np.sum(tremblay_sfh)
)
normalisation_reid = (
    np.sum(solution_optimal_lsq)
    * normalisation_this_work
    / np.sum(reid_sfh)
    / 0.5
)

fig1 = plt.figure(100, figsize=(8, 8))
plt.clf()
ax1 = plt.gca()
# plot data from this work
ax1.step(
    partial_age_optimal,
    solution_optimal_lsq * normalisation_this_work / bin_norm_this_work,
    where="mid",
    color="black",
    label="This work",
)
ax1.errorbar(
    partial_age_optimal,
    solution_optimal_lsq * normalisation_this_work / bin_norm_this_work,
    yerr=[
        (solution_optimal - solution_lower)
        * normalisation_this_work
        / bin_norm_this_work,
        (solution_upper - solution_optimal)
        * normalisation_this_work
        / bin_norm_this_work,
    ],
    fmt="+",
    color="black",
)

# plot Cignoni+ data
ax1.step(
    cignoni_time,
    cignoni_sfh * normalisation_cignoni,
    where="mid",
    color="C0",
    linestyle="dashed",
)
ax1.vlines(
    cignoni_time,
    (cignoni_sfh - cignoni_sigma_low) * normalisation_cignoni,
    (cignoni_sfh + cignoni_sigma_up) * normalisation_cignoni,
    color="C0",
    linestyle="dashed",
    label="Cignoni+ 2006",
)

# plot Reid+ data
ax1.step(
    reid_time,
    reid_sfh * normalisation_reid,
    where="mid",
    color="C1",
    linestyle="dashed",
    label="Reid+ 2007",
)


# plot Tremblay+ data
ax1.step(
    tremblay_time,
    tremblay_sfh * normalisation_tremblay,
    where="mid",
    color="C2",
    linestyle="dashed",
)
ax1.vlines(
    tremblay_time,
    (tremblay_sfh - tremblay_sigma_low) * normalisation_tremblay,
    (tremblay_sfh + tremblay_sigma_up) * normalisation_tremblay,
    linestyle="dashed",
    color="C2",
    label="Tremblay+ 2014",
)

"""
# plot Isern data
ax1.step(
    isern_time,
    isern_sfh,
    where="mid",
    color="C3",
)
ax1.vlines(
    isern_time,
    isern_sfh - isern_sigma_low,
    isern_sfh + isern_sigma_up,
    color="C3",
    label="Isern 2019",
)
"""

# plot Mor+ data
ax1.step(
    mor_time,
    mor_sfh * normalisation_mor,
    where="mid",
    color="C4",
    linestyle="-.",
)
ax1.vlines(
    mor_time,
    (mor_sfh - mor_sigma_low) * normalisation_mor,
    (mor_sfh + mor_sigma_up) * normalisation_mor,
    linestyle="-.",
    color="C4",
    label="Mor+ 2019",
)

"""
# plot Torres+ data
ax1.step(
    torres_time,
    torres_sfh,
    where="mid",
    color="C5",
    label="Torres+ 2021",
)
"""

ax1.grid()
ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 14)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Lookback time [Gyr]")
ax1.set_ylabel(r"Star Formation Rate [N Gyr$^{-1}$ pc$^{-3}$]")
ax1.legend()


plt.tight_layout()

fig1.savefig(
    os.path.join(
        figure_folder,
        "fig_06_compare_sfh.png",
    )
)
