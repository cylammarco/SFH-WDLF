import os

from matplotlib import pyplot as plt
import numpy as np
from spectresc import spectres

plt.rcParams.update({"font.size": 12})

figure_folder = "SFH-WDLF-article/figures"
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


(
    partial_age_optimal,
    partial_age_duration,
    solution_optimal,
    solution_lower,
    solution_upper,
) = np.load("SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy").T
(
    partial_age_optimal,
    partial_age_duration,
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

_, _s, _vh = np.linalg.svd(solution_optimal_jac, full_matrices=False)
tol = np.finfo(float).eps * _s[0] * max(solution_optimal_jac.shape)
_w = _s > tol
cov = (_vh[_w].T / _s[_w] ** 2) @ _vh[_w]  # robust covariance matrix
stdev = np.sqrt(np.diag(cov))

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.insert(np.load("SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy"), 0, 0)
mag_obs_optimal, resolution_optimal = np.load("SFH-WDLF-article/figure_data/mbol_resolution.npy").T
mag_obs_optimal_bin_edges = np.append(
    mag_obs_optimal - resolution_optimal * 0.5,
    mag_obs_optimal[-1] + resolution_optimal[-1] * 0.5,
)

h_gen_optimal_20pc_subset, b_optimal_20pc_subset = np.histogram(
    gcns_wdlf_20pc_subset["Mbol"],
    bins=mag_obs_optimal_bin_edges,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf_20pc_subset["Vgen"],
)

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

# Stack up the pwdlfs to the desired resolution
partial_wdlf_optimal = []
partial_age_optimal = []

for idx in np.sort(list(set(pwdlf_mapping_bin_optimal))):
    pwdlf_temp = np.zeros_like(mag_obs_optimal)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_optimal == idx)[0]:
        pwdlf_temp += spectres(mag_obs_optimal, mag_pwdlf, data[i][:, 1], fill=0.0)
        age_temp = age[i]
    partial_wdlf_optimal.append(pwdlf_temp)
    partial_age_optimal.append(age_temp)


partial_wdlf_optimal_20pc_subset = np.vstack(partial_wdlf_optimal)
partial_wdlf_optimal = np.vstack(partial_wdlf_optimal)
partial_wdlf_optimal /= np.nansum(partial_wdlf_optimal)
partial_wdlf_optimal_20pc_subset /= np.nansum(partial_wdlf_optimal_20pc_subset)
partial_wdlf_optimal = partial_wdlf_optimal[obs_wdlf_optimal > 0.0][:, obs_wdlf_optimal > 0.0]
partial_wdlf_optimal_20pc_subset = partial_wdlf_optimal_20pc_subset[obs_wdlf_optimal_20pc_subset > 0.0][
    :, obs_wdlf_optimal_20pc_subset > 0.0
]
partial_wdlf_optimal_20pc_subset = np.array(partial_wdlf_optimal)[obs_wdlf_optimal_20pc_subset > 0.0][
    :, obs_wdlf_optimal_20pc_subset > 0.0
]
partial_age_optimal_20pc_subset = np.array(partial_age_optimal)[obs_wdlf_optimal_20pc_subset > 0.0]
#

# only compute for the wdlf integrated number density
recomputed_wdlf_optimal = np.nansum(solution_optimal * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_lsq = np.nansum(solution_optimal_lsq * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_20pc_subset = np.nansum(
    solution_optimal_20pc_subset * np.array(partial_wdlf_optimal_20pc_subset).T, axis=1
)
recomputed_wdlf_optimal_lsq_20pc_subset = np.nansum(
    solution_optimal_lsq_20pc_subset * np.array(partial_wdlf_optimal_20pc_subset).T, axis=1
)


# Cignoni+ 2006 (only relative SFH)
cignoni_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_cignoni_sfh.csv", delimiter=",")
cignoni_time = cignoni_data[:, 0]
cignoni_sfh = cignoni_data[:, 1]
cignoni_sigma_up = cignoni_data[:, 2] - cignoni_data[:, 1]
cignoni_sigma_low = cignoni_data[:, 1] - cignoni_data[:, 3]
# append for plotting the first bin
cignoni_time = np.insert(cignoni_time, 0, 2.0 * cignoni_time[0] - cignoni_time[1])
cignoni_sfh = np.insert(cignoni_sfh, 0, 0.0)
cignoni_sigma_up = np.insert(cignoni_sigma_up, 0, 0.0)
cignoni_sigma_low = np.insert(cignoni_sigma_low, 0, 0.0)
# append for plotting the last bin
cignoni_time = np.append(cignoni_time, 2.0 * cignoni_time[-1] - cignoni_time[-2])
cignoni_sfh = np.append(cignoni_sfh, 0.0)
cignoni_sigma_up = np.append(cignoni_sigma_up, 0.0)
cignoni_sigma_low = np.append(cignoni_sigma_low, 0.0)

# Isern 2019 (In mass per Gyr)
isern_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_isern_2019_sfh.csv", delimiter=",")
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
mor_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_mor_2019_sfh.csv", delimiter=",")
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
tremblay_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_tremblay_2014_sfh.csv", delimiter=",")
tremblay_time = tremblay_data[:, 0]
tremblay_sfh = tremblay_data[:, 1]
tremblay_sigma_up = tremblay_data[:, 2]
tremblay_sigma_low = tremblay_data[:, 3]
# append for plotting the first bin
tremblay_time = np.insert(tremblay_time, 0, 2.0 * tremblay_time[0] - tremblay_time[1])
tremblay_sfh = np.insert(tremblay_sfh, 0, 0.0)
tremblay_sigma_up = np.insert(tremblay_sigma_up, 0, 0.0)
tremblay_sigma_low = np.insert(tremblay_sigma_low, 0, 0.0)
# append for plotting the last bin
tremblay_time = np.append(tremblay_time, 2.0 * tremblay_time[-1] - tremblay_time[-2])
tremblay_sfh = np.append(tremblay_sfh, 0.0)
tremblay_sigma_up = np.append(tremblay_sigma_up, 0.0)
tremblay_sigma_low = np.append(tremblay_sigma_low, 0.0)

# Reid+ 2007 (only relative SFH)
reid_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_reid_2007_sfh.csv", delimiter=",")
reid_time = reid_data[:, 0]
reid_sfh = reid_data[:, 1]
# append for plotting the first bin
reid_time = np.insert(reid_time, 0, 2.0 * reid_time[0] - reid_time[1])
reid_sfh = np.insert(reid_sfh, 0, 0.0)
# append for plotting the last bin
reid_time = np.append(reid_time, 2.0 * reid_time[-1] - reid_time[-2])
reid_sfh = np.append(reid_sfh, 0.0)

# Bernard+ 2018
bernard_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_bernard_2018_sfh.csv", delimiter=",")
bernard_time = bernard_data[:, 0]
bernard_sfh = bernard_data[:, 1]
# append for plotting the first bin
bernard_time = np.insert(bernard_time, 0, 2.0 * bernard_time[0] - bernard_time[1])
bernard_sfh = np.insert(bernard_sfh, 0, 0.0)
# append for plotting the last bin
bernard_time = np.append(bernard_time, 2.0 * bernard_time[-1] - bernard_time[-2])
bernard_sfh = np.append(bernard_sfh, 0.0)


# Torres+ 2021
torres_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_torres_2021_sfh.csv", delimiter=",")
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
xiang_sfh, xiang_time_bin_edges = np.histogram(xiang_age, bins=75, range=(0, 14))
xiang_time = np.diff(xiang_time_bin_edges) * 0.5 + xiang_time_bin_edges[:-1]


# Rowell 2013
rowell_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_rowell_2013_sfh.txt")
rowell_time = rowell_data[:, 0] / 1e9
rowell_sfh = rowell_data[:, 1] * 1e9

# Rowell 2023
rowell_2023_data = np.loadtxt(r"SFH-WDLF-article/figure_data/fig_11_rowell_2023_sfh.txt")
rowell_2023_time = rowell_2023_data[:, 0] / 1e9
rowell_2023_sfh = rowell_2023_data[:, 2] * 1e9

# Alzate+ 2021
alzate_data = np.loadtxt("SFH-WDLF-article/figure_data/fig_11_alzate_fig6d.csv")
alzate_2021_time = alzate_data[:, 0]
alzate_2021_sfh = alzate_data[:, 1] + alzate_data[:, 2] + alzate_data[:, 3] + alzate_data[:, 4]
alzate_2021_time = np.append(alzate_2021_time, 15.0)
alzate_2021_sfh = np.append(alzate_2021_sfh, 0.0)

# Gallart+ 2024
gallart_data = np.loadtxt("SFH-WDLF-article/figure_data/fig_11_gallart.csv", comments="#", delimiter=",")
gallart_age = gallart_data[:, 0]
gallart_sfh = gallart_data[:, 1] * 1e4

# Nataf+ 2024
nataf_data = np.loadtxt("SFH-WDLF-article/figure_data/fig_11_nataf.csv", comments="#", delimiter=",", dtype=str)
nataf_mass = np.array(nataf_data[:, 10]).astype("float")
nataf_age = 10.0**np.array(nataf_data[:, 13]).astype("float") * 1e-9

# Alcazar+ 2025
alcazar_data = np.loadtxt("SFH-WDLF-article/figure_data/fig_11_delAlcazarJulia.csv")
alcazar_2025_age_low = alcazar_data[:, 0]
alcazar_2025_age_high = alcazar_data[:, 1]
alcazar_2025_sfh = alcazar_data[:, 2]

alcazar_2025_age_low = np.append(alcazar_2025_age_low, alcazar_2025_age_high[-1])
alcazar_2025_age_high = np.append(alcazar_2025_age_high, 15.0)
alcazar_2025_sfh = np.append(alcazar_2025_sfh, 0.0)

"""
# Fantin+ 2019
fantin_data = np.loadtxt(
    r"SFH-WDLF-article/figure_data/fig_11_fantin_2019_sfh.csv", delimiter=","
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
        (np.diff(partial_age_optimal)[:-1] + np.diff(partial_age_optimal)[1:]) / 2.0,
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

normalisation_this_work = np.sum(obs_wdlf_optimal * resolution_optimal) / np.sum(recomputed_wdlf_optimal_lsq)
normalisation_this_work_20pc_subset = np.sum(obs_wdlf_optimal_20pc_subset * resolution_optimal) / np.sum(
    recomputed_wdlf_optimal_lsq_20pc_subset
)

# These are to normalise to match the GCNS WDLF integrated number density
normalisation_cignoni = np.sum(obs_wdlf_optimal) / (cignoni_sfh @ cignoni_time) / 0.6
normalisation_mor = 1.0
normalisation_tremblay = np.sum(obs_wdlf_optimal) / np.sum(tremblay_sfh @ tremblay_time)
normalisation_reid = np.sum(solution_optimal_lsq) / np.sum(reid_sfh) * 0.1

# get the bootstrapped SFH

sfh_list = []
wdlf_list = []
sfh_20pc_list = []
wdlf_20pc_list = []

for i in range(1000):
    sfh = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_sfh_sample_{i}.csv").T[1]
    wdlf = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_reconstructed_wdlf_sample_{i}.csv").T[1]
    sfh_list.append(sfh)
    wdlf_list.append(wdlf)


sfh_age = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_sfh_sample_{i}.csv").T[0]
sfh_mean = np.mean(sfh_list, axis=0)
sfh_stdev = np.std(sfh_list, axis=0)
wdlf_mag = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_reconstructed_wdlf_sample_{i}.csv").T[0]
wdlf_mean = np.mean(wdlf_list, axis=0)
wdlf_stdev = np.std(wdlf_list, axis=0)


fig6, (ax1, ax2, ax3, ax4) = plt.subplots(
    nrows=4, ncols=1, figsize=(10, 15), height_ratios=(10, 10, 10, 10), sharex=True
)
# ax1 for SFH in the unit of N / Gyr / pc^3
# ax2 for SFH in the unit of M_sun / Gyr / pc^3
# ax3 for SFH in the unit of number count

# plot data from this work
ax1.step(
    sfh_age,
    solution_optimal_lsq / np.sum(solution_optimal) * normalisation_this_work,
    where="mid",
    color="grey",
    label="pWDLF best-fit (this work)",
)
ax1.step(
    sfh_age,
    sfh_mean / np.nansum(solution_optimal) * normalisation_this_work,
    where="mid",
    color="black",
    label="pWDLF bootstrap mean (this work)",
)


# ax1.step(
#    rowell_time,
#    rowell_sfh,
#    where="mid",
#    label="Rowell 2013",
#    color="grey",
# )

ax1.step(
    rowell_2023_time,
    rowell_2023_sfh,
    where="mid",
    label="Rowell (this work)",
    color="blue",
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

ax2.plot(
    gallart_age,
    gallart_sfh / (4.0 / 3.0 * np.pi * 1e6) / 5.0,
    color="C3",
    label=r"Gallart+ 2024 [$\times$0.2]",
)

# plot Mor+ data
ax3.step(
    mor_time,
    mor_sfh,
    where="mid",
    color="C4",
    label="Mor+ 2019",
)

# plot Alcazar+ 2025 data
ax3.step(
    alcazar_2025_age_low,
    alcazar_2025_sfh,
    color="C5",
    where="post",
    label=r"del Alc$\grave{a}$zar-Juli$\grave{a}$+2025",
)


# Number count


# plot Reid+ data
ax4.step(
    reid_time,
    reid_sfh / np.nanmax(reid_sfh),
    where="mid",
    label="Reid+ 2007",
)

# plot Tremblay+ data
ax4.step(
    tremblay_time,
    tremblay_sfh / np.nanmax(tremblay_sfh),
    where="mid",
    label="Tremblay+ 2014",
)

# plot Torres+ data (N)
ax4.step(
    torres_time,
    torres_sfh / np.nanmax(torres_sfh),
    where="mid",
    label="Torres+ 2021",
)


# plot Xiang+ data (N)
ax4.step(
    xiang_time,
    xiang_sfh / np.nanmax(xiang_sfh),
    where="mid",
    label="Xiang+ 2022",
)

ax4.step(alzate_2021_time, alzate_2021_sfh / np.max(alzate_2021_sfh), where="mid", label="Alzate+ 2021")

h, b = np.histogram(nataf_age, bins=75, range=(0, 15))
ax4.step(b[:-1], h/max(h), where="post", label="Nataf+ 2024")

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 14)

ax1.set_ylim(0, 0.0055)
ax2.set_ylim(0, 0.0055)
ax3.set_ylim(0, 15.0)
ax4.set_ylim(0, 1.25)


ax4.set_xlabel("Lookback time [Gyr]")

ax1.set_ylabel(r"Star Formation Rate [N Gyr$^{-1}$ pc$^{-3}$]")
ax2.set_ylabel(r"Star Formation Rate [M$_{\odot}$ Gyr$^{-1}$ pc$^{-3}$]")
ax3.set_ylabel(r"Star Formation Rate [M$_{\odot}$ Gyr$^{-1}$ pc$^{-2}$]")
ax4.set_ylabel(r"Star Formation Rate [N (renormalised)]")


ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")
ax4.legend(loc="upper right", ncol=3)

plt.tight_layout()
plt.subplots_adjust(hspace=0)


fig6.savefig(
    os.path.join(
        figure_folder,
        "fig_10_compare_sfh.png",
    )
)
