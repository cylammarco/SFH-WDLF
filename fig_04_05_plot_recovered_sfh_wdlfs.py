import os

from matplotlib import pyplot as plt
import numpy as np
from pynverse import inversefunc
from spectres import spectres
from scipy import interpolate


plt.rcParams.update({"font.size": 12})

figure_folder = "SFH-WDLF-article/figures"

# Load the GCNS data
gcns_wdlf = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]
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

mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag_pwdlf[np.argmax(d[:, 1])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=len(age) / 150, k=5
)
age_resolution_itp = inversefunc(mag_resolution_itp)


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

_, _s_20pc_subset, _vh_20pc_subset = np.linalg.svd(
    solution_optimal_jac_20pc_subset, full_matrices=False
)
tol_20pc_subset = (
    np.finfo(float).eps
    * _s_20pc_subset[0]
    * max(solution_optimal_jac_20pc_subset.shape)
)
_w_20pc_subset = _s_20pc_subset > tol_20pc_subset
cov_20pc_subset = (
    _vh_20pc_subset[_w_20pc_subset].T / _s_20pc_subset[_w_20pc_subset] ** 2
) @ _vh_20pc_subset[
    _w_20pc_subset
]  # robust covariance matrix
stdev_20pc_subset = np.sqrt(np.diag(cov_20pc_subset))

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

plt.figure(1, figsize=(8, 6))
plt.clf()
for i, _wdlf in enumerate(partial_wdlf_optimal):
    plt.plot(
        mag_obs_optimal,
        _wdlf,
        color="C0",
        alpha=0.3 + 0.7 * i / len(partial_wdlf_optimal),
        lw=0.5,
    )

plt.xlim(6.0, 18.0)
plt.ylim(1e5, 3e9)
plt.xlabel("Mbol [mag]")
plt.ylabel("Arbitrary number density")
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    os.path.join(
        figure_folder,
        "fig_04_basis_pwdlf.png",
    )
)


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


wdlf_err_low = np.nansum(
    (solution_lower) * np.array(partial_wdlf_optimal).T, axis=1
)
wdlf_err_high = np.nansum(
    (solution_upper) * np.array(partial_wdlf_optimal).T, axis=1
)
wdlf_err_low_20pc_subset = np.nansum(
    (solution_lower_20pc_subset) * np.array(partial_wdlf_optimal).T, axis=1
)
wdlf_err_high_20pc_subset = np.nansum(
    (solution_upper_20pc_subset) * np.array(partial_wdlf_optimal).T, axis=1
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


normalisation_this_work = (
    np.sum(obs_wdlf_optimal) / np.sum(recomputed_wdlf_optimal_lsq) * 1e9
)
normalisation_this_work_20pc_subset = (
    np.sum(obs_wdlf_optimal_20pc_subset)
    / np.sum(recomputed_wdlf_optimal_lsq_20pc_subset)
    * 1e9
)
bin_norm_this_work = np.concatenate(
    [
        [partial_age_optimal[1] - partial_age_optimal[0]],
        (np.diff(partial_age_optimal)[:-1] + np.diff(partial_age_optimal)[1:])
        / 2.0,
        [partial_age_optimal[-1] - partial_age_optimal[-2]],
    ]
)

fig1, (ax2, ax_dummy1, ax1) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(15, 2, 10)
)

ax1.step(
    partial_age_optimal,
    solution_optimal * normalisation_this_work / bin_norm_this_work,
    where="mid",
    label="MCMC",
)
ax1.step(
    partial_age_optimal,
    solution_optimal_lsq * normalisation_this_work / bin_norm_this_work,
    where="mid",
    label="lsq",
)
ax1.fill_between(
    partial_age_optimal,
    solution_lower * normalisation_this_work / bin_norm_this_work,
    solution_upper * normalisation_this_work / bin_norm_this_work,
    step="mid",
    color="lightgrey",
)


ax1.step(
    partial_age_optimal,
    solution_optimal_20pc_subset
    * 100
    * normalisation_this_work_20pc_subset
    / bin_norm_this_work,
    where="mid",
    label="MCMC (20pc subset) [x100]",
)
ax1.step(
    partial_age_optimal,
    solution_optimal_lsq_20pc_subset
    * 100
    * normalisation_this_work_20pc_subset
    / bin_norm_this_work,
    where="mid",
    label="lsq (20pc subset) [x100]",
)
ax1.fill_between(
    partial_age_optimal,
    solution_lower_20pc_subset
    * normalisation_this_work_20pc_subset
    / bin_norm_this_work,
    solution_upper_20pc_subset
    * normalisation_this_work_20pc_subset
    / bin_norm_this_work,
    step="mid",
    color="lightgrey",
)


ax1.grid()
ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 14)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Lookback time [Gyr]")
ax1.set_ylabel("Star Formation Rate [N Gyr$^{-1}$ pc$^{-3}$]")
ax1.legend()

ax_dummy1.axis("off")
ax2.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal
    / np.nansum(recomputed_wdlf_optimal)
    * np.nansum(obs_wdlf_optimal),
    label="Reconstructed WDLF (MCMC)",
    color="C00",
)
ax2.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal_lsq
    / np.nansum(recomputed_wdlf_optimal_lsq)
    * np.nansum(obs_wdlf_optimal),
    label="Reconstructed WDLF (lsq)",
    color="C01",
)
ax2.errorbar(
    mag_obs_optimal,
    obs_wdlf_optimal,
    yerr=[obs_wdlf_err_optimal, obs_wdlf_err_optimal],
    fmt="+",
    markersize=5,
    label="Input WDLF",
    color="black",
    alpha=0.7,
)

ax2.fill_between(
    mag_obs_optimal,
    wdlf_err_low
    / np.nansum(recomputed_wdlf_optimal_lsq)
    * np.nansum(obs_wdlf_optimal),
    wdlf_err_high
    / np.nansum(recomputed_wdlf_optimal_lsq)
    * np.nansum(obs_wdlf_optimal),
    color="lightgrey",
)

ax2.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal_20pc_subset
    / np.nansum(recomputed_wdlf_optimal_20pc_subset)
    * np.nansum(obs_wdlf_optimal_20pc_subset),
    label="Reconstructed WDLF (MCMC, 20pc subset)",
    color="C02",
)
ax2.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal_lsq_20pc_subset
    / np.nansum(recomputed_wdlf_optimal_lsq_20pc_subset)
    * np.nansum(obs_wdlf_optimal_20pc_subset),
    label="Reconstructed WDLF (lsq, 20pc subset)",
    color="C03",
)
ax2.errorbar(
    mag_obs_optimal,
    obs_wdlf_optimal_20pc_subset,
    yerr=[obs_wdlf_err_optimal_20pc_subset, obs_wdlf_err_optimal_20pc_subset],
    fmt="+",
    markersize=5,
    color="black",
    alpha=0.3,
)

ax2.xaxis.set_ticks(np.arange(6.0, 18.1, 1.0))
ax2.set_xlabel(r"M${_\mathrm{bol}}$ [mag]")
ax2.set_ylabel("log(number density)")
ax2.set_xlim(5.75, 18.25)
ax2.set_ylim(5e-9, 3e-3)
ax2.set_yscale("log")
ax2.legend(loc="lower center")
ax2.grid()

# Get the Mbol to Age relation
age_ticks = age_resolution_itp(np.arange(6.0, 18.1, 0.5))
age_ticklabels = [f"{i:.3f}" for i in age_ticks]


# make the top axis
ax2b = ax2.twiny()
ax2b.set_xlim(ax2.get_xlim())
ax2b.set_xticks(ax2.get_xticks())
ax2b.xaxis.set_ticks(np.arange(6.0, 18.1, 0.5))
ax2b.xaxis.set_ticklabels(age_ticklabels, rotation=90)
ax2b.set_xlabel("Lookback time [Gyr]")

"""
# change colour of the upper axis
ax2b.xaxis.label.set_color('blue')
ax2b.spines['top'].set_color('blue')
ax2b.tick_params(axis='x', colors='blue')
"""

plt.subplots_adjust(top=0.9, bottom=0.06, left=0.12, right=0.98, hspace=0.01)

fig1.savefig(
    os.path.join(
        figure_folder,
        "fig_05_gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.png",
    )
)

# Prepare to output CSV of the reconstructed WDLFs
wdlf_output = (
    recomputed_wdlf_optimal_lsq_20pc_subset
    / np.nansum(recomputed_wdlf_optimal_lsq_20pc_subset)
    * np.nansum(obs_wdlf_optimal_20pc_subset)
)

wdlf_err_output = obs_wdlf_optimal_20pc_subset
wdlf_20_output = (
    recomputed_wdlf_optimal_lsq_20pc_subset
    / np.nansum(recomputed_wdlf_optimal_lsq_20pc_subset)
    * np.nansum(obs_wdlf_optimal_20pc_subset)
)

wdlf_20_err_output = obs_wdlf_optimal_20pc_subset


csv_output = np.column_stack(
    [
        mag_obs_optimal,
        wdlf_output,
        wdlf_err_output,
        wdlf_20_output,
        wdlf_20_err_output,
    ]
)

np.savetxt(
    "SFH-WDLF-article/figure_data/fig_05_gcns_reconstructed_wdlf.csv",
    csv_output,
    fmt='%.6e'
)
