import os

from matplotlib import pyplot as plt
import numpy as np
from spectres import spectres
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader


plt.rcParams.update({'font.size': 12})

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
) = np.load("gcns_sfh_optimal_resolution_bin_optimal.npy").T
# The lsq solution
lsq_res = np.load(
    "gcns_sfh_optimal_resolution_lsq_solution.npy", allow_pickle=True
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
    "gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.npy"
).T

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.insert(
    np.load("pwdlf_bin_optimal_mapping.npy"), 0, 0
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


plt.figure(1, figsize=(8, 8))
plt.clf()
for i in partial_wdlf_optimal:
    plt.plot(mag_obs_optimal, i, color="black", lw=0.5)

plt.xlim(6.0, 18.0)
plt.xlabel("Mbol / mag")
plt.ylabel("Arbitrary number density")
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    os.path.join(
        figure_folder,
        "fig_05_basis_pwdlf.png",
    )
)


recomputed_wdlf_optimal = np.nansum(
    solution_optimal * np.array(partial_wdlf_optimal).T, axis=1
)
recomputed_wdlf_optimal_lsq = np.nansum(
    solution_optimal_lsq * np.array(partial_wdlf_optimal).T, axis=1
)

wdlf_err_low = np.nansum(
    (solution_lower) * np.array(partial_wdlf_optimal).T, axis=1
)
wdlf_err_high = np.nansum(
    (solution_upper) * np.array(partial_wdlf_optimal).T, axis=1
)

fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(10, 3, 15)
)

ax1.step(partial_age_optimal, solution_optimal, where="post", label="MCMC")
ax1.step(partial_age_optimal, solution_optimal_lsq, where="post", label="lsq")
ax1.fill_between(
    partial_age_optimal,
    solution_lower,
    solution_upper,
    step="post",
    color="lightgrey",
)

ax1.grid()
ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 15)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Lookback time / Gyr")
ax1.set_ylabel("Relative Star Formation Rate")
ax1.legend()

ax_dummy1.axis("off")
ax2.errorbar(
    mag_obs_optimal,
    obs_wdlf_optimal,
    yerr=[obs_wdlf_err_optimal, obs_wdlf_err_optimal],
    fmt="+",
    markersize=5,
    label="Input WDLF",
)
ax2.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal
    / np.nansum(recomputed_wdlf_optimal)
    * np.nansum(obs_wdlf_optimal),
    label="Reconstructed WDLF (MCMC)",
)
ax2.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal_lsq
    / np.nansum(recomputed_wdlf_optimal_lsq)
    * np.nansum(obs_wdlf_optimal),
    label="Reconstructed WDLF (lsq)",
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

ax2.xaxis.set_ticks(np.arange(6.0, 18.1, 1.0))
ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_xlim(5.75, 18.25)
ax2.set_ylim(1e-6, 5e-3)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()

# Get the Mbol to Age relation
atm = AtmosphereModelReader()
Mbol_to_age = atm.interp_am(dependent="age")
age_ticks = Mbol_to_age(8.0, np.arange(6.0, 18.1, 1.0))
age_ticklabels = [f"{i/1e9:.3f}" for i in age_ticks]

# make the top axis
ax2b = ax2.twiny()
ax2b.set_xlim(ax2.get_xlim())
ax2b.set_xticks(ax2.get_xticks())
ax2b.xaxis.set_ticklabels(age_ticklabels, rotation=90)

plt.subplots_adjust(
    top=0.995, bottom=0.06, left=0.1, right=0.995, hspace=0.01
)

fig1.savefig(
    os.path.join(
        figure_folder,
        "fig_06_gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.png",
    )
)
