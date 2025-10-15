import os

from matplotlib import pyplot as plt
import numpy as np
from pynverse import inversefunc
from spectresc import spectres
from scipy.integrate import quad
from scipy import interpolate


def imf_func(m):
    m = np.ravel(m).astype(float)
    mf = m**-2.3
    mask = m < 1.0
    if mask.any():
        # 0.158 / (ln(10) * mass_ms) = 0.06861852814 / mass_ms
        # log(0.079) = -1.1023729087095586
        # 2 * 0.69**2. = 0.9522
        # Normalisation factor (at mass_ms=1) is 0.01915058
        norm = 0.01915058
        factor = 0.06861852814
        logconst = 1.1023729087095586
        sigma_sq = 0.9522
        mf[mask] = (factor / m[mask]) * np.exp(-((np.log10(m[mask]) + logconst) ** 2) / sigma_sq) / norm
    return mf


plt.rcParams.update({"font.size": 12})

figure_folder = "SFH-WDLF-article/figures"

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

mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag_pwdlf[np.argmax(d[:, 1])]


mag_resolution_itp = interpolate.UnivariateSpline(age, mag_at_peak_density, s=len(age) / 150, k=5)
age_resolution_itp = inversefunc(mag_resolution_itp)


(
    partial_age_optimal,
    partial_age_duration,
    solution_optimal,
    solution_lower,
    solution_upper,
) = np.load("SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy").T

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
# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_optimal = np.load("SFH-WDLF-article/figure_data/pwdlf_bin_optimal_mapping.npy")
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

obs_wdlf_optimal = h_gen_optimal / resolution_optimal
obs_wdlf_err_optimal = e_gen_optimal**0.5 / resolution_optimal

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


partial_wdlf_optimal = np.vstack(partial_wdlf_optimal)
partial_wdlf_optimal /= np.nansum(partial_wdlf_optimal)
partial_wdlf_optimal = partial_wdlf_optimal[obs_wdlf_optimal > 0.0][:, obs_wdlf_optimal > 0.0]

plt.figure(1, figsize=(8, 6))
plt.clf()
for i, _wdlf in enumerate(partial_wdlf_optimal[:-1]):
    plt.plot(
        mag_obs_optimal,
        _wdlf,
        color="C0",
        alpha=0.3 + 0.7 * i / len(partial_wdlf_optimal),
        lw=0.5,
    )

plt.xlim(6.0, 17.5)
plt.ylim(2e-7, 2e-2)
plt.xlabel("Mbol [mag]")
plt.ylabel(r"Arbitrary number density [N pc$^{-3}$ mag$^{-1}$]")
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    os.path.join(
        figure_folder,
        "fig_04_basis_pwdlf.png",
    )
)


recomputed_wdlf_optimal = np.nansum(solution_optimal * np.array(partial_wdlf_optimal).T, axis=1)
recomputed_wdlf_optimal_lsq = np.nansum(solution_optimal_lsq * np.array(partial_wdlf_optimal).T, axis=1)

wdlf_err_low = np.nansum((solution_lower) * np.array(partial_wdlf_optimal).T, axis=1)
wdlf_err_high = np.nansum((solution_upper) * np.array(partial_wdlf_optimal).T, axis=1)

# append for plotting the first bin
solution_optimal_lsq = np.insert(solution_optimal_lsq, 0, 0.0)
solution_optimal = np.insert(solution_optimal, 0, 0.0)
solution_upper = np.insert(solution_upper, 0, 0.0)
solution_lower = np.insert(solution_lower, 0, 0.0)
# append for plotting the last bin
solution_optimal_lsq = np.append(solution_optimal_lsq, 0.0)
solution_optimal = np.append(solution_optimal, 0.0)
solution_upper = np.append(solution_upper, 0.0)
solution_lower = np.append(solution_lower, 0.0)


imf_normalisation = quad(imf_func, 0.9, 8.0)[0]
normalisation_this_work = (
    np.sum(obs_wdlf_optimal * resolution_optimal) / np.sum(recomputed_wdlf_optimal_lsq) / imf_normalisation
)
partial_age_optimal_padded = np.insert(partial_age_optimal, 0, 0.0)
partial_age_optimal_padded = np.append(partial_age_optimal_padded, 15.0)

fig1, (ax1, ax_dummy1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), height_ratios=(15, 2, 15))


ax1.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal / np.nansum(recomputed_wdlf_optimal) * np.nansum(obs_wdlf_optimal),
    label="Reconstructed WDLF (MCMC)",
    color="C00",
)
ax1.plot(
    mag_obs_optimal,
    recomputed_wdlf_optimal_lsq / np.nansum(recomputed_wdlf_optimal_lsq) * np.nansum(obs_wdlf_optimal),
    label="Reconstructed WDLF (lsq)",
    color="C01",
    ls="dashed",
)
ax1.errorbar(
    mag_obs_optimal,
    obs_wdlf_optimal,
    yerr=[obs_wdlf_err_optimal, obs_wdlf_err_optimal],
    fmt="+",
    markersize=5,
    label="Input WDLF",
    color="black",
    alpha=0.7,
)

ax1.fill_between(
    mag_obs_optimal,
    wdlf_err_low / np.nansum(recomputed_wdlf_optimal_lsq) * np.nansum(obs_wdlf_optimal),
    wdlf_err_high / np.nansum(recomputed_wdlf_optimal_lsq) * np.nansum(obs_wdlf_optimal),
    color="lightgrey",
)


ax1.xaxis.set_ticks(np.arange(6.0, 18.1, 1.0))
ax1.set_xlabel(r"M${_\mathrm{bol}}$ [mag]")
ax1.set_ylabel("log(number density) [N pc$^{-3}$ mag$^{-1}$]")
ax1.set_xlim(5.75, 18.25)
ax1.set_ylim(1e-6, 3e-3)
ax1.set_yscale("log")
ax1.legend(loc="lower center")
ax1.grid()

# Get the Mbol to Age relation
age_ticks = age_resolution_itp(np.arange(6.0, 18.1, 0.5))
age_ticklabels = [f"{i:.3f}" for i in age_ticks]


# make the top axis
ax1b = ax1.twiny()
ax1b.set_xlim(ax1.get_xlim())
ax1b.set_xticks(ax1.get_xticks())
ax1b.xaxis.set_ticks(np.arange(6.0, 18.1, 0.5))
ax1b.xaxis.set_ticklabels(age_ticklabels, rotation=90)
ax1b.set_xlabel("Lookback time [Gyr]")

ax_dummy1.axis("off")

ax2.step(
    partial_age_optimal_padded,
    solution_optimal / np.nansum(solution_optimal) * normalisation_this_work,
    where="mid",
    label="MCMC",
)
ax2.step(
    partial_age_optimal_padded,
    solution_optimal_lsq / np.nansum(solution_optimal) * normalisation_this_work,
    where="mid",
    label="lsq",
    ls="dashed",
)
ax2.fill_between(
    partial_age_optimal_padded,
    solution_lower / np.nansum(solution_optimal) * normalisation_this_work,
    solution_upper / np.nansum(solution_optimal) * normalisation_this_work,
    step="mid",
    color="lightgrey",
)
ax2.grid()
ax2.set_xticks(np.arange(0, 15, 2))
ax2.set_xlim(0, 14)
ax2.set_ylim(bottom=0)
ax2.set_xlabel("Lookback time [Gyr]")
ax2.set_ylabel("log(number density) [N pc$^{-3}$ Gyr$^{-1}$]")
ax2.legend()

plt.subplots_adjust(top=0.9, bottom=0.06, left=0.125, right=0.98, hspace=0.00)

fig1.savefig(
    os.path.join(
        figure_folder,
        "fig_05_gcns_reconstructed_wdlf_optimal_resolution_bin_optimal.png",
    )
)

# Prepare to output CSV of the reconstructed WDLFs
wdlf_output = recomputed_wdlf_optimal_lsq / np.nansum(recomputed_wdlf_optimal_lsq) * np.nansum(obs_wdlf_optimal)

wdlf_err_output = obs_wdlf_err_optimal

csv_output = np.column_stack(
    [
        mag_obs_optimal,
        wdlf_output,
        wdlf_err_output,
    ]
)

np.savetxt(
    "SFH-WDLF-article/figure_data/fig_05_gcns_reconstructed_wdlf.csv",
    csv_output,
    fmt="%.6e",
)
