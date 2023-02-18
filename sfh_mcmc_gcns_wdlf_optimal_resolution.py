import os

import emcee
from matplotlib import pyplot as plt
import numpy as np
from spectres import spectres
from WDPhotTools import theoretical_lf


def log_prior(theta):
    if (theta < 0.0).any():
        return -np.inf
    else:
        return 0.0


def log_probability(rel_norm, obs, err, model_list):
    rel_norm /= np.sum(rel_norm)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = np.nansum(rel_norm[:, None] * model_list, axis=0)
    model /= np.nansum(model)
    obs /= np.nansum(obs)
    log_likelihood = -np.nansum(((model - obs) / err) ** 2.0)
    return log_likelihood


# Load the GCNS data
gcns_wdlf = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]
n_bin_02 = 84
n_bin_05 = 32

h_gen_02, b_02 = np.histogram(
    gcns_wdlf["Mbol"],
    bins=n_bin_02,
    range=(1.9, 18.6),
    weights=0.01 / gcns_wdlf["Vgen"],
)
h_gen_05, b_05 = np.histogram(
    gcns_wdlf["Mbol"],
    bins=n_bin_05,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf["Vgen"],
)
e_gen_02, _ = np.histogram(
    gcns_wdlf["Mbol"],
    bins=n_bin_02,
    range=(1.9, 18.6),
    weights=0.01 / gcns_wdlf["Vgen"] ** 2.0,
)
e_gen_05, _ = np.histogram(
    gcns_wdlf["Mbol"],
    bins=n_bin_05,
    range=(2.25, 18.25),
    weights=0.01 / gcns_wdlf["Vgen"] ** 2.0,
)

bin_size_02 = b_02[1] - b_02[0]
bin_size_05 = b_05[1] - b_05[0]

mag_obs_02 = np.around(b_02[1:] - 0.5 * bin_size_02, 2)
obs_wdlf_02 = h_gen_02 / bin_size_02
obs_wdlf_err_02 = e_gen_02**0.5 / bin_size_02

mag_obs_05 = np.around(b_05[1:] - 0.5 * bin_size_05, 2)
obs_wdlf_05 = h_gen_05 / bin_size_05
obs_wdlf_err_05 = e_gen_05**0.5 / bin_size_05

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

# Load the mapped pwdlf age-mag resolution
pwdlf_mapping_bin_02 = np.insert(np.load("pwdlf_bin_02_mapping.npy"), 0, 0)
pwdlf_mapping_bin_05 = np.insert(np.load("pwdlf_bin_05_mapping.npy"), 0, 0)

# Stack up the pwdlfs to the desired resolution
partial_wdlf_02 = []
partial_age_02 = []
for idx in np.sort(list(set(pwdlf_mapping_bin_02))):
    pwdlf_temp = np.zeros_like(mag_obs_02)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_02 == idx)[0]:
        pwdlf_temp += spectres(mag_obs_02, mag_pwdlf, data[i][:, 1], fill=0.0)
        age_temp += age[i]
        age_count += 1
    partial_wdlf_02.append(pwdlf_temp)
    partial_age_02.append(age_temp / age_count)


partial_wdlf_05 = []
partial_age_05 = []
for idx in np.sort(list(set(pwdlf_mapping_bin_05))):
    pwdlf_temp = np.zeros_like(mag_obs_05)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_bin_05 == idx)[0]:
        pwdlf_temp += spectres(mag_obs_05, mag_pwdlf, data[i][:, 1], fill=0.0)
        age_temp += age[i]
        age_count += 1
    partial_wdlf_05.append(pwdlf_temp)
    partial_age_05.append(age_temp / age_count)


"""
plt.figure(1)
plt.clf()
for wdlf in partial_wdlf_05:
    plt.plot(mag_obs, np.log10(wdlf), color='black')

plt.yscale("log")

plt.xlabel(r"M$_{\mathrm{bol}}$")
plt.ylabel("Relative number density")
"""


pwdlf_model_02 = np.vstack(partial_wdlf_02)[:, obs_wdlf_02 > 0.0]
pwdlf_model_05 = np.vstack(partial_wdlf_05)[:, obs_wdlf_05 > 0.0]

nwalkers_02 = 2000
nwalkers_05 = 500

ndim_02 = len(partial_wdlf_02)
ndim_05 = len(partial_wdlf_05)

rel_norm_02 = np.random.rand(nwalkers_02, ndim_02)
rel_norm_05 = np.random.rand(nwalkers_05, ndim_05)

sampler_02 = emcee.EnsembleSampler(
    nwalkers_02,
    ndim_02,
    log_probability,
    args=(
        obs_wdlf_02[obs_wdlf_02 > 0.0],
        obs_wdlf_err_02[obs_wdlf_02 > 0.0],
        pwdlf_model_02,
    ),
)
sampler_05 = emcee.EnsembleSampler(
    nwalkers_05,
    ndim_05,
    log_probability,
    args=(
        obs_wdlf_05[obs_wdlf_05 > 0.0],
        obs_wdlf_err_05[obs_wdlf_05 > 0.0],
        pwdlf_model_05,
    ),
)
sampler_02.run_mcmc(rel_norm_02, 50000, progress=True)
sampler_05.run_mcmc(rel_norm_05, 2000, progress=True)

flat_samples_02 = sampler_02.get_chain(discard=5000, thin=5, flat=True)
flat_samples_05 = sampler_05.get_chain(discard=500, thin=5, flat=True)

solution_02 = np.zeros(ndim_02)
for i in range(ndim_02):
    solution_02[i] = np.percentile(flat_samples_02[:, i], [50.0])

solution_05 = np.zeros(ndim_05)
for i in range(ndim_05):
    solution_05[i] = np.percentile(flat_samples_05[:, i], [50.0])

solution_02 /= np.nansum(solution_02)
solution_05 /= np.nansum(solution_05)

plt.figure(2)
plt.clf()
plt.step(partial_age_02, solution_02, where="post", label="MCMC")
plt.legend()
plt.grid()
plt.xticks(np.arange(0, 15, 2))
plt.xlim(0, 15)
plt.ylim(bottom=0)
plt.xlabel("Lookback time / Gyr")
plt.ylabel("Relative Star Formation Rate")
plt.tight_layout()
plt.savefig(
    os.path.join("test_case", "gcns_sfh_optimal_resolution_bin_02.png")
)

plt.figure(3)
plt.clf()
plt.step(partial_age_05, solution_05, where="post", label="MCMC")
plt.legend()
plt.grid()
plt.xticks(np.arange(0, 15, 2))
plt.xlim(0, 15)
plt.ylim(bottom=0)
plt.xlabel("Lookback time / Gyr")
plt.ylabel("Relative Star Formation Rate")
plt.tight_layout()
plt.savefig(
    os.path.join("test_case", "gcns_sfh_optimal_resolution_bin_05.png")
)

recomputed_wdlf_02 = np.nansum(
    solution_02 * np.array(partial_wdlf_02).T, axis=1
)
recomputed_wdlf_05 = np.nansum(
    solution_05 * np.array(partial_wdlf_05).T, axis=1
)

plt.figure(3)
plt.clf()
plt.errorbar(
    mag_obs_02,
    obs_wdlf_02,
    yerr=[obs_wdlf_err_02, obs_wdlf_err_02],
    fmt="+",
    markersize=5,
    label="Input WDLF",
)
plt.plot(
    mag_obs_02,
    recomputed_wdlf_02
    / np.nansum(recomputed_wdlf_02)
    * np.nansum(obs_wdlf_02),
    label="Reconstructed WDLF",
)
plt.xlabel(r"M${_\mathrm{bol}}$ / mag")
plt.ylabel("log(arbitrary number density)")
plt.xlim(0.0, 20.0)
plt.ylim(1e-7, 5e-3)
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    os.path.join(
        "test_case", "gcns_reconstructed_wdlf_optimal_resolution_bin_02.png"
    )
)

np.save(
    os.path.join("test_case", "gcns_sfh_optimal_resolution_bin_02.npy"),
    np.column_stack((partial_age_02, solution_02)),
)
np.save(
    os.path.join(
        "test_case", "gcns_reconstructed_wdlf_optimal_resolution_bin_02.npy"
    ),
    np.column_stack((mag_obs_02, obs_wdlf_02, obs_wdlf_err_02)),
)

plt.figure(4)
plt.clf()
plt.errorbar(
    mag_obs_05,
    obs_wdlf_05,
    yerr=[obs_wdlf_err_05, obs_wdlf_err_05],
    fmt="+",
    markersize=5,
    label="Input WDLF",
)
plt.plot(
    mag_obs_05,
    recomputed_wdlf_05
    / np.nansum(recomputed_wdlf_05)
    * np.nansum(obs_wdlf_05),
    label="Reconstructed WDLF",
)
plt.xlabel(r"M${_\mathrm{bol}}$ / mag")
plt.ylabel("log(arbitrary number density)")
plt.xlim(0.0, 20.0)
plt.ylim(1e-7, 5e-3)
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    os.path.join(
        "test_case", "gcns_reconstructed_wdlf_optimal_resolution_bin_05.png"
    )
)


np.save(
    os.path.join("test_case", "gcns_sfh_optimal_resolution_bin_05.npy"),
    np.column_stack((partial_age_05, solution_05)),
)
np.save(
    os.path.join(
        "test_case", "gcns_reconstructed_wdlf_optimal_resolution_bin_05.npy"
    ),
    np.column_stack((mag_obs_05, obs_wdlf_05, obs_wdlf_err_05)),
)
