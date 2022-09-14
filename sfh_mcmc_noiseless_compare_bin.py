import os

import emcee
import numpy as np
from matplotlib import pyplot as plt
from WDPhotTools import theoretical_lf


def log_prior(theta):
    if (theta <= 0.0).any():
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


# Collect all the partial WDLFs, in chunks of 0.5 Gyr
partial_wdlf = []
age = np.arange(0.5, 10.0, 0.5)
for i in age:
    filename1 = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_Mbol.csv".format(i)
    )
    filename2 = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_Mbol.csv".format(
            i + 0.1
        )
    )
    filename3 = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_Mbol.csv".format(
            i + 0.2
        )
    )
    filename4 = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_Mbol.csv".format(
            i + 0.3
        )
    )
    filename5 = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_Mbol.csv".format(
            i + 0.4
        )
    )
    n_temp = np.loadtxt(os.path.join("output", filename1), delimiter=",")[:, 1]
    n_temp += np.loadtxt(os.path.join("output", filename2), delimiter=",")[
        :, 1
    ]
    n_temp += np.loadtxt(os.path.join("output", filename3), delimiter=",")[
        :, 1
    ]
    n_temp += np.loadtxt(os.path.join("output", filename4), delimiter=",")[
        :, 1
    ]
    n_temp += np.loadtxt(os.path.join("output", filename5), delimiter=",")[
        :, 1
    ]
    n_temp /= np.nansum(n_temp)
    partial_wdlf.append(n_temp.copy())
    n_temp = None

partial_wdlf = np.array(partial_wdlf)
partial_wdlf_02_bin = np.array([[np.mean(partial_wdlf[j][i*2:i*2+2]) for i in range(100)] for j in range(19)])
partial_wdlf_05_bin = np.array([[np.mean(partial_wdlf[j][i*5:i*5+5]) for i in range(40)] for j in range(19)])
partial_wdlf_10_bin = np.array([[np.mean(partial_wdlf[j][i*10:i*10+10]) for i in range(20)] for j in range(19)])

# Get the magnitudes
mag = np.loadtxt(
    os.path.join("output", filename1),
    delimiter=",",
)[:, 0]
mag_02_bin = np.around([np.mean(mag[i*2:i*2+2]) for i in range(100)], decimals=2)
mag_05_bin = np.around([np.mean(mag[i*5:i*5+5]) for i in range(40)], decimals=2)
mag_10_bin = np.around([np.mean(mag[i*10:i*10+10]) for i in range(20)], decimals=2)



###############################################################################
#
# Two Bursts
#
###############################################################################

# Generate a WDLF with 1 Gyr burst at 2-3 Gyr and 5-6 Gyr
obs_wdlf = (
    partial_wdlf[3] + partial_wdlf[4] + partial_wdlf[9] + partial_wdlf[10]
)
obs_wdlf_02_bin = (
    partial_wdlf_02_bin[3] + partial_wdlf_02_bin[4] + partial_wdlf_02_bin[9] + partial_wdlf_02_bin[10]
)
obs_wdlf_05_bin = (
    partial_wdlf_05_bin[3] + partial_wdlf_05_bin[4] + partial_wdlf_05_bin[9] + partial_wdlf_05_bin[10]
)
obs_wdlf_10_bin = (
    partial_wdlf_10_bin[3] + partial_wdlf_10_bin[4] + partial_wdlf_10_bin[9] + partial_wdlf_10_bin[10]
)
obs_wdlf /= np.nansum(obs_wdlf)
obs_wdlf_02_bin /= np.nansum(obs_wdlf_02_bin)
obs_wdlf_05_bin /= np.nansum(obs_wdlf_05_bin)
obs_wdlf_10_bin /= np.nansum(obs_wdlf_10_bin)

# assume 2% error (S/N of 50) and a 1% noise floor
obs_wdlf_err = obs_wdlf * 0.02 + np.percentile(obs_wdlf[obs_wdlf > 0], 1.0)
obs_wdlf_err_02_bin = obs_wdlf_02_bin * 0.02 + np.percentile(obs_wdlf_02_bin[obs_wdlf_02_bin > 0], 1.0)
obs_wdlf_err_05_bin = obs_wdlf_05_bin * 0.02 + np.percentile(obs_wdlf_05_bin[obs_wdlf_05_bin > 0], 1.0)
obs_wdlf_err_10_bin = obs_wdlf_10_bin * 0.02 + np.percentile(obs_wdlf_10_bin[obs_wdlf_10_bin > 0], 1.0)

pwdlf_model = partial_wdlf[:, obs_wdlf > 0.0]
pwdlf_model_02_bin = partial_wdlf_02_bin[:, obs_wdlf_02_bin > 0.0]
pwdlf_model_05_bin = partial_wdlf_05_bin[:, obs_wdlf_05_bin > 0.0]
pwdlf_model_10_bin = partial_wdlf_10_bin[:, obs_wdlf_10_bin > 0.0]

nwalkers = 500
ndim = len(partial_wdlf)
ndim_02_bin = len(partial_wdlf_02_bin)
ndim_05_bin = len(partial_wdlf_05_bin)
ndim_10_bin = len(partial_wdlf_10_bin)

rel_norm = np.random.rand(nwalkers, ndim)
rel_norm_02_bin = np.random.rand(nwalkers, ndim_02_bin)
rel_norm_05_bin = np.random.rand(nwalkers, ndim_05_bin)
rel_norm_10_bin = np.random.rand(nwalkers, ndim_10_bin)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(obs_wdlf[obs_wdlf > 0.0], obs_wdlf_err[obs_wdlf > 0.0], pwdlf_model),
)
sampler_02_bin = emcee.EnsembleSampler(
    nwalkers,
    ndim_02_bin,
    log_probability,
    args=(obs_wdlf_02_bin[obs_wdlf_02_bin > 0.0], obs_wdlf_err_02_bin[obs_wdlf_02_bin > 0.0], pwdlf_model_02_bin),
)
sampler_05_bin = emcee.EnsembleSampler(
    nwalkers,
    ndim_05_bin,
    log_probability,
    args=(obs_wdlf_05_bin[obs_wdlf_05_bin > 0.0], obs_wdlf_err_05_bin[obs_wdlf_05_bin > 0.0], pwdlf_model_05_bin),
)
sampler_10_bin = emcee.EnsembleSampler(
    nwalkers,
    ndim_10_bin,
    log_probability,
    args=(obs_wdlf_10_bin[obs_wdlf_10_bin > 0.0], obs_wdlf_err_10_bin[obs_wdlf_10_bin > 0.0], pwdlf_model_10_bin),
)

sampler.run_mcmc(rel_norm, 10000, progress=True)
sampler_02_bin.run_mcmc(rel_norm_02_bin, 10000, progress=True)
sampler_05_bin.run_mcmc(rel_norm_05_bin, 10000, progress=True)
sampler_10_bin.run_mcmc(rel_norm_10_bin, 10000, progress=True)

flat_samples = sampler.get_chain(discard=1000, thin=5, flat=True)
flat_samples_02_bin = sampler_02_bin.get_chain(discard=1000, thin=5, flat=True)
flat_samples_05_bin = sampler_05_bin.get_chain(discard=1000, thin=5, flat=True)
flat_samples_10_bin = sampler_10_bin.get_chain(discard=1000, thin=5, flat=True)

solution = np.zeros(ndim)
solution_02_bin = np.zeros(ndim_02_bin)
solution_05_bin = np.zeros(ndim_05_bin)
solution_10_bin = np.zeros(ndim_10_bin)

for i in range(ndim):
    solution[i] = np.percentile(flat_samples[:, i], [50.0])

for i in range(ndim_02_bin):
    solution_02_bin[i] = np.percentile(flat_samples_02_bin[:, i], [50.0])

for i in range(ndim_05_bin):
    solution_05_bin[i] = np.percentile(flat_samples_05_bin[:, i], [50.0])

for i in range(ndim_10_bin):
    solution_10_bin[i] = np.percentile(flat_samples_10_bin[:, i], [50.0])



real_sfh = np.zeros_like(age)
real_sfh[3] = 1.0
real_sfh[4] = 1.0
real_sfh[9] = 1.0
real_sfh[10] = 1.0

real_sfh /= np.nansum(real_sfh)
solution /= np.nansum(solution)
solution_02_bin /= np.nansum(solution_02_bin)
solution_05_bin /= np.nansum(solution_05_bin)
solution_10_bin /= np.nansum(solution_10_bin)

plt.figure(1)
plt.clf()
plt.step(age, solution, where="post", label="bin size = 0.1 mag")
plt.step(age, solution_02_bin, where="post", label="bin size = 0.2 mag")
plt.step(age, solution_05_bin, where="post", label="bin size = 0.5 mag")
plt.step(age, solution_10_bin, where="post", label="bin size = 1.0 mag")
plt.step(age, real_sfh, where="post", label="Input")
plt.legend()
plt.grid()
plt.xlabel("Lookback time / Gyr")
plt.ylabel("Relative Star Formation Rate")
plt.tight_layout()
plt.savefig(os.path.join("test_case", "two_bursts_sfh_compare_bin_size.png"))

recomputed_wdlf = np.nansum(solution * np.array(partial_wdlf).T, axis=1)
recomputed_wdlf_02_bin = np.nansum(solution_02_bin * np.array(partial_wdlf_02_bin).T, axis=1)
recomputed_wdlf_05_bin = np.nansum(solution_05_bin * np.array(partial_wdlf_05_bin).T, axis=1)
recomputed_wdlf_10_bin = np.nansum(solution_10_bin * np.array(partial_wdlf_10_bin).T, axis=1)

plt.figure(2)
plt.clf()
plt.plot(mag, np.log10(obs_wdlf), label="Input")
plt.plot(mag, np.log10(recomputed_wdlf), label="bin size = 0.1 mag")
plt.plot(mag_02_bin, np.log10(recomputed_wdlf_02_bin), label="bin size = 0.2 mag")
plt.plot(mag_05_bin, np.log10(recomputed_wdlf_05_bin), label="bin size = 0.5 mag")
plt.plot(mag_10_bin, np.log10(recomputed_wdlf_10_bin), label="bin size = 1.0 mag")
plt.xlabel(r"M${_\mathrm{bol}}$ / mag")
plt.ylabel("log(arbitrary number density)")
plt.xlim(0, 20)
plt.ylim(-5.5, 1)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join("test_case", "two_bursts_wdlf_compare_bin_size.png"))

