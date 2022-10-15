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
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    log_likelihood = 0.0
    for i, j, k in zip(obs, err, model_list):
        model = np.nansum(rel_norm[:, None] * k, axis=0)
        log_likelihood += -np.nansum(((model - i) / j) ** 2.0)
    return log_likelihood


# Collect all the partial WDLFs, in chunks of 0.5 Gyr
partial_wdlf_G = []
partial_wdlf_BP = []
partial_wdlf_RP = []

age = np.arange(0.5, 10.0, 0.5)

# Note that n_temp_BP and n_temp_RP are normalised by n_temp_G because
# the relative normalisation is important in preserving the colour information
for i in age:
    filename1_G = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3.csv".format(i)
    )
    filename2_G = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3.csv".format(i + 0.1)
    )
    filename3_G = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3.csv".format(i + 0.2)
    )
    filename4_G = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3.csv".format(i + 0.3)
    )
    filename5_G = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3.csv".format(i + 0.4)
    )
    n_temp_G = np.loadtxt(os.path.join("output", filename1_G), delimiter=",")[
        :, 1
    ]
    n_temp_G += np.loadtxt(os.path.join("output", filename2_G), delimiter=",")[
        :, 1
    ]
    n_temp_G += np.loadtxt(os.path.join("output", filename3_G), delimiter=",")[
        :, 1
    ]
    n_temp_G += np.loadtxt(os.path.join("output", filename4_G), delimiter=",")[
        :, 1
    ]
    n_temp_G += np.loadtxt(os.path.join("output", filename5_G), delimiter=",")[
        :, 1
    ]
    norm_factor = np.nansum(n_temp_G)
    n_temp_G /= norm_factor
    partial_wdlf_G.append(n_temp_G.copy())
    n_temp_G = None
    filename1_BP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_BP.csv".format(i)
    )
    filename2_BP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_BP.csv".format(
            i + 0.1
        )
    )
    filename3_BP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_BP.csv".format(
            i + 0.2
        )
    )
    filename4_BP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_BP.csv".format(
            i + 0.3
        )
    )
    filename5_BP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_BP.csv".format(
            i + 0.4
        )
    )
    n_temp_BP = np.loadtxt(
        os.path.join("output", filename1_BP), delimiter=","
    )[:, 1]
    n_temp_BP += np.loadtxt(
        os.path.join("output", filename2_BP), delimiter=","
    )[:, 1]
    n_temp_BP += np.loadtxt(
        os.path.join("output", filename3_BP), delimiter=","
    )[:, 1]
    n_temp_BP += np.loadtxt(
        os.path.join("output", filename4_BP), delimiter=","
    )[:, 1]
    n_temp_BP += np.loadtxt(
        os.path.join("output", filename5_BP), delimiter=","
    )[:, 1]
    n_temp_BP /= norm_factor
    partial_wdlf_BP.append(n_temp_BP.copy())
    n_temp_BP = None
    filename1_RP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_RP.csv".format(i)
    )
    filename2_RP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_RP.csv".format(
            i + 0.1
        )
    )
    filename3_RP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_RP.csv".format(
            i + 0.2
        )
    )
    filename4_RP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_RP.csv".format(
            i + 0.3
        )
    )
    filename5_RP = (
        "montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_RP.csv".format(
            i + 0.4
        )
    )
    n_temp_RP = np.loadtxt(
        os.path.join("output", filename1_RP), delimiter=","
    )[:, 1]
    n_temp_RP += np.loadtxt(
        os.path.join("output", filename2_RP), delimiter=","
    )[:, 1]
    n_temp_RP += np.loadtxt(
        os.path.join("output", filename3_RP), delimiter=","
    )[:, 1]
    n_temp_RP += np.loadtxt(
        os.path.join("output", filename4_RP), delimiter=","
    )[:, 1]
    n_temp_RP += np.loadtxt(
        os.path.join("output", filename5_RP), delimiter=","
    )[:, 1]
    n_temp_RP /= norm_factor
    partial_wdlf_RP.append(n_temp_RP.copy())
    n_temp_RP = None


partial_wdlf_G = np.array(partial_wdlf_G)
partial_wdlf_BP = np.array(partial_wdlf_BP)
partial_wdlf_RP = np.array(partial_wdlf_RP)

partial_wdlf_G_02_bin = np.array(
    [
        [np.mean(partial_wdlf_G[j][i * 2 : i * 2 + 2]) for i in range(100)]
        for j in range(19)
    ]
)
partial_wdlf_BP_02_bin = np.array(
    [
        [np.mean(partial_wdlf_BP[j][i * 2 : i * 2 + 2]) for i in range(100)]
        for j in range(19)
    ]
)
partial_wdlf_RP_02_bin = np.array(
    [
        [np.mean(partial_wdlf_RP[j][i * 2 : i * 2 + 2]) for i in range(100)]
        for j in range(19)
    ]
)

partial_wdlf_G_05_bin = np.array(
    [
        [np.mean(partial_wdlf_G[j][i * 5 : i * 5 + 5]) for i in range(40)]
        for j in range(19)
    ]
)
partial_wdlf_BP_05_bin = np.array(
    [
        [np.mean(partial_wdlf_BP[j][i * 5 : i * 5 + 5]) for i in range(40)]
        for j in range(19)
    ]
)
partial_wdlf_RP_05_bin = np.array(
    [
        [np.mean(partial_wdlf_RP[j][i * 5 : i * 5 + 5]) for i in range(40)]
        for j in range(19)
    ]
)

partial_wdlf_G_10_bin = np.array(
    [
        [np.mean(partial_wdlf_G[j][i * 10 : i * 10 + 10]) for i in range(20)]
        for j in range(19)
    ]
)
partial_wdlf_BP_10_bin = np.array(
    [
        [np.mean(partial_wdlf_BP[j][i * 10 : i * 10 + 10]) for i in range(20)]
        for j in range(19)
    ]
)
partial_wdlf_RP_10_bin = np.array(
    [
        [np.mean(partial_wdlf_RP[j][i * 10 : i * 10 + 10]) for i in range(20)]
        for j in range(19)
    ]
)


# Get the magnitude
mag = np.loadtxt(
    os.path.join("output", filename1_G),
    delimiter=",",
)[:, 0]


###############################################################################
#
# Two Bursts
#
###############################################################################

#
# 0.1 mag bin
#

# Generate a WDLF with 1 Gyr burst at 2-3 Gyr and 5-6 Gyr
obs_wdlf_G = (
    partial_wdlf_G[3]
    + partial_wdlf_G[4]
    + partial_wdlf_G[9]
    + partial_wdlf_G[10]
)
obs_wdlf_BP = (
    partial_wdlf_BP[3]
    + partial_wdlf_BP[4]
    + partial_wdlf_BP[9]
    + partial_wdlf_BP[10]
)
obs_wdlf_RP = (
    partial_wdlf_RP[3]
    + partial_wdlf_RP[4]
    + partial_wdlf_RP[9]
    + partial_wdlf_RP[10]
)
# assume 2% error (S/N of 50) and a 1% noise floor
obs_wdlf_err_G = obs_wdlf_G * 0.02 + np.percentile(
    obs_wdlf_G[obs_wdlf_G > 0], 1.0
)
obs_wdlf_err_BP = obs_wdlf_BP * 0.02 + np.percentile(
    obs_wdlf_BP[obs_wdlf_BP > 0], 1.0
)
obs_wdlf_err_RP = obs_wdlf_RP * 0.02 + np.percentile(
    obs_wdlf_RP[obs_wdlf_RP > 0], 1.0
)

pwdlf_model_G = partial_wdlf_G[:, obs_wdlf_G > 0.0]
pwdlf_model_BP = partial_wdlf_BP[:, obs_wdlf_BP > 0.0]
pwdlf_model_RP = partial_wdlf_RP[:, obs_wdlf_RP > 0.0]

nwalkers = 500
ndim = len(partial_wdlf_G)

rel_norm = np.random.rand(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(
        [
            obs_wdlf_G[obs_wdlf_G > 0.0],
            obs_wdlf_BP[obs_wdlf_BP > 0.0],
            obs_wdlf_RP[obs_wdlf_RP > 0.0],
        ],
        [
            obs_wdlf_err_G[obs_wdlf_G > 0.0],
            obs_wdlf_err_BP[obs_wdlf_BP > 0.0],
            obs_wdlf_err_RP[obs_wdlf_RP > 0.0],
        ],
        [pwdlf_model_G, pwdlf_model_BP, pwdlf_model_RP],
    ),
)
sampler.run_mcmc(rel_norm, 10000, progress=True)

flat_samples = sampler.get_chain(discard=1000, thin=5, flat=True)

solution = np.zeros(ndim)
err_lower = np.zeros(ndim)
err_upper = np.zeros(ndim)
for i in range(ndim):
    err_lower[i], solution[i], err_upper[i] = np.percentile(flat_samples[:, i], [15.8655, 50.0, 84.1345])


#
# 0.2 mag bin
#

# Generate a WDLF with 1 Gyr burst at 2-3 Gyr and 5-6 Gyr
obs_wdlf_G_02_bin = (
    partial_wdlf_G_02_bin[3]
    + partial_wdlf_G_02_bin[4]
    + partial_wdlf_G_02_bin[9]
    + partial_wdlf_G_02_bin[10]
)
obs_wdlf_BP_02_bin = (
    partial_wdlf_BP_02_bin[3]
    + partial_wdlf_BP_02_bin[4]
    + partial_wdlf_BP_02_bin[9]
    + partial_wdlf_BP_02_bin[10]
)
obs_wdlf_RP_02_bin = (
    partial_wdlf_RP_02_bin[3]
    + partial_wdlf_RP_02_bin[4]
    + partial_wdlf_RP_02_bin[9]
    + partial_wdlf_RP_02_bin[10]
)
# assume 2% error (S/N of 50) and a 1% noise floor
obs_wdlf_err_G_02_bin = obs_wdlf_G_02_bin * 0.02 + np.percentile(
    obs_wdlf_G_02_bin[obs_wdlf_G_02_bin > 0], 1.0
)
obs_wdlf_err_BP_02_bin = obs_wdlf_BP_02_bin * 0.02 + np.percentile(
    obs_wdlf_BP_02_bin[obs_wdlf_BP_02_bin > 0], 1.0
)
obs_wdlf_err_RP_02_bin = obs_wdlf_RP_02_bin * 0.02 + np.percentile(
    obs_wdlf_RP_02_bin[obs_wdlf_RP_02_bin > 0], 1.0
)

pwdlf_model_G_02_bin = partial_wdlf_G_02_bin[:, obs_wdlf_G_02_bin > 0.0]
pwdlf_model_BP_02_bin = partial_wdlf_BP_02_bin[:, obs_wdlf_BP_02_bin > 0.0]
pwdlf_model_RP_02_bin = partial_wdlf_RP_02_bin[:, obs_wdlf_RP_02_bin > 0.0]

ndim_02_bin = len(partial_wdlf_G_02_bin)

rel_norm_02_bin = np.random.rand(nwalkers, ndim_02_bin)

sampler_02_bin = emcee.EnsembleSampler(
    nwalkers,
    ndim_02_bin,
    log_probability,
    args=(
        [
            obs_wdlf_G_02_bin[obs_wdlf_G_02_bin > 0.0],
            obs_wdlf_BP_02_bin[obs_wdlf_BP_02_bin > 0.0],
            obs_wdlf_RP_02_bin[obs_wdlf_RP_02_bin > 0.0],
        ],
        [
            obs_wdlf_err_G_02_bin[obs_wdlf_G_02_bin > 0.0],
            obs_wdlf_err_BP_02_bin[obs_wdlf_BP_02_bin > 0.0],
            obs_wdlf_err_RP_02_bin[obs_wdlf_RP_02_bin > 0.0],
        ],
        [pwdlf_model_G_02_bin, pwdlf_model_BP_02_bin, pwdlf_model_RP_02_bin],
    ),
)
sampler_02_bin.run_mcmc(rel_norm_02_bin, 10000, progress=True)

flat_samples_02_bin = sampler_02_bin.get_chain(discard=1000, thin=5, flat=True)

solution_02_bin = np.zeros(ndim_02_bin)
err_lower_02_bin = np.zeros(ndim_02_bin)
err_upper_02_bin = np.zeros(ndim_02_bin)
for i in range(ndim_02_bin):
    err_lower_02_bin[i], solution_02_bin[i], err_upper_02_bin[i] = np.percentile(flat_samples_02_bin[:, i], [15.8655, 50.0, 84.1345])


#
# 0.5 mag bin
#

# Generate a WDLF with 1 Gyr burst at 2-3 Gyr and 5-6 Gyr
obs_wdlf_G_05_bin = (
    partial_wdlf_G_05_bin[3]
    + partial_wdlf_G_05_bin[4]
    + partial_wdlf_G_05_bin[9]
    + partial_wdlf_G_05_bin[10]
)
obs_wdlf_BP_05_bin = (
    partial_wdlf_BP_05_bin[3]
    + partial_wdlf_BP_05_bin[4]
    + partial_wdlf_BP_05_bin[9]
    + partial_wdlf_BP_05_bin[10]
)
obs_wdlf_RP_05_bin = (
    partial_wdlf_RP_05_bin[3]
    + partial_wdlf_RP_05_bin[4]
    + partial_wdlf_RP_05_bin[9]
    + partial_wdlf_RP_05_bin[10]
)
# assume 2% error (S/N of 50) and a 1% noise floor
obs_wdlf_err_G_05_bin = obs_wdlf_G_05_bin * 0.02 + np.percentile(
    obs_wdlf_G_05_bin[obs_wdlf_G_05_bin > 0], 1.0
)
obs_wdlf_err_BP_05_bin = obs_wdlf_BP_05_bin * 0.02 + np.percentile(
    obs_wdlf_BP_05_bin[obs_wdlf_BP_05_bin > 0], 1.0
)
obs_wdlf_err_RP_05_bin = obs_wdlf_RP_05_bin * 0.02 + np.percentile(
    obs_wdlf_RP_05_bin[obs_wdlf_RP_05_bin > 0], 1.0
)

pwdlf_model_G_05_bin = partial_wdlf_G_05_bin[:, obs_wdlf_G_05_bin > 0.0]
pwdlf_model_BP_05_bin = partial_wdlf_BP_05_bin[:, obs_wdlf_BP_05_bin > 0.0]
pwdlf_model_RP_05_bin = partial_wdlf_RP_05_bin[:, obs_wdlf_RP_05_bin > 0.0]

ndim_05_bin = len(partial_wdlf_G_05_bin)

rel_norm_05_bin = np.random.rand(nwalkers, ndim_05_bin)

sampler_05_bin = emcee.EnsembleSampler(
    nwalkers,
    ndim_05_bin,
    log_probability,
    args=(
        [
            obs_wdlf_G_05_bin[obs_wdlf_G_05_bin > 0.0],
            obs_wdlf_BP_05_bin[obs_wdlf_BP_05_bin > 0.0],
            obs_wdlf_RP_05_bin[obs_wdlf_RP_05_bin > 0.0],
        ],
        [
            obs_wdlf_err_G_05_bin[obs_wdlf_G_05_bin > 0.0],
            obs_wdlf_err_BP_05_bin[obs_wdlf_BP_05_bin > 0.0],
            obs_wdlf_err_RP_05_bin[obs_wdlf_RP_05_bin > 0.0],
        ],
        [pwdlf_model_G_05_bin, pwdlf_model_BP_05_bin, pwdlf_model_RP_05_bin],
    ),
)
sampler_05_bin.run_mcmc(rel_norm_05_bin, 10000, progress=True)

flat_samples_05_bin = sampler_05_bin.get_chain(discard=1000, thin=5, flat=True)

solution_05_bin = np.zeros(ndim_05_bin)
err_lower_05_bin = np.zeros(ndim_05_bin)
err_upper_05_bin = np.zeros(ndim_05_bin)
for i in range(ndim_05_bin):
    err_lower_05_bin[i], solution_05_bin[i], err_upper_05_bin[i] = np.percentile(flat_samples_05_bin[:, i], [15.8655, 50.0, 84.1345])


#
# 1.0 mag bin
#

# Generate a WDLF with 1 Gyr burst at 2-3 Gyr and 5-6 Gyr
obs_wdlf_G_10_bin = (
    partial_wdlf_G_10_bin[3]
    + partial_wdlf_G_10_bin[4]
    + partial_wdlf_G_10_bin[9]
    + partial_wdlf_G_10_bin[10]
)
obs_wdlf_BP_10_bin = (
    partial_wdlf_BP_10_bin[3]
    + partial_wdlf_BP_10_bin[4]
    + partial_wdlf_BP_10_bin[9]
    + partial_wdlf_BP_10_bin[10]
)
obs_wdlf_RP_10_bin = (
    partial_wdlf_RP_10_bin[3]
    + partial_wdlf_RP_10_bin[4]
    + partial_wdlf_RP_10_bin[9]
    + partial_wdlf_RP_10_bin[10]
)
# assume 2% error (S/N of 50) and a 1% noise floor
obs_wdlf_err_G_10_bin = obs_wdlf_G_10_bin * 0.02 + np.percentile(
    obs_wdlf_G_10_bin[obs_wdlf_G_10_bin > 0], 1.0
)
obs_wdlf_err_BP_10_bin = obs_wdlf_BP_10_bin * 0.02 + np.percentile(
    obs_wdlf_BP_10_bin[obs_wdlf_BP_10_bin > 0], 1.0
)
obs_wdlf_err_RP_10_bin = obs_wdlf_RP_10_bin * 0.02 + np.percentile(
    obs_wdlf_RP_10_bin[obs_wdlf_RP_10_bin > 0], 1.0
)

pwdlf_model_G_10_bin = partial_wdlf_G_10_bin[:, obs_wdlf_G_10_bin > 0.0]
pwdlf_model_BP_10_bin = partial_wdlf_BP_10_bin[:, obs_wdlf_BP_10_bin > 0.0]
pwdlf_model_RP_10_bin = partial_wdlf_RP_10_bin[:, obs_wdlf_RP_10_bin > 0.0]

ndim_10_bin = len(partial_wdlf_G_10_bin)

rel_norm_10_bin = np.random.rand(nwalkers, ndim_10_bin)

sampler_10_bin = emcee.EnsembleSampler(
    nwalkers,
    ndim_10_bin,
    log_probability,
    args=(
        [
            obs_wdlf_G_10_bin[obs_wdlf_G_10_bin > 0.0],
            obs_wdlf_BP_10_bin[obs_wdlf_BP_10_bin > 0.0],
            obs_wdlf_RP_10_bin[obs_wdlf_RP_10_bin > 0.0],
        ],
        [
            obs_wdlf_err_G_10_bin[obs_wdlf_G_10_bin > 0.0],
            obs_wdlf_err_BP_10_bin[obs_wdlf_BP_10_bin > 0.0],
            obs_wdlf_err_RP_10_bin[obs_wdlf_RP_10_bin > 0.0],
        ],
        [pwdlf_model_G_10_bin, pwdlf_model_BP_10_bin, pwdlf_model_RP_10_bin],
    ),
)
sampler_10_bin.run_mcmc(rel_norm_10_bin, 10000, progress=True)

flat_samples_10_bin = sampler_10_bin.get_chain(discard=1000, thin=5, flat=True)

solution_10_bin = np.zeros(ndim_10_bin)
err_lower_10_bin = np.zeros(ndim_10_bin)
err_upper_10_bin = np.zeros(ndim_10_bin)
for i in range(ndim_10_bin):
    err_lower_10_bin[i], solution_10_bin[i], err_upper_10_bin[i] = np.percentile(flat_samples_10_bin[:, i], [15.8655, 50.0, 84.1345])


real_sfh = np.zeros_like(age)
real_sfh[3] = 1.0
real_sfh[4] = 1.0
real_sfh[9] = 1.0
real_sfh[10] = 1.0

plt.figure(1)
plt.clf()
plt.step(age, solution, where="mid", label="bin size = 0.1 mag")
plt.step(age, solution_02_bin, where="mid", label="bin size = 0.2 mag")
plt.step(age, solution_05_bin, where="mid", label="bin size = 0.5 mag")
plt.step(age, solution_10_bin, where="mid", label="bin size = 1.0 mag")

plt.errorbar(age, solution, yerr=[solution-err_lower, err_upper-solution], fmt=' ', color='C0')
plt.errorbar(age, solution_02_bin, yerr=[solution_02_bin-err_lower_02_bin, err_upper_02_bin-solution_02_bin], fmt=' ', color='C1')
plt.errorbar(age, solution_05_bin, yerr=[solution_05_bin-err_lower_05_bin, err_upper_05_bin-solution_05_bin], fmt=' ', color='C2')
plt.errorbar(age, solution_10_bin, yerr=[solution_10_bin-err_lower_10_bin, err_upper_10_bin-solution_10_bin], fmt=' ', color='C3')

plt.step(age, real_sfh, where="mid", label="input", color='k', ls=":")
plt.legend()
plt.grid()
plt.xlabel("Lookback time / Gyr")
plt.ylabel("Relative Star Formation Rate")
plt.tight_layout()
plt.savefig(
    os.path.join("test_case", "two_bursts_sfh_compare_bin_multi_band.png")
)

# recomputed_wdlf_G = np.nansum(solution * np.array(partial_wdlf_G).T, axis=1)
# recomputed_wdlf_BP = np.nansum(solution * np.array(partial_wdlf_BP).T, axis=1)
# recomputed_wdlf_RP = np.nansum(solution * np.array(partial_wdlf_RP).T, axis=1)

# plt.figure(2)
# plt.clf()
# plt.plot(mag, np.log10(obs_wdlf_RP), color='Red', label="Input WDLF (G3 RP)")
# plt.plot(mag, np.log10(obs_wdlf_G), color='Green', label="Input WDLF (G3)")
# plt.plot(mag, np.log10(obs_wdlf_BP), color='Blue', label="Input WDLF (G3 BP)")
# plt.plot(mag, np.log10(recomputed_wdlf_RP), color='Red', ls=':', label="Reconstructed")
# plt.plot(mag, np.log10(recomputed_wdlf_G), color='Green', ls=':', label="Reconstructed")
# plt.plot(mag, np.log10(recomputed_wdlf_BP), color='Blue', ls=':', label="Reconstructed")
# plt.xlabel(r"M${_\mathrm{bol}}$ / mag")
# plt.ylabel("log(arbitrary number density)")
# plt.xlim(6, 18)
# plt.ylim(-5.5, 1)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join("test_case", "two_bursts_wdlf_compare_bin_multi_band.png"))
