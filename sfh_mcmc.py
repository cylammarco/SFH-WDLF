import emcee
import numpy as np
from matplotlib import pyplot as plt
from WDLFBuilder import theoretical_lf


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
        "{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20".format(i)
        + "_montreal_co_da_20_montreal_co_da_20.csv"
    )
    filename2 = (
        "{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20".format(i + 0.1)
        + "_montreal_co_da_20_montreal_co_da_20.csv"
    )
    filename3 = (
        "{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20".format(i + 0.2)
        + "_montreal_co_da_20_montreal_co_da_20.csv"
    )
    filename4 = (
        "{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20".format(i + 0.3)
        + "_montreal_co_da_20_montreal_co_da_20.csv"
    )
    filename5 = (
        "{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20".format(i + 0.4)
        + "_montreal_co_da_20_montreal_co_da_20.csv"
    )
    n_temp = np.loadtxt(filename1, delimiter=",")[:, 1]
    n_temp += np.loadtxt(filename2, delimiter=",")[:, 1]
    n_temp += np.loadtxt(filename3, delimiter=",")[:, 1]
    n_temp += np.loadtxt(filename4, delimiter=",")[:, 1]
    n_temp += np.loadtxt(filename5, delimiter=",")[:, 1]
    n_temp /= np.nansum(n_temp)
    partial_wdlf.append(n_temp.copy())
    n_temp = None

partial_wdlf = np.array(partial_wdlf)
# Get the magnitude
mag = np.loadtxt(
    "0.10Gyr_burst_C16_C08_montreal_co_da_20_montreal_co_da_20_"
    + "montreal_co_da_20.csv",
    delimiter=",",
)[:, 0]

# Generate a WDLF with 1 Gyr burst at 2-3 Gyr and 5-6 Gyr
obs_wdlf = (
    partial_wdlf[3] + partial_wdlf[4] + partial_wdlf[9] + partial_wdlf[10]
)
obs_wdlf /= np.nansum(obs_wdlf)

# assume 2% error (S/N of 50) and a 1% noise floor
obs_wdlf_err = obs_wdlf * 0.02 + np.percentile(obs_wdlf[obs_wdlf > 0], 1.0)
pwdlf_model = partial_wdlf[:, obs_wdlf > 0.0]

nwalkers = 500
ndim = len(partial_wdlf)

rel_norm = np.random.rand(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(obs_wdlf[obs_wdlf > 0.0], obs_wdlf_err[obs_wdlf > 0.0], pwdlf_model),
)
sampler.run_mcmc(rel_norm, 10000, progress=True)

flat_samples = sampler.get_chain(discard=1000, thin=5, flat=True)

solution = np.zeros(ndim)
for i in range(ndim):
    solution[i] = np.percentile(flat_samples[:, i], [50.0])

real_sfh = np.zeros_like(age)
real_sfh[3] = 1.0
real_sfh[4] = 1.0
real_sfh[9] = 1.0
real_sfh[10] = 1.0

solution /= np.nansum(solution)
real_sfh /= np.nansum(real_sfh)

plt.figure(1)
plt.clf()
plt.step(age, solution, where="post", label="MCMC")
plt.step(age, real_sfh, where="post", label="input")
plt.legend()
plt.grid()
plt.xlabel("Lookback time / Gyr")
plt.ylabel("Relative Star Formation Rate")
plt.tight_layout()
plt.savefig("two_bursts_sfh.png")

recomputed_wdlf = np.nansum(solution * np.array(partial_wdlf).T, axis=1)

plt.figure(2)
plt.clf()
plt.plot(mag, np.log10(obs_wdlf), label="Input WDLF")
plt.plot(mag, np.log10(recomputed_wdlf), label="Reconstructed WDLF")
plt.xlabel(r"M${_\mathrm{bol}}$ / mag")
plt.ylabel("log(arbitrary number density)")
plt.xlim(0, 20)
plt.ylim(-7.5, -1)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("two_bursts_wdlf.png")

# Decay SFH profile
wdlf = theoretical_lf.WDLF()
wdlf.compute_cooling_age_interpolator()
wdlf.set_sfr_model(mode="decay", age=6e9)

_, obs_wdlf2 = wdlf.compute_density(Mag=mag, n_points=10)
# assume 2% error (S/N of 50) and a 1% noise floor
obs_wdlf_err2 = obs_wdlf2 * 0.02 + np.percentile(obs_wdlf2[obs_wdlf2 > 0], 1.0)
pwdlf_model2 = partial_wdlf[:, obs_wdlf2 > 0.0]

nwalkers = 500
ndim = len(partial_wdlf)

rel_norm2 = np.random.rand(nwalkers, ndim)

sampler2 = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(
        obs_wdlf2[obs_wdlf2 > 0.0],
        obs_wdlf_err2[obs_wdlf2 > 0.0],
        pwdlf_model2,
    ),
)
sampler2.run_mcmc(rel_norm2, 50000, progress=True)

flat_samples2 = sampler2.get_chain(discard=5000, thin=5, flat=True)

sfh_solution2 = np.zeros(ndim)
for i in range(ndim):
    sfh_solution2[i] = np.percentile(flat_samples2[:, i], [50.0])

age2 = np.linspace(0, 10e9, 1000) / 1e9
real_sfh2 = wdlf.sfr(np.linspace(0, 10e9, 1000))
real_sfh2 /= np.nansum(real_sfh2) * np.diff(age2)[0]
sfh_solution2 /= np.nansum(sfh_solution2)

plt.figure(3)
plt.clf()
plt.step(age, sfh_solution2 / 0.5, where="post", label="MCMC")
plt.plot(age2, real_sfh2, label="input")
plt.xlim(0, 10.0)
plt.legend()
plt.grid()
plt.xlabel("Lookback time / Gyr")
plt.ylabel("Relative Star Formation Rate")
plt.tight_layout()
plt.savefig("decay_sfh.png")

recomputed_wdlf2 = np.nansum(sfh_solution2 * np.array(partial_wdlf).T, axis=1)

plt.figure(4)
plt.clf()
plt.plot(mag, np.log10(obs_wdlf2), label="Input WDLF")
plt.plot(mag, np.log10(recomputed_wdlf2), label="Reconstructed WDLF")
plt.xlabel(r"M$_bol$ / mag")
plt.ylabel("log(arbitrary number density)")
plt.xlim(0, 20)
plt.ylim(-7, 0)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("decay_sfh_wdlf.png")

# Decay and a burst between 2 and 3 Gyr
obs_wdlf3 = obs_wdlf2 + partial_wdlf[3] + partial_wdlf[4]
obs_wdlf_err3 = obs_wdlf3 * 0.02 + np.percentile(obs_wdlf3[obs_wdlf3 > 0], 1.0)
pwdlf_model3 = partial_wdlf[:, obs_wdlf3 > 0.0]

nwalkers = 200
ndim = len(partial_wdlf)

rel_norm3 = np.random.rand(nwalkers, ndim)

sampler3 = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    log_probability,
    args=(
        obs_wdlf3[obs_wdlf3 > 0.0],
        obs_wdlf_err3[obs_wdlf3 > 0.0],
        pwdlf_model3,
    ),
)
sampler3.run_mcmc(rel_norm3, 200000, progress=True)

flat_samples3 = sampler3.get_chain(discard=20000, flat=True)

sfh_solution3 = np.zeros(ndim)
for i in range(ndim):
    sfh_solution3[i] = np.percentile(flat_samples3[:, i], [50.0])

sfh_solution3 /= np.nansum(sfh_solution3)
recomputed_wdlf3 = np.nansum(sfh_solution3 * np.array(partial_wdlf).T, axis=1)
norm3 = np.sum(obs_wdlf3) / np.sum(recomputed_wdlf3)

age3 = np.linspace(0, 10e9, 1000) / 1e9
real_sfh3 = wdlf.sfr(np.linspace(0, 10e9, 1000))
real_sfh3[np.argmax(age3 >= 2) : np.argmin(age3 < 3)] += np.nansum(
    real_sfh3
) / (len(real_sfh3) / len(sfh_solution3))
real_sfh3 /= np.nansum(real_sfh3) / len(real_sfh3) * len(sfh_solution3)

plt.figure(5)
plt.clf()
plt.step(age, sfh_solution3, where="post", label="MCMC")
plt.plot(age3, real_sfh3, label="input")
plt.xlim(0, 10.0)
plt.legend()
plt.grid()
plt.xlabel("Lookback time / Gyr")
plt.ylabel("Relative Star Formation Rate")
plt.tight_layout()
plt.savefig("decay_plus_burst_sfh.png")

plt.figure(6)
plt.clf()
plt.plot(mag, np.log10(obs_wdlf3), label="Input WDLF")
plt.plot(mag, np.log10(recomputed_wdlf3 * norm3), label="Reconstructed WDLF")
plt.xlabel(r"M$_bol$ / mag")
plt.ylabel("log(arbitrary number density)")
plt.xlim(0, 20)
plt.ylim(-7, 0)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("decay_plus_burst_sfh_wdlf.png")

plt.show()
