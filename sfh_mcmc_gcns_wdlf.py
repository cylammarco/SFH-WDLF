import os
from unicodedata import decimal

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
# Get the magnitude
mag = np.loadtxt(
    os.path.join("output", filename1),
    delimiter=",",
)[:, 0]


data = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]
n_bin = 200

h_gen, b = np.histogram(
    data["Mbol"], bins=n_bin, range=(-0.05, 19.95), weights=0.01 / data["Vgen"]
)
e_gen, _ = np.histogram(
    data["Mbol"],
    bins=n_bin,
    range=(-0.05, 19.95),
    weights=0.01 / data["Vgen"] ** 2.0,
)

bin_size = b[1] - b[0]

m = np.around(b[1:] - 0.5 * bin_size, 2)
obs_wdlf = h_gen / bin_size
obs_wdlf_err = e_gen**0.5 / bin_size


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
sampler.run_mcmc(rel_norm, 50000, progress=True)

flat_samples = sampler.get_chain(discard=1000, thin=5, flat=True)

solution = np.zeros(ndim)
for i in range(ndim):
    solution[i] = np.percentile(flat_samples[:, i], [50.0])

solution /= np.nansum(solution)

plt.figure(1)
plt.clf()
plt.step(age, solution, where="post", label="MCMC")
plt.legend()
plt.grid()
plt.xlabel("Lookback time / Gyr")
plt.ylabel("Relative Star Formation Rate")
plt.tight_layout()
plt.savefig(os.path.join("test_case", "gcns_sfh.png"))

recomputed_wdlf = np.nansum(solution * np.array(partial_wdlf).T, axis=1)

plt.figure(2)
plt.clf()
plt.errorbar(
    m,
    obs_wdlf,
    yerr=[obs_wdlf_err, obs_wdlf_err],
    fmt="+",
    markersize=5,
    label="Input WDLF",
)
plt.plot(
    mag,
    recomputed_wdlf / np.nansum(recomputed_wdlf) * np.nansum(obs_wdlf),
    label="Reconstructed WDLF",
)
plt.xlabel(r"M${_\mathrm{bol}}$ / mag")
plt.ylabel("log(arbitrary number density)")
plt.xlim(5.0, 18.0)
plt.ylim(1e-6, 5e-3)
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join("test_case", "gcns_reconstructed_wdlf.png"))
