import os

import emcee
import numpy as np
from matplotlib import pyplot as plt
from WDPhotTools import theoretical_lf

from matplotlib.pyplot import *

ion()


def log_prior(theta):
    if (theta < 0.0).any():
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


'''
# Collect all the partial WDLFs, in chunks of 0.5 Gyr
partial_wdlf_G = []
partial_wdlf_BP = []
partial_wdlf_RP = []

age = np.arange(0.1, 15.0, 0.5)
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

# Get the magnitude
mag = np.loadtxt(
    os.path.join("output", filename1_RP),
    delimiter=",",
)[:, 0]


data_gcns = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]
data_edr3 = np.load(
    "pubgcnswdlf-crossmatched-full-edr3-catalogue.npz", allow_pickle=True
)["data"]

dm = np.array([5.0 * (np.log10(d) - 1.0) for d in data_gcns["dpc"]])

gcns_to_edr3_mapping = np.array(
    [
        np.where(data_edr3["source_id"] == i)[0][0]
        for i in data_gcns["sourceID"]
    ]
)

g = (
    np.array([data_edr3[i]["phot_g_mean_mag"] for i in gcns_to_edr3_mapping])
    - dm
)
g_bp = (
    np.array([data_edr3[i]["phot_bp_mean_mag"] for i in gcns_to_edr3_mapping])
    - dm
)
g_rp = (
    np.array([data_edr3[i]["phot_rp_mean_mag"] for i in gcns_to_edr3_mapping])
    - dm
)


n_bin = 200

h_gen, b = np.histogram(
    data_gcns["Mbol"],
    bins=n_bin,
    range=(-0.05, 19.95),
    weights=0.01 / data_gcns["Vgen"],
)
obs_wdlf_G, _ = np.histogram(
    g, bins=n_bin, range=(-0.05, 19.95), weights=0.01 / data_gcns["Vgen"]
)
obs_wdlf_BP, _ = np.histogram(
    g_bp, bins=n_bin, range=(-0.05, 19.95), weights=0.01 / data_gcns["Vgen"]
)
obs_wdlf_RP, _ = np.histogram(
    g_rp, bins=n_bin, range=(-0.05, 19.95), weights=0.01 / data_gcns["Vgen"]
)
e_gen, _ = np.histogram(
    data_gcns["Mbol"],
    bins=n_bin,
    range=(-0.05, 19.95),
    weights=0.01 / data_gcns["Vgen"] ** 2.0,
)
obs_wdlf_err_G, _ = np.histogram(
    g,
    bins=n_bin,
    range=(-0.05, 19.95),
    weights=0.01 / data_gcns["Vgen"] ** 2.0,
)
obs_wdlf_err_BP, _ = np.histogram(
    g_bp,
    bins=n_bin,
    range=(-0.05, 19.95),
    weights=0.01 / data_gcns["Vgen"] ** 2.0,
)
obs_wdlf_err_RP, _ = np.histogram(
    g_rp,
    bins=n_bin,
    range=(-0.05, 19.95),
    weights=0.01 / data_gcns["Vgen"] ** 2.0,
)

bin_size = b[1] - b[0]


figure(1, figsize=(10, 6))
clf()
errorbar(
    b[1:],
    h_gen / bin_size,
    yerr=[e_gen**0.5 / bin_size, e_gen**0.5 / bin_size],
    fmt="+",
    markersize=5,
    color="black",
)
errorbar(
    b[1:],
    obs_wdlf_G / bin_size,
    yerr=[obs_wdlf_err_G**0.5 / bin_size, obs_wdlf_err_G**0.5 / bin_size],
    fmt="+",
    markersize=5,
    color="green",
)
errorbar(
    b[1:],
    obs_wdlf_BP / bin_size,
    yerr=[
        obs_wdlf_err_BP**0.5 / bin_size,
        obs_wdlf_err_BP**0.5 / bin_size,
    ],
    fmt="+",
    markersize=5,
    color="blue",
)
errorbar(
    b[1:],
    obs_wdlf_RP / bin_size,
    yerr=[
        obs_wdlf_err_RP**0.5 / bin_size,
        obs_wdlf_err_RP**0.5 / bin_size,
    ],
    fmt="+",
    markersize=5,
    color="red",
)
yscale("log")

xlabel("Mbol / mag")
ylabel(r"N / pc$^3$ / Mbol")
xlim(5.0, 18.0)
ylim(1e-6, 5e-3)
grid()
tight_layout()


m = np.around(b[1:] - 0.5 * bin_size, 2)

nwalkers = 1000
ndim = len(partial_wdlf_G)
'''
prior = np.array([0.05109938, 0.03133296, 0.03178364, 0.0501848 , 0.06141645,
       0.0498405 , 0.0324508 , 0.03566725, 0.02552374, 0.02701787,
       0.03131913, 0.02504585, 0.02587348, 0.02415062, 0.04583976,
       0.04213541, 0.04331451, 0.04067248, 0.03550198, 0.02821787,
       0.03126558, 0.04093755, 0.03498278, 0.03557601, 0.02362598,
       0.0297468 , 0.0264834 , 0.01821633, 0.02077708])

for i in range(10):

    rel_norm = np.random.rand(nwalkers, ndim) * prior

    mask_G = (obs_wdlf_G > 0.) & (obs_wdlf_err_G > 0.)
    mask_BP = (obs_wdlf_BP > 0.) & (obs_wdlf_err_BP > 0.)
    mask_RP = (obs_wdlf_RP > 0.) & (obs_wdlf_err_RP > 0.)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            [
                obs_wdlf_G[mask_G],
                obs_wdlf_BP[mask_BP],
                obs_wdlf_RP[mask_RP],
            ],
            [
                obs_wdlf_err_G[mask_G]**0.5,
                obs_wdlf_err_BP[mask_BP]**0.5,
                obs_wdlf_err_RP[mask_RP]**0.5,
            ],
            [partial_wdlf_G[:, mask_G], partial_wdlf_BP[:, mask_BP], partial_wdlf_RP[:, mask_RP]],
        ),
    )
    sampler.run_mcmc(rel_norm, 10000, progress=True)

    flat_samples = sampler.get_chain(discard=2000, flat=True)

    solution = np.zeros(ndim)
    for i in range(ndim):
        solution[i] = np.percentile(flat_samples[:, i], [50.0])

    solution /= np.nansum(solution)

    plt.figure(1)
    plt.clf()
    plt.step(age[:-1], solution, where="post", label="MCMC")
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0, 15, 2))
    plt.xlim(0, 15)
    plt.ylim(bottom=0)
    plt.xlabel("Lookback time / Gyr")
    plt.ylabel("Relative Star Formation Rate")
    plt.tight_layout()
    plt.savefig(os.path.join("test_case", "gcns_sfh_multi_band.png"))

    recomputed_wdlf_G = np.nansum(solution * np.array(partial_wdlf_G).T, axis=1)
    recomputed_wdlf_BP = np.nansum(solution * np.array(partial_wdlf_BP).T, axis=1)
    recomputed_wdlf_RP = np.nansum(solution * np.array(partial_wdlf_RP).T, axis=1)

    plt.figure(2)
    plt.clf()
    plt.errorbar(
        m,
        obs_wdlf_G / bin_size,
        yerr=[obs_wdlf_err_G**0.5 / bin_size, obs_wdlf_err_G**0.5 / bin_size],
        fmt="+",
        markersize=5,
        label="Input observed WDLF G",
        color="green",
    )
    plt.errorbar(
        m,
        obs_wdlf_BP / bin_size,
        yerr=[
            obs_wdlf_err_BP**0.5 / bin_size,
            obs_wdlf_err_BP**0.5 / bin_size,
        ],
        fmt="+",
        markersize=5,
        label="Input observed WDLF BP",
        color="blue",
    )
    plt.errorbar(
        m,
        obs_wdlf_RP / bin_size,
        yerr=[
            obs_wdlf_err_RP**0.5 / bin_size,
            obs_wdlf_err_RP**0.5 / bin_size,
        ],
        fmt="+",
        markersize=5,
        label="Input observed WDLF RP",
        color="red",
    )
    plt.plot(
        mag,
        recomputed_wdlf_G
        / np.nansum(recomputed_wdlf_G)
        * np.nansum(obs_wdlf_G)
        / bin_size,
        label="Reconstructed WDLF G",
        color="green",
    )
    plt.plot(
        mag,
        recomputed_wdlf_BP
        / np.nansum(recomputed_wdlf_BP)
        * np.nansum(obs_wdlf_BP)
        / bin_size,
        label="Reconstructed WDLF BP",
        color="blue",
    )
    plt.plot(
        mag,
        recomputed_wdlf_RP
        / np.nansum(recomputed_wdlf_RP)
        * np.nansum(obs_wdlf_RP)
        / bin_size,
        label="Reconstructed WDLF RP",
        color="red",
    )
    plt.xlabel(r"M${_\mathrm{bol}}$ / mag")
    plt.ylabel("log(arbitrary number density)")
    plt.xlim(7.5, 20.0)
    plt.ylim(1e-7, 5e-3)
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        os.path.join("test_case", "gcns_reconstructed_wdlf_multi_band.png")
    )

    prior = solution
