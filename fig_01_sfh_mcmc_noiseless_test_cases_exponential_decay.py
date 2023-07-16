import os

import emcee
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import least_squares
from spectres import spectres
from WDPhotTools import theoretical_lf


figure_folder = "SFH-WDLF-article/figures"


def log_prior(theta):
    if (theta <= 0.0).any():
        return -np.inf
    else:
        return 0.0


def log_probability(rel_norm, obs_normed, err_normed, model_list):
    rel_norm /= np.sum(rel_norm)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = rel_norm[:, None].T @ model_list
    model /= np.nansum(model)
    log_likelihood = -np.nansum(((model - obs_normed) / err_normed) ** 2.0)
    return log_likelihood


# Load the pwdlfs
data = []
age_list_1 = np.arange(0.049, 0.100, 0.001)
age_list_2 = np.arange(0.100, 0.350, 0.005)
age_list_3 = np.arange(0.35, 15.01, 0.01)
age_list_3dp = np.concatenate((age_list_1, age_list_2))
age_list_2dp = age_list_3

age = np.concatenate((age_list_3dp, age_list_2dp))

for i in age_list_3dp:
    print(i)
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
    print(i)
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


###############################################################################
#
# Exponentially incrasing with a truncation profile - an exponential profile
# with an decay coefficient of 3 Gyr and a truncation at 2 Gyr
#
###############################################################################
def exponential_profile(t, coeff, truncation):
    profile = np.ones_like(t)
    profile *= np.exp(-t / coeff)
    profile[t < truncation] = 0.0
    profile /= max(profile)
    return profile


sfh_1 = exponential_profile(age * 1e9, 2.0e9, 2.0e9)
sfh_2 = exponential_profile(age * 1e9, 2.0e9, 4.0e9)
sfh_3 = exponential_profile(age * 1e9, 2.0e9, 6.0e9)
sfh_4 = exponential_profile(age * 1e9, 2.0e9, 8.0e9)
sfh_5 = exponential_profile(age * 1e9, 2.0e9, 1.0e10)

sfh_1 /= max(sfh_1)
sfh_2 /= max(sfh_2)
sfh_3 /= max(sfh_3)
sfh_4 /= max(sfh_4)
sfh_5 /= max(sfh_5)

wdlf_1_reconstructed = np.array(data)[:, :, 1].T @ sfh_1
wdlf_2_reconstructed = np.array(data)[:, :, 1].T @ sfh_2
wdlf_3_reconstructed = np.array(data)[:, :, 1].T @ sfh_3
wdlf_4_reconstructed = np.array(data)[:, :, 1].T @ sfh_4
wdlf_5_reconstructed = np.array(data)[:, :, 1].T @ sfh_5

mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag_pwdlf[np.argmax(d[:, 1])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=50, k=5
)

# Load the "resolution" of the Mbol solution
mbol_err_upper_bound = interpolate.UnivariateSpline(
    *np.load("mbol_err_upper_bound.npy").T, s=0
)

mag_bin = []
bin_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
bin_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
bin_number = 0
stop = False
mag_bin.append(mag_resolution_itp(age[0]))
for i, a in enumerate(age):
    if i < j:
        continue
    start = float(mag_resolution_itp(a))
    end = start
    j = i
    carry_on = True
    while carry_on:
        tmp = mag_resolution_itp(age[j + 1]) - mag_resolution_itp(age[j])
        print(j, end + tmp - start)
        if end + tmp - start < mbol_err_upper_bound(end):
            end = end + tmp
            bin_pwdlf[j] = bin_number
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    mag_bin.append(end)
    bin_idx.append(i)
    if stop:
        break


# these the the bin edges
mag_bin = np.array(mag_bin)
# get the bin centre
mag_bin = (mag_bin[1:] + mag_bin[:-1]) / 2.0
pwdlf_mapping_mag_bin = np.insert(bin_pwdlf, 0, 0)


wdlf_1_number_density = spectres(mag_bin, data[0][:, 0], wdlf_1_reconstructed)
wdlf_2_number_density = spectres(mag_bin, data[0][:, 0], wdlf_2_reconstructed)
wdlf_3_number_density = spectres(mag_bin, data[0][:, 0], wdlf_3_reconstructed)
wdlf_4_number_density = spectres(mag_bin, data[0][:, 0], wdlf_4_reconstructed)
wdlf_5_number_density = spectres(mag_bin, data[0][:, 0], wdlf_5_reconstructed)


wdlf_1_reconstructed /= np.sum(wdlf_1_reconstructed)
wdlf_2_reconstructed /= np.sum(wdlf_2_reconstructed)
wdlf_3_reconstructed /= np.sum(wdlf_3_reconstructed)
wdlf_4_reconstructed /= np.sum(wdlf_4_reconstructed)
wdlf_5_reconstructed /= np.sum(wdlf_5_reconstructed)

partial_wdlf = []
partial_age = []
for idx in np.sort(list(set(pwdlf_mapping_mag_bin))):
    pwdlf_temp = np.zeros_like(mag_bin)
    age_temp = 0.0
    age_count = 0
    for i in np.where(pwdlf_mapping_mag_bin == idx)[0]:
        pwdlf_temp += spectres(mag_bin, data[i][:, 0], data[i][:, 1], fill=0.0)
        age_temp += age[i]
        age_count += 1
    partial_wdlf.append(pwdlf_temp)
    partial_age.append(age_temp / age_count)


partial_wdlf = np.array(partial_wdlf)
pwdlf_model_1 = np.vstack(partial_wdlf)[:, wdlf_1_number_density > 0.0]
pwdlf_model_1 /= np.nansum(pwdlf_model_1) * np.nanmax(wdlf_1_number_density)

pwdlf_model_2 = np.vstack(partial_wdlf)[:, wdlf_2_number_density > 0.0]
pwdlf_model_2 /= np.nansum(pwdlf_model_2) * np.nanmax(wdlf_2_number_density)

pwdlf_model_3 = np.vstack(partial_wdlf)[:, wdlf_3_number_density > 0.0]
pwdlf_model_3 /= np.nansum(pwdlf_model_3) * np.nanmax(wdlf_3_number_density)

pwdlf_model_4 = np.vstack(partial_wdlf)[:, wdlf_4_number_density > 0.0]
pwdlf_model_4 /= np.nansum(pwdlf_model_4) * np.nanmax(wdlf_4_number_density)

pwdlf_model_5 = np.vstack(partial_wdlf)[:, wdlf_5_number_density > 0.0]
pwdlf_model_5 /= np.nansum(pwdlf_model_5) * np.nanmax(wdlf_5_number_density)


nwalkers = 250
ndim = len(partial_wdlf)
init_guess_1 = np.array(
    [
        6.03909927e-03,
        9.52464096e-17,
        4.05605911e-03,
        1.83607987e-03,
        4.99306714e-03,
        2.75306681e-02,
        3.87335445e-02,
        1.77359395e-02,
        1.59503758e-02,
        1.94370698e-02,
        7.91543421e-03,
        4.74583112e-03,
        9.09920996e-02,
        1.00000000e00,
        8.73254168e-01,
        6.62034450e-01,
        7.06664572e-01,
        5.56878848e-01,
        5.82155328e-01,
        4.51109574e-01,
        3.95621413e-01,
        4.56247024e-01,
        3.43895094e-01,
        3.30177814e-01,
        3.22824999e-01,
        2.70438762e-01,
        2.12595764e-01,
        1.03243284e-01,
        2.90468375e-02,
        3.17579364e-02,
        2.21137220e-02,
        1.65816128e-02,
        1.37677613e-02,
        9.64941759e-03,
        7.75583233e-03,
        5.60830221e-03,
        4.63151350e-03,
        3.70202394e-03,
        3.68317635e-03,
        1.84547863e-03,
        2.71747532e-03,
        2.24252076e-03,
        1.77059050e-03,
        2.50178255e-03,
        2.45583365e-03,
        2.01096077e-03,
        2.43563559e-03,
        2.38655692e-03,
        2.43095645e-03,
        2.00165143e-03,
        2.37123157e-03,
    ]
)
init_guess_2 = np.array(
    [
        4.22821910e-04,
        6.17968583e-16,
        3.80802086e-04,
        7.33978786e-04,
        4.83973511e-04,
        1.27761593e-03,
        1.10149850e-03,
        1.14376502e-03,
        7.01406154e-03,
        4.74390749e-03,
        1.92175045e-03,
        2.13126059e-04,
        2.41582964e-03,
        2.89946858e-03,
        8.53699254e-04,
        4.96020693e-03,
        3.38933997e-03,
        2.56493788e-03,
        4.34687035e-03,
        5.36656166e-03,
        1.52690927e-02,
        8.28007848e-02,
        1.00000000e00,
        4.29366797e-02,
        5.36711868e-01,
        3.80393539e-01,
        3.36066122e-01,
        1.74348851e-01,
        3.84703240e-02,
        5.35406512e-02,
        2.58099692e-02,
        3.12953566e-02,
        1.91172624e-02,
        1.70535491e-02,
        1.06285832e-02,
        1.00631304e-02,
        5.47251621e-03,
        7.42381231e-03,
        4.66994939e-03,
        4.44925491e-03,
        2.12998945e-03,
        4.13632844e-03,
        3.72238087e-03,
        4.19242408e-03,
        3.62637219e-03,
        2.53995848e-03,
        4.24097504e-03,
        4.39536291e-03,
        3.84542114e-03,
        3.73166756e-03,
        3.56286319e-03,
    ]
)
init_guess_3 = np.array(
    [
        1.87988095e-03,
        4.60706692e-04,
        3.35560698e-04,
        2.93998374e-16,
        8.13495009e-04,
        1.71859446e-03,
        1.43270293e-03,
        4.95064187e-04,
        3.26364579e-03,
        1.45280526e-03,
        1.54213691e-03,
        9.99661031e-04,
        1.37244596e-03,
        1.41154782e-03,
        3.56481585e-03,
        2.84740949e-03,
        2.43943943e-03,
        3.55635864e-03,
        5.62407615e-03,
        2.29731022e-03,
        3.68768757e-03,
        2.49431500e-03,
        4.96620095e-03,
        7.63202203e-03,
        6.99257883e-03,
        8.35265983e-03,
        3.79474160e-03,
        1.00000000e00,
        8.25577584e-02,
        5.31572079e-01,
        1.03161817e-01,
        1.94297384e-01,
        1.34652647e-01,
        9.08493410e-02,
        7.62066545e-02,
        5.38530993e-02,
        3.71858413e-02,
        4.79882057e-02,
        2.84049661e-02,
        2.25215771e-02,
        1.62592255e-03,
        2.74063023e-02,
        2.80963698e-02,
        2.68394478e-02,
        2.17206421e-02,
        2.33127023e-02,
        2.82922200e-02,
        2.74513249e-02,
        3.68525791e-02,
        1.10116214e-02,
        8.87897546e-03,
    ]
)
init_guess_4 = np.array(
    [
        5.65460834e-04,
        1.13590354e-03,
        2.39443435e-04,
        1.38021993e-04,
        5.41423486e-04,
        4.89114745e-04,
        1.98946142e-04,
        9.62669489e-04,
        6.86181032e-04,
        2.13772393e-03,
        6.09084800e-04,
        3.95467409e-06,
        2.80822639e-04,
        2.75460698e-17,
        1.15306243e-03,
        2.47110489e-04,
        1.21152754e-03,
        8.04182984e-04,
        6.42431738e-04,
        3.14402022e-03,
        9.05264822e-04,
        3.84839519e-03,
        6.69033216e-03,
        2.21558655e-03,
        6.06783079e-03,
        4.43721979e-03,
        7.78259681e-04,
        9.23221158e-03,
        6.92626527e-01,
        1.00000000e00,
        3.52251799e-02,
        4.45181143e-01,
        2.64545939e-01,
        1.93260230e-01,
        1.53364827e-01,
        1.10932369e-01,
        8.08444432e-02,
        9.12746645e-02,
        6.46259041e-02,
        4.61637870e-02,
        2.65293804e-05,
        6.16976587e-02,
        4.71788446e-02,
        4.06740922e-02,
        4.40515966e-02,
        5.16811782e-02,
        5.01560013e-02,
        4.03327863e-02,
        6.09883197e-02,
        2.39804186e-02,
        2.35660781e-02,
    ]
)
init_guess_5 = np.array(
    [
        4.98789830e-19,
        4.85949391e-19,
        4.10816002e-19,
        3.17461960e-19,
        3.56702834e-19,
        9.41792902e-19,
        6.42165106e-19,
        1.93869286e-19,
        1.98690830e-19,
        9.16055324e-19,
        1.49137733e-18,
        8.82348388e-19,
        5.50996746e-19,
        7.16959322e-19,
        1.38978219e-19,
        1.50849790e-18,
        1.04775109e-18,
        1.20130193e-19,
        1.08157928e-18,
        2.45087282e-19,
        1.09284279e-18,
        2.25850489e-19,
        5.09042046e-19,
        5.37070366e-19,
        1.38081254e-19,
        1.45484727e-19,
        1.24352321e-19,
        2.38606044e-19,
        9.78714555e-19,
        6.46969695e-19,
        2.55653230e-19,
        1.00000000e00,
        5.56194357e-01,
        4.62209114e-01,
        3.66317707e-01,
        2.68382748e-01,
        2.02160021e-01,
        1.08189908e-01,
        1.12569790e-01,
        2.80464842e-01,
        1.72218800e-01,
        1.58249066e-01,
        9.81354702e-02,
        4.44081502e-02,
        7.71788555e-02,
        6.03612945e-02,
        1.20821691e-02,
        2.89286946e-02,
        3.34959468e-02,
        1.43000603e-02,
        9.08255788e-03,
    ]
)

rel_norm_1 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_1
rel_norm_2 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_2
rel_norm_3 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_3
rel_norm_4 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_4
rel_norm_5 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_5

n_wd_1 = wdlf_1_number_density[wdlf_1_number_density > 1e-10]
n_wd_2 = wdlf_2_number_density[wdlf_2_number_density > 1e-10]
n_wd_3 = wdlf_3_number_density[wdlf_3_number_density > 1e-10]
n_wd_4 = wdlf_4_number_density[wdlf_4_number_density > 1e-10]
n_wd_5 = wdlf_5_number_density[wdlf_5_number_density > 1e-10]

n_wd_err_1 = wdlf_1_number_density[wdlf_1_number_density > 1e-10] * 0.1
n_wd_err_2 = wdlf_2_number_density[wdlf_2_number_density > 1e-10] * 0.1
n_wd_err_3 = wdlf_3_number_density[wdlf_3_number_density > 1e-10] * 0.1
n_wd_err_4 = wdlf_4_number_density[wdlf_4_number_density > 1e-10] * 0.1
n_wd_err_5 = wdlf_5_number_density[wdlf_5_number_density > 1e-10] * 0.1

mag_bin_1 = mag_bin[wdlf_1_number_density > 1e-10]
mag_bin_2 = mag_bin[wdlf_2_number_density > 1e-10]
mag_bin_3 = mag_bin[wdlf_3_number_density > 1e-10]
mag_bin_4 = mag_bin[wdlf_4_number_density > 1e-10]
mag_bin_5 = mag_bin[wdlf_5_number_density > 1e-10]

n_wd_err_1 /= np.nansum(n_wd_1)
n_wd_err_2 /= np.nansum(n_wd_2)
n_wd_err_3 /= np.nansum(n_wd_3)
n_wd_err_4 /= np.nansum(n_wd_4)
n_wd_err_5 /= np.nansum(n_wd_5)

n_wd_1 /= np.nansum(n_wd_1)
n_wd_2 /= np.nansum(n_wd_2)
n_wd_3 /= np.nansum(n_wd_3)
n_wd_4 /= np.nansum(n_wd_4)
n_wd_5 /= np.nansum(n_wd_5)

sfh_real_1 = exponential_profile(np.array(partial_age) * 1e9, 2.0e9, 2.0e9)
sfh_real_2 = exponential_profile(np.array(partial_age) * 1e9, 2.0e9, 4.0e9)
sfh_real_3 = exponential_profile(np.array(partial_age) * 1e9, 2.0e9, 6.0e9)
sfh_real_4 = exponential_profile(np.array(partial_age) * 1e9, 2.0e9, 8.0e9)
sfh_real_5 = exponential_profile(np.array(partial_age) * 1e9, 2.0e9, 1.0e10)

for i in range(10):
    sampler_1 = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            n_wd_1,
            n_wd_err_1,
            pwdlf_model_1,
        ),
    )
    sampler_2 = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            n_wd_2,
            n_wd_err_2,
            pwdlf_model_2,
        ),
    )
    sampler_3 = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            n_wd_3,
            n_wd_err_3,
            pwdlf_model_3,
        ),
    )
    sampler_4 = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            n_wd_4,
            n_wd_err_4,
            pwdlf_model_4,
        ),
    )
    sampler_5 = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            n_wd_5,
            n_wd_err_5,
            pwdlf_model_5,
        ),
    )
    #
    sampler_1.run_mcmc(rel_norm_1, 10000, progress=True)
    sampler_2.run_mcmc(rel_norm_2, 10000, progress=True)
    sampler_3.run_mcmc(rel_norm_3, 10000, progress=True)
    sampler_4.run_mcmc(rel_norm_4, 10000, progress=True)
    sampler_5.run_mcmc(rel_norm_5, 10000, progress=True)
    #
    flat_samples_1 = sampler_1.get_chain(discard=1000, flat=True)
    flat_samples_2 = sampler_2.get_chain(discard=1000, flat=True)
    flat_samples_3 = sampler_3.get_chain(discard=1000, flat=True)
    flat_samples_4 = sampler_4.get_chain(discard=1000, flat=True)
    flat_samples_5 = sampler_5.get_chain(discard=1000, flat=True)
    #
    solution_1 = np.zeros(ndim)
    solution_2 = np.zeros(ndim)
    solution_3 = np.zeros(ndim)
    solution_4 = np.zeros(ndim)
    solution_5 = np.zeros(ndim)
    for i in range(ndim):
        solution_1[i] = np.percentile(flat_samples_1[:, i], [50.0])
        solution_2[i] = np.percentile(flat_samples_2[:, i], [50.0])
        solution_3[i] = np.percentile(flat_samples_3[:, i], [50.0])
        solution_4[i] = np.percentile(flat_samples_4[:, i], [50.0])
        solution_5[i] = np.percentile(flat_samples_5[:, i], [50.0])
    #
    solution_1 /= np.nanmax(solution_1)
    solution_2 /= np.nanmax(solution_2)
    solution_3 /= np.nanmax(solution_3)
    solution_4 /= np.nanmax(solution_4)
    solution_5 /= np.nanmax(solution_5)
    #
    rel_norm_1 = (
        (np.random.rand(nwalkers, ndim) - 0.5) * 0.01 + 1
    ) * solution_1
    rel_norm_2 = (
        (np.random.rand(nwalkers, ndim) - 0.5) * 0.01 + 1
    ) * solution_2
    rel_norm_3 = (
        (np.random.rand(nwalkers, ndim) - 0.5) * 0.01 + 1
    ) * solution_3
    rel_norm_4 = (
        (np.random.rand(nwalkers, ndim) - 0.5) * 0.01 + 1
    ) * solution_4
    rel_norm_5 = (
        (np.random.rand(nwalkers, ndim) - 0.5) * 0.01 + 1
    ) * solution_5
    #
    recomputed_wdlf_1 = np.nansum(
        solution_1 * np.array(pwdlf_model_1).T, axis=1
    )
    recomputed_wdlf_2 = np.nansum(
        solution_2 * np.array(pwdlf_model_2).T, axis=1
    )
    recomputed_wdlf_3 = np.nansum(
        solution_3 * np.array(pwdlf_model_3).T, axis=1
    )
    recomputed_wdlf_4 = np.nansum(
        solution_4 * np.array(pwdlf_model_4).T, axis=1
    )
    recomputed_wdlf_5 = np.nansum(
        solution_5 * np.array(pwdlf_model_5).T, axis=1
    )
    #
    fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
        nrows=3, ncols=1, figsize=(8, 10), height_ratios=(5, 1, 8)
    )
    #
    ax_dummy1.axis("off")
    ax1.plot(age / 1e9, sfh_1 / np.nanmax(sfh_1), ls=":", color="C0")
    ax1.plot(age / 1e9, sfh_2 / np.nanmax(sfh_2), ls=":", color="C1")
    ax1.plot(age / 1e9, sfh_3 / np.nanmax(sfh_3), ls=":", color="C2")
    ax1.plot(age / 1e9, sfh_4 / np.nanmax(sfh_4), ls=":", color="C3")
    ax1.plot(age / 1e9, sfh_5 / np.nanmax(sfh_5), ls=":", color="C4")
    ax1.step(partial_age, solution_1, where="mid", color="C0")
    ax1.step(partial_age, solution_2, where="mid", color="C1")
    ax1.step(partial_age, solution_3, where="mid", color="C2")
    ax1.step(partial_age, solution_4, where="mid", color="C3")
    ax1.step(partial_age, solution_5, where="mid", color="C4")
    ax1.grid()
    ax1.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
    ax1.plot([-1, -1], [-1, -2], ls="-", color="black", label="MCMC")
    ax1.set_xticks(np.arange(0, 15, 2))
    ax1.set_xlim(0, 15)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Lookback t / Gyr")
    ax1.set_ylabel("Relative SFR")
    ax1.legend()
    #
    ax2.plot(
        mag_bin,
        wdlf_1_number_density,
        ls=":",
        color="C0",
    )
    ax2.plot(
        mag_bin,
        wdlf_2_number_density,
        ls=":",
        color="C1",
    )
    ax2.plot(
        mag_bin,
        wdlf_3_number_density,
        ls=":",
        color="C2",
    )
    ax2.plot(
        mag_bin,
        wdlf_4_number_density,
        ls=":",
        color="C3",
    )
    ax2.plot(
        mag_bin,
        wdlf_5_number_density,
        ls=":",
        color="C4",
    )
    #
    ax2.plot(
        mag_bin_1,
        recomputed_wdlf_1
        / np.nansum(recomputed_wdlf_1)
        * np.nansum(wdlf_1_number_density),
        color="C0",
    )
    ax2.plot(
        mag_bin_2,
        recomputed_wdlf_2
        / np.nansum(recomputed_wdlf_2)
        * np.nansum(wdlf_1_number_density),
        color="C1",
    )
    ax2.plot(
        mag_bin_3,
        recomputed_wdlf_3
        / np.nansum(recomputed_wdlf_3)
        * np.nansum(wdlf_3_number_density),
        color="C2",
    )
    ax2.plot(
        mag_bin_4,
        recomputed_wdlf_4
        / np.nansum(recomputed_wdlf_4)
        * np.nansum(wdlf_4_number_density),
        color="C3",
    )
    ax2.plot(
        mag_bin_5,
        recomputed_wdlf_5
        / np.nansum(recomputed_wdlf_5)
        * np.nansum(wdlf_5_number_density),
        color="C4",
    )
    ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
    ax2.set_ylabel("log(arbitrary number density)")
    ax2.set_xlim(6.75, 18.25)
    ax2.set_ylim(1e-5, 5e-1)
    ax2.set_yscale("log")
    ax2.plot([1, 1], [1, 2], ls=":", color="black", label="Input")
    ax2.plot(
        [1, 1], [1, 2], ls="-", color="black", label="Reconstructed (MCMC)"
    )
    ax2.legend()
    ax2.grid()
    #
    plt.subplots_adjust(
        top=0.975, bottom=0.075, left=0.125, right=0.975, hspace=0.075
    )
    plt.savefig(
        os.path.join(figure_folder, "fig_01_exponential_decay_wdlf.png")
    )


sfh_mcmc_lower_1 = np.zeros(ndim)
sfh_mcmc_lower_2 = np.zeros(ndim)
sfh_mcmc_lower_3 = np.zeros(ndim)
sfh_mcmc_lower_4 = np.zeros(ndim)
sfh_mcmc_lower_5 = np.zeros(ndim)
#
sfh_mcmc_1 = np.zeros(ndim)
sfh_mcmc_2 = np.zeros(ndim)
sfh_mcmc_3 = np.zeros(ndim)
sfh_mcmc_4 = np.zeros(ndim)
sfh_mcmc_5 = np.zeros(ndim)
#
sfh_mcmc_upper_1 = np.zeros(ndim)
sfh_mcmc_upper_2 = np.zeros(ndim)
sfh_mcmc_upper_3 = np.zeros(ndim)
sfh_mcmc_upper_4 = np.zeros(ndim)
sfh_mcmc_upper_5 = np.zeros(ndim)
#
for i in range(ndim):
    sfh_mcmc_lower_1[i], sfh_mcmc_1[i], sfh_mcmc_upper_1[i] = np.percentile(
        flat_samples_1[:, i], [31.7310508, 50.0, 68.2689492]
    )
    sfh_mcmc_lower_2[i], sfh_mcmc_2[i], sfh_mcmc_upper_2[i] = np.percentile(
        flat_samples_2[:, i], [31.7310508, 50.0, 68.2689492]
    )
    sfh_mcmc_lower_3[i], sfh_mcmc_3[i], sfh_mcmc_upper_3[i] = np.percentile(
        flat_samples_3[:, i], [31.7310508, 50.0, 68.2689492]
    )
    sfh_mcmc_lower_4[i], sfh_mcmc_4[i], sfh_mcmc_upper_4[i] = np.percentile(
        flat_samples_4[:, i], [31.7310508, 50.0, 68.2689492]
    )
    sfh_mcmc_lower_5[i], sfh_mcmc_5[i], sfh_mcmc_upper_5[i] = np.percentile(
        flat_samples_5[:, i], [31.7310508, 50.0, 68.2689492]
    )


sfh_mcmc_lower_1 /= np.nanmax(sfh_mcmc_1)
sfh_mcmc_lower_2 /= np.nanmax(sfh_mcmc_2)
sfh_mcmc_lower_3 /= np.nanmax(sfh_mcmc_3)
sfh_mcmc_lower_4 /= np.nanmax(sfh_mcmc_4)
sfh_mcmc_lower_5 /= np.nanmax(sfh_mcmc_5)

sfh_mcmc_upper_1 /= np.nanmax(sfh_mcmc_1)
sfh_mcmc_upper_2 /= np.nanmax(sfh_mcmc_2)
sfh_mcmc_upper_3 /= np.nanmax(sfh_mcmc_3)
sfh_mcmc_upper_4 /= np.nanmax(sfh_mcmc_4)
sfh_mcmc_upper_5 /= np.nanmax(sfh_mcmc_5)

# Finally refining with a minimizer
solution_lsq_1 = least_squares(
    log_probability,
    solution_1,
    args=(
        n_wd_1,
        n_wd_err_1,
        pwdlf_model_1,
    ),
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
).x
solution_lsq_2 = least_squares(
    log_probability,
    solution_2,
    args=(
        n_wd_2,
        n_wd_err_2,
        pwdlf_model_2,
    ),
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
).x
solution_lsq_3 = least_squares(
    log_probability,
    solution_3,
    args=(
        n_wd_3,
        n_wd_err_3,
        pwdlf_model_3,
    ),
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
).x
solution_lsq_4 = least_squares(
    log_probability,
    solution_4,
    args=(
        n_wd_4,
        n_wd_err_4,
        pwdlf_model_4,
    ),
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
).x
solution_lsq_5 = least_squares(
    log_probability,
    solution_5,
    args=(
        n_wd_5,
        n_wd_err_5,
        pwdlf_model_5,
    ),
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
).x
#
solution_lsq_1 /= max(solution_lsq_1)
solution_lsq_2 /= max(solution_lsq_2)
solution_lsq_3 /= max(solution_lsq_3)
solution_lsq_4 /= max(solution_lsq_4)
solution_lsq_5 /= max(solution_lsq_5)

recomputed_wdlf_lsq_1 = np.nansum(
    solution_lsq_1 * np.array(pwdlf_model_1).T, axis=1
)
recomputed_wdlf_lsq_2 = np.nansum(
    solution_lsq_2 * np.array(pwdlf_model_2).T, axis=1
)
recomputed_wdlf_lsq_3 = np.nansum(
    solution_lsq_3 * np.array(pwdlf_model_3).T, axis=1
)
recomputed_wdlf_lsq_4 = np.nansum(
    solution_lsq_4 * np.array(pwdlf_model_4).T, axis=1
)
recomputed_wdlf_lsq_5 = np.nansum(
    solution_lsq_5 * np.array(pwdlf_model_5).T, axis=1
)

np.save("fig_01_partial_wdlf", partial_wdlf)

np.save("fig_01_sfh_1", sfh_1)
np.save("fig_01_sfh_2", sfh_2)
np.save("fig_01_sfh_3", sfh_3)
np.save("fig_01_sfh_4", sfh_4)
np.save("fig_01_sfh_5", sfh_5)

np.save("fig_01_input_wdlf_1", wdlf_1_reconstructed)
np.save("fig_01_input_wdlf_2", wdlf_2_reconstructed)
np.save("fig_01_input_wdlf_3", wdlf_3_reconstructed)
np.save("fig_01_input_wdlf_4", wdlf_4_reconstructed)
np.save("fig_01_input_wdlf_5", wdlf_5_reconstructed)

np.save("fig_01_solution_1", np.column_stack([partial_age, solution_1]))
np.save("fig_01_solution_2", np.column_stack([partial_age, solution_2]))
np.save("fig_01_solution_3", np.column_stack([partial_age, solution_3]))
np.save("fig_01_solution_4", np.column_stack([partial_age, solution_4]))
np.save("fig_01_solution_5", np.column_stack([partial_age, solution_5]))

np.save(
    "fig_01_solution_lsq_1", np.column_stack([partial_age, solution_lsq_1])
)
np.save(
    "fig_01_solution_lsq_2", np.column_stack([partial_age, solution_lsq_2])
)
np.save(
    "fig_01_solution_lsq_3", np.column_stack([partial_age, solution_lsq_3])
)
np.save(
    "fig_01_solution_lsq_4", np.column_stack([partial_age, solution_lsq_4])
)
np.save(
    "fig_01_solution_lsq_5", np.column_stack([partial_age, solution_lsq_5])
)


np.save(
    "fig_01_recomputed_wdlf_mcmc_1",
    np.column_stack([mag_bin_1, recomputed_wdlf_1]),
)
np.save(
    "fig_01_recomputed_wdlf_mcmc_2",
    np.column_stack([mag_bin_2, recomputed_wdlf_2]),
)
np.save(
    "fig_01_recomputed_wdlf_mcmc_3",
    np.column_stack([mag_bin_3, recomputed_wdlf_3]),
)
np.save(
    "fig_01_recomputed_wdlf_mcmc_4",
    np.column_stack([mag_bin_4, recomputed_wdlf_4]),
)
np.save(
    "fig_01_recomputed_wdlf_mcmc_5",
    np.column_stack([mag_bin_5, recomputed_wdlf_5]),
)

np.save(
    "fig_01_recomputed_wdlf_lsq_1",
    np.column_stack([mag_bin_1, recomputed_wdlf_lsq_1]),
)
np.save(
    "fig_01_recomputed_wdlf_lsq_2",
    np.column_stack([mag_bin_2, recomputed_wdlf_lsq_2]),
)
np.save(
    "fig_01_recomputed_wdlf_lsq_3",
    np.column_stack([mag_bin_3, recomputed_wdlf_lsq_3]),
)
np.save(
    "fig_01_recomputed_wdlf_lsq_4",
    np.column_stack([mag_bin_4, recomputed_wdlf_lsq_4]),
)
np.save(
    "fig_01_recomputed_wdlf_lsq_5",
    np.column_stack([mag_bin_5, recomputed_wdlf_lsq_5]),
)

fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(5, 1, 8)
)
#
ax1.plot(age, sfh_1 / np.nanmax(sfh_1), ls=":", color="C0")
ax1.plot(age, sfh_2 / np.nanmax(sfh_2), ls=":", color="C1")
ax1.plot(age, sfh_3 / np.nanmax(sfh_3), ls=":", color="C2")
ax1.plot(age, sfh_4 / np.nanmax(sfh_4), ls=":", color="C3")
ax1.plot(age, sfh_5 / np.nanmax(sfh_5), ls=":", color="C4")
#
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_1,
    sfh_mcmc_upper_1,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_2,
    sfh_mcmc_upper_2,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_3,
    sfh_mcmc_upper_3,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_4,
    sfh_mcmc_upper_4,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_5,
    sfh_mcmc_upper_5,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
#
ax1.step(partial_age, solution_1, ls="dashed", where="mid", color="C0")
ax1.step(partial_age, solution_2, ls="dashed", where="mid", color="C1")
ax1.step(partial_age, solution_3, ls="dashed", where="mid", color="C2")
ax1.step(partial_age, solution_4, ls="dashed", where="mid", color="C3")
ax1.step(partial_age, solution_5, ls="dashed", where="mid", color="C4")
#
ax1.step(partial_age, solution_lsq_1, where="mid", color="C0")
ax1.step(partial_age, solution_lsq_2, where="mid", color="C1")
ax1.step(partial_age, solution_lsq_3, where="mid", color="C2")
ax1.step(partial_age, solution_lsq_4, where="mid", color="C3")
ax1.step(partial_age, solution_lsq_5, where="mid", color="C4")
#
ax1.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
ax1.plot([-1, -1], [-1, -2], ls="dashed", color="black", label="MCMC")
ax1.plot([-1, -1], [-1, -2], ls="-", color="black", label="lsq-refined")
#
ax1.grid()
ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 15)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Lookback t / Gyr")
ax1.set_ylabel("Relative SFR")
ax1.legend()
#
ax_dummy1.axis("off")
#
ax2.plot(
    mag_bin,
    wdlf_1_number_density / np.nansum(wdlf_1_number_density),
    ls=":",
    color="C0",
)
ax2.plot(
    mag_bin,
    wdlf_2_number_density / np.nansum(wdlf_2_number_density),
    ls=":",
    color="C1",
)
ax2.plot(
    mag_bin,
    wdlf_3_number_density / np.nansum(wdlf_3_number_density),
    ls=":",
    color="C2",
)
ax2.plot(
    mag_bin,
    wdlf_4_number_density / np.nansum(wdlf_4_number_density),
    ls=":",
    color="C3",
)
ax2.plot(
    mag_bin,
    wdlf_5_number_density / np.nansum(wdlf_5_number_density),
    ls=":",
    color="C4",
)
#
ax2.plot(
    mag_bin_1,
    recomputed_wdlf_1 / np.nansum(recomputed_wdlf_1),
    ls="dashed",
    color="C0",
)
ax2.plot(
    mag_bin_2,
    recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2),
    ls="dashed",
    color="C1",
)
ax2.plot(
    mag_bin_3,
    recomputed_wdlf_3 / np.nansum(recomputed_wdlf_3),
    ls="dashed",
    color="C2",
)
ax2.plot(
    mag_bin_4,
    recomputed_wdlf_4 / np.nansum(recomputed_wdlf_4),
    ls="dashed",
    color="C3",
)
ax2.plot(
    mag_bin_5,
    recomputed_wdlf_5 / np.nansum(recomputed_wdlf_5),
    ls="dashed",
    color="C4",
)
#
ax2.plot(
    mag_bin_1,
    recomputed_wdlf_lsq_1 / np.nansum(recomputed_wdlf_lsq_1),
    color="C0",
)
ax2.plot(
    mag_bin_2,
    recomputed_wdlf_lsq_2 / np.nansum(recomputed_wdlf_lsq_2),
    color="C1",
)
ax2.plot(
    mag_bin_3,
    recomputed_wdlf_lsq_3 / np.nansum(recomputed_wdlf_lsq_3),
    color="C2",
)
ax2.plot(
    mag_bin_4,
    recomputed_wdlf_lsq_4 / np.nansum(recomputed_wdlf_lsq_4),
    color="C3",
)
ax2.plot(
    mag_bin_5,
    recomputed_wdlf_lsq_5 / np.nansum(recomputed_wdlf_lsq_5),
    color="C4",
)
#
ax2.plot([1, 1], [1, 2], ls=":", color="black", label="Input")
ax2.plot(
    [1, 1], [1, 2], ls="dashed", color="black", label="Reconstructed (MCMC)"
)
ax2.plot([1, 1], [1, 2], ls="-", color="black", label="Reconstructed (lsq)")
#
ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_xlim(6.75, 18.25)
ax2.set_ylim(5e-5, 3e-1)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
#
plt.subplots_adjust(
    top=0.975, bottom=0.075, left=0.125, right=0.975, hspace=0.075
)
plt.savefig(os.path.join(figure_folder, "fig_01_exponential_decay_wdlf.png"))
