import os

import emcee
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import least_squares
from spectres import spectres
from WDPhotTools import theoretical_lf
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader


figure_folder = "SFH-WDLF-article/figures"
plt.rcParams.update({"font.size": 12})


def log_prior(theta):
    if (theta <= 0.0).any():
        return -np.inf
    else:
        return 0.0


def log_probability(rel_norm, obs, err, model_list):
    rel_norm /= np.sum(rel_norm)
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = rel_norm[:, None].T @ model_list
    model /= np.nansum(model)
    err /= np.nansum(obs)
    obs /= np.nansum(obs)
    log_likelihood = -np.nansum(((model - obs) / err) ** 2.0)
    return log_likelihood


# Load the pwdlfs
data = []
age_list_1 = np.arange(0.049, 0.100, 0.001)
age_list_2 = np.arange(0.100, 0.350, 0.005)
age_list_3 = np.arange(0.35, 14.01, 0.01)
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
# Two Bursts - an intensity burst at 2 Gyr, a broader one one at 8 Gyr
#
###############################################################################


def gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function.
    Parameters
    ----------
    x: float or 1-d numpy array
        The data to evaluate the Gaussian over
    a: float
        the amplitude
    b: float
        the constant offset
    x0: float
        the center of the Gaussian
    sigma: float
        the width of the Gaussian
    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + b


sfh_broad = gaus(age * 1e9, 0.3, 0.0, 8.0e9, 1.5e9)
sfh_1 = gaus(age * 1e9, 1.0, 0.0, 2.0e9, 0.25e9)
sfh_2 = gaus(age * 1e9, 1.0, 0.0, 4.0e9, 0.25e9)
sfh_3 = gaus(age * 1e9, 1.0, 0.0, 6.0e9, 0.25e9)
sfh_4 = gaus(age * 1e9, 1.0, 0.0, 8.0e9, 0.25e9)
sfh_5 = gaus(age * 1e9, 1.0, 0.0, 10.0e9, 0.25e9)
sfh_total_1 = sfh_1 + sfh_broad
sfh_total_2 = sfh_2 + sfh_broad
sfh_total_3 = sfh_3 + sfh_broad
sfh_total_4 = sfh_4 + sfh_broad
sfh_total_5 = sfh_5 + sfh_broad
sfh_total_combined = sfh_1 + sfh_2 + sfh_3 + sfh_4 + sfh_5 + sfh_broad

wdlf_reconstructed_1 = np.array(data)[:, :, 1].T @ sfh_total_1
wdlf_reconstructed_2 = np.array(data)[:, :, 1].T @ sfh_total_2
wdlf_reconstructed_3 = np.array(data)[:, :, 1].T @ sfh_total_3
wdlf_reconstructed_4 = np.array(data)[:, :, 1].T @ sfh_total_4
wdlf_reconstructed_5 = np.array(data)[:, :, 1].T @ sfh_total_5
wdlf_reconstructed_combined = np.array(data)[:, :, 1].T @ sfh_total_combined

mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag_pwdlf[np.argmax(d[:, 1])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=len(age) / 150, k=5
)

# Load the "resolution" of the Mbol solution
# mbol_err_upper_bound = interpolate.UnivariateSpline(
#    *np.load("mbol_err_upper_bound.npy").T, s=0
# )
mbol_err_upper_bound = interpolate.UnivariateSpline(
    *np.load("SFH-WDLF-article/figure_data/mbol_err_mean.npy").T, s=0
)

mag_bin = []
bin_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
mag_bin_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
bin_number = 0
stop = False
# the bin_optimal is the bin edges
mag_bin.append(mag_at_peak_density[0])
for i, a in enumerate(age):
    if i < j:
        continue
    start = float(mag_resolution_itp(a))
    end = start
    j = i
    carry_on = True
    if j >= len(age) - 1:
        carry_on = False
    while carry_on:
        tmp = mag_resolution_itp(age[j + 1]) - mag_resolution_itp(age[j])
        print(j, end + tmp - start)
        mag_bin_pwdlf[j] = bin_number
        # Nyquist sampling: sample 2 times for every FWHM (i.e. 2.355 sigma)
        if end + tmp - start < mbol_err_upper_bound(start) * 1.1775:
            end = end + tmp
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

# if magnitudes don't reach 18, pad to 18 in 0.2 mag increment
if max(mag_bin) < 18.0:
    _n = int(np.round((18.0 - mag_bin[-1]) / 0.2))
    mag_bin = np.append(mag_bin[:-1], np.linspace(mag_bin[-1], 18.0, _n))

# get the bin centre
mag_bin = (mag_bin[1:] + mag_bin[:-1]) / 2.0
pwdlf_mapping_mag_bin = np.insert(mag_bin_pwdlf, 0, 0)

wdlf_number_density_1 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_1)
wdlf_number_density_2 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_2)
wdlf_number_density_3 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_3)
wdlf_number_density_4 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_4)
wdlf_number_density_5 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_5)
wdlf_number_density_combined = spectres(
    mag_bin, data[0][:, 0], wdlf_reconstructed_combined
)

wdlf_number_density_1 /= np.sum(wdlf_number_density_1)
wdlf_number_density_2 /= np.sum(wdlf_number_density_2)
wdlf_number_density_3 /= np.sum(wdlf_number_density_3)
wdlf_number_density_4 /= np.sum(wdlf_number_density_4)
wdlf_number_density_5 /= np.sum(wdlf_number_density_5)
wdlf_number_density_combined /= np.sum(wdlf_number_density_combined)

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
pwdlf_model_1 = np.vstack(partial_wdlf)[:, wdlf_number_density_1 > 0.0]
pwdlf_model_1 /= np.nansum(pwdlf_model_1) * np.nanmax(wdlf_number_density_1)

pwdlf_model_2 = np.vstack(partial_wdlf)[:, wdlf_number_density_2 > 0.0]
pwdlf_model_2 /= np.nansum(pwdlf_model_2) * np.nanmax(wdlf_number_density_2)

pwdlf_model_3 = np.vstack(partial_wdlf)[:, wdlf_number_density_3 > 0.0]
pwdlf_model_3 /= np.nansum(pwdlf_model_3) * np.nanmax(wdlf_number_density_3)

pwdlf_model_4 = np.vstack(partial_wdlf)[:, wdlf_number_density_4 > 0.0]
pwdlf_model_4 /= np.nansum(pwdlf_model_4) * np.nanmax(wdlf_number_density_4)

pwdlf_model_5 = np.vstack(partial_wdlf)[:, wdlf_number_density_5 > 0.0]
pwdlf_model_5 /= np.nansum(pwdlf_model_5) * np.nanmax(wdlf_number_density_5)

pwdlf_model_combined = np.vstack(partial_wdlf)[
    :, wdlf_number_density_combined > 0.0
]
pwdlf_model_combined /= np.nansum(pwdlf_model_combined) * np.nanmax(
    wdlf_number_density_combined
)

nwalkers = 250
ndim = len(partial_wdlf)
init_guess_1 = np.array([2.51207737e-02, 1.63992915e-02, 9.29823145e-03, 1.13673706e-02,
       1.64840213e-02, 1.27459312e-02, 1.59139221e-01, 1.00000000e+00,
       7.27969300e-01, 1.15928054e-02, 1.00658795e-02, 4.53701736e-03,
       2.93105114e-03, 5.88583040e-03, 6.10183116e-03, 1.07696334e-02,
       1.54322727e-02, 2.35562770e-02, 3.41788915e-02, 3.74469431e-02,
       6.86011546e-02, 9.78065154e-02, 5.86490219e-02, 2.43614434e-01,
       3.50492247e-01, 2.75040491e-01, 3.24478047e-01, 2.15351363e-01,
       1.73431812e-01, 1.36594478e-01, 8.32070844e-02, 6.12688187e-02,
       4.81886618e-02, 2.90234218e-02, 2.00944703e-02, 1.14855921e-02,
       6.57845977e-03, 3.23611778e-03, 1.27319458e-03, 4.88811181e-04,
       6.02632525e-04, 7.70514477e-04, 9.61031980e-04])
init_guess_2 = np.array([1.10236968e-02, 5.91305430e-03, 4.96477165e-03, 5.45993716e-03,
       4.47783637e-03, 4.73426473e-03, 4.15746437e-03, 5.09318067e-03,
       4.96869946e-03, 6.20857002e-03, 7.32366159e-03, 1.83193079e-02,
       1.99900774e-02, 1.34564416e-01, 1.00000000e+00, 9.72681232e-01,
       7.11312225e-01, 1.25942113e-01, 6.32356895e-02, 3.00175080e-02,
       5.41289802e-02, 1.18771977e-01, 5.37971986e-02, 2.19087797e-01,
       3.67491889e-01, 2.49918631e-01, 3.26201672e-01, 2.14910140e-01,
       1.75753776e-01, 1.15113516e-01, 9.19665713e-02, 6.23207952e-02,
       4.63026318e-02, 2.70319185e-02, 2.04945676e-02, 1.12280542e-02,
       6.11832278e-03, 2.83896526e-03, 1.23060555e-03, 5.44969300e-04,
       6.08158645e-04, 7.67298779e-04, 9.80414931e-04])
init_guess_3 = np.array([1.23772377e-02, 5.62087158e-03, 5.10116407e-03, 6.07611924e-03,
       5.13830859e-03, 4.87420558e-03, 7.51551725e-03, 8.24315853e-03,
       6.06453748e-03, 4.07499424e-03, 4.61597071e-03, 3.33538813e-03,
       4.32810270e-03, 6.06271217e-03, 9.16702295e-03, 2.53227574e-02,
       3.85360687e-02, 6.57964441e-02, 5.17502798e-02, 1.62736940e-02,
       2.15238556e-01, 7.23057733e-01, 3.38186711e-01, 1.00000000e+00,
       3.43707074e-01, 1.94021161e-01, 4.90162155e-01, 2.20078475e-01,
       1.91217069e-01, 1.72583457e-01, 1.04406794e-01, 6.18161978e-02,
       6.04228271e-02, 4.63470822e-02, 2.04723578e-02, 5.47670874e-03,
       5.00448690e-03, 6.98611707e-03, 2.19741045e-03, 8.07563166e-04,
       9.25746128e-04, 1.40830511e-03, 1.65521923e-03])
init_guess_4 = np.array([6.40160070e-03, 3.88401418e-03, 3.00239996e-03, 2.89974270e-03,
       3.39560438e-03, 2.51322766e-03, 2.52238174e-03, 2.16629695e-03,
       2.68597189e-03, 2.07657579e-03, 2.73541855e-03, 3.08616347e-03,
       2.76357056e-03, 4.58026780e-03, 5.26572552e-03, 8.58078806e-03,
       1.31873334e-02, 1.92040352e-02, 3.61021245e-02, 2.14065024e-02,
       5.79406496e-02, 7.01924629e-02, 3.10455215e-02, 1.68237536e-01,
       4.55455801e-01, 1.00000000e+00, 2.14527012e-01, 1.46529497e-01,
       1.73803281e-01, 9.97256520e-02, 6.93150690e-02, 5.82957507e-02,
       3.90836140e-02, 2.29978248e-02, 1.71721465e-02, 9.45394819e-03,
       4.97181901e-03, 2.15816634e-03, 1.25742289e-03, 7.49973645e-04,
       7.72218299e-04, 9.52241666e-04, 9.27664188e-04])
init_guess_5 = np.array([5.85174331e-03, 3.40194444e-03, 2.19767782e-03, 2.36118890e-03,
       2.10511366e-03, 2.40158857e-03, 2.16905287e-03, 1.81568587e-03,
       2.04007348e-03, 2.52114454e-03, 2.28801808e-03, 2.57206557e-03,
       2.48890023e-03, 4.03006880e-03, 5.50846514e-03, 7.69283341e-03,
       1.22584289e-02, 1.68079885e-02, 2.96234666e-02, 1.85028160e-02,
       5.68540676e-02, 5.99070583e-02, 3.43220093e-02, 2.01439975e-01,
       2.66875695e-01, 2.03108457e-01, 2.56948961e-01, 1.27459699e-01,
       4.58614423e-01, 1.00000000e+00, 2.72823770e-01, 4.51742503e-02,
       3.24436157e-02, 2.71899724e-02, 1.38995397e-02, 7.72902703e-03,
       5.19093435e-03, 2.25073552e-03, 1.04319969e-03, 6.19648749e-04,
       5.87359409e-04, 9.30905529e-04, 1.03637651e-03])
init_guess_combined = np.array([2.94769779e-02, 1.90439418e-02, 1.14677243e-02, 1.36856826e-02,
       1.30788831e-02, 1.83138967e-02, 1.16867488e-01, 7.34521375e-01,
       5.40651957e-01, 1.56581305e-02, 1.43672498e-02, 1.51315293e-02,
       2.06419004e-02, 8.43885137e-02, 7.68832107e-01, 7.48334015e-01,
       5.45587734e-01, 1.08500957e-01, 5.61554104e-02, 2.80058525e-02,
       8.26630143e-02, 5.44726864e-01, 3.31680236e-01, 3.92725715e-01,
       4.37166576e-01, 7.60843560e-01, 2.67025635e-01, 9.07092003e-02,
       4.73620154e-01, 1.00000000e+00, 2.74580849e-01, 4.09127715e-02,
       3.92498669e-02, 2.34865886e-02, 1.47571660e-02, 5.53674472e-03,
       3.05724195e-03, 2.23294956e-03, 1.19953783e-03, 8.38360164e-04,
       1.05728533e-03, 1.49494163e-03, 1.85405644e-03])


#init_guess_1 = np.random.random(ndim)
#init_guess_2 = np.random.random(ndim)
#init_guess_3 = np.random.random(ndim)
#init_guess_4 = np.random.random(ndim)
#init_guess_5 = np.random.random(ndim)
#init_guess_combined = np.random.random(ndim)

rel_norm_1 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_1
rel_norm_2 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_2
rel_norm_3 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_3
rel_norm_4 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_4
rel_norm_5 = ((np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1) * init_guess_5
rel_norm_combined = (
    (np.random.rand(nwalkers, ndim) - 0.5) * 0.05 + 1
) * init_guess_combined

n_wd_1 = wdlf_number_density_1[wdlf_number_density_1 > 0.0]
n_wd_2 = wdlf_number_density_2[wdlf_number_density_2 > 0.0]
n_wd_3 = wdlf_number_density_3[wdlf_number_density_3 > 0.0]
n_wd_4 = wdlf_number_density_4[wdlf_number_density_4 > 0.0]
n_wd_5 = wdlf_number_density_5[wdlf_number_density_5 > 0.0]
n_wd_combined = wdlf_number_density_combined[
    wdlf_number_density_combined > 0.0
]

n_wd_err_1 = n_wd_1 * 0.05
n_wd_err_2 = n_wd_2 * 0.05
n_wd_err_3 = n_wd_3 * 0.05
n_wd_err_4 = n_wd_4 * 0.05
n_wd_err_5 = n_wd_5 * 0.05
n_wd_err_combined = n_wd_combined * 0.05

n_step = 10000
n_burn = 1000

for i in range(1):
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
    sampler_combined = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(
            n_wd_combined,
            n_wd_err_combined,
            pwdlf_model_combined,
        ),
    )
    #
    sampler_1.run_mcmc(rel_norm_1, n_step, progress=True)
    sampler_2.run_mcmc(rel_norm_2, n_step, progress=True)
    sampler_3.run_mcmc(rel_norm_3, n_step, progress=True)
    sampler_4.run_mcmc(rel_norm_4, n_step, progress=True)
    sampler_5.run_mcmc(rel_norm_5, n_step, progress=True)
    sampler_combined.run_mcmc(rel_norm_combined, n_step, progress=True)
    #
    flat_samples_1 = sampler_1.get_chain(discard=n_burn, flat=True)
    flat_samples_2 = sampler_2.get_chain(discard=n_burn, flat=True)
    flat_samples_3 = sampler_3.get_chain(discard=n_burn, flat=True)
    flat_samples_4 = sampler_4.get_chain(discard=n_burn, flat=True)
    flat_samples_5 = sampler_5.get_chain(discard=n_burn, flat=True)
    flat_samples_combined = sampler_combined.get_chain(
        discard=n_burn, flat=True
    )
    #
    solution_1 = np.zeros(ndim)
    solution_2 = np.zeros(ndim)
    solution_3 = np.zeros(ndim)
    solution_4 = np.zeros(ndim)
    solution_5 = np.zeros(ndim)
    solution_combined = np.zeros(ndim)
    for i in range(ndim):
        solution_1[i] = np.nanpercentile(flat_samples_1[:, i], [50.0])
        solution_2[i] = np.nanpercentile(flat_samples_2[:, i], [50.0])
        solution_3[i] = np.nanpercentile(flat_samples_3[:, i], [50.0])
        solution_4[i] = np.nanpercentile(flat_samples_4[:, i], [50.0])
        solution_5[i] = np.nanpercentile(flat_samples_5[:, i], [50.0])
        solution_combined[i] = np.nanpercentile(
            flat_samples_combined[:, i], [50.0]
        )
    solution_1 /= np.nanmax(solution_1)
    solution_2 /= np.nanmax(solution_2)
    solution_3 /= np.nanmax(solution_3)
    solution_4 /= np.nanmax(solution_4)
    solution_5 /= np.nanmax(solution_5)
    solution_combined /= np.nanmax(solution_combined)
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
    rel_norm_combined = (
        (np.random.rand(nwalkers, ndim) - 0.5) * 0.01 + 1
    ) * solution_combined
    #
    recomputed_wdlf_1 = np.nansum(
        solution_1 * np.array(partial_wdlf).T, axis=1
    )
    recomputed_wdlf_2 = np.nansum(
        solution_2 * np.array(partial_wdlf).T, axis=1
    )
    recomputed_wdlf_3 = np.nansum(
        solution_3 * np.array(partial_wdlf).T, axis=1
    )
    recomputed_wdlf_4 = np.nansum(
        solution_4 * np.array(partial_wdlf).T, axis=1
    )
    recomputed_wdlf_5 = np.nansum(
        solution_5 * np.array(partial_wdlf).T, axis=1
    )
    recomputed_wdlf_combined = np.nansum(
        solution_combined * np.array(partial_wdlf).T, axis=1
    )
    #
    fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
        nrows=3, ncols=1, figsize=(8, 10), height_ratios=(10, 3, 15)
    )
    #
    ax1.plot(
        age,
        sfh_total_1 / np.nanmax(sfh_total_1),
        ls=":",
        color="C0",
    )
    ax1.plot(
        age,
        sfh_total_2 / np.nanmax(sfh_total_2),
        ls=":",
        color="C1",
    )
    ax1.plot(
        age,
        sfh_total_3 / np.nanmax(sfh_total_3),
        ls=":",
        color="C2",
    )
    ax1.plot(
        age,
        sfh_total_4 / np.nanmax(sfh_total_4),
        ls=":",
        color="C3",
    )
    ax1.plot(
        age,
        sfh_total_5 / np.nanmax(sfh_total_5),
        ls=":",
        color="C4",
    )
    ax1.plot(
        age,
        sfh_total_combined / np.nanmax(sfh_total_combined),
        ls=":",
        color="C5",
    )
    #
    ax1.step(partial_age, solution_1, where="post", color="C0")
    ax1.step(partial_age, solution_2, where="post", color="C1")
    ax1.step(partial_age, solution_3, where="post", color="C2")
    ax1.step(partial_age, solution_4, where="post", color="C3")
    ax1.step(partial_age, solution_5, where="post", color="C4")
    ax1.step(partial_age, solution_combined, where="post", color="C5")
    #
    ax1.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
    ax1.plot([-1, -1], [-1, -2], ls="dashed", color="black", label="MCMC")
    #
    ax1.grid()
    ax1.set_xticks(np.arange(0, 15, 2))
    ax1.set_xlim(0, 15)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Lookback time / Gyr")
    ax1.set_ylabel("Relative Star Formation Rate")
    ax1.legend()
    #
    ax_dummy1.axis("off")
    #
    ax2.plot(mag_bin, n_wd_1, ls=":", color="C0")
    ax2.plot(mag_bin, n_wd_2, ls=":", color="C1")
    ax2.plot(mag_bin, n_wd_3, ls=":", color="C2")
    ax2.plot(mag_bin, n_wd_4, ls=":", color="C3")
    ax2.plot(mag_bin, n_wd_5, ls=":", color="C4")
    ax2.plot(mag_bin, n_wd_combined, ls=":", color="C5")
    #
    ax2.plot(
        mag_bin,
        recomputed_wdlf_1 / np.nansum(recomputed_wdlf_1) * np.nansum(n_wd_1),
        color="C0",
    )
    ax2.plot(
        mag_bin,
        recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2) * np.nansum(n_wd_2),
        color="C1",
    )
    ax2.plot(
        mag_bin,
        recomputed_wdlf_3 / np.nansum(recomputed_wdlf_3) * np.nansum(n_wd_3),
        color="C2",
    )
    ax2.plot(
        mag_bin,
        recomputed_wdlf_4 / np.nansum(recomputed_wdlf_4) * np.nansum(n_wd_4),
        color="C3",
    )
    ax2.plot(
        mag_bin,
        recomputed_wdlf_5 / np.nansum(recomputed_wdlf_5) * np.nansum(n_wd_5),
        color="C4",
    )
    ax2.plot(
        mag_bin,
        recomputed_wdlf_combined
        / np.nansum(recomputed_wdlf_combined)
        * np.nansum(n_wd_combined),
        color="C5",
    )
    #
    ax2.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
    ax2.plot([-1, -1], [-1, -2], ls="-", color="black", label="Reconstructed")
    #
    ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
    ax2.set_ylabel("log(arbitrary number density)")
    ax2.set_xlim(6.75, 18.25)
    ax2.set_ylim(1e-5, 5e-1)
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid()
    #
    plt.subplots_adjust(
        top=0.975, bottom=0.075, left=0.125, right=0.975, hspace=0.075
    )
    plt.savefig(os.path.join(figure_folder, "fig_02_two_bursts_wdlf.png"))


sfh_mcmc_lower_1 = np.zeros(ndim)
sfh_mcmc_lower_2 = np.zeros(ndim)
sfh_mcmc_lower_3 = np.zeros(ndim)
sfh_mcmc_lower_4 = np.zeros(ndim)
sfh_mcmc_lower_5 = np.zeros(ndim)
sfh_mcmc_lower_combined = np.zeros(ndim)
#
sfh_mcmc_1 = np.zeros(ndim)
sfh_mcmc_2 = np.zeros(ndim)
sfh_mcmc_3 = np.zeros(ndim)
sfh_mcmc_4 = np.zeros(ndim)
sfh_mcmc_5 = np.zeros(ndim)
sfh_mcmc_combined = np.zeros(ndim)
#
sfh_mcmc_upper_1 = np.zeros(ndim)
sfh_mcmc_upper_2 = np.zeros(ndim)
sfh_mcmc_upper_3 = np.zeros(ndim)
sfh_mcmc_upper_4 = np.zeros(ndim)
sfh_mcmc_upper_5 = np.zeros(ndim)
sfh_mcmc_upper_combined = np.zeros(ndim)
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
    (
        sfh_mcmc_lower_combined[i],
        sfh_mcmc_combined[i],
        sfh_mcmc_upper_combined[i],
    ) = np.percentile(
        flat_samples_combined[:, i], [31.7310508, 50.0, 68.2689492]
    )


sfh_mcmc_1 /= np.nanmax(sfh_mcmc_1)
sfh_mcmc_2 /= np.nanmax(sfh_mcmc_2)
sfh_mcmc_3 /= np.nanmax(sfh_mcmc_3)
sfh_mcmc_4 /= np.nanmax(sfh_mcmc_4)
sfh_mcmc_5 /= np.nanmax(sfh_mcmc_5)
sfh_mcmc_combined /= np.nanmax(sfh_mcmc_combined)

sfh_mcmc_lower_1 /= np.nanmax(sfh_mcmc_1)
sfh_mcmc_lower_2 /= np.nanmax(sfh_mcmc_2)
sfh_mcmc_lower_3 /= np.nanmax(sfh_mcmc_3)
sfh_mcmc_lower_4 /= np.nanmax(sfh_mcmc_4)
sfh_mcmc_lower_5 /= np.nanmax(sfh_mcmc_5)
sfh_mcmc_lower_combined /= np.nanmax(sfh_mcmc_combined)

sfh_mcmc_upper_1 /= np.nanmax(sfh_mcmc_1)
sfh_mcmc_upper_2 /= np.nanmax(sfh_mcmc_2)
sfh_mcmc_upper_3 /= np.nanmax(sfh_mcmc_3)
sfh_mcmc_upper_4 /= np.nanmax(sfh_mcmc_4)
sfh_mcmc_upper_5 /= np.nanmax(sfh_mcmc_5)
sfh_mcmc_upper_combined /= np.nanmax(sfh_mcmc_combined)


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
solution_lsq_combined = least_squares(
    log_probability,
    solution_combined,
    args=(
        n_wd_combined,
        n_wd_err_combined,
        pwdlf_model_combined,
    ),
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
).x
solution_lsq_1 /= np.nanmax(solution_lsq_1)
solution_lsq_2 /= np.nanmax(solution_lsq_2)
solution_lsq_3 /= np.nanmax(solution_lsq_3)
solution_lsq_4 /= np.nanmax(solution_lsq_4)
solution_lsq_5 /= np.nanmax(solution_lsq_5)
solution_lsq_combined /= np.nanmax(solution_lsq_combined)

recomputed_wdlf_lsq_1 = np.nansum(
    solution_lsq_1 * np.array(partial_wdlf).T, axis=1
)
recomputed_wdlf_lsq_2 = np.nansum(
    solution_lsq_2 * np.array(partial_wdlf).T, axis=1
)
recomputed_wdlf_lsq_3 = np.nansum(
    solution_lsq_3 * np.array(partial_wdlf).T, axis=1
)
recomputed_wdlf_lsq_4 = np.nansum(
    solution_lsq_4 * np.array(partial_wdlf).T, axis=1
)
recomputed_wdlf_lsq_5 = np.nansum(
    solution_lsq_5 * np.array(partial_wdlf).T, axis=1
)
recomputed_wdlf_lsq_combined = np.nansum(
    solution_lsq_combined * np.array(partial_wdlf).T, axis=1
)


np.save("SFH-WDLF-article/figure_data/fig_02_partial_wdlf", partial_wdlf)

np.save("SFH-WDLF-article/figure_data/fig_02_sfh_total_1", sfh_total_1)
np.save("SFH-WDLF-article/figure_data/fig_02_sfh_total_2", sfh_total_2)
np.save("SFH-WDLF-article/figure_data/fig_02_sfh_total_3", sfh_total_3)
np.save("SFH-WDLF-article/figure_data/fig_02_sfh_total_4", sfh_total_4)
np.save("SFH-WDLF-article/figure_data/fig_02_sfh_total_5", sfh_total_5)
np.save("SFH-WDLF-article/figure_data/fig_02_sfh_total_combined", sfh_total_combined)

np.save("SFH-WDLF-article/figure_data/fig_02_input_wdlf_1", wdlf_number_density_1)
np.save("SFH-WDLF-article/figure_data/fig_02_input_wdlf_2", wdlf_number_density_2)
np.save("SFH-WDLF-article/figure_data/fig_02_input_wdlf_3", wdlf_number_density_3)
np.save("SFH-WDLF-article/figure_data/fig_02_input_wdlf_4", wdlf_number_density_4)
np.save("SFH-WDLF-article/figure_data/fig_02_input_wdlf_5", wdlf_number_density_5)
np.save("SFH-WDLF-article/figure_data/fig_02_input_wdlf_combined", wdlf_number_density_combined)

np.save("SFH-WDLF-article/figure_data/fig_02_solution_1", np.column_stack([partial_age, solution_1]))
np.save("SFH-WDLF-article/figure_data/fig_02_solution_2", np.column_stack([partial_age, solution_2]))
np.save("SFH-WDLF-article/figure_data/fig_02_solution_3", np.column_stack([partial_age, solution_3]))
np.save("SFH-WDLF-article/figure_data/fig_02_solution_4", np.column_stack([partial_age, solution_4]))
np.save("SFH-WDLF-article/figure_data/fig_02_solution_5", np.column_stack([partial_age, solution_5]))
np.save(
    "SFH-WDLF-article/figure_data/fig_02_solution_combined",
    np.column_stack([partial_age, solution_combined]),
)

np.save(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_1", np.column_stack([partial_age, solution_lsq_1])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_2", np.column_stack([partial_age, solution_lsq_2])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_3", np.column_stack([partial_age, solution_lsq_3])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_4", np.column_stack([partial_age, solution_lsq_4])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_5", np.column_stack([partial_age, solution_lsq_5])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_solution_lsq_combined",
    np.column_stack([partial_age, solution_lsq_combined]),
)

np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_1",
    np.column_stack([mag_bin, recomputed_wdlf_1]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_2",
    np.column_stack([mag_bin, recomputed_wdlf_2]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_3",
    np.column_stack([mag_bin, recomputed_wdlf_3]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_4",
    np.column_stack([mag_bin, recomputed_wdlf_4]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_5",
    np.column_stack([mag_bin, recomputed_wdlf_5]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_mcmc_combined",
    np.column_stack([mag_bin, recomputed_wdlf_combined]),
)

np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_1",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_1]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_2",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_2]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_3",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_3]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_4",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_4]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_5",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_5]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_02_recomputed_wdlf_lsq_combined",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_combined]),
)

# Figure here
fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(10, 3, 15)
)
ax_dummy1.axis("off")
#
ax1.plot(age, sfh_total_1 / np.nanmax(sfh_total_1), ls=":", color="C0")
ax1.plot(age, sfh_total_2 / np.nanmax(sfh_total_2), ls=":", color="C1")
ax1.plot(age, sfh_total_3 / np.nanmax(sfh_total_3), ls=":", color="C2")
ax1.plot(age, sfh_total_4 / np.nanmax(sfh_total_4), ls=":", color="C3")
ax1.plot(age, sfh_total_5 / np.nanmax(sfh_total_5), ls=":", color="C4")
ax1.plot(
    age, sfh_total_combined / np.nanmax(sfh_total_combined), ls=":", color="C1"
)
ax1.fill_between(
    partial_age,
    solution_lsq_1 - sfh_mcmc_lower_1,
    solution_lsq_1 + sfh_mcmc_upper_1,
    step="mid",
    color="C0",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_2 - sfh_mcmc_lower_2,
    solution_lsq_2 + sfh_mcmc_upper_2,
    step="mid",
    color="C1",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_3 - sfh_mcmc_lower_3,
    solution_lsq_3 + sfh_mcmc_upper_3,
    step="mid",
    color="C2",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_4 - sfh_mcmc_lower_4,
    solution_lsq_4 + sfh_mcmc_upper_4,
    step="mid",
    color="C3",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_5 - sfh_mcmc_lower_5,
    solution_lsq_5 + sfh_mcmc_upper_5,
    step="mid",
    color="C4",
    alpha=0.3,
)
ax1.fill_between(
    partial_age,
    solution_lsq_combined - sfh_mcmc_lower_combined,
    solution_lsq_combined + sfh_mcmc_upper_combined,
    step="mid",
    color="C5",
    alpha=0.3,
)
ax1.step(partial_age, sfh_mcmc_1, ls="dashed", where="mid", color="C0")
ax1.step(partial_age, sfh_mcmc_2, ls="dashed", where="mid", color="C1")
ax1.step(partial_age, sfh_mcmc_3, ls="dashed", where="mid", color="C2")
ax1.step(partial_age, sfh_mcmc_4, ls="dashed", where="mid", color="C3")
ax1.step(partial_age, sfh_mcmc_5, ls="dashed", where="mid", color="C4")
ax1.step(partial_age, sfh_mcmc_combined, ls="dashed", where="mid", color="C5")
ax1.step(partial_age, solution_lsq_1, where="mid", color="C0")
ax1.step(partial_age, solution_lsq_2, where="mid", color="C1")
ax1.step(partial_age, solution_lsq_3, where="mid", color="C2")
ax1.step(partial_age, solution_lsq_4, where="mid", color="C3")
ax1.step(partial_age, solution_lsq_5, where="mid", color="C4")
ax1.step(partial_age, solution_lsq_combined, where="mid", color="C5")
ax1.plot([-1, -1], [-1, -2], ls=":", color="black", label="Input")
ax1.plot([-1, -1], [-1, -2], ls="dashed", color="black", label="MCMC")
ax1.plot([-1, -1], [-1, -2], ls="-", color="black", label="lsq-refined")
ax1.grid()
ax1.set_xticks(np.arange(0, 15, 2))
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 1.4)
ax1.set_xlabel("Lookback time / Gyr")
ax1.set_ylabel("Relative SFR")
ax1.legend()
#
ax2.plot(mag_bin, n_wd_1, ls=":", color="C0")
ax2.plot(mag_bin, n_wd_2 * 0.2, ls=":", color="C1")
ax2.plot(mag_bin, n_wd_3 * 0.2**2, ls=":", color="C2")
ax2.plot(mag_bin, n_wd_4 * 0.2**3, ls=":", color="C3")
ax2.plot(mag_bin, n_wd_5 * 0.2**4, ls=":", color="C4")
ax2.plot(mag_bin, n_wd_combined * 0.2**5, ls=":", color="C5")
#
ax2.plot(
    mag_bin,
    recomputed_wdlf_1 / np.nansum(recomputed_wdlf_1) * np.nansum(n_wd_1),
    ls="dashed",
    color="C0",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2) * np.nansum(n_wd_2) * 0.2,
    ls="dashed",
    color="C1",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_3 / np.nansum(recomputed_wdlf_3) * np.nansum(n_wd_3) * 0.2**2,
    ls="dashed",
    color="C2",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_4 / np.nansum(recomputed_wdlf_4) * np.nansum(n_wd_4) * 0.2**3,
    ls="dashed",
    color="C3",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_5 / np.nansum(recomputed_wdlf_5) * np.nansum(n_wd_5) * 0.2**4,
    ls="dashed",
    color="C4",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_combined
    / np.nansum(recomputed_wdlf_combined)
    * np.nansum(n_wd_combined) * 0.2**5,
    ls="dashed",
    color="C5",
)
#
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_1
    / np.nansum(recomputed_wdlf_lsq_1)
    * np.nansum(n_wd_1),
    color="C0",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_2
    / np.nansum(recomputed_wdlf_lsq_2)
    * np.nansum(n_wd_2) * 0.2,
    color="C1",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_3
    / np.nansum(recomputed_wdlf_lsq_3)
    * np.nansum(n_wd_3) * 0.2**2,
    color="C2",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_4
    / np.nansum(recomputed_wdlf_lsq_4)
    * np.nansum(n_wd_4) * 0.2**3,
    color="C3",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_5
    / np.nansum(recomputed_wdlf_lsq_5)
    * np.nansum(n_wd_5) * 0.2**4,
    color="C4",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_combined
    / np.nansum(recomputed_wdlf_lsq_combined)
    * np.nansum(n_wd_combined) * 0.2**5,
    color="C5",
)
#
ax2.plot([1, 1], [1, 2], ls=":", color="black", label="Input")
ax2.plot(
    [1, 1], [1, 2], ls="dashed", color="black", label="Reconstructed (MCMC)"
)
ax2.plot([1, 1], [1, 2], ls="-", color="black", label="Reconstructed (lsq)")
ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
ax2.set_ylabel("log(arbitrary number density)")
ax2.set_xlim(6, 18)
ax2.set_ylim(1e-8, 2e-1)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
plt.subplots_adjust(
    top=0.975, bottom=0.075, left=0.125, right=0.975, hspace=0.075
)

# Get the Mbol to Age relation
atm = AtmosphereModelReader()
Mbol_to_age = atm.interp_am(dependent="age")
age_ticks = Mbol_to_age(8.0, ax2.get_xticks())
age_ticklabels = [f"{i/1e9:.3f}" for i in age_ticks]

# make the top axis
ax2b = ax2.twiny()
ax2b.set_xlim(ax2.get_xlim())
ax2b.set_xticks(ax2.get_xticks())
ax2b.xaxis.set_ticklabels(age_ticklabels, rotation=90)

plt.subplots_adjust(top=0.99, bottom=0.06, left=0.1, right=0.98, hspace=0.01)

plt.savefig(os.path.join(figure_folder, "fig_02_two_bursts_wdlf.png"))
