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

wdlf_1_number_density = spectres(mag_bin, data[0][:, 0], wdlf_1_reconstructed)
wdlf_2_number_density = spectres(mag_bin, data[0][:, 0], wdlf_2_reconstructed)
wdlf_3_number_density = spectres(mag_bin, data[0][:, 0], wdlf_3_reconstructed)
wdlf_4_number_density = spectres(mag_bin, data[0][:, 0], wdlf_4_reconstructed)
wdlf_5_number_density = spectres(mag_bin, data[0][:, 0], wdlf_5_reconstructed)

wdlf_1_number_density /= np.sum(wdlf_1_number_density)
wdlf_2_number_density /= np.sum(wdlf_2_number_density)
wdlf_3_number_density /= np.sum(wdlf_3_number_density)
wdlf_4_number_density /= np.sum(wdlf_4_number_density)
wdlf_5_number_density /= np.sum(wdlf_5_number_density)

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
init_guess_1 = np.array([0.03214147, 0.02171529, 0.0202689 , 0.02915921, 0.03080308,
       0.02256727, 0.03227703, 0.1056474 , 1.        , 0.70979569,
       0.7706346 , 0.60453861, 0.53279217, 0.50430006, 0.40478685,
       0.40049906, 0.36746668, 0.29120413, 0.29857828, 0.23469354,
       0.21470497, 0.18009904, 0.13762508, 0.10018739, 0.07074451,
       0.04590578, 0.03275272, 0.02719798, 0.01838592, 0.0201698 ,
       0.01551107, 0.01207459, 0.01106962, 0.00978645, 0.00841503,
       0.0081717 , 0.00664562, 0.00537495, 0.00480811, 0.00343975,
       0.00313964, 0.00295774, 0.00312936])
init_guess_2 = np.array([0.0238459 , 0.01171534, 0.01115995, 0.01212029, 0.01003765,
       0.01348464, 0.0103316 , 0.01124767, 0.01148089, 0.0108833 ,
       0.01425526, 0.01713299, 0.01713851, 0.02755686, 0.05739602,
       0.61695405, 1.        , 0.76946287, 0.95869761, 0.66446092,
       0.34231586, 0.77733585, 0.37678595, 0.23250603, 0.26375079,
       0.12904827, 0.10557626, 0.06760662, 0.0623153 , 0.05786561,
       0.04657554, 0.03549451, 0.03579686, 0.02707825, 0.02481947,
       0.02086221, 0.02144489, 0.01546898, 0.01390259, 0.00998112,
       0.00952388, 0.0076898 , 0.01011626])
init_guess_3 = np.array([0.01342993, 0.0077217 , 0.00804255, 0.00803677, 0.00660698,
       0.00756447, 0.00775368, 0.00595212, 0.00867202, 0.00954278,
       0.0095292 , 0.00634626, 0.00806494, 0.00796258, 0.00745069,
       0.00974759, 0.0118472 , 0.01915955, 0.03147927, 0.02500376,
       0.03491546, 0.08069153, 0.15118708, 1.        , 0.72735744,
       0.31121672, 0.30111836, 0.22514708, 0.14012856, 0.16708825,
       0.12976163, 0.09422377, 0.08699263, 0.09268703, 0.06860845,
       0.05820231, 0.05265711, 0.04337584, 0.0331974 , 0.03285987,
       0.02510272, 0.0235949 , 0.02385734])
init_guess_4 = np.array([0.01449089, 0.01078932, 0.00502821, 0.00485475, 0.00576641,
       0.00475509, 0.00538911, 0.00390691, 0.00257336, 0.00499954,
       0.00303439, 0.00341936, 0.00503381, 0.00356576, 0.00621965,
       0.00457599, 0.00563462, 0.00466003, 0.00669617, 0.00662319,
       0.00651447, 0.00931094, 0.01337996, 0.02424725, 0.05498467,
       0.33703238, 1.        , 0.27384914, 0.49135028, 0.38433786,
       0.23286242, 0.2424411 , 0.18531148, 0.21418857, 0.15196529,
       0.15575253, 0.11539804, 0.11349563, 0.08038686, 0.07042802,
       0.06344372, 0.04758427, 0.0563571 ])
init_guess_5 = np.array([0.0120994 , 0.00574493, 0.00478746, 0.00619167, 0.00595476,
       0.00426484, 0.00377914, 0.00261088, 0.00523437, 0.00396058,
       0.00358869, 0.00345676, 0.00354453, 0.00461832, 0.00387676,
       0.00408484, 0.00620757, 0.00518128, 0.00435009, 0.00611753,
       0.00472386, 0.00388765, 0.00755502, 0.00888859, 0.00855109,
       0.01657078, 0.02675188, 0.05737822, 0.07342006, 0.38607854,
       1.        , 0.89316167, 0.58629485, 0.47648982, 0.7577839 ,
       0.28773225, 0.4119619 , 0.39635847, 0.20537342, 0.2693909 ,
       0.22493477, 0.16111312, 0.10599248])


#init_guess_1 = np.random.random(ndim)
#init_guess_2 = np.random.random(ndim)
#init_guess_3 = np.random.random(ndim)
#init_guess_4 = np.random.random(ndim)
#init_guess_5 = np.random.random(ndim)

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

n_wd_err_1 = wdlf_1_number_density[wdlf_1_number_density > 1e-10] * 0.05
n_wd_err_2 = wdlf_2_number_density[wdlf_2_number_density > 1e-10] * 0.05
n_wd_err_3 = wdlf_3_number_density[wdlf_3_number_density > 1e-10] * 0.05
n_wd_err_4 = wdlf_4_number_density[wdlf_4_number_density > 1e-10] * 0.05
n_wd_err_5 = wdlf_5_number_density[wdlf_5_number_density > 1e-10] * 0.05

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
    #
    sampler_1.run_mcmc(rel_norm_1, n_step, progress=True)
    sampler_2.run_mcmc(rel_norm_2, n_step, progress=True)
    sampler_3.run_mcmc(rel_norm_3, n_step, progress=True)
    sampler_4.run_mcmc(rel_norm_4, n_step, progress=True)
    sampler_5.run_mcmc(rel_norm_5, n_step, progress=True)
    #
    flat_samples_1 = sampler_1.get_chain(discard=n_burn, flat=True)
    flat_samples_2 = sampler_2.get_chain(discard=n_burn, flat=True)
    flat_samples_3 = sampler_3.get_chain(discard=n_burn, flat=True)
    flat_samples_4 = sampler_4.get_chain(discard=n_burn, flat=True)
    flat_samples_5 = sampler_5.get_chain(discard=n_burn, flat=True)
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
        nrows=3, ncols=1, figsize=(8, 10), height_ratios=(10, 3, 15)
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
    ax1.set_xlim(0, 14)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Lookback t / Gyr")
    ax1.set_ylabel("Relative SFR")
    ax1.legend()
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
    ax2.set_xlabel(r"M${_\mathrm{bol}}$ / mag")
    ax2.set_ylabel("log(arbitrary number density)")
    ax2.set_xlim(6.00, 18.00)
    ax2.set_ylim(1e-5, 5e-1)
    ax2.set_yscale("log")
    ax2.plot([1, 1], [1, 2], ls=":", color="black", label="Input")
    ax2.plot(
        [1, 1], [1, 2], ls="-", color="black", label="Reconstructed (MCMC)"
    )
    ax2.legend()
    ax2.grid()
    #
    atm = AtmosphereModelReader()
    Mbol_to_age = atm.interp_am(dependent="age")
    age_ticks = Mbol_to_age(8.0, ax2.get_xticks())
    age_ticklabels = [f"{i/1e9:.3f}" for i in age_ticks]
    # make the top axis
    ax2b = ax2.twiny()
    ax2b.set_xlim(ax2.get_xlim())
    ax2b.set_xticks(ax2.get_xticks())
    ax2b.xaxis.set_ticklabels(age_ticklabels, rotation=90)
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

np.save("SFH-WDLF-article/figure_data/fig_01_partial_wdlf", partial_wdlf)

np.save("SFH-WDLF-article/figure_data/fig_01_sfh_1", sfh_1)
np.save("SFH-WDLF-article/figure_data/fig_01_sfh_2", sfh_2)
np.save("SFH-WDLF-article/figure_data/fig_01_sfh_3", sfh_3)
np.save("SFH-WDLF-article/figure_data/fig_01_sfh_4", sfh_4)
np.save("SFH-WDLF-article/figure_data/fig_01_sfh_5", sfh_5)

np.save("SFH-WDLF-article/figure_data/fig_01_input_wdlf_1", wdlf_1_number_density)
np.save("SFH-WDLF-article/figure_data/fig_01_input_wdlf_2", wdlf_2_number_density)
np.save("SFH-WDLF-article/figure_data/fig_01_input_wdlf_3", wdlf_3_number_density)
np.save("SFH-WDLF-article/figure_data/fig_01_input_wdlf_4", wdlf_4_number_density)
np.save("SFH-WDLF-article/figure_data/fig_01_input_wdlf_5", wdlf_5_number_density)

np.save("SFH-WDLF-article/figure_data/fig_01_solution_1", np.column_stack([partial_age, solution_1]))
np.save("SFH-WDLF-article/figure_data/fig_01_solution_2", np.column_stack([partial_age, solution_2]))
np.save("SFH-WDLF-article/figure_data/fig_01_solution_3", np.column_stack([partial_age, solution_3]))
np.save("SFH-WDLF-article/figure_data/fig_01_solution_4", np.column_stack([partial_age, solution_4]))
np.save("SFH-WDLF-article/figure_data/fig_01_solution_5", np.column_stack([partial_age, solution_5]))

np.save(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_1", np.column_stack([partial_age, solution_lsq_1])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_2", np.column_stack([partial_age, solution_lsq_2])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_3", np.column_stack([partial_age, solution_lsq_3])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_4", np.column_stack([partial_age, solution_lsq_4])
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_solution_lsq_5", np.column_stack([partial_age, solution_lsq_5])
)


np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_1",
    np.column_stack([mag_bin_1, recomputed_wdlf_1]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_2",
    np.column_stack([mag_bin_2, recomputed_wdlf_2]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_3",
    np.column_stack([mag_bin_3, recomputed_wdlf_3]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_4",
    np.column_stack([mag_bin_4, recomputed_wdlf_4]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_mcmc_5",
    np.column_stack([mag_bin_5, recomputed_wdlf_5]),
)

np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_1",
    np.column_stack([mag_bin_1, recomputed_wdlf_lsq_1]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_2",
    np.column_stack([mag_bin_2, recomputed_wdlf_lsq_2]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_3",
    np.column_stack([mag_bin_3, recomputed_wdlf_lsq_3]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_4",
    np.column_stack([mag_bin_4, recomputed_wdlf_lsq_4]),
)
np.save(
    "SFH-WDLF-article/figure_data/fig_01_recomputed_wdlf_lsq_5",
    np.column_stack([mag_bin_5, recomputed_wdlf_lsq_5]),
)

fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(10, 3, 15)
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
ax1.set_xlim(0, 14)
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
    wdlf_2_number_density / np.nansum(wdlf_2_number_density) * 0.2,
    ls=":",
    color="C1",
)
ax2.plot(
    mag_bin,
    wdlf_3_number_density / np.nansum(wdlf_3_number_density) * 0.2**2,
    ls=":",
    color="C2",
)
ax2.plot(
    mag_bin,
    wdlf_4_number_density / np.nansum(wdlf_4_number_density) * 0.2**3,
    ls=":",
    color="C3",
)
ax2.plot(
    mag_bin,
    wdlf_5_number_density / np.nansum(wdlf_5_number_density) * 0.2**4,
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
    recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2) * 0.2,
    ls="dashed",
    color="C1",
)
ax2.plot(
    mag_bin_3,
    recomputed_wdlf_3 / np.nansum(recomputed_wdlf_3) * 0.2**2,
    ls="dashed",
    color="C2",
)
ax2.plot(
    mag_bin_4,
    recomputed_wdlf_4 / np.nansum(recomputed_wdlf_4) * 0.2**3,
    ls="dashed",
    color="C3",
)
ax2.plot(
    mag_bin_5,
    recomputed_wdlf_5 / np.nansum(recomputed_wdlf_5) * 0.2**4,
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
    recomputed_wdlf_lsq_2 / np.nansum(recomputed_wdlf_lsq_2) * 0.2,
    color="C1",
)
ax2.plot(
    mag_bin_3,
    recomputed_wdlf_lsq_3 / np.nansum(recomputed_wdlf_lsq_3) * 0.2**2,
    color="C2",
)
ax2.plot(
    mag_bin_4,
    recomputed_wdlf_lsq_4 / np.nansum(recomputed_wdlf_lsq_4) * 0.2**3,
    color="C3",
)
ax2.plot(
    mag_bin_5,
    recomputed_wdlf_lsq_5 / np.nansum(recomputed_wdlf_lsq_5) * 0.2**4,
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
ax2.set_xlim(6.00, 18.00)
ax2.set_ylim(1e-8, 2e-1)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
#
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

plt.subplots_adjust(top=0.99, bottom=0.06, left=0.1, right=0.985, hspace=0.01)

plt.savefig(os.path.join(figure_folder, "fig_01_exponential_decay_wdlf.png"))
