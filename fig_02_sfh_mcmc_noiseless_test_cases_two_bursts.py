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
    age, mag_at_peak_density, s=50, k=5
)

# Load the "resolution" of the Mbol solution
mbol_err_upper_bound = interpolate.UnivariateSpline(
    *np.load("mbol_err_upper_bound.npy").T, s=0
)

mag_bin = []
mag_bin_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
mag_bin_pwdlf = np.zeros_like(age[1:]).astype("int")
j = 0
bin_number = 0
stop = False
mag_bin.append(mag_at_peak_density[0])
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
        if end + tmp - start < mbol_err_upper_bound(end) * 1.1775:
            end = end + tmp
            mag_bin_pwdlf[j] = bin_number
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    mag_bin.append(end)
    mag_bin_idx.append(i)
    if stop:
        break


# these the the bin edges
mag_bin = np.array(mag_bin)
# get the bin centre
mag_bin = (mag_bin[1:] + mag_bin[:-1]) / 2.0
pwdlf_mapping_mag_bin = np.insert(mag_bin_pwdlf, 0, 0)

wdlf_number_density_1 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_1)
wdlf_number_density_1 /= np.sum(wdlf_number_density_1)

wdlf_number_density_2 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_2)
wdlf_number_density_2 /= np.sum(wdlf_number_density_2)

wdlf_number_density_3 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_3)
wdlf_number_density_3 /= np.sum(wdlf_number_density_3)

wdlf_number_density_4 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_4)
wdlf_number_density_4 /= np.sum(wdlf_number_density_4)

wdlf_number_density_5 = spectres(mag_bin, data[0][:, 0], wdlf_reconstructed_5)
wdlf_number_density_5 /= np.sum(wdlf_number_density_5)

wdlf_number_density_combined = spectres(
    mag_bin, data[0][:, 0], wdlf_reconstructed_combined
)
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
init_guess_1 = np.array([1.12845923e-02, 1.29153575e-02, 1.59912453e-02, 1.57379811e-02,
       2.05786404e-02, 2.69153472e-02, 1.20046964e-02, 2.31524590e-02,
       3.78283598e-02, 4.49438092e-02, 6.64626862e-02, 1.98016892e-01,
       1.00000000e+00, 8.79149104e-01, 3.15664313e-02, 8.90989712e-03,
       5.41901406e-03, 4.55039948e-03, 6.73469490e-03, 7.44246015e-03,
       9.18210027e-03, 1.32513974e-02, 1.45722001e-02, 6.69788850e-03,
       2.14902587e-02, 1.26235728e-02, 1.64011888e-02, 3.58142096e-01,
       1.76271413e-01, 3.59582678e-01, 7.22618863e-02, 1.46385988e-01,
       4.84678799e-02, 2.68796190e-02, 7.92611407e-03, 5.85426472e-03,
       9.30549198e-04, 8.69583032e-04, 8.12248145e-04, 7.39990206e-04,
       6.55062973e-12, 6.40888715e-04, 1.10623583e-03, 6.05224026e-04,
       1.18015459e-03, 9.82047017e-04, 1.37520854e-03, 1.18101773e-03,
       1.08865089e-03, 2.03825462e-03, 4.81935507e-03])
init_guess_2 = np.array([1.49777812e-02, 6.20820589e-03, 7.99579636e-03, 9.18817083e-03,
       6.30974476e-03, 9.27751635e-03, 9.93582715e-03, 1.00901450e-02,
       8.59358169e-03, 1.45596020e-02, 1.00309740e-02, 4.77342396e-03,
       5.73996006e-03, 5.33782817e-03, 7.50631662e-03, 8.37943236e-03,
       9.22082341e-03, 3.24986605e-02, 1.09235947e-02, 3.04070710e-02,
       5.18496348e-01, 1.00000000e+00, 1.18715753e-01, 4.83373481e-01,
       7.31584009e-02, 9.94442610e-03, 1.32485174e-02, 2.41646405e-01,
       6.05853623e-02, 2.77341427e-01, 5.37468212e-02, 5.48680308e-02,
       4.72328394e-02, 1.51811550e-02, 4.88373601e-03, 1.62497565e-03,
       8.04589845e-04, 1.13250225e-03, 3.77308865e-04, 3.47215758e-04,
       9.47720172e-12, 5.55982583e-04, 8.59652348e-04, 6.11548946e-04,
       4.07087546e-04, 4.89429362e-04, 1.78899594e-03, 8.48975713e-04,
       7.46260544e-04, 1.09236983e-03, 5.96016877e-03])
init_guess_3 = np.array([1.60482963e-02, 9.97988939e-03, 1.24547275e-02, 1.18743959e-02,
       5.10036203e-03, 1.70930372e-02, 1.55803105e-02, 1.34299869e-02,
       1.85955228e-02, 1.83399228e-02, 1.48057613e-02, 1.18506962e-02,
       8.98979647e-03, 1.19014468e-02, 1.14411522e-02, 7.72389915e-03,
       1.35113092e-02, 1.54749022e-02, 1.14951756e-02, 1.95697192e-02,
       2.61407460e-02, 3.53818066e-02, 2.88526244e-02, 4.46691267e-02,
       7.06464533e-02, 6.68795006e-02, 2.23191354e-02, 1.00000000e+00,
       8.85896859e-02, 3.55695296e-01, 1.98158472e-01, 1.37472929e-01,
       6.81404219e-02, 2.83731495e-02, 6.16472002e-03, 2.60434220e-03,
       2.40304666e-03, 1.95247821e-03, 7.47916416e-04, 4.87876169e-04,
       3.95014501e-11, 2.54196058e-03, 1.60382641e-03, 1.34576396e-03,
       2.95009964e-03, 2.91657388e-03, 3.37390746e-03, 3.25713722e-03,
       2.51090096e-03, 8.73829189e-03, 1.55909970e-02])
init_guess_4 = np.array([1.73438350e-02, 9.53419444e-03, 9.06690901e-03, 1.04422777e-02,
       1.22994346e-02, 8.25214291e-03, 1.22469193e-02, 1.35413837e-02,
       1.05838689e-02, 1.09881708e-02, 1.01865386e-02, 6.40813861e-03,
       4.43705607e-03, 5.78336626e-03, 6.55508522e-03, 3.01342462e-03,
       5.71363798e-03, 7.79847288e-03, 6.39488921e-03, 1.16793544e-02,
       1.61133505e-02, 1.58977367e-02, 1.40183052e-02, 2.21539245e-02,
       1.58729590e-02, 1.58628666e-02, 1.05780274e-02, 5.74607132e-01,
       1.00000000e+00, 2.18361989e-01, 1.44826165e-01, 1.30999803e-01,
       7.88924355e-02, 2.65587882e-02, 1.25918582e-02, 9.75323908e-03,
       5.91643439e-04, 1.66307603e-03, 1.18375481e-03, 3.47118092e-04,
       1.06263051e-11, 2.38567753e-03, 8.63136797e-04, 1.24287665e-03,
       1.12452658e-03, 2.50545495e-03, 2.71597568e-03, 1.42909269e-03,
       1.88556339e-03, 1.69849036e-03, 2.55515703e-03])
init_guess_5 = np.array([1.59427164e-02, 6.68376764e-03, 7.87264689e-03, 1.05240964e-02,
       8.13987692e-03, 1.16520484e-02, 1.05636818e-02, 1.23556520e-02,
       7.48722484e-03, 1.20922914e-02, 8.72377897e-03, 6.90022988e-03,
       2.82022019e-03, 4.69121407e-03, 7.29393077e-03, 5.51320174e-03,
       5.63455602e-03, 6.40058758e-03, 7.99182442e-03, 1.24387785e-02,
       1.45669959e-02, 1.51786121e-02, 1.94627046e-02, 1.61462606e-02,
       2.77730669e-02, 1.54153577e-02, 1.54694686e-02, 3.63555995e-01,
       2.12515853e-01, 8.48873107e-01, 3.73812296e-01, 1.00000000e+00,
       4.00610084e-02, 2.63555583e-02, 1.30278768e-02, 1.49817653e-03,
       8.80363382e-04, 1.46242233e-03, 2.97297911e-04, 4.44151452e-04,
       1.70535371e-11, 1.16853445e-03, 9.88381332e-04, 2.14928216e-03,
       2.40995799e-03, 1.09528583e-03, 2.25033361e-03, 1.54936094e-03,
       1.54914373e-03, 4.20247190e-03, 8.02028562e-03])
init_guess_combined = np.array([3.90975690e-02, 2.17415517e-02, 2.58197425e-02, 2.39704594e-02,
       3.02068670e-02, 3.98300243e-02, 3.60795895e-02, 3.23498472e-02,
       2.82160264e-02, 6.19184393e-02, 6.42936272e-02, 1.01222638e-01,
       7.85716203e-01, 6.27066615e-01, 3.76176786e-02, 2.05749002e-02,
       2.18049621e-02, 4.33989921e-02, 5.51889569e-02, 1.51338654e-01,
       4.79968517e-01, 1.00000000e+00, 6.95607954e-01, 2.46463762e-01,
       9.01115447e-02, 4.08270567e-02, 9.06909899e-02, 7.40770267e-01,
       2.28311767e-01, 2.16659440e-01, 8.17381604e-01, 4.58390962e-01,
       3.78781320e-02, 1.72474853e-02, 6.21385041e-03, 3.18756762e-03,
       2.12360964e-03, 1.68792607e-03, 2.27805755e-03, 1.65880283e-03,
       1.26835345e-10, 2.87919410e-03, 2.05078140e-03, 2.40824863e-03,
       3.29622598e-03, 3.46760700e-03, 4.68927087e-03, 3.54997854e-03,
       3.46588817e-03, 4.10638837e-03, 1.17168811e-02])

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
n_wd_combined = wdlf_number_density_combined[wdlf_number_density_combined > 0.0]

n_wd_err_1 = n_wd_1 * 0.1
n_wd_err_2 = n_wd_2 * 0.1
n_wd_err_3 = n_wd_3 * 0.1
n_wd_err_4 = n_wd_4 * 0.1
n_wd_err_5 = n_wd_5 * 0.1
n_wd_err_combined = n_wd_combined * 0.1

n_step = 10000
n_burn = 1000

for i in range(100):
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
    flat_samples_combined = sampler_combined.get_chain(discard=n_burn, flat=True)
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


np.save("fig_02_partial_wdlf", partial_wdlf)

np.save("fig_02_sfh_total_1", sfh_total_1)
np.save("fig_02_sfh_total_2", sfh_total_2)
np.save("fig_02_sfh_total_3", sfh_total_3)
np.save("fig_02_sfh_total_4", sfh_total_4)
np.save("fig_02_sfh_total_5", sfh_total_5)
np.save("fig_02_sfh_total_combined", sfh_total_combined)

np.save("fig_02_input_wdlf_1", wdlf_reconstructed_1)
np.save("fig_02_input_wdlf_2", wdlf_reconstructed_2)
np.save("fig_02_input_wdlf_3", wdlf_reconstructed_3)
np.save("fig_02_input_wdlf_4", wdlf_reconstructed_4)
np.save("fig_02_input_wdlf_5", wdlf_reconstructed_5)
np.save("fig_02_input_wdlf_combined", wdlf_reconstructed_combined)

np.save("fig_02_solution_1", np.column_stack([partial_age, solution_1]))
np.save("fig_02_solution_2", np.column_stack([partial_age, solution_2]))
np.save("fig_02_solution_3", np.column_stack([partial_age, solution_3]))
np.save("fig_02_solution_4", np.column_stack([partial_age, solution_4]))
np.save("fig_02_solution_5", np.column_stack([partial_age, solution_5]))
np.save(
    "fig_02_solution_combined",
    np.column_stack([partial_age, solution_combined]),
)

np.save(
    "fig_02_solution_lsq_1", np.column_stack([partial_age, solution_lsq_1])
)
np.save(
    "fig_02_solution_lsq_2", np.column_stack([partial_age, solution_lsq_2])
)
np.save(
    "fig_02_solution_lsq_3", np.column_stack([partial_age, solution_lsq_3])
)
np.save(
    "fig_02_solution_lsq_4", np.column_stack([partial_age, solution_lsq_4])
)
np.save(
    "fig_02_solution_lsq_5", np.column_stack([partial_age, solution_lsq_5])
)
np.save(
    "fig_02_solution_lsq_combined",
    np.column_stack([partial_age, solution_lsq_combined]),
)

np.save(
    "fig_02_recomputed_wdlf_mcmc_1",
    np.column_stack([mag_bin, recomputed_wdlf_1]),
)
np.save(
    "fig_02_recomputed_wdlf_mcmc_2",
    np.column_stack([mag_bin, recomputed_wdlf_2]),
)
np.save(
    "fig_02_recomputed_wdlf_mcmc_3",
    np.column_stack([mag_bin, recomputed_wdlf_3]),
)
np.save(
    "fig_02_recomputed_wdlf_mcmc_4",
    np.column_stack([mag_bin, recomputed_wdlf_4]),
)
np.save(
    "fig_02_recomputed_wdlf_mcmc_5",
    np.column_stack([mag_bin, recomputed_wdlf_5]),
)
np.save(
    "fig_02_recomputed_wdlf_mcmc_combined",
    np.column_stack([mag_bin, recomputed_wdlf_combined]),
)

np.save(
    "fig_02_recomputed_wdlf_lsq_1",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_1]),
)
np.save(
    "fig_02_recomputed_wdlf_lsq_2",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_2]),
)
np.save(
    "fig_02_recomputed_wdlf_lsq_3",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_3]),
)
np.save(
    "fig_02_recomputed_wdlf_lsq_4",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_4]),
)
np.save(
    "fig_02_recomputed_wdlf_lsq_5",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_5]),
)
np.save(
    "fig_02_recomputed_wdlf_lsq_combined",
    np.column_stack([mag_bin, recomputed_wdlf_lsq_combined]),
)

fig1, (ax1, ax_dummy1, ax2) = plt.subplots(
    nrows=3, ncols=1, figsize=(8, 10), height_ratios=(10, 3, 15)
)
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
    ls="dashed",
    color="C0",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_2 / np.nansum(recomputed_wdlf_2) * np.nansum(n_wd_2),
    ls="dashed",
    color="C1",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_3 / np.nansum(recomputed_wdlf_3) * np.nansum(n_wd_3),
    ls="dashed",
    color="C2",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_4 / np.nansum(recomputed_wdlf_4) * np.nansum(n_wd_4),
    ls="dashed",
    color="C3",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_5 / np.nansum(recomputed_wdlf_5) * np.nansum(n_wd_5),
    ls="dashed",
    color="C4",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_combined
    / np.nansum(recomputed_wdlf_combined)
    * np.nansum(n_wd_combined),
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
    * np.nansum(n_wd_2),
    color="C1",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_3
    / np.nansum(recomputed_wdlf_lsq_3)
    * np.nansum(n_wd_3),
    color="C2",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_4
    / np.nansum(recomputed_wdlf_lsq_4)
    * np.nansum(n_wd_4),
    color="C3",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_5
    / np.nansum(recomputed_wdlf_lsq_5)
    * np.nansum(n_wd_5),
    color="C4",
)
ax2.plot(
    mag_bin,
    recomputed_wdlf_lsq_combined
    / np.nansum(recomputed_wdlf_lsq_combined)
    * np.nansum(n_wd_combined),
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
ax2.set_xlim(6.75, 18.25)
ax2.set_ylim(1e-4, 5e-1)
ax2.set_yscale("log")
ax2.legend()
ax2.grid()
ax1.plot(age, sfh_total_1 / np.nanmax(sfh_total_1), ls=":", color="C0")
ax1.plot(age, sfh_total_2 / np.nanmax(sfh_total_2), ls=":", color="C1")
ax1.plot(age, sfh_total_3 / np.nanmax(sfh_total_3), ls=":", color="C1")
ax1.plot(age, sfh_total_4 / np.nanmax(sfh_total_4), ls=":", color="C1")
ax1.plot(age, sfh_total_5 / np.nanmax(sfh_total_5), ls=":", color="C1")
ax1.plot(
    age, sfh_total_combined / np.nanmax(sfh_total_combined), ls=":", color="C1"
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_1 - sfh_mcmc_1 + solution_lsq_1,
    sfh_mcmc_upper_1 - sfh_mcmc_1 + solution_lsq_1,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_2 - sfh_mcmc_2 + solution_lsq_2,
    sfh_mcmc_upper_2 - sfh_mcmc_2 + solution_lsq_2,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_3 - sfh_mcmc_3 + solution_lsq_3,
    sfh_mcmc_upper_3 - sfh_mcmc_3 + solution_lsq_3,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_4 - sfh_mcmc_4 + solution_lsq_4,
    sfh_mcmc_upper_4 - sfh_mcmc_4 + solution_lsq_4,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_5 - sfh_mcmc_5 + solution_lsq_5,
    sfh_mcmc_upper_5 - sfh_mcmc_5 + solution_lsq_5,
    step="mid",
    color="darkgrey",
    alpha=0.5,
)
ax1.fill_between(
    partial_age,
    sfh_mcmc_lower_combined - sfh_mcmc_combined + solution_lsq_combined,
    sfh_mcmc_upper_combined - sfh_mcmc_combined + solution_lsq_combined,
    step="mid",
    color="darkgrey",
    alpha=0.5,
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
ax1.set_xlim(0, 15)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Lookback time / Gyr")
ax1.set_ylabel("Relative SFR")
ax1.legend()
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

plt.subplots_adjust(top=0.995, bottom=0.06, left=0.1, right=0.995, hspace=0.01)

plt.savefig(os.path.join(figure_folder, "fig_02_two_bursts_wdlf.png"))
