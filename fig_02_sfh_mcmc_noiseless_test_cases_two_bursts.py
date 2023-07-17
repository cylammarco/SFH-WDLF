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
# mbol_err_upper_bound = interpolate.UnivariateSpline(
#    *np.load("mbol_err_upper_bound.npy").T, s=0
# )
mbol_err_upper_bound = interpolate.UnivariateSpline(
    *np.load("mbol_err_median.npy").T, s=0
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
init_guess_1 = np.array(
    [
        2.47732719e-02,
        6.65942566e-15,
        2.61619032e-02,
        2.12316253e-02,
        3.04348956e-02,
        3.07907078e-02,
        3.67869700e-02,
        2.51799770e-02,
        2.88735693e-02,
        3.08047079e-02,
        3.10077052e-02,
        3.41437170e-02,
        7.96399479e-01,
        1.00000000e00,
        7.09654530e-01,
        8.95413902e-01,
        4.73860968e-01,
        6.78211704e-01,
        6.28562444e-01,
        2.97640557e-01,
        6.09749748e-01,
        1.94725255e-01,
        7.39335921e-01,
        2.52912237e-01,
        3.70740802e-01,
        9.95214133e-02,
        6.17370833e-01,
        3.31579622e-01,
        5.34930715e-02,
        1.05225354e-01,
        3.48250019e-01,
        1.29103496e-01,
        1.58790554e-02,
        5.88602539e-01,
        3.35803427e-01,
        1.43076912e-01,
        2.46687123e-01,
        1.67728193e-01,
        1.71451690e-01,
        6.96921375e-02,
        8.17298066e-02,
        2.57449126e-02,
        1.14589822e-02,
        6.18849467e-02,
        1.93545537e-02,
        3.14460180e-02,
        1.31248348e-02,
        1.65921717e-02,
        1.05223724e-02,
        6.19842043e-03,
        2.45827025e-02,
        1.52980218e-02,
        1.26419659e-02,
        8.11312554e-03,
        1.13570830e-02,
        6.70369529e-03,
        7.78657451e-03,
        5.41755182e-03,
        3.69713038e-03,
        5.04459496e-03,
        4.20614326e-03,
        1.08191941e-03,
        1.99387185e-03,
        1.26657623e-03,
        2.01214632e-03,
        2.94021929e-03,
        1.58431832e-03,
        2.89119061e-03,
        1.95536072e-03,
        4.78117270e-03,
        1.40994569e-03,
        3.35656894e-03,
        3.03150915e-03,
        1.96622145e-03,
        2.11262093e-03,
        2.17319843e-03,
    ]
)
init_guess_2 = np.array(
    [
        1.27924961e-02,
        4.84152685e-14,
        4.65368141e-03,
        5.04498194e-03,
        2.43356763e-03,
        1.08550522e-02,
        1.14676439e-02,
        1.12482328e-02,
        1.04918564e-02,
        1.38225758e-02,
        1.44347317e-02,
        5.49623470e-03,
        1.10571310e-02,
        1.10950596e-02,
        1.33206261e-02,
        2.98764163e-02,
        3.64906235e-03,
        7.84417729e-03,
        7.85937293e-03,
        1.74481215e-02,
        6.09018629e-02,
        4.17797017e-02,
        5.70562101e-02,
        9.08176933e-02,
        3.88900045e-01,
        6.45387940e-02,
        1.27159183e-01,
        7.15183062e-01,
        4.74409021e-02,
        3.80705585e-02,
        1.00000000e00,
        6.90637868e-02,
        5.32335073e-02,
        6.69120076e-03,
        3.84046488e-01,
        8.77143228e-02,
        1.75531482e-01,
        3.43574461e-01,
        2.70041668e-02,
        1.58912099e-01,
        1.15947811e-02,
        5.29746378e-02,
        5.16850952e-02,
        5.11559550e-03,
        3.67048873e-03,
        2.95326722e-02,
        4.45332997e-02,
        1.32163338e-03,
        9.83179124e-03,
        2.05093374e-02,
        2.35113726e-02,
        1.25518131e-02,
        8.62841079e-03,
        1.13515358e-02,
        1.12378984e-02,
        6.83306505e-03,
        6.31331975e-03,
        6.21065041e-03,
        3.17172796e-03,
        5.45949134e-03,
        3.76856407e-03,
        1.93459733e-03,
        1.91947587e-03,
        2.16971911e-03,
        1.56524756e-03,
        3.40428610e-03,
        2.75783733e-03,
        1.95302182e-03,
        1.80407537e-03,
        2.89980396e-03,
        2.15500776e-03,
        2.93061785e-03,
        3.93728453e-03,
        1.16714093e-03,
        2.64936975e-03,
        4.57399490e-03,
    ]
)
init_guess_3 = np.array(
    [
        5.35898658e-03,
        4.74960445e-03,
        6.22451041e-03,
        3.14398317e-15,
        4.15671864e-03,
        1.18955915e-02,
        5.28572092e-03,
        1.62483414e-03,
        5.99664766e-03,
        7.02302098e-03,
        8.65048310e-03,
        4.78544686e-03,
        5.44381392e-03,
        9.35220641e-03,
        4.39148037e-03,
        5.66938606e-03,
        5.92695593e-03,
        4.71778800e-03,
        8.90156254e-03,
        6.88992919e-03,
        5.64639090e-03,
        1.10486157e-02,
        8.51638583e-03,
        5.81221290e-03,
        9.32051169e-03,
        1.27128637e-02,
        1.33994385e-02,
        1.35911327e-02,
        2.20954251e-02,
        1.69583881e-02,
        1.58377244e-02,
        3.60967344e-02,
        1.51330057e-02,
        2.65871706e-02,
        4.50696534e-02,
        2.07949054e-02,
        5.71181419e-02,
        1.86495975e-02,
        1.00762258e-01,
        1.00000000e00,
        1.90420002e-02,
        4.35288534e-01,
        4.19247370e-02,
        6.50462367e-02,
        2.49352842e-01,
        1.38975846e-01,
        1.13645251e-01,
        8.20554855e-02,
        7.36727056e-02,
        4.23362701e-02,
        1.79556829e-01,
        8.01409817e-02,
        2.41191895e-02,
        1.12328916e-01,
        2.40644347e-02,
        7.04521766e-02,
        1.77995750e-02,
        4.28404563e-02,
        1.87340144e-02,
        2.96066830e-02,
        2.87689580e-02,
        4.49487320e-03,
        1.64636163e-02,
        7.53028049e-03,
        1.18752246e-02,
        2.33153636e-02,
        1.86528965e-02,
        8.96585914e-03,
        1.60640869e-02,
        1.92179272e-02,
        2.06038075e-02,
        1.09283844e-02,
        6.05200795e-03,
        1.43066804e-02,
        1.07267877e-02,
        1.71637769e-02,
    ]
)
init_guess_4 = np.array(
    [
        1.12741883e-02,
        3.12248238e-03,
        3.73047048e-04,
        3.05344103e-03,
        7.04529359e-03,
        5.70941296e-03,
        5.08657363e-03,
        8.19692859e-03,
        4.34717604e-03,
        5.79218654e-03,
        2.56185406e-03,
        6.40142756e-04,
        1.41235257e-03,
        3.45215616e-16,
        3.87004637e-03,
        2.71149175e-03,
        3.40509863e-03,
        4.02479764e-03,
        1.96558581e-03,
        2.26824325e-03,
        4.10954389e-03,
        7.75989082e-03,
        5.52896718e-03,
        4.97956147e-03,
        5.87043918e-03,
        9.11655712e-03,
        5.51258234e-03,
        3.89984287e-03,
        1.19010336e-02,
        9.78127742e-03,
        5.59752505e-03,
        1.10694145e-02,
        5.85716731e-03,
        9.13036545e-03,
        7.14161039e-03,
        8.37051776e-03,
        7.68543526e-03,
        6.93404717e-03,
        5.66354134e-03,
        1.65137696e-02,
        6.62172467e-04,
        1.00000000e00,
        1.75997401e-01,
        8.23188535e-02,
        7.00379365e-01,
        2.01033544e-01,
        4.47852764e-02,
        2.41111688e-01,
        3.12965414e-01,
        1.07226877e-01,
        1.95246214e-01,
        7.33647410e-03,
        3.31049455e-01,
        2.98431285e-02,
        5.83205259e-02,
        2.19436791e-01,
        4.00183132e-02,
        1.31254794e-02,
        1.10762162e-01,
        3.26834169e-02,
        1.81857895e-02,
        4.48161936e-02,
        4.27171891e-02,
        3.49960187e-02,
        4.70297633e-03,
        3.03050675e-02,
        3.06448907e-02,
        2.37869685e-02,
        2.39485156e-02,
        1.21063113e-02,
        2.06999687e-02,
        2.70128987e-03,
        2.05742434e-02,
        1.09624385e-02,
        3.02735143e-02,
        1.96156703e-02,
    ]
)
init_guess_5 = np.array(
    [
        1.76231610e-15,
        1.31049938e-16,
        6.23966300e-17,
        2.23004178e-16,
        6.27672382e-16,
        1.81378424e-16,
        2.48944840e-16,
        1.33001203e-16,
        1.98704534e-16,
        3.96030812e-16,
        4.90838101e-16,
        9.33794886e-16,
        3.23967622e-16,
        1.15443689e-15,
        1.11842218e-17,
        1.10451680e-15,
        4.41489145e-16,
        9.02569876e-17,
        7.36364185e-16,
        4.18859576e-16,
        2.50836441e-16,
        8.80590925e-17,
        3.68142487e-16,
        8.52626062e-17,
        1.24344003e-16,
        4.51102903e-17,
        6.60312075e-18,
        2.57137304e-17,
        6.48961751e-16,
        7.56847294e-16,
        1.73400122e-16,
        9.78336225e-03,
        8.92693923e-03,
        1.09238400e-02,
        7.94279096e-03,
        9.11333075e-03,
        7.86018387e-03,
        7.70280021e-03,
        4.19029198e-03,
        3.57897287e-03,
        9.82255199e-03,
        1.38966514e-02,
        2.56012997e-02,
        2.70311245e-02,
        4.07860275e-02,
        1.16138322e-01,
        2.72166473e-01,
        1.03256472e-01,
        6.61102304e-01,
        8.84658412e-02,
        1.00000000e00,
        4.30842543e-01,
        5.52895502e-01,
        1.35890041e-01,
        4.82306766e-01,
        2.07639130e-01,
        2.21244237e-01,
        2.11862781e-01,
        1.23562397e-01,
        1.93221152e-01,
        7.09511808e-02,
        9.18699686e-02,
        1.40449250e-01,
        6.30446716e-02,
        1.66190563e-02,
        1.09323597e-01,
        7.63490039e-02,
        1.61190125e-01,
        3.61345200e-02,
        6.39203091e-02,
        8.71409398e-03,
        2.52752764e-02,
        7.41472720e-02,
        2.52930789e-02,
        1.61621177e-01,
        9.59775798e-03,
    ]
)
init_guess_combined = np.array(
    [
        5.35898658e-03,
        4.74960445e-03,
        6.22451041e-03,
        3.14398317e-15,
        4.15671864e-03,
        1.18955915e-02,
        5.28572092e-03,
        1.62483414e-03,
        5.99664766e-03,
        7.02302098e-03,
        8.65048310e-03,
        4.78544686e-03,
        5.44381392e-03,
        9.35220641e-03,
        4.39148037e-03,
        5.66938606e-03,
        5.92695593e-03,
        4.71778800e-03,
        8.90156254e-03,
        6.88992919e-03,
        5.64639090e-03,
        1.10486157e-02,
        8.51638583e-03,
        5.81221290e-03,
        9.32051169e-03,
        1.27128637e-02,
        1.33994385e-02,
        1.35911327e-02,
        2.20954251e-02,
        1.69583881e-02,
        1.58377244e-02,
        3.60967344e-02,
        1.51330057e-02,
        2.65871706e-02,
        4.50696534e-02,
        2.07949054e-02,
        5.71181419e-02,
        1.86495975e-02,
        1.00762258e-01,
        1.00000000e00,
        1.90420002e-02,
        4.35288534e-01,
        4.19247370e-02,
        6.50462367e-02,
        2.49352842e-01,
        1.38975846e-01,
        1.13645251e-01,
        8.20554855e-02,
        7.36727056e-02,
        4.23362701e-02,
        1.79556829e-01,
        8.01409817e-02,
        2.41191895e-02,
        1.12328916e-01,
        2.40644347e-02,
        7.04521766e-02,
        1.77995750e-02,
        4.28404563e-02,
        1.87340144e-02,
        2.96066830e-02,
        2.87689580e-02,
        4.49487320e-03,
        1.64636163e-02,
        7.53028049e-03,
        1.18752246e-02,
        2.33153636e-02,
        1.86528965e-02,
        8.96585914e-03,
        1.60640869e-02,
        1.92179272e-02,
        2.06038075e-02,
        1.09283844e-02,
        6.05200795e-03,
        1.43066804e-02,
        1.07267877e-02,
        1.71637769e-02,
    ]
)

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
