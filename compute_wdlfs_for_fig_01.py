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


t = np.arange(0.001, 15.0, 0.001) * 1e9
sfh_1 = exponential_profile(t, 2.0e9, 2.0e9)
sfh_2 = exponential_profile(t, 2.0e9, 4.0e9)
sfh_3 = exponential_profile(t, 2.0e9, 6.0e9)
sfh_4 = exponential_profile(t, 2.0e9, 8.0e9)
sfh_5 = exponential_profile(t, 2.0e9, 1.0e10)


sfh_spline_1 = interpolate.UnivariateSpline(t, sfh_1, s=0, ext="zeros")
sfh_spline_2 = interpolate.UnivariateSpline(t, sfh_2, s=0, ext="zeros")
sfh_spline_3 = interpolate.UnivariateSpline(t, sfh_3, s=0, ext="zeros")
sfh_spline_4 = interpolate.UnivariateSpline(t, sfh_4, s=0, ext="zeros")
sfh_spline_5 = interpolate.UnivariateSpline(t, sfh_5, s=0, ext="zeros")


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


wdlf_1 = theoretical_lf.WDLF()
wdlf_1.set_sfr_model(mode="manual", sfr_model=sfh_spline_1)
wdlf_1.set_imf_model(model="K01")
wdlf_1.set_ms_model(model="PARSECz0014")
wdlf_1.compute_density(
    mag_bin,
    passband="Mbol",
    interpolator="CT",
    epsabs=1e-8,
    epsrel=1e-8
)
np.save(
    "fig_01_input_wdlf_1.npy",
    np.column_stack([wdlf_1.mag, wdlf_1.number_density]),
)
print("wdlf_1 computed")

wdlf_2 = theoretical_lf.WDLF()
wdlf_2.set_sfr_model(mode="manual", sfr_model=sfh_spline_2)
wdlf_2.set_imf_model(model="K01")
wdlf_2.set_ms_model(model="PARSECz0014")
wdlf_2.compute_density(
    mag_bin,
    passband="Mbol",
    interpolator="CT",
    epsabs=1e-10,
    epsrel=1e-10,
    limit=100000,
    n_points=200,
)
np.save(
    "fig_01_input_wdlf_2.npy",
    np.column_stack([wdlf_2.mag, wdlf_2.number_density]),
)
print("wdlf_2 computed")

wdlf_3 = theoretical_lf.WDLF()
wdlf_3.set_sfr_model(mode="manual", sfr_model=sfh_spline_3)
wdlf_3.set_imf_model(model="K01")
wdlf_3.set_ms_model(model="PARSECz0014")
wdlf_3.compute_density(
    mag_bin,
    passband="Mbol",
    interpolator="CT",
    epsabs=1e-10,
    epsrel=1e-10,
    limit=100000,
    n_points=200,
)
np.save(
    "fig_01_input_wdlf_3.npy",
    np.column_stack([wdlf_3.mag, wdlf_3.number_density]),
)
print("wdlf_3 computed")

wdlf_4 = theoretical_lf.WDLF()
wdlf_4.set_sfr_model(mode="manual", sfr_model=sfh_spline_4)
wdlf_4.set_imf_model(model="K01")
wdlf_4.set_ms_model(model="PARSECz0014")
wdlf_4.compute_density(
    mag_bin,
    passband="Mbol",
    interpolator="CT",
    epsabs=1e-10,
    epsrel=1e-10,
    limit=100000,
    n_points=200,
)
np.save(
    "fig_01_input_wdlf_4.npy",
    np.column_stack([wdlf_4.mag, wdlf_4.number_density]),
)
print("wdlf_4 computed")

wdlf_5 = theoretical_lf.WDLF()
wdlf_5.set_sfr_model(mode="manual", sfr_model=sfh_spline_5)
wdlf_5.set_imf_model(model="K01")
wdlf_5.set_ms_model(model="PARSECz0014")
wdlf_5.compute_density(
    mag_bin,
    passband="Mbol",
    interpolator="CT",
    epsabs=1e-12,
    epsrel=1e-12,
    limit=10000,
    n_points=500,
)
np.save(
    "fig_01_input_wdlf_5.npy",
    np.column_stack([wdlf_5.mag, wdlf_5.number_density]),
)
print("wdlf_5 computed")



sfh_6 = exponential_profile(t, 2.0e9, 8.5e9)
sfh_spline_6 = interpolate.UnivariateSpline(t, sfh_6, s=0, ext="zeros")

wdlf_6 = theoretical_lf.WDLF()
wdlf_6.set_sfr_model(mode="manual", sfr_model=sfh_spline_6)
wdlf_6.set_imf_model(model="K01")
wdlf_6.set_ms_model(model="PARSECz0014")
wdlf_6.compute_density(
    mag_bin,
    passband="Mbol",
    interpolator="CT",
    epsabs=1e-10,
    epsrel=1e-10,
    limit=1000,
    n_points=200,
)



sfh_7 = exponential_profile(t, 2.0e9, 9.0e9)
sfh_spline_7 = interpolate.UnivariateSpline(t, sfh_7, s=0, ext="zeros")

wdlf_7 = theoretical_lf.WDLF()
wdlf_7.set_sfr_model(mode="manual", sfr_model=sfh_spline_7)
wdlf_7.set_imf_model(model="K01")
wdlf_7.set_ms_model(model="PARSECz0014")
wdlf_7.compute_density(
    mag_bin,
    passband="Mbol",
    interpolator="CT",
    epsabs=1e-10,
    epsrel=1e-10,
    limit=100,
    n_points=20,
)



sfh_8 = exponential_profile(t, 2.0e9, 1.1e10)
sfh_spline_8 = interpolate.UnivariateSpline(t, sfh_8, s=0, ext="zeros")

wdlf_8 = theoretical_lf.WDLF()
wdlf_8.set_sfr_model(mode="burst", age=1.1e10, duration=0.1e9)
wdlf_8.set_imf_model(model="K01")
wdlf_8.set_ms_model(model="PARSECz0014")
wdlf_8.compute_density(
    mag_bin,
    passband="Mbol",
    interpolator="CT",
    epsabs=1e-12,
    epsrel=1e-12,
    limit=1000000,
    n_points=50,
)

from matplotlib.pyplot import *
ion()

clf()
plot(wdlf_5.mag, wdlf_5.number_density)
plot(wdlf_6.mag, wdlf_6.number_density)
plot(wdlf_7.mag, wdlf_7.number_density)
plot(wdlf_8.mag, wdlf_8.number_density)
yscale('log')
