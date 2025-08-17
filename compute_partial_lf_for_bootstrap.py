import os
import sys

import numpy as np
from scipy.interpolate import interp1d

from WDPhotTools import theoretical_lf
from WDPhotTools.util import load_ms_lifetime_datatable

(
    imf_samples,
    ms_samples,
    ifmr_m_samples,
    ifmr_c_samples,
    cooling_samples,
) = np.load("bootstrap_table.npy").T

wdlf = theoretical_lf.WDLF()

mag_obs_optimal, resolution_optimal = np.load("SFH-WDLF-article/figure_data/mbol_resolution.npy").T

ms_lieftime_datatable = load_ms_lifetime_datatable("PARSECz0017.csv")

# imf_model="C03",
# ifmr_model="C08",
# low_mass_cooling_model="montreal_co_da_20",
# intermediate_mass_cooling_model="montreal_co_da_20",
# high_mass_cooling_model="montreal_co_da_20",
# ms_model="PARSECz0017",


def imf_function(delta):
    # Precompute constants for log-normal part
    log_0_079 = -1.1023729087095586
    two_sigma_sq = 0.9522
    norm_factor = 0.01915058
    const_factor = 0.06861852814  # 0.158 / (ln(10))

    def imf_func(m):
        m = np.ravel(m).astype(float)
        mass_function = m ** -(2.3 + delta)

        m_mask = m < 1.0
        if m_mask.any():
            log_term = np.log10(m[m_mask]) + log_0_079
            mass_function[m_mask] = (const_factor / m[m_mask]) * np.exp(-(log_term**2) / two_sigma_sq) / norm_factor

        return mass_function

    return imf_func


def ms_function(delta):
    massi = np.ravel(ms_lieftime_datatable[:, 0]).astype(np.float64)
    time = np.ravel(ms_lieftime_datatable[:, 1]).astype(np.float64) * (1 + delta)

    interpolator = interp1d(massi, time, kind="cubic", fill_value="extrapolate")

    def ms_func(m):
        m = np.ravel(m).astype(float)
        return interpolator(m)

    return ms_func


def ifmr_function(delta_m, delta_c):
    coeff_m = 0.117 * (1 + delta_m)
    coeff_c = 0.384 * (1 + delta_c)

    def ifmr_func(m):
        m = np.ravel(m).astype(float)
        mass = coeff_m * m + coeff_c
        mass = np.where(mass < 0.4349, 0.4349, mass)
        return mass

    return ifmr_func


Mag = np.arange(5.0, 18.0, 0.1)
(
    partial_age_optimal,
    partial_age_duration,
    _,
    _,
    _,
) = np.load("SFH-WDLF-article/figure_data/gcns_sfh_optimal_resolution_bin_optimal.npy").T

idx = int(sys.argv[1])
print(f"Processing bootstrap sample {idx}...")

delta_imf = imf_samples[idx]
delta_ms = ms_samples[idx]
delta_ifmr_m = ifmr_m_samples[idx]
delta_ifmr_c = ifmr_c_samples[idx]
delta_cooling = cooling_samples[idx]

print(
    f"Currently computing WDLF with imf delta={delta_imf:.4f}, ms delta={delta_ms:.4f}, "
    f"ifmr_m delta={delta_ifmr_m:.4f}, ifmr_c delta={delta_ifmr_c:.4f}, cooling delta={delta_cooling:.4f}."
)
wdlf.set_imf_model(model="manual", imf_function=imf_function(delta_imf))
wdlf.set_ms_model(model="manual", ms_function=ms_function(delta_ms))
wdlf.set_ifmr_model(
    model="manual",
    ifmr_function=ifmr_function(delta_ifmr_m, delta_ifmr_c),
)
# Construct the interpolator
wdlf.compute_cooling_age_interpolator(interpolator="RBF", scaling_factor=(1.0 + delta_cooling))

for age, duration in zip(partial_age_optimal, partial_age_duration):

    print(f"Currently computing {age} Gyr population.")
    wdlf.set_sfr_model(mode="burst", age=age * 1e9, duration=duration * 1e9)

    os.makedirs(f"./bootstrap_sample_folder/sample_{idx}", exist_ok=True)
    # WDLF in Mbol
    wdlf.compute_density(
        Mag,
        interpolator="RBF",
        normed=False,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=10000,
        n_points=50,
        folder=f"./bootstrap_sample_folder/sample_{idx}",
        filename=f"montreal_co_da_20_C03_PARSECz0017_C08_{age:.4f}.csv",
        save_csv=True,
    )
    wdlf.plot_wdlf(
        display=False,
        savefig=True,
        folder=f"./bootstrap_sample_folder/sample_{idx}",
        filename=f"montreal_co_da_20_C03_PARSECz0017_C08_{age:.4f}",
    )
