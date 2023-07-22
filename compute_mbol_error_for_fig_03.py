import os

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader
from WDPhotTools.fitter import WDfitter

from scipy.interpolate import splrep, splev


def get_envelope_v1(x, y):
    x_list, y_list = list(x), list(y)
    assert len(x_list) == len(y_list)
    # First data
    ui, ux, uy = [0], [x_list[0]], [y_list[0]]
    li, lx, ly = [0], [x_list[0]], [y_list[0]]
    # Find upper peaks and lower peaks
    for i in range(1, len(x_list) - 1):
        if y_list[i] >= y_list[i - 1] and y_list[i] >= y_list[i + 1]:
            ui.append(i)
            ux.append(x_list[i])
            uy.append(y_list[i])
        if y_list[i] <= y_list[i - 1] and y_list[i] <= y_list[i + 1]:
            li.append(i)
            lx.append(x_list[i])
            ly.append(y_list[i])
    # Last data
    ui.append(len(x_list) - 1)
    ux.append(x_list[-1])
    uy.append(y_list[-1])
    li.append(len(y_list) - 1)
    lx.append(x_list[-1])
    ly.append(y_list[-1])
    if len(ux) == 2 or len(lx) == 2:
        return [], []
    else:
        func_ub = interp1d(ux, uy, kind="cubic", bounds_error=False)
        func_lb = interp1d(lx, ly, kind="cubic", bounds_error=False)
        ub, lb = [], []
        for i in x_list:
            ub = func_ub(x_list)
            lb = func_lb(x_list)
        ub = np.array([y, ub]).max(axis=0)
        lb = np.array([y, lb]).min(axis=0)
        return ub, lb


data_edr3 = np.load(
    "pubgcnswdlf-crossmatched-full-edr3-catalogue.npz", allow_pickle=True
)["data"]

gcns_wd = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]

G3 = data_edr3["phot_g_mean_mag"]
G3_BP = data_edr3["phot_bp_mean_mag"]
G3_RP = data_edr3["phot_rp_mean_mag"]

# the extra uncertainties come from
# https://ui.adsabs.harvard.edu/abs/2006AJ....132.1221H/abstract
# using u+g+r as proxy-G_BP
# using u+g+r+i+z as proxy-G
# using r+i+z as proxy-G_RP
G3_ERR = np.sqrt(
    1.0 / data_edr3["phot_g_mean_flux_over_error"] ** 2.0
    + ((0.038 + 0.09 + 0.0308) / 3 / 2.355) ** 2.0
)
G3_BP_ERR = np.sqrt(
    1.0 / data_edr3["phot_bp_mean_flux_over_error"] ** 2.0
    + ((0.038 + 0.09 + 0.0308 + 0.0308 + 0.0414) / 5 / 2.355) ** 2.0
)
G3_RP_ERR = np.sqrt(
    1.0 / data_edr3["phot_rp_mean_flux_over_error"] ** 2.0
    + ((0.0308 + 0.0308 + 0.0414) / 3 / 2.355) ** 2.0
)

distance = 1000.0 / data_edr3["parallax"]
distance_err = (
    (1000.0 / (data_edr3["parallax"] - data_edr3["parallax_error"]))
    - (1000.0 / (data_edr3["parallax"] + data_edr3["parallax_error"]))
) / 2

gcns_Mbol = gcns_wd["Mbol"]
vmax = gcns_wd['Vmax']

Mbol_err = {}
Mbol_initial = {}
one_over_vmax_mean = {}
Mbol = {}
Mbol_upper_1_sigma = {}
Mbol_upper_2_sigma = {}
Mbol_upper_3_sigma = {}
Mbol_lower_1_sigma = {}
Mbol_lower_2_sigma = {}
Mbol_lower_3_sigma = {}

# https://ui.adsabs.harvard.edu/abs/2014ApJ...796..128G/abstract
# Comparison of Atmospheric Parameters Determined from Spectroscopy and Photometry for DA White Dwarfs in the Sloan Digital Sky Survey
# log(g_phot) = 7.994 ± 0.404
# log(g_spec) = 8.037 ± 0.206
logg_mean = 7.994
logg_sigma = 0.404
logg = {
    "low_3": logg_mean - 3 * logg_sigma,
    "low_2": logg_mean - 2 * logg_sigma,
    "low_1": logg_mean - logg_sigma,
    "mid": logg_mean,
    "high_1": logg_mean + logg_sigma,
    "high_2": logg_mean + 2 * logg_sigma,
    "high_3": logg_mean + 3 * logg_sigma,
}


try:
    Mbol = np.load(
        "SFH-WDLF-article/figure_data/Mbol_recomputed.npy", allow_pickle=True
    ).item()
    if len(Mbol) == 1:
        Mbol = {}
        raise ValueError("Mbol_recomputed.npy is empty")
    Mbol_upper_1_sigma = np.load(
        "SFH-WDLF-article/figure_data/Mbol_upper_1_sigma_recomputed.npy",
        allow_pickle=True,
    ).item()
    Mbol_upper_2_sigma = np.load(
        "SFH-WDLF-article/figure_data/Mbol_upper_2_sigma_recomputed.npy",
        allow_pickle=True,
    ).item()
    Mbol_upper_3_sigma = np.load(
        "SFH-WDLF-article/figure_data/Mbol_upper_3_sigma_recomputed.npy",
        allow_pickle=True,
    ).item()
    Mbol_lower_1_sigma = np.load(
        "SFH-WDLF-article/figure_data/Mbol_lower_1_sigma_recomputed.npy",
        allow_pickle=True,
    ).item()
    Mbol_lower_2_sigma = np.load(
        "SFH-WDLF-article/figure_data/Mbol_lower_2_sigma_recomputed.npy",
        allow_pickle=True,
    ).item()
    Mbol_lower_3_sigma = np.load(
        "SFH-WDLF-article/figure_data/Mbol_lower_3_sigma_recomputed.npy",
        allow_pickle=True,
    ).item()
    Mbol_err = np.load(
        "SFH-WDLF-article/figure_data/Mbol_error_recomputed.npy",
        allow_pickle=True,
    ).item()
    one_over_vmax_mean = np.load(
        "SFH-WDLF-article/figure_data/one_over_vmax_mean.npy",
        allow_pickle=True,
    ).item()
    print("Pre-computed files loaded successfully.")
except Exception as e:
    print(e)
    print("Computing the post-hoc errors.")
    for k, g in logg.items():
        ftr = WDfitter()
        for i, source_id in enumerate(data_edr3["source_id"]):
            Mbol_initial_guess = np.nanmean(
                gcns_Mbol[gcns_wd["sourceID"] == source_id]
            )
            Mbol_initial[source_id] = Mbol_initial_guess
            one_over_vmax_mean[source_id] = 1.0 / np.nanmean(
                vmax[gcns_wd["sourceID"] == source_id]
            )

            ftr.fit(
                atmosphere="H",
                filters=[
                    "G3",
                    "G3_BP",
                    "G3_RP",
                ],
                mags=[
                    G3[i],
                    G3_BP[i],
                    G3_RP[i],
                ],
                mag_errors=[
                    G3_ERR[i],
                    G3_BP_ERR[i],
                    G3_RP_ERR[i],
                ],
                independent=["Mbol"],
                logg=g,
                atmosphere_interpolator="RBF",
                reuse_interpolator=True,
                distance=distance[i],
                distance_err=distance_err[i],
                initial_guess=[Mbol_initial_guess],
                method="least_squares",
            )
            if k == "low_1":
                Mbol_lower_1_sigma[source_id] = ftr.best_fit_params["H"][
                    "Mbol"
                ]
            if k == "low_2":
                Mbol_lower_2_sigma[source_id] = ftr.best_fit_params["H"][
                    "Mbol"
                ]
            if k == "low_3":
                Mbol_lower_3_sigma[source_id] = ftr.best_fit_params["H"][
                    "Mbol"
                ]
            if k == "mid":
                Mbol_err[source_id] = ftr.best_fit_params["H"]["Mbol_err"]
                Mbol[source_id] = ftr.best_fit_params["H"]["Mbol"]
            if k == "high_1":
                Mbol_upper_1_sigma[source_id] = ftr.best_fit_params["H"][
                    "Mbol"
                ]
            if k == "high_2":
                Mbol_upper_2_sigma[source_id] = ftr.best_fit_params["H"][
                    "Mbol"
                ]
            if k == "high_3":
                Mbol_upper_3_sigma[source_id] = ftr.best_fit_params["H"][
                    "Mbol"
                ]

    np.save("SFH-WDLF-article/figure_data/Mbol_recomputed.npy", Mbol)
    np.save(
        "SFH-WDLF-article/figure_data/Mbol_upper_1_sigma_recomputed.npy",
        Mbol_upper_1_sigma,
    )
    np.save(
        "SFH-WDLF-article/figure_data/Mbol_upper_2_sigma_recomputed.npy",
        Mbol_upper_2_sigma,
    )
    np.save(
        "SFH-WDLF-article/figure_data/Mbol_upper_3_sigma_recomputed.npy",
        Mbol_upper_3_sigma,
    )
    np.save(
        "SFH-WDLF-article/figure_data/Mbol_lower_1_sigma_recomputed.npy",
        Mbol_lower_1_sigma,
    )
    np.save(
        "SFH-WDLF-article/figure_data/Mbol_lower_2_sigma_recomputed.npy",
        Mbol_lower_2_sigma,
    )
    np.save(
        "SFH-WDLF-article/figure_data/Mbol_lower_3_sigma_recomputed.npy",
        Mbol_lower_3_sigma,
    )
    np.save("SFH-WDLF-article/figure_data/Mbol_error_recomputed.npy", Mbol_err)
    np.save("SFH-WDLF-article/figure_data/one_over_vmax_mean.npy", one_over_vmax_mean)


(
    _mbol_err,
    _mbol,
    _mbol_upper_1_sigma,
    _mbol_upper_2_sigma,
    _mbol_upper_3_sigma,
    _mbol_lower_1_sigma,
    _mbol_lower_2_sigma,
    _mbol_lower_3_sigma,
    _vmax_mean
) = np.array(
    [
        (
            Mbol_err[k],
            v,
            Mbol_upper_1_sigma[k],
            Mbol_upper_2_sigma[k],
            Mbol_upper_3_sigma[k],
            Mbol_lower_1_sigma[k],
            Mbol_lower_2_sigma[k],
            Mbol_lower_3_sigma[k],
            one_over_vmax_mean[k]
        )
        for k, v in Mbol.items()
        if v < 18.0
    ]
).T
_arg = np.argsort(_mbol)


mbol_diff = (
    _mbol
    - _mbol_upper_1_sigma
    + (_mbol - _mbol_upper_2_sigma) / 2.0
    + (_mbol - _mbol_upper_3_sigma) / 3.0
    + _mbol_lower_1_sigma
    - _mbol
    + (_mbol_lower_2_sigma - _mbol) / 2.0
    + (_mbol_lower_3_sigma - _mbol) / 3.0
) / 6.0

# mbol_diff = np.abs(_mbol_lower_1_sigma - _mbol_upper_1_sigma).astype("float") / 2.0

mbol_total_err = np.sqrt(mbol_diff**2.0 + _mbol_err.astype("float") ** 2.0)
mbol_total_err = mbol_total_err[_arg]
vmax_mean = _vmax_mean[_arg]

_mbol = _mbol[_arg]


mag_bin = np.concatenate(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        np.arange(5.00, 6.75, 0.5),
        np.arange(6.75, 17.41, 0.2),
        [17.6],
    ]
)
mag_bin_center = (mag_bin[1:] + mag_bin[:-1]) / 2.0
idx = np.digitize(_mbol, mag_bin)
sdv_bin = np.zeros_like(mag_bin)
mean_bin = np.zeros_like(mag_bin)
for i in list(set(idx)):
    sdv_bin[i] = 1.4826 * np.median(
        np.abs(np.median(mbol_total_err[idx == i]) - mbol_total_err[idx == i])
    )
    mean_bin[i] = np.average(mbol_total_err[idx == i], weights=vmax_mean[idx == i])


tck = splrep(mag_bin_center, mean_bin[1:], k=3, s=1./len(mean_bin))
# tck = splrep(mag_bin_center, median_bin[1:] + sdv_bin[1:] * 3, k=2, s=0.015)

output_mean = np.column_stack(
    (
        np.arange(2.0, 19.0, 0.1),
        splev(np.arange(2.0, 19.0, 0.1), tck),
    )
)

output_upper_bound = np.column_stack(
    (
        np.arange(2.0, 19.0, 0.1),
        splev(np.arange(2.0, 19.0, 0.1), tck),
    )
)
np.save("SFH-WDLF-article/figure_data/mbol_err_mean", output_mean)
np.save(
    "SFH-WDLF-article/figure_data/mbol_err_upper_bound", output_upper_bound
)
