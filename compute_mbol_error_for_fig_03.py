import os

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
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


figure_folder = "SFH-WDLF-article/figures"

data_edr3 = np.load(
    "pubgcnswdlf-crossmatched-full-edr3-catalogue.npz", allow_pickle=True
)["data"]

gcns_wd = np.load(
    "pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz"
)["data"]

G3 = data_edr3["phot_g_mean_mag"]
G3_BP = data_edr3["phot_bp_mean_mag"]
G3_RP = data_edr3["phot_rp_mean_mag"]

G3_ERR = 1.0 / data_edr3["phot_g_mean_flux_over_error"]
G3_BP_ERR = 1.0 / data_edr3["phot_bp_mean_flux_over_error"]
G3_RP_ERR = 1.0 / data_edr3["phot_rp_mean_flux_over_error"]

distance = 1000.0 / data_edr3["parallax"]
distance_err = (
    (1000.0 / (data_edr3["parallax"] - data_edr3["parallax_error"]))
    - (1000.0 / (data_edr3["parallax"] + data_edr3["parallax_error"]))
) / 2

gcns_Mbol = gcns_wd["Mbol"]

Mbol_err = {}
Mbol_initial = {}
Mbol = {}
Mbol_upper = {}
Mbol_lower = {}

# https://ui.adsabs.harvard.edu/abs/2014ApJ...796..128G/abstract
# Comparison of Atmospheric Parameters Determined from Spectroscopy and Photometry for DA White Dwarfs in the Sloan Digital Sky Survey
# log(g) = 7.994 ± 0.404
logg_mean = 7.994
logg_sigma = 0.404
logg = {
    "low": logg_mean - logg_sigma,
    "mid": logg_mean,
    "high": logg_mean + logg_sigma,
}


try:
    Mbol = np.load("Mbol_recomputed.npy", allow_pickle=True).item()
    if len(Mbol) == 1:
        Mbol = {}
        raise ValueError("Mbol_recomputed.npy is empty")
    Mbol_upper = np.load("Mbol_upper_recomputed.npy", allow_pickle=True).item()
    Mbol_lower = np.load("Mbol_lower_recomputed.npy", allow_pickle=True).item()
    Mbol_err = np.load("Mbol_error_recomputed.npy", allow_pickle=True).item()
    print("Pre-computed files loaded successfully.")
except:
    print("Computing the post-hoc errors.")
    for k, g in logg.items():
        ftr = WDfitter()
        for i, source_id in enumerate(data_edr3["source_id"]):
            Mbol_initial_guess = np.nanmean(
                gcns_Mbol[gcns_wd["sourceID"] == source_id]
            )
            Mbol_initial[source_id] = Mbol_initial_guess
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
            if k == "low":
                Mbol_lower[source_id] = ftr.best_fit_params["H"]["Mbol"]
            if k == "mid":
                Mbol_err[source_id] = ftr.best_fit_params["H"]["Mbol_err"]
                Mbol[source_id] = ftr.best_fit_params["H"]["Mbol"]
            if k == "high":
                Mbol_upper[source_id] = ftr.best_fit_params["H"]["Mbol"]

    np.save("Mbol_recomputed.npy", Mbol)
    np.save("Mbol_upper_recomputed.npy", Mbol_upper)
    np.save("Mbol_lower_recomputed.npy", Mbol_lower)
    np.save("Mbol_error_recomputed.npy", Mbol_err)


_mbol_err, _mbol, _mbol_upper, _mbol_lower = np.array(
    [
        (Mbol_err[k], v, Mbol_upper[k], Mbol_lower[k])
        for k, v in Mbol.items()
    ]
).T
_arg = np.argsort(_mbol)

mbol_diff = np.abs(_mbol_lower - _mbol_upper).astype("float") / 2.0

mbol_total_err = np.sqrt(mbol_diff**2.0 + _mbol_err.astype("float") ** 2.0)
mbol_total_err = mbol_total_err[_arg]


_mbol = _mbol[_arg]


mag_bin = np.concatenate(
    [
        [1.5],
        np.arange(5.25, 6.75, 0.5),
        np.arange(6.75, 9.75, 0.25),
        np.arange(9.75, 15.745, 0.1),
        np.arange(15.75, 17.76, 0.5),
        [18.5],
    ]
)
mag_bin_center = (mag_bin[1:] + mag_bin[:-1]) / 2.0
idx = np.digitize(_mbol, mag_bin)
sdv_bin = np.zeros_like(mag_bin)
median_bin = np.zeros_like(mag_bin)
for i in list(set(idx)):
    sdv_bin[i] = 1.4826 * np.median(
        np.abs(np.median(mbol_total_err[idx == i]) - mbol_total_err[idx == i])
    )
    median_bin[i] = np.median(mbol_total_err[idx == i])


tck = splrep(mag_bin_center, median_bin[1:] + sdv_bin[1:] * 3, k=2, s=0.015)
"""
figure(1, figsize=(8, 6))
clf()
scatter(_mbol, mbol_total_err, s=1)
plot(mag_bin_center, median_bin[1:], color='black', ls="dashed")
plot(mag_bin_center, median_bin[1:] + sdv_bin[1:] * 3, color='C1')
plot(np.arange(2.0, 19.0, 0.1), splev(np.arange(2.0, 19.0, 0.1), tck), color='black')
yscale('log')
ylim(5e-3, 3e0)
xlim(2, 19.0)
tight_layout()
grid()
savefig('mbol_err_upper_bound.png')
"""

output_median = np.column_stack(
    (
        mag_bin,
        median_bin,
    )
)

output_upper_bound = np.column_stack(
    (
        np.arange(2.0, 19.0, 0.1),
        splev(np.arange(2.0, 19.0, 0.1), tck),
    )
)
np.save("mbol_err_median", output_median)
np.save("mbol_err_upper_bound", output_upper_bound)
