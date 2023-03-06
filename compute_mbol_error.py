import os

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from WDPhotTools.fitter import WDfitter

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
gcns_vmax = gcns_wd["Vgen"]

Mbol_err = {}
Mbol_initial = {}
Mbol = {}
Mbol_upper = {}
Mbol_lower = {}
vmax = {}

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

for k, g in logg.items():
    ftr = WDfitter()
    for i, source_id in enumerate(data_edr3["source_id"]):
        Mbol_initial_guess = np.nanmean(
            gcns_Mbol[gcns_wd["sourceID"] == source_id]
        )
        Mbol_initial[source_id] = Mbol_initial_guess
        vmax[source_id] = np.nanmean(
            gcns_vmax[gcns_wd["sourceID"] == source_id]
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
            atmosphere_interpolator="CT",
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


_mbol_err, _mbol, _mbol_upper, _mbol_lower, _vmax = np.array(
    [
        (Mbol_err[k], v, Mbol_upper[k], Mbol_lower[k], vmax[k])
        for k, v in Mbol.items()
    ]
).T
_arg = np.argsort(_mbol)

mbol_diff = np.abs(_mbol_lower - _mbol_upper).astype("float") / 2.0

mbol_total_err = np.sqrt(mbol_diff**2.0 + _mbol_err.astype("float") ** 2.0)
# mbol_total_err = mbol_diff

mbol_total_err[_arg] = medfilt(mbol_total_err[_arg], 5)
mbol_total_err_smoothed = np.concatenate(
    (
        mbol_total_err[_arg][_mbol[_arg] <= 11.0],
        lowess(
            mbol_total_err[_arg][(_mbol[_arg] > 11.0) & (_mbol[_arg] <= 16.0)],
            _mbol[_arg][(_mbol[_arg] > 11.0) & (_mbol[_arg] <= 16.0)],
            frac=0.05,
            it=10,
        )[:, 1],
        mbol_total_err[_arg][_mbol[_arg] > 16.0],
    )
)


_spline = UnivariateSpline(
    np.log10(_mbol[_arg].astype("float")),
    np.log10(mbol_total_err_smoothed),
    s=0.22,
    k=3
)

# 99%
# np.sum(10.0**_spline(np.log10(_mbol[_arg].astype('float'))) + 3.34e-1 > _mbol_err[_arg].astype('float'))

shift = 3.34e-1

output = np.column_stack(
    (
        np.arange(2, 18, 0.01),
        10.0 ** _spline(np.log10(np.arange(2, 18, 0.01))) + shift,
    )
)
np.save("mbol_sigma", output)

