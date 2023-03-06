import os

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

figure_folder = "SFH-WDLF-article/figures"


Mbol = np.load("Mbol_recomputed.npy", allow_pickle=True).item()
Mbol_upper = np.load("Mbol_upper_recomputed.npy", allow_pickle=True).item()
Mbol_lower = np.load("Mbol_lower_recomputed.npy", allow_pickle=True).item()
Mbol_err = np.load("Mbol_error_recomputed.npy", allow_pickle=True).item()


_mbol_err, _mbol, _mbol_upper, _mbol_lower = np.array(
    [
        (Mbol_err[k], v, Mbol_upper[k], Mbol_lower[k])
        for k, v in Mbol.items()
    ]
).T
_arg = np.argsort(_mbol)

mbol_diff = np.abs(_mbol_lower - _mbol_upper).astype("float") / 2.0

mbol_total_err = np.sqrt(mbol_diff**2.0 + _mbol_err.astype("float") ** 2.0)
# mbol_total_err = mbol_diff


plt.figure(1, figsize=(8, 6))
plt.clf()
plt.scatter(
    _mbol,
    mbol_total_err,
    s=1,
    label=r"Post-hoc M$_{\mathrm{bol}}$ uncertainties",
)


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

"""
plt.scatter(
    _mbol[_arg],
    mbol_total_err_smoothed,
    s=0.5,
    label="Smoothed uncertainties",
)
"""
_spline = UnivariateSpline(
    np.log10(_mbol[_arg].astype("float")),
    np.log10(mbol_total_err_smoothed),
    s=0.22,
    k=3,
)
plt.plot(
    np.arange(2, 18, 0.01),
    10.0 ** _spline(np.log10(np.arange(2, 18, 0.01))),
    lw=1,
    label="Fitted median",
    color="black",
)

# 99%
# np.sum(10.0**_spline(np.log10(_mbol[_arg].astype('float'))) + 3.34e-1 > _mbol_err[_arg].astype('float'))

shift = 3.34e-1
plt.plot(
    np.arange(2, 18, 0.01),
    10.0 ** _spline(np.log10(np.arange(2, 18, 0.01))) + shift,
    lw=1,
    label="Fitted median shifted (Above 90% of data)",
    color="black",
    ls="dashed",
)
plt.yscale("log")
plt.xlabel(r"M$_{\mathrm{bol}}$ (mag)")
plt.ylabel(r"$\Delta$M$_{\mathrm{bol}}$ (mag)")
plt.xlim(2, 18)
plt.ylim(1e-3, 10)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "fig_03_mbol_sigma.png"))

output = np.column_stack(
    (
        np.arange(2, 18, 0.01),
        10.0 ** _spline(np.log10(np.arange(2, 18, 0.01))) + shift,
    )
)
np.save("mbol_sigma", output)
