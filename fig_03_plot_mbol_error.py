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

# get the total error
_mbol_err, _mbol, _mbol_upper, _mbol_lower = np.array(
    [(Mbol_err[k], v, Mbol_upper[k], Mbol_lower[k]) for k, v in Mbol.items()]
).T
_arg = np.argsort(_mbol)

mbol_diff = np.abs(_mbol_lower - _mbol_upper).astype("float") / 2.0

mbol_total_err = np.sqrt(mbol_diff**2.0 + _mbol_err.astype("float") ** 2.0)
mbol_total_err = mbol_total_err[_arg]
_mbol = _mbol[_arg]

# get the precomputed envelope
mag_upper_bound, err_upper_bound = np.load("mbol_err_upper_bound.npy").T
mag_median, err_median = np.load("mbol_err_median.npy").T

plt.figure(1, figsize=(8, 6))
plt.clf()
# plot the data
plt.scatter(
    _mbol,
    mbol_total_err,
    s=0.1,
    label=r"Post-hoc M$_{\mathrm{bol}}$ uncertainties",
)
# plot the median
plt.plot(
    mag_median[mag_median>5.5], err_median[mag_median>5.5], lw=1, label=r"Median", color="black", ls="dashed"
)

# plot the envelope
plt.plot(
    mag_upper_bound,
    err_upper_bound,
    lw=1,
    label=r"Line 3$\sigma$ above median",
    color="C1",
)

plt.yscale("log")
plt.xlabel(r"M$_{\mathrm{bol}}$ (mag)")
plt.ylabel(r"$\Delta$M$_{\mathrm{bol}}$ (mag)")
plt.xlim(2, 18)
plt.ylim(1e-3, 10)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "fig_03_mbol_sigma.png"))
