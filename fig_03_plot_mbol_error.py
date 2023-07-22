import os

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

figure_folder = "SFH-WDLF-article/figures"


Mbol = np.load(
    "SFH-WDLF-article/figure_data/Mbol_recomputed.npy", allow_pickle=True
).item()
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
    "SFH-WDLF-article/figure_data/Mbol_error_recomputed.npy", allow_pickle=True
).item()



(
    _mbol_err,
    _mbol,
    _mbol_upper_1_sigma,
    _mbol_upper_2_sigma,
    _mbol_upper_3_sigma,
    _mbol_lower_1_sigma,
    _mbol_lower_2_sigma,
    _mbol_lower_3_sigma,
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
        )
        for k, v in Mbol.items()
        if v < 18.0
    ]
).T
_arg = np.argsort(_mbol)


# get the total error
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

_mbol = _mbol[_arg]

# get the precomputed envelope
# mag_upper_bound, err_upper_bound = np.load("mbol_err_upper_bound.npy").T
mag_mean, err_mean = np.load(
    "SFH-WDLF-article/figure_data/mbol_err_mean.npy"
).T

plt.figure(1, figsize=(8, 6))
plt.clf()
# plot the data
plt.scatter(
    _mbol,
    mbol_total_err,
    s=0.1,
    label=r"Post-hoc M$_{\mathrm{bol}}$ uncertainties",
)
# plot the mean
plt.plot(
    mag_mean[(mag_mean > 2.0) & (mag_mean < 18.25)],
    err_mean[(mag_mean > 2.0) & (mag_mean < 18.25)],
    lw=1,
    label=r"Median",
    color="black",
    ls="dashed",
)

# plot the envelope
# plt.plot(
#     mag_upper_bound,
#     err_upper_bound,
#     lw=1,
#     label=r"Line 3$\sigma$ above mean",
#     color="C1",
# )

plt.yscale("log")
plt.xlabel(r"M$_{\mathrm{bol}}$ (mag)")
plt.ylabel(r"$\Delta$M$_{\mathrm{bol}}$ (mag)")
plt.xlim(2, 18)
plt.ylim(1e-3, 10)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "fig_03_mbol_sigma.png"))
