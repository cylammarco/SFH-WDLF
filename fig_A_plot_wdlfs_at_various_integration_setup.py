import pathlib

import numpy as np
from matplotlib import pyplot as plt

base_folder = pathlib.Path("SFH-WDLF-article/figure_data/")

age_list = np.array([1e9, 1e10]).astype(int)
duration_list = np.array([1e6, 1e7, 1e8]).astype(int)


tol_list = ["1e-05", "1e-06", "1e-07", "1e-08", "1e-09"]
n_points_list = ["100", "200", "500", "1000", "2000", "5000", "10000"]

for age in age_list:

    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey="row")
    for i, duration in enumerate(duration_list):
        # vary tolerance
        t = []
        for tol in tol_list:
            t.append(
                np.loadtxt(
                    base_folder
                    / f"fig_A_age_{age}.0_duration_{duration}.0_tolerance_{tol}_n_points_10000_limit_1000000.csv",
                    delimiter=",",
                )
            )

        # Varying n_points
        n_p = []
        for n_points in n_points_list:
            n_p.append(
                np.loadtxt(
                    base_folder
                    / f"fig_A_age_{age}.0_duration_{duration}.0_tolerance_1e-06_n_points_{n_points}_limit_1000000.csv",
                    delimiter=",",
                )
            )

        ax1[i].plot(t[0][:, 0], t[0][:, 1] / t[4][:, 1], label="tol = 1E-5")
        ax1[i].plot(t[1][:, 0], t[1][:, 1] / t[4][:, 1], label="tol = 1E-6")
        ax1[i].plot(t[2][:, 0], t[2][:, 1] / t[4][:, 1], label="tol = 1E-7")
        ax1[i].plot(t[3][:, 0], t[3][:, 1] / t[4][:, 1], label="tol = 1E-8")
        ax1[i].plot(t[4][:, 0], t[4][:, 1] / t[4][:, 1], label="tol = 1E-9")

        n_norm = np.nanmax(n_p[6][:, 1])
        ax2[i].plot(n_p[0][:, 0], n_p[0][:, 1] / n_norm, label=r"$\mathrm{n_{points}} = 100$")
        ax2[i].plot(n_p[1][:, 0], n_p[1][:, 1] / n_norm, label=r"$\mathrm{n_{points}} = 200$")
        ax2[i].plot(n_p[2][:, 0], n_p[2][:, 1] / n_norm, label=r"$\mathrm{n_{points}} = 500$")
        ax2[i].plot(n_p[3][:, 0], n_p[3][:, 1] / n_norm, label=r"$\mathrm{n_{points}} = 1000$")
        ax2[i].plot(n_p[4][:, 0], n_p[4][:, 1] / n_norm, label=r"$\mathrm{n_{points}} = 2000$")
        ax2[i].plot(n_p[5][:, 0], n_p[5][:, 1] / n_norm, label=r"$\mathrm{n_{points}} = 5000$")
        ax2[i].plot(n_p[6][:, 0], n_p[6][:, 1] / n_norm, label=r"$\mathrm{n_{points}} = 10000$")

        # ax1.text(5, 1.1, r"$\mathrm{limit} = 1000000, \mathrm{n_{points}} = 2000$")
        # ax2.text(5, 1e5, r"$\mathrm{limit} = 1000000, tol = 1\times10^{-6}$")

        ax1[i].set_xlim(3.5, 18.5)
        ax2[i].set_xlim(3.5, 18.5)

        ax1[i].set_ylim(1 - 1e-6, 1 + 1e-6)
        ax2[i].set_ylim(1e-4, 1.5)

        ax1[i].set_yscale("linear")
        ax2[i].set_yscale("log")

        ax2[i].set_xlabel(r"$\mathrm{M}_{\mathrm{bol}}$ / mag")

        ax1[i].set_title(f"Burst Duration = {int(duration/1000000)} Myr")

        if duration == 1000000:
            ax1[i].legend()
            ax2[i].legend()
            ax1[i].set_ylabel(r"$n / n_{tol=1e-9}$")
            ax2[i].set_ylabel("Arbitrary number density")

        # plt.suptitle(f"{age // 1e9} Gyr old population with {duration // 1e6} Myr" " starburst")
        plt.subplots_adjust(top=0.96, bottom=0.055, left=0.05, right=0.995, hspace=0, wspace=0)
        plt.savefig(f"SFH-WDLF-article/figures/fig_A_age_{age}.png")
