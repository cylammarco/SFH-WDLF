import numpy as np
from WDPhotTools import theoretical_lf


Mag = np.arange(4.0, 18.0, 0.1)
tolerance_list = 10.0 ** np.arange(-5, -10, -1)
n_points_list = [100, 200, 500, 1000, 2000, 5000, 10000]

# Default
tolerance = 1e-6
n_points = 10000
limit = 1000000


for age in [1e9, 1e10]:

    for duration in [1e6, 1e7, 1e8]:

        """
        for n in n_points_list:
            wdlf = theoretical_lf.WDLF()
            # Construct the interpolator
            wdlf.compute_cooling_age_interpolator()
            wdlf.set_sfr_model(mode="burst", age=age, duration=duration)
            print(f"n_points: {n}")
            wdlf.compute_density(
                Mag,
                interpolator="CT",
                normed=False,
                epsabs=tolerance,
                epsrel=tolerance,
                limit=limit,
                n_points=n,
                folder="SFH-WDLF-article/figure_data/",
                filename=f"fig_A_age_{age}_duration_{duration}_tolerance_{tolerance}_n_points_{n}_limit_{limit}.csv",
                save_csv=True,
            )

        """
        for t in tolerance_list:
            wdlf = theoretical_lf.WDLF()
            # Construct the interpolator
            wdlf.compute_cooling_age_interpolator()
            wdlf.set_sfr_model(mode="burst", age=age, duration=duration)
            # WDLF in Mbol
            print(f"tolerance: {t}")
            wdlf.compute_density(
                Mag,
                interpolator="CT",
                normed=False,
                epsabs=t,
                epsrel=t,
                limit=limit,
                n_points=n_points,
                folder="SFH-WDLF-article/figure_data/",
                filename=f"fig_A_age_{age}_duration_{duration}_tolerance_{t}_n_points_{n_points}_limit_{limit}.csv",
                save_csv=True,
            )
