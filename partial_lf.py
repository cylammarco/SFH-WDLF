import numpy as np
from WDPhotTools import theoretical_lf

wdlf = theoretical_lf.WDLF()

# Construct the interpolator
wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0.0, 20.0, 1.0)
age_list = 1e9 * np.arange(0.1, 15.0, 0.1)

for age in age_list:
    wdlf.set_sfr_model(mode="burst", age=age, duration=1e8)
    # WDLF in Mbol
    wdlf.compute_density(
        Mag=Mag,
        normed=False,
        epsabs=1e-12,
        epsrel=1e-11,
        n_points=1000,
        folder="output",
        filename="montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_Mbol.csv".format(
            age / 1e9
        ),
        save_csv=True,
    )
    wdlf.plot_wdlf(
        display=False,
        savefig=True,
        folder="output",
        filename="montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_Mbol".format(
            age / 1e9
        ),
    )
    # WDLF in GDR3 G
    wdlf.compute_density(
        Mag=Mag,
        passband="G3",
        normed=False,
        epsabs=1e-12,
        epsrel=1e-11,
        n_points=1000,
        folder="output",
        filename="montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3.csv".format(
            age / 1e9
        ),
        save_csv=True,
    )
    wdlf.plot_wdlf(
        display=False,
        savefig=True,
        folder="output",
        filename="montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3".format(
            age / 1e9
        ),
    )
    # WDLF in GDR3 BP
    wdlf.compute_density(
        Mag=Mag,
        passband="G3_BP",
        normed=False,
        epsabs=1e-12,
        epsrel=1e-11,
        n_points=1000,
        folder="output",
        filename="montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_BP.csv".format(
            age / 1e9
        ),
        save_csv=True,
    )
    wdlf.plot_wdlf(
        display=False,
        savefig=True,
        folder="output",
        filename="montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_BP".format(
            age / 1e9
        ),
    )
    # WDLF in GDR3 RP
    wdlf.compute_density(
        Mag=Mag,
        passband="G3_RP",
        normed=False,
        epsabs=1e-12,
        epsrel=1e-11,
        n_points=1000,
        folder="output",
        filename="montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_RP.csv".format(
            age / 1e9
        ),
        save_csv=True,
    )
    wdlf.plot_wdlf(
        display=False,
        savefig=True,
        folder="output",
        filename="montreal_co_da_20_K01_PARSECz0014_C08_{0:.2f}_G3_RP".format(
            age / 1e9
        ),
    )
