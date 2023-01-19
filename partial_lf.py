from mpi4py import MPI
import numpy as np
import os
import sys
from WDPhotTools import theoretical_lf


comm = MPI.COMM_WORLD
size = comm.Get_size()
my_rank = comm.Get_rank()

if my_rank == 0:
    sys.stdout.write(f"{size} processes in total.")
    sys.stdout.write("")

sys.stdout.write(f"Rank {my_rank} started.")
sys.stdout.write("")

wdlf = theoretical_lf.WDLF()

# Construct the interpolator
wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0.0, 20.0, 0.1)
age_list_1 = 1e9 * np.arange(0.001, 0.100, 0.001)
age_list_2 = 1e9 * np.arange(0.100, 0.350, 0.005)
age_list_3 = 1e9 * np.arange(0.350, 15.01, 0.01)
# age_list = np.concatenate((age_list_1, age_list_2, age_list_3))
age_list = np.concatenate((age_list_1, age_list_2))

age_list_per_rank = np.array_split(age_list, size)[my_rank]

for age in age_list_per_rank:
    sys.stdout.write(f"Currently computing {age / 1e9} Gyr population.")
    sys.stdout.write("")
    sys.stdout.flush()
    wdlf.set_sfr_model(mode="burst", age=age, duration=1e7)
    # WDLF in Mbol
    wdlf.compute_density(
        Mag,
        interpolator="CT",
        normed=False,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=1000000,
        n_points=10000,
        folder="output",
        filename=f"montreal_co_da_20_K01_PARSECz0014_C08_{age / 1e9:.3f}_Mbol.csv",
        save_csv=True,
    )
    wdlf.plot_wdlf(
        display=False,
        savefig=True,
        folder="output",
        filename=f"montreal_co_da_20_K01_PARSECz0014_C08_{age / 1e9:.3f}_Mbol.png",
    )
    """
    # WDLF in GDR3 G
    wdlf.compute_density(
        Mag,
        passband="G3",
        normed=False,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=1000000,
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
        Mag,
        passband="G3_BP",
        normed=False,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=1000000,
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
        Mag,
        passband="G3_RP",
        normed=False,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=1000000,
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
    """
