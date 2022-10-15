from mpi4py import MPI
import numpy as np
import sys
from WDPhotTools import theoretical_lf


comm = MPI.COMM_WORLD
size = comm.Get_size()
my_rank = comm.Get_rank()

if my_rank == 0:
    sys.stdout.write("{} processes in total.".format(size))
    sys.stdout.write("")

sys.stdout.write("Rank {} started.".format(my_rank))
sys.stdout.write("")

wdlf = theoretical_lf.WDLF()

# Construct the interpolator
wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0.0, 20.0, 0.01)
age_list = np.array_split(1e9 * np.arange(15.0, 15.1, 0.01), size)[my_rank]

for age in age_list:
    sys.stdout.write(
        "Currently computing {} Gyr population.".format(age / 1e9)
    )
    sys.stdout.write("")
    sys.stdout.flush()
    wdlf.set_sfr_model(mode="burst", age=age, duration=1e8)
    # WDLF in Mbol
    wdlf.compute_density(
        Mag=Mag,
        normed=False,
        epsabs=1e-16,
        epsrel=1e-16,
        n_points=100000,
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
        epsabs=1e-16,
        epsrel=1e-16,
        n_points=100000,
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
        epsabs=1e-16,
        epsrel=1e-16,
        n_points=100000,
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
        epsabs=1e-16,
        epsrel=1e-16,
        n_points=100000,
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
