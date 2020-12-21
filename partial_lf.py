import numpy as np
from WDLFBuilder import theoretical_lf

wdlf = theoretical_lf.WDLF()

wdlf.set_ifmr_model('C08')
wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0., 20., 0.01)
age_list = 1E9 * np.arange(8.8, 15., 0.1)

for age in age_list:
    wdlf.set_sfr_model(mode='burst', age=age, duration=1e8)
    wdlf.compute_density(Mag=Mag,
                         normed=False,
                         save_csv=True)

    wdlf.plot_wdlf(display=False,
                   savefig=True,
                   filename='{}Gyr_burst.png'.format(age))
