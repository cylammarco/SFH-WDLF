import pickle
import numpy as np
import healpy.pixelfunc as hp
from scipy import interpolate as itp


# Compute the splines
spl = []
for j in range(0, hp.nside2npix(2)):
    data = np.load('phot_healpix_' + str(j) + '.npy')
    mag = data['phot_g_mean_mag']
    par_err = data['parallax_error']
    bin_size = 0.2
    mag_range = np.arange(4.9, 20.5, bin_size)
    mag_mask = [((mag>i) & (mag<i+bin_size)) for i in mag_range[:-1]]
    par_median = [np.median(par_err[i]) for i in mag_mask]
    spl.append(itp.UnivariateSpline(mag_range[:-1]+bin_size/2, par_median, k=3, s=0.0002))
    #plot(mag_range[:-1]+bin_size, spl[j](mag_range[:-1]+bin_size/2))


# Save the splines
for k, splinefit in enumerate(spl):
    pickle.dump(splinefit, open('par_err_pix_' + str(k) + '.pkl', 'wb+'))

# to read
# pickle.load(open('/Users/marcolam/Desktop/Gaia_phot/nside_2/par_err_pix_14.pkl', 'br'))

