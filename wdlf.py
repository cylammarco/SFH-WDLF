import pickle
import numpy as np
from numpy import sin, cos
from scipy.integrate import quad
from scipy import interpolate as itp
from scipy.optimize import minimize
#import mwdust
from scipy.signal import medfilt
import healpy.pixelfunc as hp
#from numba import jit
from matplotlib.pyplot import *
ion()

def eq2gal(ra, dec, intype='degrees', outtype='degrees'):
    """
    Code to convert equatorial coordinates to galactic coordinates
    in J2000. Constants (Galactic north pole, and galactic longitude
    of celestial pole) are taken from Funamental Astronomy 5th Edition,
    Karttunen, Kroger, Poutanen, Donner.
    Parameters:
    ------------
    _dp: J2000 Declination of the NGP
    _ap: J2000 Right Acension of the NGP
    _ln: J2000 position angle of the galactic centre
         (galactic longitude of the NCP)
    ra: (float or np array) Source(s) right ascension(s) in J2000
    dec: (float or np array) Source(s) declination(s) in J2000
    radians: (boolean) set to True if ra and dec are already in radians
            or False if they are left in degrees.
    fullcircle: (boolean) set to True if return coordinates are to be from
                0:360 degrees. False if [-180:180].
    Return:
    --------
    (lgal,bgal): tuple
    lgal: (float or np array) Source(s) Galactic longitude coordinate(s)
    bgal: (float or np array) Source(s) Galactic lattitude coordinate(s)
    """
    _dp = np.radians(27.128336)
    _ap = np.radians(192.859508)
    _ln = np.radians(122.9319)
    if (intype == 'degrees'):
        ra = np.radians(ra)
        dec = np.radians(dec)
    b1 = (cos(dec) * cos(_dp) * cos(ra - _ap) + sin(dec) * sin(_dp))
    bgal = np.arcsin(b1)
    ly = cos(dec) * sin(ra - _ap)
    lx = (-cos(dec) * sin(_dp) * cos(ra - _ap) + sin(dec) * cos(_dp))
    lgal = _ln - np.arctan2(ly, lx)
    try:
        ind = np.where(lgal < 0.)
        lgal[ind] = lgal[ind] + 2. * np.pi
    except:
        if lgal < 0.:
            lgal = lgal + 2. * np.pi
    if (outtype == 'degrees'):
        lgal = np.degrees(lgal)
        bgal = np.degrees(bgal)
    return np.array([lgal, bgal])


# Analytical solution for integrating
#   /int_0^D [ r**2. * np.exp( -np.abs(r * sinb + z) / H) ] dr
# with constant sinb, z & H
def volume(D, sinb, z, H):
    xi = H / sinb
    exponent = (D * sinb + z) / H
    if exponent >= 0:
        v = -np.exp(-exponent) * (2*xi**3. + 2*D*xi**2. + D**2.*xi) + 2.*xi**3. * np.exp(-z/H)
    else:
        v = np.exp(exponent) * (2*xi**3. - 2*D*xi**2. + D**2.*xi) - 2.*xi**3. * np.exp(z/H)
    return v

#@jit
def find_d_limit(max_dist, parallax, mag, par_itp, sig, mag_low, mag_high):
    '''
    The function to be minimised to find the maximum distnace limit based on the magnitude and the parallax uncertainties
    

    Parameter
    ---------
    max_dist : float
        The upper limit of the distance, particularly important for a
        distance limited sample
    parallax : float
        Measured parallax
    mag : float
        Observed magnitude
    par_itp : scipy.interpolate.interp1d object
        Parallax uncertainty as a function of mag
    sig : float
        Required parallax significance for finding
    mag_low : float
        Bright magnitude limit (smallest magnitude value)
    mag_high : float
        Faint magnitude limit (Largest magnitude value)

    Return
    ------
    Absolute difference between the parallax and the parallax uncertainties at
    the given sig significance level.

    '''

    if (max_dist > 100.):
        return np.inf
    mod = np.log10(1000. / parallax)*5. - 5.
    max_mod = np.log10(max_dist)*5. - 5.
    mag_new = mag + max_mod - mod
    if (mag_new > mag_low) & (mag_new < mag_high):
        parallax_error = par_itp(mag_new)
        diff = np.abs(parallax - sig*parallax_error)
        return diff
    else:
        return np.inf


'''
extinction = mwdust.Combined15()
extinction(30.,3.,np.array([1.,2.,3.,10.,100.]))
'''

'''
(1) Paralllax Error as a function of (a) HEALPix position; (b) G mag
(2) 4.0 < G / mag < 20.3. (spline constructed with 4.9 to 20.5)
(3) scaleheight / pc = 200, 250, 300, 350
(4) compare different distance limit at 10, 20, 25, 50, 75, 100, full sample

'''

# GDR2 data
data = np.genfromtxt('/Users/marcolam/git/gaia_wdlf/kilic2018-result.csv', delimiter=",", names=True)

# parallax error as a function of g_mean_mag
#argsorted = np.argsort(data['phot_g_mean_mag'])
#par_itp = itp.interp1d(data['phot_g_mean_mag'][argsorted], data['parallax_error'][argsorted], fill_value='extrapolate')
par_err_spl = np.array([['']] * hp.nside2npix(2), dtype='object')
for i in range(hp.nside2npix(2)):
    par_err_spl[i] = pickle.load(open('/Users/marcolam/Desktop/Gaia_phot/nside_2/par_err_pix_' + str(i) + '.pkl', 'rb'))


l,b = eq2gal(data['ra'], data['dec'])
sinb = np.sin(np.radians(b))
pix = hp.ang2pix(2, np.radians(90.-data['dec']), np.radians(data['ra']))

# Each data point has Npix dmax, vol and vmax
dmax = np.zeros((len(data), len(par_err_spl)))
vol = np.zeros((len(data), len(par_err_spl)))
vmax = np.zeros((len(data), len(par_err_spl)))

# Finding the distance limits at N sigma confidence parallax limit as
# brightness decreases
# Calculate the volume and the maximum volume
# Loop through each data point
for i in range(len(data)):
    print(i)
    for j in range(len(par_err_spl)):
        dmax[i,j] = minimize(find_d_limit, 50.,
            args=(data['parallax'][i],
                data['phot_g_mean_mag'][i],
                par_err_spl[j][0],
                10.,
                5.,
                20.3)
            ).x
        vol[i,j] = volume(1000./data['parallax'][i], sinb[i], 20., 250.) * 4. * np.pi
        vmax[i,j] = volume(dmax[i,j], sinb[i], 20., 250.) * 4. * np.pi

# G = 20.3 mag limit across the sky (temp)

figure(2)
clf()
hist(np.sum(vol, axis=1)/np.sum(vmax, axis=1), bins=40, range=(0,1))
xlabel('V/Vmax')
xlim(0,1)

num, mag = np.histogram(
    data['phot_g_mean_mag'] - np.log10(1000./data['parallax'])*5. + 5,
    weights=48.*4./np.sum(vmax, axis=1),
    bins=60,
    range=(5,20))
mag = mag[1:] + 0.25

figure(3)
clf()
plot(mag, np.log10(num))
scatter(mag, np.log10(num))
xlabel('G / mag')
ylabel(r'Number Density / N pc$^{-3}$ mag$^{-1}$')
xlim(7, 19)
ylim(-6.0, -2.5)
grid()
tight_layout()
#savefig('gaia_wdlf.png')

