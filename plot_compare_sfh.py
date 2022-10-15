from matplotlib import pyplot as plt
import numpy as np

plt.ion()


lam_age, lam_sfh = np.load('test_case/gcns_sfh.npy').T

rowell_age, rowell_sfh, _ = np.genfromtxt('test_case/fig_17.txt', delimiter=' ').T
rowell_age /= 1.0e+9

mor_age = np.array([0.05, 0.55, 1.5, 2.5, 4.0, 6.0, 7.5, 8.5, 9.5])
mor_sfh = np.array([0.16, 2.2, 6.5, 8.7, 12.1, 7.7, 5.8, 7.4, 9.8])

isern_age_s = np.array([0.05, 0.12, 0.41, 0.97, 1.80, 2.59, 3.53, 4.58, 5.75, 7.06, 8.88, 11.95, 14.13])
isern_sfh_s = 10.0**np.array([-2.794, -2.553, -2.655, -2.780, -2.546, -2.468, -2.600, -2.747, -2.753, -2.667, -2.885, -3.403, -4.123])

isern_age_ns = np.array([0.05, 0.12, 0.41, 0.91, 1.53, 2.13, 2.86, 3.75, 4.82, 6.09, 7.89, 10.96, 13.14])
isern_sfh_ns = 10.0**np.array([-2.794, -2.553, -2.643, -2.678, -2.418, -2.350, -2.508, -2.694, -2.728, -2.660, -2.884, -3.403, -4.130])

_lam_steps = plt.step(lam_age, lam_sfh, where='mid', label="This work")
_rowell_steps = plt.step(rowell_age, rowell_sfh, where='mid', label="Rowell 2013")
_mor_steps = plt.step(mor_age, mor_sfh, where='mid', label="Mor et al. 2019")
_isern_s_steps = plt.step(isern_age_s, isern_sfh_s, where='mid', label="Isern 2019")
_isern_ns_steps = plt.step(isern_age_ns, isern_sfh_ns, where='mid', label="Isern 2019 (NS)")

lam_age_width = np.diff(np.append(_lam_steps[0]._path._vertices[:,0][::2],_lam_steps[0]._path._vertices[:,0][-1]))
rowell_age_width = np.diff(np.append(_rowell_steps[0]._path._vertices[:,0][::2],_rowell_steps[0]._path._vertices[:,0][-1]))
mor_age_width = np.diff(np.append(_mor_steps[0]._path._vertices[:,0][::2],_mor_steps[0]._path._vertices[:,0][-1]))
isern_s_age_width = np.diff(np.append(_isern_s_steps[0]._path._vertices[:,0][::2],_isern_s_steps[0]._path._vertices[:,0][-1]))
isern_ns_age_width = np.diff(np.append(_isern_ns_steps[0]._path._vertices[:,0][::2],_isern_ns_steps[0]._path._vertices[:,0][-1]))

lam_sfh /= np.sum(lam_sfh * lam_age_width)
rowell_sfh /= np.sum(rowell_sfh * rowell_age_width)
mor_sfh /= np.sum(mor_sfh * mor_age_width)
isern_sfh_s /= np.sum(isern_sfh_s * isern_s_age_width)
isern_sfh_ns /= np.sum(isern_sfh_ns * isern_ns_age_width)

plt.figure(1)
plt.clf()
plt.step(lam_age, lam_sfh, where='mid', label="This work", color='black', lw=2)
plt.step(rowell_age, rowell_sfh, where='mid', label="Rowell 2013")
plt.step(mor_age, mor_sfh, where='mid', label="Mor et al. 2019")
plt.step(isern_age_s, isern_sfh_s, where='mid', label="Isern 2019")
plt.step(isern_age_ns, isern_sfh_ns, where='mid', label="Isern 2019 (NS)")
plt.legend()
plt.xlabel('Lookback time / Gyr')
plt.ylabel('Relative Star Formation Rate')
plt.savefig('compare_sfh.png')
