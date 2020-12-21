import emcee
import corner
import numpy as np
from scipy import interpolate as itp
from matplotlib.pyplot import *
from WDLFBuilder import theoretical_lf
ion()



def log_prior(theta):
    if (theta <= 0.0).any():
        return -np.inf
    else:
        return 0.


def log_probability(rel_norm, obs, err, model_list):
    if not np.isfinite(log_prior(rel_norm)):
        return -np.inf
    model = np.nansum(rel_norm[:, None] * model_list, axis=0)
    # Assume Poisson noise in both model and observed WDLFs
    err[err<=0.] = 1e-10
    log_likelihood = -np.nansum(((model-obs)/err)**2.)
    if np.isfinite(log_likelihood):
        return log_likelihood
    else:
        return -np.inf


# Collect all the partial WDLFs, in chunks of 0.5 Gyr
partial_wdlf = []
age = np.arange(2.0, 10.0, 0.5)
for i in age:
    filename1 = '{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.csv'.format(i)
    filename2 = '{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.csv'.format(i+0.1)
    filename3 = '{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.csv'.format(i+0.2)
    filename4 = '{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.csv'.format(i+0.3)
    filename5 = '{0:.2f}Gyr_burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.csv'.format(i+0.4)
    n_temp = np.loadtxt(filename1, delimiter=',')[:,1]
    n_temp += np.loadtxt(filename2, delimiter=',')[:,1]
    n_temp += np.loadtxt(filename3, delimiter=',')[:,1]
    n_temp += np.loadtxt(filename4, delimiter=',')[:,1]
    n_temp += np.loadtxt(filename5, delimiter=',')[:,1]
    #n_temp /= np.max(n_temp)
    partial_wdlf.append(n_temp.copy())
    n_temp = None


# Get the magnitude
mag = np.loadtxt('0.10Gyr_burst_C16_C08_montreal_co_da_20_montreal_co_da_20_montreal_co_da_20.csv', delimiter=',')[:,0]

# Generate a WDLF with 1 Gyr burst at 2-3 Gyr and 5-6 Gyr
obs_wdlf = partial_wdlf[2] + partial_wdlf[3] + partial_wdlf[6] + partial_wdlf[7]

obs_wdlf_err = np.sqrt(obs_wdlf)
obs_wdlf_err[obs_wdlf<=0.] = np.sqrt(min(obs_wdlf[obs_wdlf>0.]))

nwalkers = 100
ndim = len(partial_wdlf)

rel_norm = np.random.rand(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(obs_wdlf, obs_wdlf_err, partial_wdlf))
sampler.run_mcmc(rel_norm, 100000, progress=True)

flat_samples = sampler.get_chain(discard=10000, thin=15, flat=True)

#fig = corner.corner(flat_samples)

solution = np.zeros(ndim)
for i in range(ndim):
    solution[i] = np.percentile(flat_samples[:, i], [50.])

real_sfh = np.zeros_like(age)
real_sfh[2] = 1.
real_sfh[3] = 1.
real_sfh[6] = 1.
real_sfh[7] = 1.

figure(1)
scatter(age, solution, label='MCMC')
scatter(age, real_sfh, label='input')
legend()
grid()
xlabel('Lookback time / Gyr') 
ylabel('Relative Star Formation Rate') 
tight_layout()
savefig('two_bursts_sfh.png')

sfh_model = itp.interp1d(age, solution)
wdlf = theoretical_lf.WDLF()
wdlf.set_ifmr_model('C08')
wdlf.compute_cooling_age_interpolator()

wdlf.set_sfr_model(mode='manual', age=max(age), sfr_model=sfh_model)
_, recomputed_wdlf = wdlf.compute_density(Mag=mag,
                         normed=True,
                         save_csv=False)

figure(2)
plot(mag, np.log10(obs_wdlf), label='Input WDLF')
plot(mag, np.log10(recomputed_wdlf), label='Extracted WDLF')
xlabel(r'M$_bol$ / mag')
ylabel('log(arbitrary number density)')
xlim(7.5, 20)
ylim(-5, 0)
legend()
grid()
tight_layout()
savefig('two_bursts_wdlf.png')


# Decay SFH profile
nwalkers = 100
ndim = len(partial_wdlf)

rel_norm = np.random.rand(nwalkers, ndim)

wdlf = theoretical_lf.WDLF()

wdlf.compute_cooling_age_interpolator()
wdlf.set_sfr_model(mode='decay', age=6E9) 

_, obs_wdlf2 = wdlf.compute_density(Mag=mag, n_points=10)
obs_wdlf_err2 = np.sqrt(obs_wdlf2)
obs_wdlf_err2[obs_wdlf2<=0.] = np.sqrt(min(obs_wdlf2[obs_wdlf2>0.]))

sampler2 = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(obs_wdlf2, obs_wdlf_err2, partial_wdlf))
sampler2.run_mcmc(rel_norm, 100000, progress=True)

flat_samples2 = sampler2.get_chain(discard=10000, thin=15, flat=True)

solution2 = np.zeros(ndim)
for i in range(ndim):
    solution2[i] = np.percentile(flat_samples2[:, i], [50.])


figure(3)
scatter(age, solution2/max(solution2), label='MCMC')
plot(np.linspace(0, 10e9, 1000)/1e9, wdlf.sfr(np.linspace(0, 10e9, 1000)), label='input')
xlim(0, 10.)
legend()
grid()
xlabel('Lookback time / Gyr') 
ylabel('Relative Star Formation Rate') 
tight_layout()
savefig('decay_sfh.png')


sfh_model2 = itp.interp1d(age*1E9, solution2)
wdlf2 = theoretical_lf.WDLF()
wdlf2.set_ifmr_model('C08')
wdlf2.compute_cooling_age_interpolator()

wdlf2.set_sfr_model(mode='manual', age=6E9, sfr_model=sfh_model2)
_, recomputed_wdlf2 = wdlf.compute_density(Mag=mag,
                         normed=True)

figure(4)
plot(mag, np.log10(obs_wdlf2), label='Input WDLF')
plot(mag, np.log10(recomputed_wdlf2), label='Extracted WDLF')
xlabel(r'M$_bol$ / mag')
ylabel('log(arbitrary number density)')
xlim(7.5, 20)
ylim(-5, 0)
legend()
grid()
tight_layout()
savefig('decay_sfh_wdlf.png')



