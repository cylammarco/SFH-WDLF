import numpy as np

rng = np.random.RandomState(846513)


# These are the same as in Roberts+25
# stdev on the exponent
sigma_imf = 0.1  # El-Badry+18; Cunningham+24
# percentage of the ms lifetime
sigma_ms = 0.048  # Hurley+00
# percentage of the intial-final mass relation
sigma_m_ifmr = 0.004  # Catalan+08 ~3%
sigma_c_ifmr = (
    0.011  # Catalan+08 ~3%; Hollands+24 has 1-10% depending on the mass, less than 4% in the range of 1-5 M_sun
)
# perentage of the cooling time
sigma_cooling = 0.06  # CUkanovaite+23; Cunningham+24; Pathak+24

imf_samples = rng.normal(loc=0.0, scale=sigma_imf, size=1000)
ms_samples = rng.normal(loc=0.0, scale=sigma_ms, size=1000)
ifmr_m_samples = rng.normal(loc=0.0, scale=sigma_m_ifmr, size=1000)
ifmr_c_samples = rng.normal(loc=0.0, scale=sigma_c_ifmr, size=1000)
cooling_samples = rng.normal(loc=0.0, scale=sigma_cooling, size=1000)

bootstrap_table = np.column_stack(
    (
        imf_samples,
        ms_samples,
        ifmr_m_samples,
        ifmr_c_samples,
        cooling_samples,
    )
)
np.save("bootstrap_table.npy", bootstrap_table)
