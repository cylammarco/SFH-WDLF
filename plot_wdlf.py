import numpy as np
from matplotlib.pyplot import *
ion()

data = np.load('pubgcnswdlf-h366pc-dpdf-samples-hp5-maglim80-vgen-grp-rdc-srt.npz')['data']
n_bin = 200

h_gen, b = np.histogram(data['Mbol'], bins=n_bin, range=(0, 20), weights=0.01/data['Vgen'])
e_gen, _ = np.histogram(data['Mbol'], bins=n_bin, range=(0, 20), weights=0.01/data['Vgen']**2.)

bin_size = b[1]-b[0]

figure(1, figsize=(10, 6))
clf()
errorbar(b[1:], h_gen / bin_size, yerr=[e_gen**0.5/ bin_size, e_gen**0.5/ bin_size], fmt="+", markersize=5)
yscale('log')

xlabel('Mbol / mag')
ylabel(r'N / pc$^3$ / Mbol')
xlim(5.0, 18.0)
ylim(1e-6, 5e-3)
grid()
tight_layout()
