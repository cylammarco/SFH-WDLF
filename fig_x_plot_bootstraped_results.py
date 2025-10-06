import numpy as np
from matplotlib import pyplot as plt
plt.ion()


figure_folder = "SFH-WDLF-article/figures"
boostrap_folder = "SFH-WDLF-article/bootstrap_sample_folder"

sfh_list = []
wdlf_list = []
sfh_20pc_list = []
wdlf_20pc_list = []

for i in range(1000):
    sfh = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_sfh_sample_{i}.csv").T[1]
    wdlf = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_reconstructed_wdlf_sample_{i}.csv").T[1]
    sfh_20pc = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_sfh_sample_{i}.csv").T[4]
    wdlf_20pc = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_reconstructed_wdlf_sample_{i}.csv").T[3]
    sfh_list.append(sfh)
    wdlf_list.append(wdlf)
    sfh_20pc_list.append(sfh_20pc)
    wdlf_20pc_list.append(wdlf_20pc)


sfh_age = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_sfh_sample_{i}.csv").T[0]
sfh_mean = np.mean(sfh_list, axis=0)
sfh_stdev = np.std(sfh_list, axis=0)
sfh_20pc_mean = np.mean(sfh_20pc_list, axis=0)
sfh_20pc_stdev = np.std(sfh_20pc_list, axis=0)
wdlf_mag = np.genfromtxt(f"{boostrap_folder}/sample_{i}/gcns_reconstructed_wdlf_sample_{i}.csv").T[0]
wdlf_mean = np.mean(wdlf_list, axis=0)
wdlf_stdev = np.std(wdlf_list, axis=0)
wdlf_20pc_mean = np.mean(wdlf_20pc_list, axis=0)
wdlf_20pc_stdev = np.std(wdlf_20pc_list, axis=0)

"""
plt.figure(1)
plt.clf()
plt.plot(wdlf_mag, wdlf_mean)
plt.fill_between(wdlf_mag, wdlf_mean - wdlf_stdev, wdlf_mean + wdlf_stdev, alpha=0.5)
plt.yscale("log")
"""

# Calculate signal-to-noise ratio (SNR)
sfh_snr = sfh_mean / sfh_stdev

# Create a 2:1 vertically stacked subplot
fig = plt.figure(2, figsize=(8, 6))
plt.clf()
gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0)

# Top subplot: current figure
ax1 = fig.add_subplot(gs[0, 0])
ax1.step(
    sfh_age,
    sfh_mean,
    where="mid",
)
ax1.step(
    sfh_age,
    sfh_mean,
    where="mid",
)

for i in range(1000):
    ax1.step(
        sfh_age,
        sfh_list[i],
        where="mid",
        color="lightgrey",
        alpha=0.01
    )

ax1.fill_between(
    sfh_age,
    sfh_mean - sfh_stdev,
    sfh_mean + sfh_stdev,
    alpha=0.5,
    step="mid",
    color="lightgrey",
)
ax1.set_ylabel("SFH Mean")
ax1.set_ylim(0.0, 0.2)
ax1.tick_params(labelbottom=False)

# Bottom subplot: signal-to-noise ratio
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(sfh_age, sfh_snr, color="orange")
ax2.set_ylabel("SNR")
ax2.set_xlabel("Age")

plt.figure(3)
plt.clf()
plt.plot(wdlf_mag, wdlf_20pc_mean)
plt.fill_between(wdlf_mag, wdlf_20pc_mean - wdlf_20pc_stdev, wdlf_20pc_mean + wdlf_20pc_stdev, alpha=0.5)
plt.yscale("log")

# Calculate signal-to-noise ratio (SNR)
sfh_20pc_snr = sfh_20pc_mean / sfh_20pc_stdev

# Create a 2:1 vertically stacked subplot
fig = plt.figure(4, figsize=(8, 6))
plt.clf()
gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0)

# Top subplot: current figure
ax3 = fig.add_subplot(gs[0, 0])
ax3.step(
    sfh_age,
    sfh_20pc_mean,
    where="mid",
)

ax3.fill_between(
    sfh_age,
    sfh_20pc_mean - sfh_20pc_stdev,
    sfh_20pc_mean + sfh_20pc_stdev,
    alpha=0.5,
    step="mid",
    color="lightgrey",
)
ax3.set_ylabel("SFH Mean")
ax3.tick_params(labelbottom=False)

# Bottom subplot: signal-to-noise ratio
ax4 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax4.plot(sfh_age, sfh_20pc_snr, color="orange")
ax4.set_ylabel("SNR")
ax4.set_xlabel("Age")
