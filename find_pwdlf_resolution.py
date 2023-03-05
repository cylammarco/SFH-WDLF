from contextlib import _AsyncGeneratorContextManager
import os

from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate


data = []
age_list_1 = np.arange(0.049, 0.100, 0.001)
age_list_2 = np.arange(0.100, 0.350, 0.005)
age_list_3 = np.arange(0.35, 15.01, 0.01)
age_list_3dp = np.concatenate((age_list_1, age_list_2))
age_list_2dp = age_list_3

age = np.concatenate((age_list_3dp, age_list_2dp))

for i in age_list_3dp:
    data.append(
        np.loadtxt(
            os.path.join(
                "output",
                f"montreal_co_da_20_K01_PARSECz0014_C08_{i:.3f}_Mbol.csv",
            ),
            delimiter=",",
        )
    )

for i in age_list_2dp:
    data.append(
        np.loadtxt(
            os.path.join(
                "output",
                f"montreal_co_da_20_K01_PARSECz0014_C08_{i:.2f}_Mbol.csv",
            ),
            delimiter=",",
        )
    )


mag = data[0][:, 0]

mag_at_peak_density = np.zeros_like(age)
for i, d in enumerate(data):
    mag_at_peak_density[i] = mag[np.argmax(d[:, 1])]


mag_resolution_itp = interpolate.UnivariateSpline(
    age, mag_at_peak_density, s=len(age) / 150, k=5
)


plt.figure(1, figsize=(8, 6))
plt.clf()
plt.plot(age, mag_at_peak_density, label="Measured")
plt.plot(age, mag_resolution_itp(age), label="Fitted")
plt.xlabel("age")
plt.ylabel("magnitude at peak density")
plt.xscale('log')
plt.tight_layout()
plt.savefig("age_peak_density.png")


plt.figure(2, figsize=(8, 6))
plt.clf()
plt.plot(
    age[1:],
    mag_resolution_itp(age[1:]) - mag_resolution_itp(age[:-1]),
    label="Fitted",
)
plt.xlabel("age")
plt.ylabel("magnitude resolution at peak density")
plt.xscale('log')
plt.tight_layout()
plt.savefig("age_peak_density_resolution.png")


# Nyquist sampling for resolving 2 gaussian peaks -> 2.355 sigma.
# For each WDLF bin of size 0.2, 0.5 mag, we want the resolving power of
# at least 0.2 and 0.5 / 2.355 = 0.0849257 and 0.2123142 mag

bin_02 = []
bin_02_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
bin_02_pwdlf = np.zeros_like(age[1:]).astype('int')
j = 0
bin_number = 0
stop = False
for i, a in enumerate(age):
    if i < j:
        continue
    start = mag_resolution_itp(a)
    end = start
    j = i
    carry_on = True
    while carry_on:
        tmp = mag_resolution_itp(age[j+1]) - mag_resolution_itp(age[j])
        print(j, end + tmp - start)
        if end + tmp - start < 0.2:
            end = end + tmp
            bin_02_pwdlf[j] = bin_number
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    if stop:
        break
    bin_02.append(end)
    bin_02_idx.append(i)


bin_05 = []
bin_05_idx = []
# to group the constituent pwdlfs into the optimal set of pwdlfs
bin_05_pwdlf = np.zeros_like(age[1:]).astype('int')
j = 0
bin_number = 0
stop = False
for i, a in enumerate(age):
    if i < j:
        continue
    start = mag_resolution_itp(a)
    end = start
    j = i
    carry_on = True
    while carry_on:
        tmp = mag_resolution_itp(age[j+1]) - mag_resolution_itp(age[j])
        if end + tmp - start < 0.5:
            end = end + tmp
            bin_05_pwdlf[j] = bin_number
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
            bin_number += 1
    if stop:
        break
    bin_05.append(end)
    bin_05_idx.append(i)


resolution_02 = np.array(bin_02[1:]) - np.array(bin_02[:-1])
resolution_05 = np.array(bin_05[1:]) - np.array(bin_05[:-1])

plt.figure(3, figsize=(8, 6))
plt.scatter(bin_02[:-1], resolution_02, label='0.2 mag bin', s=5)
plt.scatter(bin_05[:-1], resolution_05, label='0.5 mag bin', s=5)
plt.legend()
plt.ylabel('magnitude resolution')
plt.xlabel('magnitude')
plt.tight_layout()
plt.savefig('magnitude_resoltuion.png')


# save the pdwdlf mapping
np.save('pwdlf_bin_02_mapping.npy', bin_02_pwdlf)
np.save('pwdlf_bin_05_mapping.npy', bin_05_pwdlf)
