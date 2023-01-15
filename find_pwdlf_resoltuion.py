from contextlib import _AsyncGeneratorContextManager
import os

from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate


data = []
age = np.arange(0.01, 15.00, 0.01)
for i in age:
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
    age, mag_at_peak_density, s=len(age) / 100
)


plt.figure(1)
plt.plot(age, mag_at_peak_density, label="Measured")
plt.plot(age, mag_resolution_itp(age), label="Fitted")
plt.xlabel("age")
plt.ylabel("magnitude at peak density")
plt.savefig("age_peak_density.png")


plt.figure(2)
plt.plot(
    age[31:],
    mag_resolution_itp(age[31:]) - mag_resolution_itp(age[30:-1]),
    label="Fitted",
)
plt.xlabel("age")
plt.ylabel("magnitude resolution at peak density")
plt.savefig("age_peak_density_resolution.png")


# Nyquist sampling for resolving 2 gaussian peaks -> 2.355 sigma.
# For each WDLF bin of size 0.2, 0.5 mag, we want the resolving power of
# at least 0.2 and 0.5 / 2.355 = 0.0849257 and 0.2123142 mag

bin_02 = []
bin_02_idx = []
j = 0
stop = False
for i, a in enumerate(age):
    if i < j:
        continue
    print(i)
    start = mag_resolution_itp(a)
    end = start
    j = i
    carry_on = True
    while carry_on:
        tmp = mag_resolution_itp(age[j+1]) - mag_resolution_itp(age[j])
        print(j, end, tmp, end+tmp)
        if end + tmp - start < 0.0849257:
            end = end + tmp
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
    if stop:
        break
    bin_02.append(end)
    bin_02_idx.append(i)

print(bin_02)
print(bin_02_idx)
print(age[bin_02_idx])


bin_05 = []
bin_05_idx = []
j = 0
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
        if end + tmp - start < 0.2123142:
            end = end + tmp
            j += 1
            if j >= len(age) - 1:
                carry_on = False
                stop = True
        else:
            carry_on = False
    if stop:
        break
    bin_05.append(end)
    bin_05_idx.append(i)

print(bin_05)
print(bin_05_idx)
print(age[bin_05_idx])

resolution_02 = np.array(bin_02[1:]) - np.array(bin_02[:-1])
resolution_05 = np.array(bin_05[1:]) - np.array(bin_05[:-1])

plt.figure(3)
plt.scatter(bin_02[:-1], resolution_02, label='0.2 mag bin', s=5)
plt.scatter(bin_05[:-1], resolution_05, label='0.5 mag bin', s=5)
plt.ylabel('magnitude resolution')
plt.xlabel('magnitude')
plt.savefig('magnitude_resoltuion.png')

