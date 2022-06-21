import os

from matplotlib import pyplot as plt
import numpy as np

file_extension = (
    "_burst_C16_C08_montreal_co_da_20_montreal_co_da_20_"
    + "montreal_co_da_20.csv"
)

data1 = np.loadtxt(
    os.path.join("output", os.sep, "0.10Gyr" + file_extension, delimiter=",")
)
data2 = np.loadtxt(
    os.path.join("output", os.sep, "0.20Gyr" + file_extension, delimiter=",")
)
data3 = np.loadtxt(
    os.path.join("output", os.sep, "0.30Gyr" + file_extension, delimiter=",")
)
data4 = np.loadtxt(
    os.path.join("output", os.sep, "0.40Gyr" + file_extension, delimiter=",")
)
data5 = np.loadtxt(
    os.path.join("output", os.sep, "0.50Gyr" + file_extension, delimiter=",")
)
data6 = np.loadtxt(
    os.path.join("output", os.sep, "0.60Gyr" + file_extension, delimiter=",")
)
data7 = np.loadtxt(
    os.path.join("output", os.sep, "0.70Gyr" + file_extension, delimiter=",")
)
data8 = np.loadtxt(
    os.path.join("output", os.sep, "0.80Gyr" + file_extension, delimiter=",")
)
data9 = np.loadtxt(
    os.path.join("output", os.sep, "0.90Gyr" + file_extension, delimiter=",")
)
data10 = np.loadtxt(
    os.path.join("output", os.sep, "1.00Gyr" + file_extension, delimiter=",")
)
data20 = np.loadtxt(
    os.path.join("output", os.sep, "2.00Gyr" + file_extension, delimiter=",")
)
data50 = np.loadtxt(
    os.path.join("output", os.sep, "5.00Gyr" + file_extension, delimiter=",")
)
data100 = np.loadtxt(
    os.path.join("output", os.sep, "10.00Gyr" + file_extension, delimiter=",")
)

plt.figure(1, figsize=(10, 6))
plt.clf()
plt.plot(data1[:, 0], data1[:, 1] / np.sum(data1[:, 1]), label="0.1 Gyr")
plt.plot(data2[:, 0], data2[:, 1] / np.sum(data2[:, 1]), label="0.2 Gyr")
plt.plot(data3[:, 0], data3[:, 1] / np.sum(data3[:, 1]), label="0.3 Gyr")
plt.plot(data4[:, 0], data4[:, 1] / np.sum(data4[:, 1]), label="0.4 Gyr")
plt.plot(data5[:, 0], data5[:, 1] / np.sum(data5[:, 1]), label="0.5 Gyr")
plt.plot(data6[:, 0], data6[:, 1] / np.sum(data6[:, 1]), label="0.6 Gyr")
plt.plot(data7[:, 0], data7[:, 1] / np.sum(data7[:, 1]), label="0.7 Gyr")
plt.plot(data8[:, 0], data8[:, 1] / np.sum(data8[:, 1]), label="0.8 Gyr")
plt.plot(data9[:, 0], data9[:, 1] / np.sum(data9[:, 1]), label="0.9 Gyr")
plt.plot(data10[:, 0], data10[:, 1] / np.sum(data10[:, 1]), label="1.0 Gyr")
plt.plot(data20[:, 0], data20[:, 1] / np.sum(data20[:, 1]), label="2.0 Gyr")
plt.plot(data50[:, 0], data50[:, 1] / np.sum(data50[:, 1]), label="5.0 Gyr")
plt.plot(
    data100[:, 0], data100[:, 1] / np.sum(data100[:, 1]), label="10.0 Gyr"
)
plt.yscale("log")

plt.xlabel(r"M$_{\mathrm{bol}}$")
plt.ylabel("Probability Distribution")
plt.ylim(
    bottom=1e-6,
)
plt.xlim(0, 20)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("burst_partial_wdlf.png")

data21 = np.loadtxt(
    os.path.join("output", os.sep, "2.10Gyr" + file_extension, delimiter=",")
)
data22 = np.loadtxt(
    os.path.join("output", os.sep, "2.20Gyr" + file_extension, delimiter=",")
)
data23 = np.loadtxt(
    os.path.join("output", os.sep, "2.30Gyr" + file_extension, delimiter=",")
)
data24 = np.loadtxt(
    os.path.join("output", os.sep, "2.40Gyr" + file_extension, delimiter=",")
)
data25 = np.loadtxt(
    os.path.join("output", os.sep, "2.50Gyr" + file_extension, delimiter=",")
)
data26 = np.loadtxt(
    os.path.join("output", os.sep, "2.60Gyr" + file_extension, delimiter=",")
)
data27 = np.loadtxt(
    os.path.join("output", os.sep, "2.70Gyr" + file_extension, delimiter=",")
)
data28 = np.loadtxt(
    os.path.join("output", os.sep, "2.80Gyr" + file_extension, delimiter=",")
)
data29 = np.loadtxt(
    os.path.join("output", os.sep, "2.90Gyr" + file_extension, delimiter=",")
)
data30 = np.loadtxt(
    os.path.join("output", os.sep, "3.00Gyr" + file_extension, delimiter=",")
)

data41 = np.loadtxt(
    os.path.join("output", os.sep, "4.10Gyr" + file_extension, delimiter=",")
)
data42 = np.loadtxt(
    os.path.join("output", os.sep, "4.20Gyr" + file_extension, delimiter=",")
)
data43 = np.loadtxt(
    os.path.join("output", os.sep, "4.30Gyr" + file_extension, delimiter=",")
)
data44 = np.loadtxt(
    os.path.join("output", os.sep, "4.40Gyr" + file_extension, delimiter=",")
)
data45 = np.loadtxt(
    os.path.join("output", os.sep, "4.50Gyr" + file_extension, delimiter=",")
)
data46 = np.loadtxt(
    os.path.join("output", os.sep, "4.60Gyr" + file_extension, delimiter=",")
)
data47 = np.loadtxt(
    os.path.join("output", os.sep, "4.70Gyr" + file_extension, delimiter=",")
)
data48 = np.loadtxt(
    os.path.join("output", os.sep, "4.80Gyr" + file_extension, delimiter=",")
)
data49 = np.loadtxt(
    os.path.join("output", os.sep, "4.90Gyr" + file_extension, delimiter=",")
)
data50 = np.loadtxt(
    os.path.join("output", os.sep, "5.00Gyr" + file_extension, delimiter=",")
)

data91 = np.loadtxt(
    os.path.join("output", os.sep, "9.10Gyr" + file_extension, delimiter=",")
)
data92 = np.loadtxt(
    os.path.join("output", os.sep, "9.20Gyr" + file_extension, delimiter=",")
)
data93 = np.loadtxt(
    os.path.join("output", os.sep, "9.30Gyr" + file_extension, delimiter=",")
)
data94 = np.loadtxt(
    os.path.join("output", os.sep, "9.40Gyr" + file_extension, delimiter=",")
)
data95 = np.loadtxt(
    os.path.join("output", os.sep, "9.50Gyr" + file_extension, delimiter=",")
)
data96 = np.loadtxt(
    os.path.join("output", os.sep, "9.60Gyr" + file_extension, delimiter=",")
)
data97 = np.loadtxt(
    os.path.join("output", os.sep, "9.70Gyr" + file_extension, delimiter=",")
)
data98 = np.loadtxt(
    os.path.join("output", os.sep, "9.80Gyr" + file_extension, delimiter=",")
)
data99 = np.loadtxt(
    os.path.join("output", os.sep, "9.90Gyr" + file_extension, delimiter=",")
)
data100 = np.loadtxt(
    os.path.join("output", os.sep, "10.00Gyr" + file_extension, delimiter=",")
)

plt.figure(2, figsize=(10, 6))
plt.clf()
plt.plot(
    data1[:, 0],
    data1[:, 1] + data2[:, 1] + data3[:, 1] + data4[:, 1] + data5[:, 1],
    label="0.0-0.5 Gyr",
)
plt.plot(
    data6[:, 0],
    data6[:, 1] + data7[:, 1] + data8[:, 1] + data9[:, 1] + data10[:, 1],
    label="0.5-1.0 Gyr",
)
plt.plot(
    data1[:, 0],
    data1[:, 1]
    + data2[:, 1]
    + data3[:, 1]
    + data4[:, 1]
    + data5[:, 1]
    + data6[:, 1]
    + data7[:, 1]
    + data8[:, 1]
    + data9[:, 1]
    + data10[:, 1],
    label="0.0-1.0 Gyr",
)
plt.plot(
    data1[:, 0],
    data21[:, 1]
    + data22[:, 1]
    + data23[:, 1]
    + data24[:, 1]
    + data25[:, 1]
    + data26[:, 1]
    + data27[:, 1]
    + data28[:, 1]
    + data29[:, 1]
    + data30[:, 1],
    label="2.0-3.0 Gyr",
)
plt.plot(
    data1[:, 0],
    data41[:, 1]
    + data42[:, 1]
    + data43[:, 1]
    + data44[:, 1]
    + data45[:, 1]
    + data46[:, 1]
    + data47[:, 1]
    + data48[:, 1]
    + data49[:, 1]
    + data50[:, 1],
    label="4.0-5.0 Gyr",
)
plt.plot(
    data1[:, 0],
    data91[:, 1]
    + data92[:, 1]
    + data93[:, 1]
    + data94[:, 1]
    + data95[:, 1]
    + data96[:, 1]
    + data97[:, 1]
    + data98[:, 1]
    + data99[:, 1]
    + data100[:, 1],
    label="9.0-10.0 Gyr",
)
plt.yscale("log")

plt.xlabel(r"M$_{\mathrm{bol}}$")
plt.ylabel("Probability Distribution")
plt.ylim(bottom=1e4)
plt.xlim(0, 20.0)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("burst_partial_wdlf_added.png")
