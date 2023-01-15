import os

from matplotlib import pyplot as plt
import numpy as np

file_body = "montreal_co_da_20_K01_PARSECz0014_C08_"

data1 = np.loadtxt(
    os.path.join("output", file_body + "0.10_Mbol.csv"), delimiter=","
)

data2 = np.loadtxt(
    os.path.join("output", file_body + "0.20_Mbol.csv"), delimiter=","
)
data3 = np.loadtxt(
    os.path.join("output", file_body + "0.30_Mbol.csv"), delimiter=","
)

data4 = np.loadtxt(
    os.path.join("output", file_body + "0.40_Mbol.csv"), delimiter=","
)

data5 = np.loadtxt(
    os.path.join("output", file_body + "0.50_Mbol.csv"), delimiter=","
)

data6 = np.loadtxt(
    os.path.join("output", file_body + "0.60_Mbol.csv"), delimiter=","
)

data7 = np.loadtxt(
    os.path.join("output", file_body + "0.70_Mbol.csv"), delimiter=","
)

data8 = np.loadtxt(
    os.path.join("output", file_body + "0.80_Mbol.csv"), delimiter=","
)

data9 = np.loadtxt(
    os.path.join("output", file_body + "0.90_Mbol.csv"), delimiter=","
)

data10 = np.loadtxt(
    os.path.join("output", file_body + "1.00_Mbol.csv"), delimiter=","
)

data20 = np.loadtxt(
    os.path.join("output", file_body + "2.00_Mbol.csv"), delimiter=","
)

data50 = np.loadtxt(
    os.path.join("output", file_body + "5.00_Mbol.csv"), delimiter=","
)

data100 = np.loadtxt(
    os.path.join("output", file_body + "10.00_Mbol.csv"), delimiter=","
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


data_00_05 = np.zeros_like(data1[:, 1])
for i in np.arange(0.01, 0.50, 0.01):
    data_00_05 += np.loadtxt(
        os.path.join("output", file_body + f"{i:.2f}_Mbol.csv"), delimiter=","
    )[:, 1]

data_05_10 = np.zeros_like(data1[:, 1])
for i in np.arange(0.50, 1.00, 0.01):
    data_05_10 += np.loadtxt(
        os.path.join("output", file_body + f"{i:.2f}_Mbol.csv"), delimiter=","
    )[:, 1]

data_20_30 = np.zeros_like(data1[:, 1])
for i in np.arange(2.01, 3.00, 0.01):
    data_20_30 += np.loadtxt(
        os.path.join("output", file_body + f"{i:.2f}_Mbol.csv"), delimiter=","
    )[:, 1]

data_40_50 = np.zeros_like(data1[:, 1])
for i in np.arange(4.01, 5.00, 0.01):
    data_40_50 += np.loadtxt(
        os.path.join("output", file_body + f"{i:.2f}_Mbol.csv"), delimiter=","
    )[:, 1]

data_90_100 = np.zeros_like(data1[:, 1])
for i in np.arange(9.01, 10.00, 0.01):
    data_90_100 += np.loadtxt(
        os.path.join("output", file_body + f"{i:.2f}_Mbol.csv"), delimiter=","
    )[:, 1]


plt.figure(2, figsize=(10, 6))
plt.clf()
plt.plot(
    data1[:, 0],
    data_00_05,
    label="0.0-0.5 Gyr",
)
plt.plot(
    data6[:, 0],
    data_05_10,
    label="0.5-1.0 Gyr",
)
plt.plot(
    data1[:, 0],
    data_00_05 + data_05_10,
    label="0.0-1.0 Gyr",
)
plt.plot(
    data1[:, 0],
    data_20_30,
    label="2.0-3.0 Gyr",
)
plt.plot(
    data1[:, 0],
    data_40_50,
    label="4.0-5.0 Gyr",
)
plt.plot(
    data1[:, 0],
    data_90_100,
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
