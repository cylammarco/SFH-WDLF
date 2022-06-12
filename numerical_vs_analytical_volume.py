import time
import numpy as np
from scipy.integrate import quad


# Analytical solution for integrating
#   /int_0^D [ r**2. * np.exp( -np.abs(r * sinb + z) / H) ] dr
# with constant sinb, z & H
def volume(D, sinb, z, H):
    xi = H / sinb
    exponent = (D * sinb + z) / H
    if exponent >= 0:
        v = -np.exp(-exponent) * (
            2 * xi**3.0 + 2 * D * xi**2.0 + D**2.0 * xi
        ) + 2.0 * xi**3.0 * np.exp(-z / H)
    else:
        v = np.exp(exponent) * (
            2 * xi**3.0 - 2 * D * xi**2.0 + D**2.0 * xi
        ) - 2.0 * xi**3.0 * np.exp(z / H)
    return v


# Numerical integrand
def integrand(D, sinb, z, H):
    return D**2.0 * np.exp(-np.abs(D * sinb + z) / H)


analytical = np.zeros((100, 100, 40, 25))
numerical = np.zeros((100, 100, 40, 25))


time1 = time.time()
# Test different distance
for i, D in enumerate(np.linspace(10, 1100, 100)):
    # Test different sinb
    for j, sinb in enumerate(np.linspace(-np.pi / 2, np.pi / 2, 100)):
        # Test different z
        for z in np.arange(-20, 20):
            # Test different scaleheight
            for k, H in enumerate(np.linspace(100, 2500, 25)):
                analytical[i, j, z, k] = volume(D, sinb, z, H)


time2 = time.time()
# Test different distance
for i, D in enumerate(np.linspace(10, 1100, 100)):
    # Test different sinb
    for j, sinb in enumerate(np.linspace(-np.pi / 2, np.pi / 2, 100)):
        # Test different z
        for z in np.arange(-20, 20):
            # Test different scaleheight
            for k, H in enumerate(np.linspace(100, 2500, 25)):
                numerical[i, j, z, k] = quad(
                    integrand, 0, D, args=(sinb, z, H)
                )[0]


time3 = time.time()

print(
    "Mean time taken to compute the analytical solution: "
    + str((time2 - time1) / 100.0 / 10.0 / 10.0 / 25.0)
    + " s"
)
print(
    "Mean time taken to compute the numerical solution: "
    + str((time3 - time2) / 100.0 / 10.0 / 10.0 / 25.0)
    + " s"
)
print(
    "Analytical is "
    + str((time3 - time2) / (time2 - time1))
    + " times faster."
)

print(
    "Minimum difference between numerical and analystical solution is "
    + str(abs(np.min(analytical / numerical) - 1.0) * 100.0)
    + " %"
)
print(
    "Maximum difference between numerical and analystical solution is "
    + str(abs(np.max(analytical / numerical) - 1.0) * 100.0)
    + " %"
)
print(
    "Mean difference between numerical and analystical solution is "
    + str(abs(np.mean(analytical / numerical) - 1.0) * 100.0)
    + " %"
)
print(
    "Median difference between numerical and analystical solution is "
    + str(abs(np.median(analytical / numerical) - 1.0) * 100.0)
    + " %"
)
