import math

import matplotlib.pyplot as plt
import numpy as np

from constants import *

a_0 = 4 * np.pi * eps_0 * hbar**2 / (m_e * e**2)


def legendre(l, m, x):
    if l == 0:
        return x / x
    if l == 1:
        if m == 0:
            return x
        if m == 1:
            return -np.sqrt(1 - x**2)
        if m == -1:
            return 0.5 * np.sqrt(1 - x**2)
    else:
        return 1 / (l - m) * (x * (2 * l - 1) * legendre(l - 1, m, x) - (l + m - 1) * legendre(l - 2, m, x))


def spherical_harmonics(l, m, theta, phi):
    return (
        1
        / np.sqrt(2 * np.pi)
        * np.sqrt((2 * l + 1) / 2 * math.factorial(l - m) / math.factorial(l + m))
        * legendre(l, m, np.cos(theta))
        * np.exp(1j * m * phi)
    )


def laguerre(n, l, x):
    if n == 0:
        return x / x
    if n == 1:
        return -x + l + 1
    if n == 2:
        return 0.5 * (x**2 - 2 * (l + 2) * x + (l + 1) * (l + 2))
    if n == 3:
        return 1 / 6 * (-(x**3) + 3 * (l + 3) * x**2 - 3 * (l + 2) * (l + 3) * x + (l + 1) * (l + 2) * (l + 3))
    else:
        return 1 / n * ((2 * (n - 1) + 1 + l - x) * laguerre(n - 1, l, x) - (n - 1 + l) * laguerre(n - 2, l, x))


def radial_function(n, l, r):
    r_scaled = r  # / a_0
    Z = 1
    rho = 2 * Z * r_scaled / (n * a_0)
    norm = np.sqrt((2 * Z / (n * a_0)) ** 3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))
    radial = np.exp(-rho / 2) * rho**l
    laguerre_term = laguerre(n - l - 1, 2 * l + 1, rho)
    return norm * radial * laguerre_term


def hydrogen_wavefunction(n, l, m, r, theta, phi):
    return radial_function(n, l, r) * spherical_harmonics(l, m, theta, phi)


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))
    return r, theta, phi


def eigenenergies(n):
    return -13.6 / n**2


def superposition(n1, l1, m1, n2, l2, m2, r, theta, phi, t):
    return hydrogen_wavefunction(n1, l1, m1, r, theta, phi) * np.exp(-1j * eigenenergies(n1) * t) + hydrogen_wavefunction(
        n2, l2, m2, r, theta, phi
    ) * np.exp(-1j * eigenenergies(n2) * t)
