"""Common physics formulas."""

from __future__ import annotations

import math

from cds.scientific.constants import get_constant


def kinetic_energy(mass: float, velocity: float) -> float:
    """KE = 0.5 * m * v^2"""
    return 0.5 * mass * velocity**2


def gravitational_force(m1: float, m2: float, r: float) -> float:
    """F = G * m1 * m2 / r^2"""
    G = get_constant("G")
    if r == 0:
        raise ValueError("distance can't be zero")
    return G * m1 * m2 / r**2


def wave_frequency(wavelength: float) -> float:
    """f = c / lambda"""
    c = get_constant("c")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    return c / wavelength


def ideal_gas_pressure(n_moles: float, temperature: float, volume: float) -> float:
    """PV = nRT => P = nRT/V"""
    R = get_constant("R")
    if volume <= 0:
        raise ValueError("volume must be positive")
    return n_moles * R * temperature / volume


def photon_energy(frequency: float) -> float:
    """E = h * f"""
    h = get_constant("h")
    return h * frequency


def schwarzschild_radius(mass: float) -> float:
    """r_s = 2GM/c^2"""
    G = get_constant("G")
    c = get_constant("c")
    return 2 * G * mass / c**2


def de_broglie_wavelength(mass: float, velocity: float) -> float:
    """lambda = h / (m * v)"""
    h = get_constant("h")
    mv = mass * velocity
    if mv == 0:
        raise ValueError("momentum can't be zero")
    return h / mv


def escape_velocity(mass: float, radius: float) -> float:
    """v_esc = sqrt(2GM/r)"""
    G = get_constant("G")
    return math.sqrt(2 * G * mass / radius)
