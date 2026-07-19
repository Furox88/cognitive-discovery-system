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


def coulomb_force(q1: float, q2: float, r: float) -> float:
    """Coulomb's law: ``F = k * |q1 q2| / r^2`` (SI, returns magnitude).

    Uses ``k = 1 / (4 π ε0)`` with ``ε0`` from :mod:`cds.scientific.constants`.
    """
    if r == 0:
        raise ValueError("distance can't be zero")
    eps0 = get_constant("epsilon_0")
    k = 1.0 / (4.0 * math.pi * eps0)
    return k * abs(q1 * q2) / (r**2)


def centripetal_acceleration(velocity: float, radius: float) -> float:
    """``a = v^2 / r`` (magnitude)."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    return (velocity**2) / radius


def pendulum_period(length: float, g: float | None = None) -> float:
    """Small-angle simple pendulum period ``T = 2π √(L/g)``."""
    if length <= 0:
        raise ValueError("length must be positive")
    gravity = get_constant("g") if g is None else g
    if gravity <= 0:
        raise ValueError("g must be positive")
    return 2.0 * math.pi * math.sqrt(length / gravity)


def doppler_frequency(
    f0: float,
    v_source: float = 0.0,
    v_observer: float = 0.0,
    v_sound: float | None = None,
) -> float:
    """1-D acoustic Doppler: ``f = f0 * (v + vo) / (v + vs)``.

    Sign convention: positive ``v_source`` means the source moves **away**
    from the observer (denominator increases → lower pitch). Positive
    ``v_observer`` means the observer moves **toward** the source.
    """
    if f0 <= 0:
        raise ValueError("f0 must be positive")
    v = 343.0 if v_sound is None else v_sound
    if v <= 0:
        raise ValueError("v_sound must be positive")
    den = v + v_source
    if den == 0:
        raise ValueError("denominator (v + v_source) can't be zero")
    return f0 * (v + v_observer) / den
