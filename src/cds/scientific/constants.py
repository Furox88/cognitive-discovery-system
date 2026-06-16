"""Physical and mathematical constants."""
from __future__ import annotations

import math

CONSTANTS: dict[str, tuple[float, str]] = {
    "c": (299_792_458.0, "speed of light (m/s)"),
    "G": (6.674_30e-11, "gravitational constant (m^3 kg^-1 s^-2)"),
    "h": (6.626_070_15e-34, "Planck constant (J·s)"),
    "hbar": (1.054_571_817e-34, "reduced Planck constant (J·s)"),
    "k_B": (1.380_649e-23, "Boltzmann constant (J/K)"),
    "e": (1.602_176_634e-19, "elementary charge (C)"),
    "N_A": (6.022_140_76e23, "Avogadro number (mol^-1)"),
    "R": (8.314_462_618, "gas constant (J mol^-1 K^-1)"),
    "sigma": (5.670_374_419e-8, "Stefan-Boltzmann constant (W m^-2 K^-4)"),
    "pi": (math.pi, "pi"),
    "e_math": (math.e, "Euler's number"),
    "m_e": (9.109_383_7015e-31, "electron mass (kg)"),
    "m_p": (1.672_621_923_69e-27, "proton mass (kg)"),
}


def get_constant(name: str) -> float:
    if name not in CONSTANTS:
        raise KeyError(f"unknown constant: {name}. available: {list(CONSTANTS.keys())}")
    return CONSTANTS[name][0]
