"""Scientific computing utilities."""

from cds.scientific.constants import CONSTANTS, get_constant
from cds.scientific.formulas import (
    centripetal_acceleration,
    coulomb_force,
    de_broglie_wavelength,
    doppler_frequency,
    escape_velocity,
    gravitational_force,
    ideal_gas_pressure,
    kinetic_energy,
    pendulum_period,
    photon_energy,
    schwarzschild_radius,
    wave_frequency,
)

__all__ = [
    "CONSTANTS",
    "get_constant",
    "kinetic_energy",
    "gravitational_force",
    "wave_frequency",
    "ideal_gas_pressure",
    "schwarzschild_radius",
    "de_broglie_wavelength",
    "escape_velocity",
    "photon_energy",
    "coulomb_force",
    "centripetal_acceleration",
    "pendulum_period",
    "doppler_frequency",
]
