"""Tests for scientific module."""

import pytest

from cds.scientific.constants import CONSTANTS, get_constant
from cds.scientific.formulas import (
    de_broglie_wavelength,
    escape_velocity,
    gravitational_force,
    ideal_gas_pressure,
    kinetic_energy,
    photon_energy,
    schwarzschild_radius,
    wave_frequency,
)


def test_constants_exist() -> None:
    assert "c" in CONSTANTS
    assert "G" in CONSTANTS
    assert "h" in CONSTANTS


def test_get_constant() -> None:
    c = get_constant("c")
    assert abs(c - 299_792_458.0) < 1


def test_unknown_constant() -> None:
    with pytest.raises(KeyError):
        get_constant("does_not_exist")


def test_kinetic_energy() -> None:
    # 1 kg at 10 m/s = 50 J
    assert kinetic_energy(1.0, 10.0) == 50.0


def test_gravitational_force() -> None:
    f = gravitational_force(5.972e24, 1.0, 6.371e6)
    assert abs(f - 9.82) < 0.1  # roughly g


def test_wave_frequency() -> None:
    # visible light ~500nm
    f = wave_frequency(500e-9)
    assert abs(f - 5.996e14) < 1e12


def test_ideal_gas() -> None:
    # 1 mol at 273K in 0.0224 m^3 ~ 101325 Pa
    p = ideal_gas_pressure(1.0, 273.15, 0.02241)
    assert abs(p - 101325) < 500


def test_photon_energy() -> None:
    e = photon_energy(5e14)
    assert e > 0


def test_schwarzschild_radius_sun() -> None:
    # sun mass ~2e30 kg, rs ~3 km
    rs = schwarzschild_radius(1.989e30)
    assert abs(rs - 2953) < 10


def test_de_broglie() -> None:
    # electron at 1e6 m/s
    lam = de_broglie_wavelength(9.109e-31, 1e6)
    assert lam > 0


def test_escape_velocity_earth() -> None:
    v = escape_velocity(5.972e24, 6.371e6)
    assert abs(v - 11186) < 50  # ~11.2 km/s


def test_kinetic_energy_zero_velocity() -> None:
    assert kinetic_energy(100.0, 0.0) == 0.0


def test_gravitational_force_zero_distance() -> None:
    with pytest.raises(ValueError):
        gravitational_force(1.0, 1.0, 0.0)


def test_wave_frequency_negative() -> None:
    with pytest.raises(ValueError):
        wave_frequency(-1.0)


def test_ideal_gas_zero_volume() -> None:
    with pytest.raises(ValueError):
        ideal_gas_pressure(1.0, 300.0, 0.0)


def test_de_broglie_zero_momentum() -> None:
    with pytest.raises(ValueError):
        de_broglie_wavelength(0.0, 0.0)


def test_constants_have_descriptions() -> None:
    for name, (val, desc) in CONSTANTS.items():
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_escape_velocity() -> None:
    from cds.scientific.formulas import escape_velocity

    v = escape_velocity(5.972e24, 6.371e6)
    assert abs(v - 11186) < 50  # ~11.2 km/s


def test_photon_energy_visible_light() -> None:
    e = photon_energy(5e14)  # green light ~500 THz
    assert e > 0
    assert e < 1e-18


def test_schwarzschild_radius_sun_km() -> None:
    r = schwarzschild_radius(1.989e30)
    assert abs(r - 2953) < 10  # ~2953 meters


def test_ideal_gas_room_conditions() -> None:
    p = ideal_gas_pressure(1.0, 300.0, 0.0245)
    assert abs(p - 101660) < 500  # ~1 atm


def test_wave_frequency_radio() -> None:
    f = wave_frequency(1.0)  # 1 meter wavelength
    assert abs(f - 299792458.0) < 1.0


def test_coulomb_force() -> None:
    from cds.scientific.formulas import coulomb_force

    f = coulomb_force(1e-6, 1e-6, 0.1)
    assert f > 0
    with pytest.raises(ValueError):
        coulomb_force(1.0, 1.0, 0.0)


def test_centripetal() -> None:
    from cds.scientific.formulas import centripetal_acceleration

    assert abs(centripetal_acceleration(10.0, 5.0) - 20.0) < 1e-12


def test_pendulum_period() -> None:
    from cds.scientific.formulas import pendulum_period

    t = pendulum_period(1.0)
    assert 1.5 < t < 2.5


def test_doppler() -> None:
    from cds.scientific.formulas import doppler_frequency

    f = doppler_frequency(440.0, v_source=0.0, v_observer=0.0)
    assert abs(f - 440.0) < 1e-9
    higher = doppler_frequency(440.0, v_observer=10.0)
    assert higher > 440.0


def test_formula_errors() -> None:
    from cds.scientific.formulas import (
        centripetal_acceleration,
        doppler_frequency,
        pendulum_period,
    )

    with pytest.raises(ValueError):
        centripetal_acceleration(1.0, 0.0)
    with pytest.raises(ValueError):
        pendulum_period(0.0)
    with pytest.raises(ValueError):
        pendulum_period(1.0, g=-1.0)
    with pytest.raises(ValueError):
        doppler_frequency(0.0)
    with pytest.raises(ValueError):
        doppler_frequency(440.0, v_sound=-1.0)
    with pytest.raises(ValueError):
        doppler_frequency(440.0, v_source=-343.0)
