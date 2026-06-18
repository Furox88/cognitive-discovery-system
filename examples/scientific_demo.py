"""Scientific constants and physics formulas demo."""

from cds.scientific import (
    CONSTANTS,
    de_broglie_wavelength,
    escape_velocity,
    get_constant,
    gravitational_force,
    ideal_gas_pressure,
    kinetic_energy,
    photon_energy,
    schwarzschild_radius,
    wave_frequency,
)


def main() -> None:
    print("=== Physical Constants ===")
    c = get_constant("c")
    G = get_constant("G")
    print(f"Speed of light c = {c} m/s")
    print(f"Gravitational constant G = {G} N·m²/kg²")
    print(f"Available constants: {sorted(CONSTANTS)}")

    print("\n=== Mechanics ===")
    ke = kinetic_energy(mass=10.0, velocity=5.0)
    fg = gravitational_force(m1=5.0, m2=10.0, r=2.0)
    ev = escape_velocity(mass=5.972e24, radius=6.371e6)
    print(f"Kinetic energy (10kg @ 5m/s) = {ke:.2f} J")
    print(f"Gravitational force (5kg,10kg @ 2m) = {fg:.6e} N")
    print(f"Earth escape velocity = {ev:.1f} m/s")

    print("\n=== Waves & Photons ===")
    pe = photon_energy(frequency=5.0e14)
    wf = wave_frequency(wavelength=500e-9)
    dbw = de_broglie_wavelength(mass=9.109e-31, velocity=1.0e6)
    print(f"Photon energy @ 5e14 Hz = {pe:.4e} J")
    print(f"Frequency of 500nm light = {wf:.4e} Hz")
    print(f"de Broglie wavelength (electron @ 1e6 m/s) = {dbw:.4e} m")

    print("\n=== Astrophysics & Thermo ===")
    rs = schwarzschild_radius(mass=1.989e30)
    pg = ideal_gas_pressure(n_moles=1.0, temperature=300.0, volume=0.024)
    print(f"Sun Schwarzschild radius = {rs:.2f} m")
    print(f"Ideal gas pressure (n=1,T=300,V=0.024) = {pg:.2f} Pa")


if __name__ == "__main__":
    main()
