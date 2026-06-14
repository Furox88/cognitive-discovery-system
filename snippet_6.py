from cds.scientific import kinetic_energy, escape_velocity, get_constant

print(get_constant("c"))          # speed of light
print(kinetic_energy(10, 5))      # 125.0 J
print(escape_velocity(5.972e24, 6.371e6))  # ~11186 m/s
