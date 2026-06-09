# API Reference

## cds.quantum

### Single-Qubit Circuit

| Function | Description |
|----------|-------------|
| `QuantumCircuit()` | Create an empty single-qubit circuit |
| `hadamard()` | Hadamard gate (creates superposition) |
| `pauli_x()` | Pauli-X gate (bit flip) |
| `pauli_z()` | Pauli-Z gate (phase flip) |
| `phase_gate(theta)` | Phase rotation gate |
| `simulate(circuit, shots)` | Run circuit multiple times, return counts |

### Multi-Qubit Register

| Function | Description |
|----------|-------------|
| `QuantumRegister.zeros(n)` | Create n-qubit register in |00...0⟩ |
| `h_gate(reg, target)` | Hadamard on target qubit |
| `x_gate(reg, target)` | Pauli-X on target qubit |
| `y_gate(reg, target)` | Pauli-Y on target qubit |
| `z_gate(reg, target)` | Pauli-Z on target qubit |
| `rz_gate(reg, target, theta)` | Rz rotation on target qubit |
| `cnot(reg, control, target)` | Controlled-NOT gate |
| `cz(reg, control, target)` | Controlled-Z gate |
| `swap(reg, q1, q2)` | SWAP two qubits |
| `toffoli(reg, c1, c2, target)` | Toffoli (CCNOT) gate |
| `bell_state(which)` | Create Bell state (0-3: Φ+, Φ-, Ψ+, Ψ-) |
| `ghz_state(n)` | Create n-qubit GHZ state |
| `is_entangled(reg)` | Detect entanglement (2-qubit concurrence) |

**QuantumRegister methods:**
- `.measure(seed=None)` → int (collapse to basis state)
- `.measure_shots(shots, seed=None)` → dict[str, int]
- `.probabilities()` → list[float]
- `.normalize()` → QuantumRegister

## cds.optimization

| Function | Description |
|----------|-------------|
| `gradient_descent(f, x0, lr, tol, max_iter)` | Minimize scalar function |
| `newton_method(f, x0, tol, max_iter)` | Find root via Newton-Raphson |
| `adam(f, x0, lr, beta1, beta2, eps, tol, max_iter)` | Minimize with Adam optimizer |
| `line_search(f, a, b, tol, max_iter)` | Golden section search in [a,b] |

All return `OptResult(x, value, iterations, converged)`.

## cds.signals

| Function | Description |
|----------|-------------|
| `dft(signal)` | Discrete Fourier Transform |
| `idft(spectrum)` | Inverse DFT |
| `fft_radix2(signal)` | Cooley-Tukey radix-2 FFT (len must be power of 2) |
| `convolve(a, b)` | Linear convolution |
| `power_spectrum(signal)` | |X[k]|² / N |
| `low_pass_filter(signal, cutoff)` | Zero high-frequency components |

## cds.probability

| Function | Description |
|----------|-------------|
| `gaussian_pdf(x, mu, sigma)` | Normal distribution PDF |
| `uniform_pdf(x, a, b)` | Uniform distribution PDF |
| `exponential_pdf(x, lam)` | Exponential distribution PDF |
| `binomial_pmf(k, n, p)` | Binomial PMF: P(X=k) |
| `poisson_pmf(k, lam)` | Poisson PMF: P(X=k) |
| `uniform_sample(a, b, n, seed)` | Generate n uniform random samples |

## cds.stats

| Function | Description |
|----------|-------------|
| `mean(data)` | Arithmetic mean |
| `median(data)` | Median value |
| `variance(data, ddof=1)` | Sample or population variance |
| `stdev(data, ddof=1)` | Standard deviation |
| `correlation(x, y)` | Pearson correlation coefficient |
| `linear_regression(x, y)` | Returns `RegressionResult(slope, intercept, r_squared, predict)` |

## cds.math_utils

| Function | Description |
|----------|-------------|
| `derivative(f, x, h)` | Numerical derivative (central difference) |
| `integral(f, a, b, n)` | Numerical integral (Simpson's rule) |
| `gradient(f, x, h)` | Gradient for multi-variable functions |
| `dot(a, b)` | Dot product |
| `mat_mul(a, b)` | Matrix multiplication |
| `transpose(m)` | Matrix transpose |
| `determinant(m)` | Matrix determinant (recursive) |

## cds.data_analysis

| Function | Description |
|----------|-------------|
| `load_csv(path)` | Load CSV file into `DataTable` |
| `normalize(data)` | Min-max normalization to [0, 1] |
| `z_score(data)` | Standardize to mean=0, std=1 |
| `moving_average(data, window)` | Sliding window average |

**DataTable methods:**
- `.column(name)` → list[str]
- `.column_as_float(name)` → list[float]
- `.head(n)` → first n rows
- `.describe()` → summary stats for numeric columns

## cds.scientific

### Constants
Access via `get_constant(name)`: `c`, `G`, `h`, `k_B`, `N_A`, `R`, `epsilon_0`, `mu_0`, `sigma`, `e`, `m_e`, `m_p`

### Formulas

| Function | Description |
|----------|-------------|
| `kinetic_energy(mass, velocity)` | KE = ½mv² |
| `gravitational_force(m1, m2, r)` | F = Gm₁m₂/r² |
| `wave_frequency(wavelength)` | f = c/λ |
| `ideal_gas_pressure(n, T, V)` | P = nRT/V |
| `photon_energy(frequency)` | E = hf |
| `schwarzschild_radius(mass)` | r_s = 2GM/c² |
| `de_broglie_wavelength(mass, velocity)` | λ = h/(mv) |
| `escape_velocity(mass, radius)` | v = √(2GM/r) |

## cds.hypothesis

| Function | Description |
|----------|-------------|
| `generate_hypotheses(question, domain, n)` | Generate n hypotheses |
| `PromptTemplate.render(question, domain, n)` | Create LLM-ready prompt |
| `SimpleOfflineGenerator()` | Offline placeholder generator |
