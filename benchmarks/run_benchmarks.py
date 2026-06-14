"""
Performance benchmark suite for CDS.
Measures execution time for core scientific modules.
"""
import time
import timeit
import multiprocessing
from pathlib import Path

# --- Linear Algebra ---
def bench_linalg():
    from cds.math_utils.linalg import mat_mul, lu_decomposition
    size = 100
    # Create a non-singular matrix (slightly modified identity)
    A = [[0.0] * size for _ in range(size)]
    for i in range(size):
        A[i][i] = 1.0
        if i > 0:
            A[i][i-1] = 0.5
    
    B = [[2.0] * size for _ in range(size)]
    
    t_mul = timeit.timeit(lambda: mat_mul(A, B), number=5) / 5
    t_lu = timeit.timeit(lambda: lu_decomposition(A), number=5) / 5
    
    return {
        "Matrix Mul (100x100)": f"{t_mul:.4f}s",
        "LU Decomp (100x100)": f"{t_lu:.4f}s"
    }

# --- Monte Carlo ---
def bench_montecarlo():
    from cds.montecarlo import estimate_pi
    samples = 100_000
    
    start = time.time()
    estimate_pi(samples, seed=42)
    t_parallel = time.time() - start
    
    return {
        "Parallel Pi (100k samples)": f"{t_parallel:.4f}s",
        "CPU Cores Used": multiprocessing.cpu_count()
    }

# --- Quantum ---
def bench_quantum():
    from cds.quantum.circuit import QuantumCircuit, hadamard
    from cds.quantum.simulator import simulate
    c = QuantumCircuit()
    for _ in range(10): # 10 gates
        c.add(hadamard())
        
    shots = 100_000
    start = time.time()
    simulate(c, shots=shots)
    t_sim = time.time() - start
    
    return {
        "Quantum Sim (100k shots)": f"{t_sim:.4f}s"
    }

def run_all():
    print("Running Benchmarks...")
    results = {}
    results["Linear Algebra"] = bench_linalg()
    results["Monte Carlo"] = bench_montecarlo()
    results["Quantum"] = bench_quantum()
    
    # Generate Report
    report = "# CDS Performance Benchmarks\n\n"
    report += "This report is auto-generated to track the efficiency of pure Python implementations.\n\n"
    
    for category, metrics in results.items():
        report += f"## {category}\n"
        for k, v in metrics.items():
            report += f"- **{k}:** {v}\n"
        report += "\n"
    
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    with open(docs_dir / "BENCHMARKS.md", "w") as f:
        f.write(report)
        
    print(f"Benchmarks completed. Report saved to {docs_dir / 'BENCHMARKS.md'}")

if __name__ == "__main__":
    run_all()
