# Security Policy

## Supported Versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 0.9.x   | Yes       | Beta; current development line |
| 0.8.x   | Yes       | Last alpha release; security fixes only |
| 0.7.x   | Yes       | Stable; security fixes only |
| 0.6.x   | Yes       | Stable; security fixes only |
| < 0.6   | No        | EOL |

Stable releases (1.0.0+) will follow a stricter support window (current + 2 previous minor versions).

## Reporting a Vulnerability

If you find a security vulnerability, please **do not open a public issue.**

Instead, email the maintainer directly or use GitHub's private vulnerability reporting feature:
- GitHub: https://github.com/Furox88/cognitive-discovery-system/security/advisories/new

We'll acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days.

## Threat Model

`cognitive-discovery-platform` is a pure-Python scientific computing library distributed via PyPI.
The threat model below describes what we consider in scope and out of scope.

### In Scope

| Threat | Mitigation |
|--------|------------|
| **Supply chain: malicious dependency** | `requirements.lock` and `requirements-dev.lock` pin exact versions; CI builds run on every push and PR; Dependabot opens weekly PRs for outdated dependencies. |
| **Supply chain: malicious release artifact** | PyPI releases uploaded via local `scripts/publish.py` only; each release attaches the sdist + wheel to a GitHub Release; GitHub Actions `attest-build-provenance@v2` produces a sigstore attestation that any consumer can verify. |
| **Code execution from package install** | Build backend is `hatchling` (no setup.py execution); `setuptools_scm` is used for version derivation only (no runtime code execution on install). |
| **Untrusted input in scientific functions** | All numerical kernels are pure-Python implementations; no `eval`/`exec` of user input; CLI (`cds` command) uses Typer with strict parsing. |
| **Dependency confusion** | PyPI project name `cognitive-discovery-platform` is unique; the import name `cds` is short enough to be susceptible but we document it in the README. |

### Out of Scope

| Threat | Reasoning |
|--------|-----------|
| **Denial of service** | This is a research/education library; DoS via crafted input is acceptable. |
| **Side-channel attacks on numerical code** | Pure-Python timing is dominated by interpreter overhead; side channels are not exploitable in practice. |
| **Confidentiality of user data** | The package does not collect, store, or transmit user data. |
| **Cryptographic implementation correctness** | The package uses Python's `hashlib` and `secrets` via the standard library where applicable; it does not implement custom crypto. |

### Known Limitations

- **Pure-Python performance**: Large inputs (e.g., Monte Carlo with N>10⁷) can be slow and may leak timing information about input size. For sensitive workloads, use a dedicated numerical library.
- **No formal verification**: Numerical kernels are tested against known reference values but not formally verified. Use at your own risk for safety-critical applications.
- **Single-maintainer project**: Response time for security reports depends on the maintainer's availability (typically < 7 days).

## Security Best Practices for Users

1. **Pin versions**: Always pin `cognitive-discovery-platform==X.Y.Z` in `requirements.txt` rather than using `>=` or unpinned ranges.
2. **Verify releases**: For high-assurance environments, verify the sigstore attestation on the GitHub Release against the published wheel/sdist hashes.
3. **Keep dependencies fresh**: Run `pip install --upgrade` regularly; Dependabot PRs are opened weekly.
4. **Audit lockfiles**: `requirements.lock` and `requirements-dev.lock` are committed; diff them on every PR to spot unexpected changes.

## Acknowledgments

We thank the security researchers and contributors who have helped improve this project.
Responsible disclosures are credited in the relevant GitHub Security Advisory (with reporter consent).

## Contact

- Maintainer: Furox88 (@Furox88)
- Email: furkanarkn1451@gmail.com
- PGP key: not published (use GitHub private vulnerability reporting)
