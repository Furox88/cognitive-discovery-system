"""Theory verification and symbolic reasoning tools.

Contains research-strength components for checking internal
consistency of physical/mathematical definitions, path independence,
parameter degeneracy, etc.

Directly inspired by systematic consistency analysis performed
on a novel acceleration law (RBA) and related quantities.
"""
from cds.verification.theory import TheoryVerifier

__all__ = ["TheoryVerifier"]
