"""
spectral_entropy: Spectral Inconsistency Entropy and Combinatorial Hodge Diagnostics
for Pairwise Comparison Graphs.
"""

from .hodge_entropy import (
    HodgeOperators,
    SpectralEntropyResult,
    build_hodge_operators,
    compute_spectral_entropy,
    verify_complete_graph_degeneracy,
    single_edge_spectral_profile,
    systemic_spectral_capacity,
    localize_corrupted_edge,
    run_edit_distance_experiment,
)

__all__ = [
    "HodgeOperators",
    "SpectralEntropyResult",
    "build_hodge_operators",
    "compute_spectral_entropy",
    "verify_complete_graph_degeneracy",
    "single_edge_spectral_profile",
    "systemic_spectral_capacity",
    "localize_corrupted_edge",
    "run_edit_distance_experiment",
]
