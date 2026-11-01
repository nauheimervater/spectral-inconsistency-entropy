"""
Core implementation of Spectral Inconsistency Entropy (H) on the cycle-restricted
1-up-Laplacian of pairwise comparison graphs.

Author: Thomas Nauheimer (2026)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import networkx as nx
from scipy.linalg import null_space, eigh
from scipy.special import digamma


@dataclass
class HodgeOperators:
    """Combinatorial Hodge operators and spectral projectors on the cycle space."""
    nodes: List[int]
    edges: List[Tuple[int, int]]
    triangles: List[Tuple[int, int, int]]
    B0: np.ndarray                 # |V| x |E| node-to-edge coboundary/incidence matrix
    B1: np.ndarray                 # |E| x |T| edge-to-triangle curl matrix
    Z: np.ndarray                  # |E| x beta_1 orthonormal basis of cycle space ker(B0)
    L1_down: np.ndarray            # |E| x |E| down-Laplacian B0^T B0
    L1_up: np.ndarray              # |E| x |E| up-Laplacian B1 B1^T
    L_Z: np.ndarray                # beta_1 x beta_1 cycle-restricted 1-up-Laplacian Z^T L1_up Z
    unique_eigenvalues: np.ndarray # |Lambda| distinct eigenvalues of L_Z
    multiplicities: List[int]      # dimension d_lambda of each eigenspace
    projectors_Z: List[np.ndarray] # orthogonal projectors P_lambda in Z-coordinates
    harmonic_dim: int              # dimension of H^1(K_2(G)) = ker(B0) cap ker(B1^T)


@dataclass
class SpectralEntropyResult:
    """Result of HodgeRank + Spectral Inconsistency Entropy analysis for an edge flow w."""
    rho: float                     # Classical inconsistency ratio ||r||^2 / ||w||^2
    entropy_bits: float            # Spectral Inconsistency Entropy H(G, w) in bits (log2)
    entropy_nats: float            # Spectral Inconsistency Entropy in nats (ln)
    normalized_entropy: float      # H(G, w) / log2(|Lambda|) in [0, 1] (0 if |Lambda| <= 1)
    harmonic_energy_fraction: float # Energy fraction in the lambda=0 harmonic subspace
    curl_energy_fraction: float    # Energy fraction in the lambda>0 triangular curl space
    energy_distribution: np.ndarray # p_lambda across unique_eigenvalues
    scores: np.ndarray             # Least-squares HodgeRank node potentials s*
    gradient_flow: np.ndarray      # Consistent component grad(s*) = B0^T s*
    cyclic_residual: np.ndarray    # Inconsistent cyclic residual r = w - grad(s*)


def build_hodge_operators(G: nx.Graph, eig_tol: float = 1e-5) -> HodgeOperators:
    """
    Construct the combinatorial Hodge operators B0, B1, cycle basis Z = ker(B0),
    cycle-restricted 1-up-Laplacian L_Z = Z^T B1 B1^T Z, and eigenspace projectors.

    Edges are oriented canonically as (u, v) with u < v, and triangles as (u, v, w)
    with u < v < w.
    """
    nodes = sorted(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    edges: List[Tuple[int, int]] = []
    for u, v in G.edges():
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in edges:
            edges.append((a, b))
    edges.sort()
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    n = len(nodes)
    m = len(edges)

    # B0: |V| x |E| with (B0^T s)_{(u,v)} = s(v) - s(u)
    B0 = np.zeros((n, m), dtype=np.float64)
    for j, (u, v) in enumerate(edges):
        B0[node_to_idx[u], j] = -1.0
        B0[node_to_idx[v], j] = 1.0

    # Clique 2-simplices (triangles) u < v < w
    adj = {u: set(G.neighbors(u)) for u in nodes}
    triangles: List[Tuple[int, int, int]] = []
    for idx_u, u in enumerate(nodes):
        for idx_v in range(idx_u + 1, n):
            v = nodes[idx_v]
            if v not in adj[u]:
                continue
            common = adj[u].intersection(adj[v])
            for w in sorted(common):
                if w > v:
                    triangles.append((u, v, w))

    t = len(triangles)
    B1 = np.zeros((m, t), dtype=np.float64)
    for j, (u, v, w) in enumerate(triangles):
        B1[edge_to_idx[(u, v)], j] = 1.0
        B1[edge_to_idx[(v, w)], j] = 1.0
        B1[edge_to_idx[(u, w)], j] = -1.0

    L1_down = B0.T @ B0
    L1_up = B1 @ B1.T

    # Orthonormal basis Z for cycle space ker(B0)
    Z = null_space(B0)
    beta_1 = Z.shape[1]

    if beta_1 == 0:
        return HodgeOperators(
            nodes=nodes,
            edges=edges,
            triangles=triangles,
            B0=B0,
            B1=B1,
            Z=Z,
            L1_down=L1_down,
            L1_up=L1_up,
            L_Z=np.zeros((0, 0), dtype=np.float64),
            unique_eigenvalues=np.array([], dtype=np.float64),
            multiplicities=[],
            projectors_Z=[],
            harmonic_dim=0,
        )

    L_Z = Z.T @ L1_up @ Z
    L_Z = 0.5 * (L_Z + L_Z.T)
    evals, evecs = eigh(L_Z)

    unique_evals: List[float] = []
    eigenspaces: List[List[np.ndarray]] = []
    for i, lam in enumerate(evals):
        lam_clean = 0.0 if abs(lam) < eig_tol else float(lam)
        vec = evecs[:, i]
        placed = False
        for j, u_lam in enumerate(unique_evals):
            if abs(lam_clean - u_lam) < eig_tol:
                eigenspaces[j].append(vec)
                placed = True
                break
        if not placed:
            unique_evals.append(lam_clean)
            eigenspaces.append([vec])

    projectors_Z: List[np.ndarray] = []
    multiplicities: List[int] = []
    harmonic_dim = 0
    for u_lam, vecs in zip(unique_evals, eigenspaces):
        V = np.column_stack(vecs)
        P = V @ V.T
        projectors_Z.append(P)
        multiplicities.append(len(vecs))
        if abs(u_lam) < eig_tol:
            harmonic_dim += len(vecs)

    return HodgeOperators(
        nodes=nodes,
        edges=edges,
        triangles=triangles,
        B0=B0,
        B1=B1,
        Z=Z,
        L1_down=L1_down,
        L1_up=L1_up,
        L_Z=L_Z,
        unique_eigenvalues=np.array(unique_evals, dtype=np.float64),
        multiplicities=multiplicities,
        projectors_Z=projectors_Z,
        harmonic_dim=harmonic_dim,
    )


def compute_spectral_entropy(
    ops: HodgeOperators,
    w: np.ndarray,
    eps: float = 1e-12,
) -> SpectralEntropyResult:
    """
    Decompose an antisymmetric edge flow w in R^{|E|} into gradient and cyclic components
    and compute its Spectral Inconsistency Entropy H(G, w).
    """
    w = np.asarray(w, dtype=np.float64)
    if ops.Z.shape[1] == 0:
        s_star, _, _, _ = np.linalg.lstsq(ops.B0.T, w, rcond=None)
        grad_flow = ops.B0.T @ s_star
        return SpectralEntropyResult(
            rho=0.0,
            entropy_bits=0.0,
            entropy_nats=0.0,
            normalized_entropy=0.0,
            harmonic_energy_fraction=0.0,
            curl_energy_fraction=0.0,
            energy_distribution=np.array([], dtype=np.float64),
            scores=s_star,
            gradient_flow=grad_flow,
            cyclic_residual=np.zeros_like(w),
        )

    # Coordinate of cyclic residual in orthonormal Z-basis: c = Z^T w
    c = ops.Z.T @ w
    r = ops.Z @ c
    grad_flow = w - r
    s_star, _, _, _ = np.linalg.lstsq(ops.B0.T, grad_flow, rcond=None)
    s_star = s_star - np.mean(s_star)

    w_norm_sq = float(np.dot(w, w))
    r_norm_sq = float(np.dot(c, c))
    rho = (r_norm_sq / w_norm_sq) if w_norm_sq > eps else 0.0

    num_spaces = len(ops.projectors_Z)
    p = np.zeros(num_spaces, dtype=np.float64)
    if r_norm_sq > eps:
        for idx, P in enumerate(ops.projectors_Z):
            c_proj = P @ c
            p[idx] = max(0.0, float(np.dot(c_proj, c_proj)) / r_norm_sq)
        p_sum = float(np.sum(p))
        if p_sum > eps:
            p /= p_sum

    mask = p > eps
    entropy_bits = float(-np.sum(p[mask] * np.log2(p[mask])))
    entropy_nats = float(-np.sum(p[mask] * np.log(p[mask])))
    max_bits = np.log2(num_spaces) if num_spaces > 1 else 0.0
    normalized_entropy = (entropy_bits / max_bits) if max_bits > 0 else 0.0

    harmonic_frac = 0.0
    curl_frac = 0.0
    for u_lam, prob in zip(ops.unique_eigenvalues, p):
        if abs(u_lam) < 1e-5:
            harmonic_frac += float(prob)
        else:
            curl_frac += float(prob)

    return SpectralEntropyResult(
        rho=rho,
        entropy_bits=entropy_bits,
        entropy_nats=entropy_nats,
        normalized_entropy=normalized_entropy,
        harmonic_energy_fraction=harmonic_frac,
        curl_energy_fraction=curl_frac,
        energy_distribution=p,
        scores=s_star,
        gradient_flow=grad_flow,
        cyclic_residual=r,
    )


def verify_complete_graph_degeneracy(n: int) -> Dict[str, object]:
    """
    Verify Theorem 1 (Eckmann Identity & Complete Graph Degeneracy) on K_n:
      1. B0^T B0 + B1 B1^T = n * I_{|E|}
      2. L_Z = n * I_{beta_1(K_n)} with beta_1(K_n) = (n-1)(n-2)/2
      3. H(K_n, w) == 0 for arbitrary inconsistent edge flows w.
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    G = nx.complete_graph(n)
    ops = build_hodge_operators(G)
    m = len(ops.edges)
    full_hodge = ops.L1_down + ops.L1_up
    eckmann_err = float(np.max(np.abs(full_hodge - n * np.eye(m))))
    beta_1 = ops.Z.shape[1]
    expected_beta_1 = (n - 1) * (n - 2) // 2
    lz_err = float(np.max(np.abs(ops.L_Z - n * np.eye(beta_1))))

    rng = np.random.default_rng(12345 + n)
    max_H = 0.0
    for _ in range(10):
        w = rng.standard_normal(m)
        res = compute_spectral_entropy(ops, w)
        max_H = max(max_H, res.entropy_bits)

    return {
        "n": n,
        "num_edges": m,
        "num_triangles": len(ops.triangles),
        "beta_1": beta_1,
        "expected_beta_1": expected_beta_1,
        "unique_eigenvalues": ops.unique_eigenvalues.tolist(),
        "multiplicities": ops.multiplicities,
        "eckmann_identity_max_error": eckmann_err,
        "lz_scalar_identity_max_error": lz_err,
        "max_entropy_bits_over_random_flows": max_H,
    }


def single_edge_spectral_profile(ops: HodgeOperators) -> Dict[str, object]:
    """
    Compute the deterministic single-edge spectral entropy H(G, e_j) for every edge e_j
    that participates in at least one cycle.
    """
    m = len(ops.edges)
    entropies: List[float] = []
    cyclic_norms: List[float] = []
    for j in range(m):
        e_vec = np.zeros(m, dtype=np.float64)
        e_vec[j] = 1.0
        c = ops.Z.T @ e_vec
        norm_sq = float(np.dot(c, c))
        cyclic_norms.append(norm_sq)
        if norm_sq > 1e-10:
            res = compute_spectral_entropy(ops, e_vec)
            entropies.append(res.entropy_bits)

    return {
        "num_cyclic_edges": len(entropies),
        "mean_single_edge_entropy_bits": float(np.mean(entropies)) if entropies else 0.0,
        "min_single_edge_entropy_bits": float(np.min(entropies)) if entropies else 0.0,
        "max_single_edge_entropy_bits": float(np.max(entropies)) if entropies else 0.0,
        "single_edge_entropies": entropies,
        "cyclic_leverages": cyclic_norms,
    }


def systemic_spectral_capacity(ops: HodgeOperators) -> Dict[str, float]:
    """
    Compute the theoretical systemic entropy limits of a graph G (Theorem 3):
      1. H_iso_bits: Exact expected entropy E[H(p)] under isotropic Gaussian cycle noise
         c ~ N(0, I_{beta_1}), where p ~ Dirichlet(d_1/2, ..., d_{|Lambda|}/2):
           H_iso(G) = (1 / ln 2) * sum_{lambda} (d_lambda / beta_1) * [psi(beta_1/2 + 1) - psi(d_lambda/2 + 1)]
      2. H_sys_bits: Jensen upper bound H(E[p]) = -sum (d_lambda / beta_1) log2(d_lambda / beta_1)
      3. H_max_bits: Combinatorial upper bound log2(|Lambda|).
    """
    beta_1 = ops.Z.shape[1]
    if beta_1 == 0 or len(ops.multiplicities) <= 1:
        return {
            "beta_1": float(beta_1),
            "num_eigenspaces": float(len(ops.multiplicities)),
            "H_iso_bits": 0.0,
            "H_sys_bits": 0.0,
            "H_max_bits": 0.0,
        }
    d_arr = np.array(ops.multiplicities, dtype=np.float64)
    p_sys = d_arr / float(beta_1)
    h_iso = float(
        np.sum(p_sys * (digamma(0.5 * beta_1 + 1.0) - digamma(0.5 * d_arr + 1.0))) / np.log(2.0)
    )
    h_sys = float(-np.sum(p_sys * np.log2(p_sys)))
    h_max = float(np.log2(len(ops.multiplicities)))
    return {
        "beta_1": float(beta_1),
        "num_eigenspaces": float(len(ops.multiplicities)),
        "H_iso_bits": h_iso,
        "H_sys_bits": h_sys,
        "H_max_bits": h_max,
    }


def localize_corrupted_edge(ops: HodgeOperators, w: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """
    Identify the most likely single corrupted edge e* by maximizing the cosine alignment
    between the observed cyclic residual r = Pi_Z w and the cycle-space leverage signature
    Pi_Z e_j of each edge j:
        score(j) = |<Z^T w, Z^T e_j>| / (||Z^T w|| * ||Z^T e_j||).

    Returns (best_edge_index, best_cosine_similarity, all_cosine_scores).
    """
    c = ops.Z.T @ np.asarray(w, dtype=np.float64)
    c_norm = float(np.linalg.norm(c))
    m = len(ops.edges)
    scores = np.zeros(m, dtype=np.float64)
    if c_norm < 1e-12:
        return -1, 0.0, scores

    # Columns of Z^T are Z^T e_j (i.e. rows of Z)
    row_norms = np.linalg.norm(ops.Z, axis=1)
    valid = row_norms > 1e-10
    scores[valid] = np.abs(ops.Z[valid] @ c) / (row_norms[valid] * c_norm)
    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx]), scores


def run_edit_distance_experiment(
    n_nodes: int = 15,
    p_edge: float = 0.4,
    k_values: Sequence[int] = (1, 2, 3, 4, 5, 6, 9, 14),
    n_trials: int = 1000,
    graph_seed: int = 42,
) -> Tuple[HodgeOperators, List[Dict[str, float]]]:
    """
    Run the L0 edit-distance vs. Spectral Inconsistency Entropy experiment on G(n, p)
    with deterministic graph seed and Monte Carlo trials.
    """
    rng = np.random.default_rng(graph_seed)
    G_undir = nx.erdos_renyi_graph(n_nodes, p_edge, seed=graph_seed, directed=False)
    ops = build_hodge_operators(G_undir)
    m = len(ops.edges)
    valid_k = [k for k in k_values if 1 <= k <= m]
    rows: List[Dict[str, float]] = []
    for k in valid_k:
        h_vals: List[float] = []
        rho_vals: List[float] = []
        top1_hits = 0
        for _ in range(n_trials):
            s = rng.standard_normal(n_nodes)
            w = ops.B0.T @ s
            w_noisy = w.copy()
            edges_to_perturb = rng.choice(m, size=k, replace=False)
            perturb = rng.standard_normal(k) * 5.0
            w_noisy[edges_to_perturb] += perturb
            res = compute_spectral_entropy(ops, w_noisy)
            if np.dot(res.cyclic_residual, res.cyclic_residual) >= 1e-8:
                h_vals.append(res.entropy_bits)
                rho_vals.append(res.rho)
                if k == 1:
                    pred_idx, _, _ = localize_corrupted_edge(ops, w_noisy)
                    if pred_idx == int(edges_to_perturb[0]):
                        top1_hits += 1
        arr = np.array(h_vals)
        rows.append({
            "k": float(k),
            "mean_H": float(np.mean(arr)),
            "std_H": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "ci95_H": float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            "mean_rho": float(np.mean(rho_vals)),
            "top1_localization_acc": (top1_hits / len(h_vals)) if (k == 1 and h_vals) else float("nan"),
        })
    return ops, rows
