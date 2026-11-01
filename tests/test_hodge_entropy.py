import unittest
import numpy as np
import networkx as nx

from spectral_entropy import (
    build_hodge_operators,
    compute_spectral_entropy,
    verify_complete_graph_degeneracy,
    single_edge_spectral_profile,
    systemic_spectral_capacity,
    localize_corrupted_edge,
    run_edit_distance_experiment,
)


class TestSpectralInconsistencyEntropy(unittest.TestCase):
    def test_theorem1_complete_graph_degeneracy(self):
        """Theorem 1: On K_n (n=3..7), L1 = n*I, L_Z = n*I, and H(K_n, w) == 0."""
        for n in range(3, 8):
            res = verify_complete_graph_degeneracy(n)
            self.assertEqual(res["beta_1"], (n - 1) * (n - 2) // 2)
            self.assertEqual(len(res["unique_eigenvalues"]), 1)
            self.assertAlmostEqual(res["unique_eigenvalues"][0], float(n), places=10)
            self.assertEqual(res["multiplicities"], [(n - 1) * (n - 2) // 2])
            self.assertLess(res["eckmann_identity_max_error"], 1e-12)
            self.assertLess(res["lz_scalar_identity_max_error"], 1e-12)
            self.assertLess(res["max_entropy_bits_over_random_flows"], 1e-12)

    def test_theorem2_harmonic_curl_separation(self):
        """Theorem 2: lambda=0 eigenspace equals H^1(K_2(G)) and separates harmonic from curl flows."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)])
        ops = build_hodge_operators(G)
        self.assertEqual(ops.Z.shape[1], 2)
        self.assertEqual(ops.harmonic_dim, 1)
        np.testing.assert_allclose(ops.unique_eigenvalues, [0.0, 3.0], atol=1e-10)

        edge_idx = {e: i for i, e in enumerate(ops.edges)}
        w_4cycle = np.zeros(len(ops.edges))
        w_4cycle[edge_idx[(0, 1)]] = 1.0
        w_4cycle[edge_idx[(1, 2)]] = 1.0
        w_4cycle[edge_idx[(2, 3)]] = 1.0
        w_4cycle[edge_idx[(0, 3)]] = -1.0

        w_3cycle = np.zeros(len(ops.edges))
        w_3cycle[edge_idx[(0, 1)]] = 1.0
        w_3cycle[edge_idx[(1, 4)]] = 1.0
        w_3cycle[edge_idx[(0, 4)]] = -1.0

        w_harm = w_4cycle - (1.0 / 3.0) * w_3cycle

        res_harm = compute_spectral_entropy(ops, w_harm)
        self.assertAlmostEqual(res_harm.harmonic_energy_fraction, 1.0, places=10)
        self.assertAlmostEqual(res_harm.curl_energy_fraction, 0.0, places=10)
        self.assertAlmostEqual(res_harm.entropy_bits, 0.0, places=10)

        res_curl = compute_spectral_entropy(ops, w_3cycle)
        self.assertAlmostEqual(res_curl.harmonic_energy_fraction, 0.0, places=10)
        self.assertAlmostEqual(res_curl.curl_energy_fraction, 1.0, places=10)
        self.assertAlmostEqual(res_curl.entropy_bits, 0.0, places=10)

    def test_potential_and_scale_invariance(self):
        """Proposition 1: H(G, B0^T s + alpha * delta) == H(G, delta) for all s and alpha != 0."""
        G = nx.erdos_renyi_graph(15, 0.4, seed=42)
        ops = build_hodge_operators(G)
        rng = np.random.default_rng(99)
        delta = np.zeros(len(ops.edges))
        delta[7] = 1.0
        h_base = compute_spectral_entropy(ops, delta).entropy_bits

        for alpha in [-10.0, -0.25, 0.5, 42.0]:
            s = rng.standard_normal(len(ops.nodes)) * 100.0
            w = ops.B0.T @ s + alpha * delta
            res = compute_spectral_entropy(ops, w)
            self.assertAlmostEqual(res.entropy_bits, h_base, places=10)

    def test_theorem3_dirichlet_digamma_asymptote(self):
        """Theorem 3: Isotropic Gaussian cycle noise matches the exact Dirichlet-Digamma formula H_iso(G)."""
        G = nx.erdos_renyi_graph(15, 0.4, seed=42)
        ops = build_hodge_operators(G)
        cap = systemic_spectral_capacity(ops)

        rng = np.random.default_rng(1234)
        samples = []
        for _ in range(1500):
            c_iso = rng.standard_normal(ops.Z.shape[1])
            w_iso = ops.Z @ c_iso
            samples.append(compute_spectral_entropy(ops, w_iso).entropy_bits)
        emp_mean = float(np.mean(samples))
        self.assertAlmostEqual(emp_mean, cap["H_iso_bits"], delta=0.03)

    def test_single_fault_localization(self):
        """Single-edge corruption on G(15, 0.4) is localized with 100% accuracy via cycle-leverage alignment."""
        G = nx.erdos_renyi_graph(15, 0.4, seed=42)
        ops = build_hodge_operators(G)
        rng = np.random.default_rng(2026)
        for j in range(len(ops.edges)):
            s = rng.standard_normal(len(ops.nodes))
            w = ops.B0.T @ s
            w[j] += 4.5
            pred_j, cos_sim, _ = localize_corrupted_edge(ops, w)
            self.assertEqual(pred_j, j)
            self.assertAlmostEqual(cos_sim, 1.0, places=10)

    def test_monotonic_edit_distance_growth(self):
        """Entropy increases from localized k=1 faults to systemic k=14 faults."""
        _, rows = run_edit_distance_experiment(
            n_nodes=15,
            p_edge=0.4,
            k_values=(1, 2, 6, 14),
            n_trials=300,
            graph_seed=42,
        )
        means = [r["mean_H"] for r in rows]
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])
        self.assertLess(means[2], means[3])


if __name__ == "__main__":
    unittest.main()
