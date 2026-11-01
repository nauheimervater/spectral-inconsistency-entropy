# Spectral Inconsistency Entropy: Distinguishing Local from Systemic Contradictions in Pairwise Comparison Graphs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/Spectral_Inconsistency_Entropy.pdf)

> **Spectral Inconsistency Entropy: Distinguishing Local from Systemic Contradictions in Pairwise Comparison Graphs**  
> Thomas Nauheimer (`nauheimer.t@gmail.com`), November 2026.  
> Paper: [`paper/Spectral_Inconsistency_Entropy.pdf`](paper/Spectral_Inconsistency_Entropy.pdf) · Source: [`paper/Spectral_Inconsistency_Entropy.md`](paper/Spectral_Inconsistency_Entropy.md) · LaTeX: [`paper/Spectral_Inconsistency_Entropy.tex`](paper/Spectral_Inconsistency_Entropy.tex)

---

## Summary of Main Results

Classical HodgeRank (Jiang et al., 2011) decomposes an antisymmetric edge comparison flow $w \in \mathbb{R}^{|E|}$ into a consistent gradient flow $B_0^T s^*$ and a cyclic residual $r = \Pi_{\mathcal{Z}} w \in \mathcal{Z} = \ker(B_0)$. While the classical inconsistency index $\rho(G, w) = \|r\|^2 / \|w\|^2$ measures the *total magnitude* of inconsistency, it confounds the scale of the ground-truth potential $s^*$ with the topology of the contradictions and cannot separate a single faulty sensor ($L_0$ edit distance $k = 1$) from network-wide systemic noise.

This repository introduces **Spectral Inconsistency Entropy ($H$)**—the Shannon entropy of the cyclic residual's energy distribution across the distinct eigenspaces $\Lambda(G)$ of the cycle-restricted 1-up-Laplacian $L_{\mathcal{Z}} = Z^T B_1 B_1^T Z$—together with a **cycle-leverage cosine localizer** for pinpointing single corrupted comparisons.

1. **Theorem 1 (Universal Clique Degeneracy via Eckmann Identity):** On every complete graph $K_n$ ($n \ge 3$), the 1-Hodge Laplacian satisfies $B_0^T B_0 + B_1 B_1^T = n I_{|E|}$, forcing $L_{\mathcal{Z}} = n I_{\beta_1(K_n)}$ with a single eigenvalue $\lambda = n$ of multiplicity $\beta_1(K_n) = (n-1)(n-2)/2$. Consequently, $H(K_n, w) \equiv 0$ identically for all $n \ge 3$ and all edge flows $w$.
2. **Theorem 2 (Harmonic–Curl Separation & Potential/Scale Invariance):** The $\lambda = 0$ eigenspace of $L_{\mathcal{Z}}$ is canonically isomorphic to the first simplicial cohomology group $\mathcal{H}^1(\mathcal{K}_2(G)) = \ker(B_0) \cap \ker(B_1^T)$ (non-triangular global cycles of length $\ge 4$), while $\lambda > 0$ eigenspaces span $\operatorname{im}(B_1)$. Furthermore, $H(G, B_0^T s + \alpha \delta) = H(G, \delta)$ is strictly invariant to node potentials $s \in \mathbb{R}^{|V|}$ and fault scale $\alpha \neq 0$.
3. **Theorem 3 (Exact Isotropic Dirichlet–Digamma Asymptote):** Under isotropic Gaussian cycle noise $c \sim \mathcal{N}(0, \sigma^2 I_{\beta_1(G)})$, the eigenspace energy fractions follow $(p_\lambda)_{\lambda \in \Lambda} \sim \operatorname{Dirichlet}(d_1/2, \dots, d_{|\Lambda|}/2)$ with closed-form expected entropy:
   $$H_{\text{iso}}(G) = \frac{1}{\ln 2} \sum_{\lambda \in \Lambda(G)} \frac{d_\lambda}{\beta_1(G)} \left[ \psi\!\left(\frac{\beta_1(G)}{2} + 1\right) - \psi\!\left(\frac{d_\lambda}{2} + 1\right) \right] \le H_{\text{sys}}(G) = -\sum_{\lambda \in \Lambda(G)} \frac{d_\lambda}{\beta_1(G)} \log_2 \frac{d_\lambda}{\beta_1(G)}.$$
4. **Empirical & Structural Regimes:**
   - On heterogeneous and modular networks (`Erdős–Rényi G(15, 0.4)`, `Small-World Tournament`, `Modular League`), localized faults ($k=1$) have significantly lower spectral entropy than systemic noise, and $H(G, w)$ grows strictly monotonically with the $L_0$ edit distance $k$, converging to the exact analytical Dirichlet–Digamma asymptote $H_{\text{iso}}(G)$ ($4.1175$ bits vs. $4.1274 \pm 0.0124$ bits measured at $k=40$ on $G(15, 0.4)$).
   - On regular spatial lattices (`Triangular Sensor Grid`), where Laplacian eigenmodes are delocalized Bloch waves, a single-edge delta spike spreads across frequencies ($H(k=1) = 2.7750 > H_{\text{iso}} = 2.4634$ bits due to Porter–Thomas $\chi^2_1$ fluctuations in single Gaussian realizations, while respecting the Jensen bound $H_{\text{sys}} = 3.2516$ bits).
   - Across all five graph families, the **cycle-leverage cosine localizer** identifies a single corrupted comparison ($k=1$) with **$91.3\%$–$100.0\%$ Top-1 accuracy**.

---

## Quickstart & Verification

```bash
git clone https://github.com/nauheimervater/spectral-inconsistency-entropy.git
cd spectral-inconsistency-entropy
pip install -r requirements.txt

# Run unit tests (Theorems 1-3, invariance, localization, monotonicity)
python -m unittest discover -s tests -v

# Reproduce all tables and empirical benchmarks
python examples/reproduce_all.py
```

### Benchmark Summary across Graph Families

| Network Topology | $|V|$ | $|E|$ | $\beta_1$ | $|\Lambda|$ | Mean $H(k=1)$ (bits) | Isotropic $H_{\text{iso}}$ (bits) | Jensen $H_{\text{sys}}$ (bits) | Single-Fault Top-1 Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Complete Graph $K_8$ | 8 | 28 | 21 | 1 | $0.0000$ | $0.0000$ | $0.0000$ | **100.0%** |
| Modular League ($4 \times 5$ Caveman) | 20 | 40 | 20 | 5 | **$0.4809$** | **$1.4392$** | $1.6710$ | **97.4%** |
| Small-World Tournament (NWS 16) | 16 | 45 | 30 | 26 | **$2.9614$** | **$3.6453$** | $4.5199$ | **100.0%** |
| Erdős–Rényi $G(15, 0.4)$ | 15 | 49 | 35 | 35 | **$3.7512$** | **$4.1175$** | $5.1293$ | **100.0%** |
| Triangular Sensor Grid ($4 \times 4$) | 12 | 23 | 12 | 10 | $2.7750$ | $2.4634$ | $3.2516$ | **91.3%** |

---

## Citation

```bibtex
@misc{nauheimer2026spectral,
  author       = {Thomas Nauheimer},
  title        = {Spectral Inconsistency Entropy: Distinguishing Local from Systemic Contradictions in Pairwise Comparison Graphs},
  year         = {2026},
  month        = nov,
  url          = {https://github.com/nauheimervater/spectral-inconsistency-entropy}
}
```
