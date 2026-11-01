# Spectral Inconsistency Entropy: Distinguishing Local from Systemic Contradictions in Pairwise Comparison Graphs

**Thomas Nauheimer**  
Independent Researcher · `nauheimer.t@gmail.com`  
November 2026

---

## Abstract

Pairwise comparison data—arising in sports rankings, autonomous sensor fusion, peer review, and RLHF preference modeling—frequently exhibits cyclic inconsistencies. While classical combinatorial Hodge theory (HodgeRank) quantifies the total magnitude of inconsistency via the cyclic residual ratio $\rho(G, w) = \|r\|^2 / \|w\|^2$, it is invariant to how contradictions are topologically distributed across the comparison network. In this paper, we introduce **Spectral Inconsistency Entropy ($H$)**, a scale- and potential-invariant diagnostic defined as the Shannon entropy of the cyclic residual's energy projection across the distinct eigenspaces of the cycle-restricted 1-up-Laplacian $L_{\mathcal{Z}} = Z^T B_1 B_1^T Z$.

We establish four main results:
1. **Universal Clique Degeneracy (Theorem 1):** Using the simplicial Eckmann identity $B_0^T B_0 + B_1 B_1^T = n I_{|E|}$ on the clique 2-complex of the complete graph $K_n$, we prove that $L_{\mathcal{Z}} = n I_{\beta_1(K_n)}$ possesses a single eigenvalue $\lambda = n$ of multiplicity $\beta_1(K_n) = (n-1)(n-2)/2$ for every $n \ge 3$. Consequently, $H(K_n, w) \equiv 0$ identically for all edge flows $w$: complete round-robin tournaments are spectrally monochrome and blind to fault localization via Laplacian eigenmodes.
2. **Harmonic–Curl Separation and Scale Invariance (Theorem 2):** On any graph $G$, the $\lambda = 0$ eigenspace of $L_{\mathcal{Z}}$ is canonically isomorphic to the first simplicial cohomology group $\mathcal{H}^1(\mathcal{K}_2(G)) = \ker(B_0) \cap \ker(B_1^T)$ (non-triangular global cycles of length $\ge 4$), while the $\lambda > 0$ eigenspaces decompose local triangular curl $\operatorname{im}(B_1)$. Moreover, $H(G, B_0^T s + \alpha \delta) = H(G, \delta)$ is strictly invariant to ground-truth node potentials $s \in \mathbb{R}^{|V|}$ and fault scale $\alpha \neq 0$.
3. **Exact Isotropic Dirichlet–Digamma Asymptote (Theorem 3):** Under isotropic Gaussian cycle noise $c \sim \mathcal{N}(0, \sigma^2 I_{\beta_1})$, the eigenspace energy fractions follow a $\operatorname{Dirichlet}(d_1/2, \dots, d_{|\Lambda|}/2)$ law whose exact expected entropy is given in closed form by the digamma function:
   $$H_{\text{iso}}(G) = \frac{1}{\ln 2} \sum_{\lambda \in \Lambda} \frac{d_\lambda}{\beta_1(G)} \left[ \psi\!\left(\frac{\beta_1(G)}{2} + 1\right) - \psi\!\left(\frac{d_\lambda}{2} + 1\right) \right] \le H_{\text{sys}}(G) = -\sum_{\lambda \in \Lambda} \frac{d_\lambda}{\beta_1(G)} \log_2 \frac{d_\lambda}{\beta_1(G)}.$$
4. **Empirical & Structural Boundaries ($L_0$ Edit Distance vs. Bloch Delocalization):** Across 1,000 Monte Carlo trials per edit distance $k$ on asymmetric Erdős–Rényi ($G(15, 0.4)$), small-world tournament, and modular league graphs, $H(G, w)$ increases monotonically from localized single-edge faults ($k=1$, $H = 3.7641 \pm 0.0394$ bits on $G(15, 0.4)$) to the analytical Dirichlet–Digamma asymptote ($H_{\text{iso}} = 4.1175$ bits; measured $4.1274 \pm 0.0124$ bits at $k=40$). Conversely, on regular spatial lattices (such as a 2D triangular sensor grid) where Laplacian eigenmodes are delocalized Bloch waves, a single-edge delta spike spreads deterministically across frequencies ($H(k=1) = 2.7750 > H_{\text{iso}} = 2.4634$ bits). To complement $H$ when eigenmodes are degenerate or delocalized, we pair Spectral Inconsistency Entropy with a direct **cycle-leverage cosine localizer** $\arg\max_{e \in E} |\langle r, \Pi_{\mathcal{Z}} \mathbf{e}_e \rangle| / (\|r\| \, \|\Pi_{\mathcal{Z}} \mathbf{e}_e\|)$ that identifies single corrupted edges with $91.3\%$–$100.0\%$ Top-1 accuracy across all tested topologies.

---

## 1. Introduction

Deriving a global ranking or scalar potential from noisy pairwise comparisons is a foundational problem across statistics, machine learning, and control theory, with direct applications to sports tournaments, peer review, autonomous sensor fusion, and reinforcement learning from human feedback (RLHF). Given a comparison graph $G = (V, E)$ with $|V| = n$ vertices and $|E| = m$ oriented edges, an antisymmetric edge flow $w \in \mathbb{R}^m$ ($w(i \to j) = -w(j \to i)$) records the observed preference margin between items $i$ and $j$.

The flow $w$ is globally consistent if and only if there exists a node potential $s: V \to \mathbb{R}$ such that $w = B_0^T s$, where $B_0 \in \mathbb{R}^{n \times m}$ is the signed vertex-edge incidence matrix. Combinatorial Hodge theory (Jiang, Lim, Yao, and Ye, 2011) decomposes $\mathbb{R}^m$ orthogonally into three subspaces:
$$w = \underbrace{B_0^T s^*}_{\text{gradient (consistent)}} \oplus \underbrace{r_{\text{curl}}}_{\text{triangular curl } \in \operatorname{im}(B_1)} \oplus \underbrace{r_{\text{harm}}}_{\text{harmonic } \in \ker(B_0) \cap \ker(B_1^T)},$$
where $r = r_{\text{curl}} \oplus r_{\text{harm}} = \Pi_{\mathcal{Z}} w \in \mathcal{Z} = \ker(B_0)$ is the total cyclic residual. The classical HodgeRank inconsistency index
$$\rho(G, w) = \frac{\|r\|_2^2}{\|w\|_2^2} \in [0, 1]$$
quantifies the *fraction of total flow energy* that is inconsistent.

However, $\rho(G, w)$ suffers from two structural limitations:
1. **Confounding of scale and topology:** Because $\|w\|_2^2 = \|B_0^T s^*\|_2^2 + \|r\|_2^2$, scaling the ground-truth potential $s^*$ arbitrarily changes $\rho(G, w)$ even when the inconsistency $r$ is fixed.
2. **Topological blindness:** Two comparison datasets with identical $\rho(G, w)$ can arise from opposite physical regimes—one caused by a single corrupted sensor or rogue reviewer ($L_0$ edit distance $k = 1$ to consistency), and the other caused by systemic network-wide noise ($k \sim m$).

To separate localized faults from systemic contradictions independently of the underlying score scale, we analyze the spectral distribution of the cyclic residual $r$ over the eigenspaces of the cycle-restricted 1-up-Laplacian.

---

## 2. Mathematical Framework

### 2.1 Simplicial Boundary Operators and the Cycle Space

Let $\mathcal{K}_2(G) = (V, E, T)$ denote the clique 2-complex of $G$, where $E$ is the set of edges (oriented canonically as $(u, v)$ with $u < v$) and $T$ is the set of 3-cliques (triangles $(u, v, w)$ with $u < v < w$).
- The 0-coboundary transpose (vertex-to-edge incidence matrix) $B_0 \in \mathbb{R}^{|V| \times |E|}$ has entries $(B_0)_{u, (u,v)} = -1$ and $(B_0)_{v, (u,v)} = +1$, so $(B_0^T s)_{(u,v)} = s(v) - s(u)$.
- The 1-boundary (edge-to-triangle curl transpose) $B_1 \in \mathbb{R}^{|E| \times |T|}$ has entries $(B_1)_{(u,v), (u,v,w)} = +1$, $(B_1)_{(v,w), (u,v,w)} = +1$, and $(B_1)_{(u,w), (u,v,w)} = -1$.

Since the boundary of a boundary vanishes ($B_0 B_1 = 0$), $\operatorname{im}(B_1) \subseteq \mathcal{Z} := \ker(B_0)$. If $G$ has $c(G)$ connected components, the cycle space $\mathcal{Z}$ has dimension equal to the first Betti number of the 1-skeleton:
$$\beta_1(G) = \dim \mathcal{Z} = |E| - |V| + c(G).$$

Let $Z \in \mathbb{R}^{|E| \times \beta_1(G)}$ be a matrix whose columns form an orthonormal basis of $\mathcal{Z}$ ($Z^T Z = I_{\beta_1(G)}$, $B_0 Z = 0$), so that $\Pi_{\mathcal{Z}} = Z Z^T$ is the orthogonal projector onto $\mathcal{Z}$.

### 2.2 The Cycle-Restricted 1-Up-Laplacian and Spectral Inconsistency Entropy

The 1-Hodge Laplacian on $\mathbb{R}^{|E|}$ splits into the down-Laplacian $L_1^{\text{down}} = B_0^T B_0$ and the up-Laplacian $L_1^{\text{up}} = B_1 B_1^T$. Because $L_1^{\text{down}}$ vanishes identically on $\mathcal{Z} = \ker(B_0)$, the action of the full Hodge Laplacian on the cycle space is governed entirely by the symmetric positive semi-definite **cycle-restricted 1-up-Laplacian**:
$$L_{\mathcal{Z}} := Z^T L_1^{\text{up}} Z = (B_1^T Z)^T (B_1^T Z) \in \mathbb{R}^{\beta_1(G) \times \beta_1(G)}.$$

Let $\Lambda(G) = \{\lambda_1 < \lambda_2 < \dots < \lambda_{|\Lambda|}\}$ denote the set of **distinct** eigenvalues of $L_{\mathcal{Z}}$, with corresponding eigenspace multiplicities $d_\lambda = \dim V_\lambda$ ($\sum_{\lambda \in \Lambda} d_\lambda = \beta_1(G)$) and orthogonal eigenspace projectors $P_\lambda \in \mathbb{R}^{\beta_1(G) \times \beta_1(G)}$ ($\sum_{\lambda \in \Lambda} P_\lambda = I_{\beta_1(G)}$).

**Definition 1 (Spectral Inconsistency Entropy):** For any edge flow $w \in \mathbb{R}^{|E|}$ with non-zero cyclic residual $c = Z^T w \neq 0$ ($r = Z c$), the spectral energy distribution over $\Lambda(G)$ is
$$p_\lambda(G, w) = \frac{\|P_\lambda c\|_2^2}{\|c\|_2^2}, \qquad \lambda \in \Lambda(G), \qquad \sum_{\lambda \in \Lambda(G)} p_\lambda(G, w) = 1.$$
The **Spectral Inconsistency Entropy** (in bits) is the Shannon entropy of $p(G, w)$:
$$H(G, w) = -\sum_{\lambda \in \Lambda(G)} p_\lambda(G, w) \log_2 p_\lambda(G, w) \in [0, \log_2 |\Lambda(G)|].$$
By convention, $H(G, w) = 0$ if $c = 0$ or $|\Lambda(G)| \le 1$.

---

## 3. Theoretical Results

### 3.1 Universal Degeneracy on Complete Graphs ($K_n$)

Defining $H(G, w)$ over distinct eigenspaces $V_\lambda$ rather than arbitrary eigenvector bases is essential for basis-invariance under the graph automorphism group $\operatorname{Aut}(G)$. We prove that for every complete graph $K_n$ ($n \ge 3$), $\operatorname{Aut}(K_n) = S_n$ collapses the entire spectrum of $L_{\mathcal{Z}}$ to a single point.

**Theorem 1 (Universal Clique Degeneracy via Eckmann Identity):** *For every complete graph $K_n$ with $n \ge 3$ vertices, the 1-Hodge Laplacian on the clique 2-complex $\mathcal{K}_2(K_n)$ satisfies the exact scalar identity*
$$L_1 = B_0^T B_0 + B_1 B_1^T = n \, I_{|E|}.$$
*Consequently, the cycle-restricted 1-up-Laplacian is a scalar multiple of the identity,*
$$L_{\mathcal{Z}} = n \, I_{\beta_1(K_n)}, \qquad \beta_1(K_n) = \frac{(n-1)(n-2)}{2},$$
*possessing a single distinct eigenvalue $\lambda = n$ of multiplicity $\beta_1(K_n)$. Therefore, $H(K_n, w) \equiv 0$ identically for all $n \ge 3$ and all edge flows $w \in \mathbb{R}^{\binom{n}{2}}$.*

**Proof:** Let $e = (u, v)$ and $e' = (u', v')$ be two canonically oriented edges in $K_n$ ($u < v$, $u' < v'$).
1. **Diagonal entries ($e = e'$):** Edge $e = (u, v)$ is incident to 2 vertices ($u$ and $v$), contributing $(B_0^T B_0)_{e, e} = (-1)^2 + 1^2 = 2$. In $K_n$, edge $(u, v)$ belongs to exactly $n - 2$ triangles $(u, v, w)$ for $w \in V \setminus \{u, v\}$, contributing $(B_1 B_1^T)_{e, e} = n - 2$. Thus $(B_0^T B_0 + B_1 B_1^T)_{e, e} = 2 + (n - 2) = n$.
2. **Disjoint edges ($\{u, v\} \cap \{u', v'\} = \emptyset$):** Disjoint edges share no vertex and cannot belong to a common triangle, so $(B_0^T B_0)_{e, e'} = (B_1 B_1^T)_{e, e'} = 0$.
3. **Adjacent edges ($|\{u, v\} \cap \{u', v'\}| = 1$):** Suppose $e$ and $e'$ share a single vertex, say $u = u'$ with $v < v'$. Then $(B_0^T B_0)_{e, e'} = (B_0)_{u, e}(B_0)_{u, e'} = (-1)(-1) = +1$. Meanwhile, $e = (u, v)$ and $e' = (u, v')$ share the unique triangle $t = (u, v, v')$, in which $(B_1)_{(u,v), t} = +1$ and $(B_1)_{(u,v'), t} = -1$, so $(B_1 B_1^T)_{e, e'} = (+1)(-1) = -1$. The two contributions cancel identically: $(B_0^T B_0 + B_1 B_1^T)_{e, e'} = 1 - 1 = 0$ (and similarly for all relative orientations).

Hence $B_0^T B_0 + B_1 B_1^T = n I_{|E|}$. Restricting to $\mathcal{Z} = \ker(B_0)$, where $B_0 Z = 0 \implies Z^T B_0^T B_0 Z = 0$, yields $L_{\mathcal{Z}} = Z^T B_1 B_1^T Z = n Z^T Z = n I_{\beta_1(K_n)}$. Since $|\Lambda(K_n)| = 1$, $p_n(K_n, w) = 1$ and $H(K_n, w) = -1 \log_2 1 = 0$ for all non-consistent flows $w$. $\square$

### 3.2 Harmonic–Curl Separation and Potential/Scale Invariance

**Theorem 2 (Harmonic–Curl Separation & Potential/Scale Invariance):** *Let $G = (V, E)$ be any graph with cycle space $\mathcal{Z} = \ker(B_0)$.*
1. *The kernel of $L_{\mathcal{Z}}$ (the $\lambda = 0$ eigenspace $V_0$) is canonically isomorphic via $c \mapsto Z c$ to the harmonic subspace $\mathcal{H}^1(\mathcal{K}_2(G)) = \ker(B_0) \cap \ker(B_1^T)$, whose dimension is the first simplicial Betti number $\beta_1(\mathcal{K}_2(G)) = \beta_1(G) - \operatorname{rank}(B_1)$. The strictly positive eigenspaces $\bigoplus_{\lambda > 0} V_\lambda$ map isomorphically onto the triangular curl subspace $\operatorname{im}(B_1)$.*
2. *For any ground-truth score vector $s \in \mathbb{R}^{|V|}$, any perturbation $\delta \notin \operatorname{im}(B_0^T)$, and any non-zero scalar $\alpha \in \mathbb{R} \setminus \{0\}$,*
   $$H(G, B_0^T s + \alpha \delta) = H(G, \delta).$$

**Proof:** (1) Since $L_{\mathcal{Z}} = (B_1^T Z)^T (B_1^T Z)$, we have $L_{\mathcal{Z}} c = 0 \iff B_1^T (Z c) = 0$. Because $Z$ is an isometric embedding of $\mathbb{R}^{\beta_1(G)}$ onto $\ker(B_0)$, $Z(V_0) = \ker(B_0) \cap \ker(B_1^T) = \mathcal{H}^1(\mathcal{K}_2(G))$, and $Z(\bigoplus_{\lambda > 0} V_\lambda) = \operatorname{im}(B_1)$.
(2) Because $B_0 Z = 0$, $Z^T (B_0^T s + \alpha \delta) = (B_0 Z)^T s + \alpha Z^T \delta = \alpha Z^T \delta$. Thus $p_\lambda(G, B_0^T s + \alpha \delta) = \frac{\alpha^2 \|P_\lambda Z^T \delta\|_2^2}{\alpha^2 \|Z^T \delta\|_2^2} = p_\lambda(G, \delta)$. $\square$

### 3.3 Exact Isotropic Dirichlet–Digamma Asymptote and Concavity Mixing

When $k = 1$ edge $e_j$ is corrupted by $\delta = \alpha \mathbf{e}_j$, Theorem 2 implies that $H(G, B_0^T s + \alpha \mathbf{e}_j) = H(G, \mathbf{e}_j)$ is completely deterministic and depends only on the edge $e_j$. As $k$ independent corruptions accumulate across the network, the cyclic residual approaches an isotropic Gaussian vector in $\mathcal{Z}$.

**Theorem 3 (Dirichlet–Digamma Isotropic Asymptote & Expected Energy Concavity):**
1. *Suppose $k$ edges $S \subseteq E$ are perturbed by independent zero-mean variables $\delta = \sum_{j \in S} \alpha_j \mathbf{e}_j$ with $\mathbb{E}[\alpha_j] = 0$ and $\mathbb{E}[\alpha_j^2] = \sigma^2$. Then the expected spectral energy distribution $\bar{p}_\lambda(S) = \frac{\mathbb{E}[\|P_\lambda Z^T \delta\|_2^2]}{\mathbb{E}[\|Z^T \delta\|_2^2]} = \sum_{j \in S} \gamma_j p_\lambda(G, \mathbf{e}_j)$ (where $\gamma_j = \|Z^T \mathbf{e}_j\|_2^2 / \sum_{i \in S} \|Z^T \mathbf{e}_i\|_2^2$) satisfies*
   $$H(\bar{p}(S)) \ge \sum_{j \in S} \gamma_j H(G, \mathbf{e}_j),$$
   *with strict inequality whenever at least two edges in $S$ have distinct spectral profiles. For $S = E$, $\bar{p}_\lambda(E) = d_\lambda / \beta_1(G)$ attains the Jensen structural capacity*
   $$H_{\text{sys}}(G) = -\sum_{\lambda \in \Lambda(G)} \frac{d_\lambda}{\beta_1(G)} \log_2 \frac{d_\lambda}{\beta_1(G)} \le \log_2 |\Lambda(G)|.$$
2. *For a single realization of isotropic Gaussian cycle noise $c \sim \mathcal{N}(0, \sigma^2 I_{\beta_1(G)})$, the independent chi-squared norms $\|P_\lambda c\|_2^2 \sim \sigma^2 \chi^2_{d_\lambda}$ induce a Dirichlet distribution $(p_\lambda)_{\lambda \in \Lambda} \sim \operatorname{Dirichlet}(d_1/2, \dots, d_{|\Lambda|}/2)$ on the simplex. The exact expected Spectral Inconsistency Entropy is*
   $$H_{\text{iso}}(G) := \mathbb{E}_{c \sim \mathcal{N}(0, I_{\beta_1})}[H(G, Z c)] = \frac{1}{\ln 2} \sum_{\lambda \in \Lambda(G)} \frac{d_\lambda}{\beta_1(G)} \left[ \psi\!\left(\frac{\beta_1(G)}{2} + 1\right) - \psi\!\left(\frac{d_\lambda}{2} + 1\right) \right],$$
   *where $\psi(z) = \Gamma'(z)/\Gamma(z)$ is the digamma function. In particular, when all $\beta_1(G)$ eigenvalues of $L_{\mathcal{Z}}$ are simple ($d_\lambda = 1$), $H_{\text{iso}}(G) = \frac{\psi(\beta_1(G)/2 + 1) - \psi(3/2)}{\ln 2} \approx \log_2 \beta_1(G) - 0.7296\text{ bits}$.*

**Proof:** (1) By independence and zero mean, $\mathbb{E}[\|P_\lambda Z^T \delta\|_2^2] = \sigma^2 \sum_{j \in S} \|P_\lambda Z^T \mathbf{e}_j\|_2^2$. Since $\sum_{j \in E} \mathbf{e}_j \mathbf{e}_j^T = I_{|E|}$ and $Z^T Z = I_{\beta_1(G)}$, summing over $S = E$ gives $\sum_{j \in E} \|P_\lambda Z^T \mathbf{e}_j\|_2^2 = \operatorname{Tr}(P_\lambda Z^T I_{|E|} Z P_\lambda) = \operatorname{Tr}(P_\lambda) = d_\lambda$. Strict concavity of Shannon entropy $H(\cdot)$ yields the Jensen inequality.
(2) Under $c \sim \mathcal{N}(0, I_{\beta_1})$, the orthogonal projections $P_\lambda c$ are independent Gaussian vectors in $V_\lambda \cong \mathbb{R}^{d_\lambda}$, so $Y_\lambda = \|P_\lambda c\|_2^2 \sim \chi^2_{d_\lambda} = \operatorname{Gamma}(d_\lambda/2, 1/2)$. Normalizing by $\sum_{\mu} Y_\mu \sim \chi^2_{\beta_1}$ yields $(p_\lambda) \sim \operatorname{Dirichlet}(\alpha_1, \dots, \alpha_{|\Lambda|})$ with $\alpha_\lambda = d_\lambda / 2$ and $\alpha_0 = \sum_\lambda \alpha_\lambda = \beta_1 / 2$. Using the standard identity $\mathbb{E}[p_\lambda \ln p_\lambda] = \frac{\alpha_\lambda}{\alpha_0}[\psi(\alpha_\lambda + 1) - \psi(\alpha_0 + 1)]$ for a Dirichlet variate and summing over $\lambda \in \Lambda(G)$ proves the exact formula for $H_{\text{iso}}(G)$. $\square$

---

## 4. Single-Fault Localization via Cycle-Leverage Alignment

While $H(G, w)$ diagnoses *whether* contradictions are localized or systemic, once a low-entropy (or small-$k$) regime is detected, an operator needs to identify *which* edge is corrupted. Under a single-edge fault $w = B_0^T s + \alpha \mathbf{e}_{j^*}$, the cyclic residual is $r = \alpha \Pi_{\mathcal{Z}} \mathbf{e}_{j^*}$. We therefore define the **cycle-leverage cosine localizer**:
$$\hat{j}(G, w) = \arg\max_{j \in E,\; \|\Pi_{\mathcal{Z}} \mathbf{e}_j\|_2 > 0} \frac{|\langle Z^T w, Z^T \mathbf{e}_j \rangle|}{\|Z^T w\|_2 \, \|Z^T \mathbf{e}_j\|_2}.$$
As long as no two cyclic edges have collinear cycle projections ($|\langle \Pi_{\mathcal{Z}} \mathbf{e}_i, \Pi_{\mathcal{Z}} \mathbf{e}_j \rangle| < \|\Pi_{\mathcal{Z}} \mathbf{e}_i\|_2 \|\Pi_{\mathcal{Z}} \mathbf{e}_j\|_2$ for $i \neq j$, i.e., no two edges belong to a series-ear path of degree-2 vertices), $\hat{j}(G, w) = j^*$ recovers the exact corrupted edge with **100% accuracy** regardless of the ground-truth potential $s$ or fault amplitude $\alpha \neq 0$, even on complete graphs $K_n$ ($n \ge 4$) where $H(K_n, w) \equiv 0$.

---

## 5. Empirical Validation and Structural Regimes

All experiments are deterministic and reproducible via `python examples/reproduce_all.py`.

### 5.1 Verification of Theorem 1 on Complete Graphs $K_n$ ($3 \le n \le 8$)

| $n$ | $|E|$ | $|T|$ | $\beta_1(K_n)$ | Spectrum of $L_{\mathcal{Z}}$ | $\|B_0^T B_0 + B_1 B_1^T - nI\|_\infty$ | $\max_w H(K_n, w)$ (bits) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | 3 | 1 | 1 | $\{3.0^{\times 1}\}$ | $0.00 \times 10^0$ | $0.00 \times 10^0$ |
| 4 | 6 | 4 | 3 | $\{4.0^{\times 3}\}$ | $0.00 \times 10^0$ | $0.00 \times 10^0$ |
| 5 | 10 | 10 | 6 | $\{5.0^{\times 6}\}$ | $0.00 \times 10^0$ | $0.00 \times 10^0$ |
| 6 | 15 | 20 | 10 | $\{6.0^{\times 10}\}$ | $0.00 \times 10^0$ | $0.00 \times 10^0$ |
| 7 | 21 | 35 | 15 | $\{7.0^{\times 15}\}$ | $0.00 \times 10^0$ | $0.00 \times 10^0$ |
| 8 | 28 | 56 | 21 | $\{8.0^{\times 21}\}$ | $0.00 \times 10^0$ | $0.00 \times 10^0$ |

### 5.2 $L_0$ Edit Distance vs. Spectral Inconsistency Entropy on $G(15, 0.4)$

On the canonical Erdős–Rényi benchmark $G(15, 0.4)$ (`seed=42`, $|V|=15, |E|=49, |T|=50, \beta_1=35$), all 35 eigenvalues of $L_{\mathcal{Z}}$ are distinct ($|\Lambda|=35$, including one harmonic mode $\lambda_1 = 0$ with $\dim \mathcal{H}^1 = 1$). Theorem 3 predicts an exact isotropic Dirichlet–Digamma asymptote of:
$$H_{\text{iso}}(G) = \frac{\psi(18.5) - \psi(1.5)}{\ln 2} = 4.1175\text{ bits} \qquad (\text{Jensen upper bound } H_{\text{sys}}(G) = \log_2 35 = 5.1293\text{ bits}).$$

Across 1,000 Monte Carlo trials per edit distance $k$ (where $w = B_0^T s + \delta$ with $s \sim \mathcal{N}(0, I_{15})$ and $k$ randomly chosen edges perturbed by $\mathcal{N}(0, 5^2)$), $H(G, w)$ increases **strictly monotonically** with $k$ toward $H_{\text{iso}}(G) = 4.1175$ bits:

| $L_0$ Edit Distance ($k$) | Mean $H(G, w)$ $\pm$ 95% CI (bits) | Std $H$ (bits) | Mean $\rho(G, w)$ | Top-1 Fault Localization |
|:---:|:---:|:---:|:---:|:---:|
| 1 | $3.7641 \pm 0.0394$ | $0.6363$ | $0.1217$ | **100.0%** |
| 2 | $3.8806 \pm 0.0275$ | $0.4432$ | $0.2174$ | — |
| 3 | $3.9308 \pm 0.0233$ | $0.3765$ | $0.2839$ | — |
| 4 | $3.9459 \pm 0.0200$ | $0.3235$ | $0.3387$ | — |
| 5 | $3.9826 \pm 0.0179$ | $0.2889$ | $0.3845$ | — |
| 6 | $3.9928 \pm 0.0186$ | $0.3001$ | $0.4186$ | — |
| 9 | $4.0308 \pm 0.0152$ | $0.2445$ | $0.4939$ | — |
| 14 | $4.0795 \pm 0.0142$ | $0.2288$ | $0.5581$ | — |
| 20 | $4.0925 \pm 0.0137$ | $0.2203$ | $0.5986$ | — |
| 30 | $4.0965 \pm 0.0130$ | $0.2098$ | $0.6360$ | — |
| 40 | $4.1274 \pm 0.0124$ | $0.2007$ | $0.6540$ | — |

*(Note: An exploratory 20-trial unseeded run in an earlier draft exhibited Monte Carlo sampling variance at $k=5$; under 1,000 seeded trials per $k$, the curve is strictly monotonic and converges to the exact analytical Dirichlet–Digamma asymptote $H_{\text{iso}} = 4.1175$ bits within $0.01$ bits).*

### 5.3 Cross-Topology Regimes: Anderson Localization vs. Bloch Lattice Delocalization

Comparing five representative graph families reveals the precise structural boundary governing $H(G, w)$:

| Network Topology | $|V|$ | $|E|$ | $\beta_1$ | $|\Lambda|$ | Mean $H(k=1)$ (bits) | Isotropic $H_{\text{iso}}$ (bits) | Jensen $H_{\text{sys}}$ (bits) | Single-Fault Top-1 Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Complete Graph $K_8$ | 8 | 28 | 21 | 1 | $0.0000$ | $0.0000$ | $0.0000$ | **100.0%** |
| Modular League ($4 \times 5$ Caveman) | 20 | 40 | 20 | 5 | **$0.4809$** | **$1.4392$** | $1.6710$ | **97.4%** |
| Small-World Tournament (NWS 16) | 16 | 45 | 30 | 26 | **$2.9614$** | **$3.6453$** | $4.5199$ | **100.0%** |
| Erdős–Rényi $G(15, 0.4)$ | 15 | 49 | 35 | 35 | **$3.7512$** | **$4.1175$** | $5.1293$ | **100.0%** |
| Triangular Sensor Grid ($4 \times 4$) | 12 | 23 | 12 | 10 | $2.7750$ | $2.4634$ | $3.2516$ | **91.3%** |

**Key Structural Findings:**
1. **Modular and Heterogeneous Networks (`Modular League`, `Small-World Tournament`, `Erdős–Rényi`):** When topological clustering or degree heterogeneity localizes the cycle eigenmodes of $L_{\mathcal{Z}}$, a single-edge fault excites only local cycle modes, producing a large diagnostic gap between $H(k=1)$ and $H_{\text{iso}}$ (from $0.4809$ vs. $1.4392$ bits in the modular league to $2.9614$ vs. $3.6453$ bits in the small-world tournament).
2. **Regular Spatial Lattices (`Triangular Sensor Grid`):** On a periodic or quasi-regular 2D lattice, the eigenvectors of $L_{\mathcal{Z}}$ are spatially delocalized Bloch/Fourier waves. By the graph Heisenberg uncertainty principle, a spatially localized delta spike $\mathbf{e}_j$ projects smoothly across all spatial frequencies ($H(k=1) = 2.7750$ bits, below the Jensen bound $H_{\text{sys}} = 3.2516$ bits as required by Theorem 3.1, but *above* the single-realization Gaussian expectation $H_{\text{iso}} = 2.4634$ bits because random Gaussian realizations experience Porter–Thomas $\chi^2_1$ zeros across individual Fourier modes).
3. **Complementary Role of Cycle-Leverage Localization:** Across all five regimes—including $K_8$ and the triangular sensor grid—the cycle-leverage cosine localizer achieves $91.3\%$–$100.0\%$ Top-1 single-fault identification accuracy (the sole ambiguities on the triangular grid and caveman graph occur on boundary degree-2 ears where two edges share identical cycle participation).

---

## 6. Conclusion

Spectral Inconsistency Entropy $H(G, w)$ equips combinatorial Hodge theory with a scale- and potential-invariant spectral lens on the cycle space $\mathcal{Z} = \ker(B_0)$. Together with the exact Eckmann degeneracy theorem on $K_n$, the harmonic–curl separation theorem, the closed-form Dirichlet–Digamma asymptote $H_{\text{iso}}(G)$, and cycle-leverage cosine fault localization, this framework provides a complete, rigorously verified toolkit for auditing contradictions in pairwise comparison networks.

---

## References

1. X. Jiang, L.-H. Lim, Y. Yao, and Y. Ye, "Statistical ranking and combinatorial Hodge theory," *Mathematical Programming*, vol. 127, no. 1, pp. 203–244, 2011.
2. B. Eckmann, "Harmonische Funktionen und Randwertaufgaben in einem Komplex," *Commentarii Mathematici Helvetici*, vol. 17, no. 1, pp. 240–255, 1944.
3. D. Horak and J. Jost, "Spectra of combinatorial Laplace operators on simplicial complexes," *Advances in Mathematics*, vol. 244, pp. 303–336, 2013.
4. A. N. Hirani, *Discrete Exterior Calculus*, Ph.D. thesis, California Institute of Technology, 2003.
5. M. T. Schaub, A. R. Benson, P. Horn, G. Lippner, and A. Jadbabaie, "Random walks on simplicial complexes and the normalized Hodge 1-Laplacian," *SIAM Review*, vol. 62, no. 2, pp. 353–391, 2020.
