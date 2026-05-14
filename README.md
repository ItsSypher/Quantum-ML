# Neural Quantum States for the XXZ Chain

This repository contains a single self-contained notebook, [`Extension.ipynb`](Extension.ipynb), that trains four neural-quantum-state (NQS) architectures by variational Monte Carlo + stochastic reconfiguration on the 1D spin-$`\tfrac12`$ XXZ chain, then scans the anisotropy $`\Delta\in[0,3]`$ across the XY phase, the BKT point at $`\Delta=1`$, and into the Ising antiferromagnet.

## Model

Pauli-normalised XXZ Hamiltonian with periodic boundary conditions, sector $`S^z_\text{tot}=0`$:

```math
H_{XXZ}(\Delta) = \sum_{i=1}^{L}\left(\sigma^x_i\sigma^x_{i+1} + \sigma^y_i\sigma^y_{i+1} + \Delta\,\sigma^z_i\sigma^z_{i+1}\right).
```

| range of $`\Delta`$ | phase |
|---|---|
| $`\Delta<-1`$ | gapped ferromagnet |
| $`-1<\Delta\le 1`$ | gapless XY / Luttinger liquid |
| $`\Delta=1`$ | BKT (isotropic Heisenberg) |
| $`\Delta>1`$ | gapped Ising-like AFM |

Three numerical anchors: Lanczos at $`L=22`$ for every $`\Delta`$, the Hulthén closed form $`e_0(1)=1-4\ln 2`$, and the Bethe-ansatz integrals (Yang–Yang 1966) plotted as a continuous $`L\to\infty`$ reference.

## Architectures

| ansatz | structure | $`L=22`$ params |
|---|---|---|
| Jastrow | $`\log\psi = \sum_i v_i\sigma_i + \sum_{ij}\sigma_i J_{ij}\sigma_j`$ | 506 |
| RBM | $`\log\psi = \sum_i a_i\sigma_i + \sum_j\log\cosh(b_j + \sum_i W_{ji}\sigma_i)`$, $`\alpha=1`$ | 528 |
| RBMSymm | RBM with translation weight-tying, $`\alpha=1`$ | 24 |
| FFNN-1L | one dense complex layer $`\mathbb{C}^L\to\mathbb{C}^{2L}`$, log-cosh, sum-reduce | 1012 |
| FFNN-2L | two dense layers $`\mathbb{C}^L\to\mathbb{C}^{2L}\to\mathbb{C}^L`$, log-cosh between | 1012 |

All four reach the Lanczos energy at $`L=22,\ \Delta=1`$ to **better than $`10^{-3}`$** relative error. RBMSymm gets there with **5%** of the plain-RBM parameter count — the simplest manifestation of *information reuse* in the sense of Levine et al., PRL **122**, 065301 (2019).

## Key implementation choices

- **`PauliStringsJax` Hamiltonian.** Built directly from Pauli strings, avoiding the slower `LocalOperator` path used in the tutorial.
- **Marshall–Peierls sign rule** (Marshall 1955) applied on the bipartite even-$`L`$ chain, $`\sigma^{x,y}\to-\sigma^{x,y}`$ on the $`B`$ sublattice. The eigenvalues are unchanged but $`\tilde H`$ is **stoquastic** in the $`\sigma^z`$ basis (Bravyi–DiVincenzo–Oliveira–Terhal 2008), so the ground state is real-positive and VMC converges with the same hyperparameters across the entire $`\Delta`$ range. Observables are translated back as needed:

  ```math
  \langle\sigma^x_0\sigma^x_r\rangle = (-1)^r\,\langle\sigma^x_0\sigma^x_r\rangle_{\tilde H}.
  ```
- **Warm-start** across the $`\Delta`$ scan: parameters trained at one $`\Delta`$ initialise the next. First point uses 400–500 iterations, subsequent points 200–250.
- **Batched observables.** The connected ZZ correlator at fixed displacement $`r`$ is built as a single `PauliStrings` summing $`L`$ pairs with weight $`1/L`$, cutting $`L\cdot L/2`$ MC passes down to $`L/2`$ — ~22× speedup for the $`C^{zz}`$ measurement at $`L=22`$.
- **Disk-cached results.** Trained variational parameters are saved to `.mpack`, observable summaries to `.pkl`, and the Lanczos reference to `.pkl`. Re-running the notebook hits the cache.

## Results at a glance

End-to-end run on a regular laptop, full $`\Delta`$ grid $`\{0.0, 0.4, 0.8, 0.9, 1.0, 1.1, 1.2, 1.6, 2.0, 2.5, 3.0\}`$.

**Energy errors vs Lanczos at $`L=22`$:** every ansatz under $`0.5\%`$ across the whole scan; FFNN-2L under $`0.1\%`$ at all four representative $`\Delta`$.

**Variance separation by phase:**
- $`\Delta=0`$ (free fermions): all ansätze $`\sigma_E^2 \le 0.03`$; Jastrow saturates at $`0.000`$.
- $`\Delta=1`$ (Heisenberg): RBMSymm 0.114 < Jastrow 0.195 ≈ RBM 0.209.
- $`\Delta=2.5`$ (Ising-AFM): FFNN-2L 0.92 ≪ RBM 1.34 < RBMSymm 3.55 ≈ Jastrow 3.43.

**Néel order parameter from $`S_\text{AFM}(\pi)/L`$ at $`\Delta=3`$, RBMSymm, three sizes:** 13.3 / 17.5 / 22.0 for $`L=16/22/28`$, giving $`m_s^2 \simeq 0.79`$.

**Luttinger exponents from log-log fit of $`|C^x(r)|`$:** $`\eta_x^\text{fit}(0.4)=0.39`$ vs theory $`0.37`$; $`\eta_x^\text{fit}(0.8)=0.28`$ vs theory $`0.21`$. At $`\Delta=1`$ the bare fit gives $`0.36`$, undershooting the $`\eta=1`$ theory value because of the marginal log correction (Affleck 1998).

## Repository layout

```
.
├── Extension.ipynb           # main deliverable: notebook with all simulations
├── _build_notebook.py        # builds Extension.ipynb from a list of cells
├── Assignment/               # the project brief PDF (Aalto, 2026)
├── results/
│   ├── heisenberg/           # logs / parameters from the L=22 Delta=1 reproduction
│   ├── xxz/                  # per-ansatz Delta-scan: *.log, *.mpack, *.pkl
│   └── jax_cache/            # JAX persistent compilation cache
├── pyproject.toml            # uv project file
├── uv.lock
└── README.md
```

## Quick start

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                                  # install Python 3.12 + locked deps
uv run jupyter lab Extension.ipynb       # open the notebook
```

Or with an existing Python 3.12 virtualenv:

```bash
pip install "netket==3.21.*" "jax>=0.7,<0.8" "flax<0.12.7" \
            numpy scipy matplotlib jupyter
jupyter lab Extension.ipynb
```

Run the notebook top-to-bottom. First run does the full sweep (≈15–25 min on a laptop CPU); re-runs hit the on-disk cache and complete in seconds.

## Notebook structure

1. Setup, persistent JAX cache, results directory.
2. **§1** XXZ Hamiltonian built via `PauliStringsJax` with the Marshall sign rule.
3. **§2** Bethe-ansatz reference (Cloizeaux–Pearson, Hulthén, Yang–Yang) in Pauli normalisation, validated against Lanczos to within $`\sim 1/L^2`$.
4. **§3** Lanczos at $`L=22,\ \Delta=1`$ → $`E_0 \simeq -39.1475`$.
5. **§4** Four NQS ansätze trained at the Heisenberg point with `VMC_SR`, each followed by a trace plot. Final summary table of (params, $`E_\text{final}`$, relative error).
6. **§5** Δ-scan extension: observable builders (staggered structure factor, $`C^x(r)`$, batched $`C^{zz}(r)`$), `sweep(...)` driver with warm-start and pickle cache, runs at $`L\in\{16,22,28\}`$ for RBMSymm and at $`L=22`$ for the other ansätze, deeper FFNN at four representative $`\Delta`$, Lanczos reference curve.
7. **§6** Plots: $`E_0/L`$ vs $`\Delta`$ vs Bethe / Lanczos / four ansätze, $`S_\text{AFM}(\pi)/L`$ for three sizes, $`\log|C^x(r)|`$ vs $`\log r`$ at three $`\Delta`$, $`\sigma_E^2`$ vs $`\Delta`$.
8. **§7** Discussion: BKT location at finite size, phase hallmarks tied to actual measured values, ansatz hierarchy tied back to Levine et al.

## References

The full numbered reference list lives inside [`Extension.ipynb`](Extension.ipynb), §7. Headline sources:

- Y. Levine, O. Sharir, N. Cohen, A. Shashua, *Quantum entanglement in deep learning architectures*, **Phys. Rev. Lett. 122, 065301 (2019)**, [arXiv:1803.09780](https://arxiv.org/abs/1803.09780).
- G. Carleo, M. Troyer, *Solving the quantum many-body problem with artificial neural networks*, **Science 355, 602 (2017)**.
- C. N. Yang, C. P. Yang, *One-dimensional chain of anisotropic spin-spin interactions. II*, **Phys. Rev. 150, 327 (1966)**.
- I. Affleck, *Exact correlation amplitude for the $`S=\tfrac12`$ Heisenberg antiferromagnetic chain*, **J. Phys. A 31, 4573 (1998)**, [cond-mat/9802045](https://arxiv.org/abs/cond-mat/9802045).
- F. Vicentini *et al.*, *NetKet 3: machine learning toolbox for many-body quantum systems*, **SciPost Phys. Codebases 7 (2022)**.
- NetKet tutorial: [Ground-State: Heisenberg model](https://netket.readthedocs.io/en/latest/tutorials/gs-heisenberg.html).

## Acknowledgements

- Assignment and tutorial scaffolding by Anouar Moustaj (Aalto University, Department of Applied Physics).
- The NetKet `Jastrow`, `RBM`, `RBMSymm`, and FFNN code patterns follow the official NetKet *Ground-State: Heisenberg* tutorial.
