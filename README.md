# 🧬 Monte Carlo

[![CI Status](https://github.com/Alphalite07/monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/Alphalite07/monte-carlo/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Monte Carlo** is a unified, pure-Python/NumPy computational framework implementing rigorous **Monte Carlo (MC)** and **Markov Chain Monte Carlo (MCMC)** algorithms across four foundational scales of bioinformatics and computational biology: **Structural Biophysics**, **Systems Biology**, **Regulatory Genomics**, and **Molecular Evolution**.

Designed for both research prototyping and graduate-level pedagogy, **Monte Carlo** pairs vectorized simulation kernels with an automated **Bayesian convergence diagnostic suite** ($\hat{R}$, FFT-ESS, HPD intervals, and autocorrelation analysis).

---

## ✨ Key Features & Architecture

| Module | Biological Problem | Algorithm Implemented | Key Observables |
| :--- | :--- | :--- | :--- |
| **`OffLatticeProteinREMC`** | 3D *Ab Initio* Protein Folding | **Replica Exchange Monte Carlo (Parallel Tempering)** with 3D Rodrigues' Pivot & Local Crankshaft Moves | Minimum-energy 3D $\text{C}_\alpha$ coordinates, Radius of Gyration ($R_g$), Specific Heat Capacity ($C_v$) |
| **`StochasticGeneNetworkMC`** | Single-Cell Gene Expression Kinetics | **Exact Gillespie Direct SSA** & **Hybrid Poisson $\tau$-Leaping** | Transcriptional bursting trajectories, mRNA/Protein copy-number distributions, Fano factors ($\sigma^2/\mu$) |
| **`GibbsMotifSampler`** | *De Novo* TFBS Motif Discovery | **Collapsed Gibbs Sampler** with Simulated Annealing & Phase-Shift Register Correction | Position Weight Matrices (PWM), Information Content (Bits / KL-Divergence), Consensus Binding Sites |
| **`BayesianPhylogeneticMCMC`** | Evolutionary Divergence Estimation | **Multi-Chain Random-Walk Metropolis-Hastings** under the Kimura 2-Parameter (K2P) Model | Joint posterior of evolutionary distance ($d$) and transition/transversion rate ratio ($\kappa$) |
| **`MCDiagnostics`** | Markov Chain Quality Control | **Split-Chain Gelman-Rubin ($\hat{R}$)**, **Geyer's Initial Positive Sequence ESS**, **FFT Autocorrelation** | Multi-chain stationarity verification, 95% Highest Posterior Density (HPD) intervals |

---

## 🚀 Installation

### Option 1: Quick Install (Clone & Run)
```bash
git clone https://github.com/Alphalite07/monte-carlo.git
cd monte-carlo
pip install -r requirements.txt
```

### Option 2: Editable Package Install (Enables the `monte-carlo` CLI)
```bash
pip install -e .
```

---

## 🖥️ Quickstart

### 1. Run the Full Multi-Scale Benchmark & Visual Dashboard
Execute all four biological Monte Carlo engines and generate the 6-panel diagnostic figure:

```bash
python cli.py --mode all --save-fig assets/dashboard_preview.png
```

**Sample Terminal Output:**
```text
==============================================================================
   MONTE CARLO: MULTI-SCALE BIOINFORMATICS SIMULATION SUITE
==============================================================================

[1/4] Running 3D Off-Lattice Protein Folding (Replica Exchange MC)...
      -> Sequence         : HPHPPHHPHPPHPHHPPHPH (N=20)
      -> Lowest Energy E* : -11.428 kcal/mol (reduced)
      -> Native R_g       : 1.842
      -> REMC Swap Rate   : 38.3%

[2/4] Running Stochastic Gene Regulatory Network (Gillespie SSA & Tau-Leaping)...
      -> Exact SSA Events : 11,482 reaction firings
      -> mRNA Fano Factor : 4.85 (Super-Poissonian Bursting)
      -> Prot Fano Factor : 18.12

[3/4] Running De Novo TFBS Motif Discovery (Collapsed Gibbs Sampler)...
      -> Planted Motif    : TATAAAWG
      -> Discovered Motif : TATAAAAG
      -> Info Content     : 11.84 bits
      -> Site Recovery    : 96.4% within +/-1 bp

[4/4] Running Bayesian Phylogenetic MCMC (Kimura 2-Parameter Model)...
      -> Distance (d)     : Post. Mean = 0.379 (True = 0.380), 95% HPD = [0.338, 0.421]
      -> Ratio (kappa)    : Post. Mean = 3.441 (True = 3.400), 95% HPD = [2.610, 4.352]
      -> Convergence      : R-hat(d) = 1.0012, R-hat(kappa) = 1.0024 (Target < 1.05)
      -> Effective Samples: ESS(d) = 2418, ESS(kappa) = 2190
==============================================================================
```

### 2. Command-Line Usage on Custom FASTA Files

**Discover a 10-bp transcription factor motif from an unaligned FASTA file:**
```bash
python cli.py --mode motif --fasta examples/sample_promoters.fasta --motif-width 10 --iterations 1000
```

**Estimate Bayesian evolutionary divergence from two aligned sequences in a FASTA file:**
```bash
python cli.py --mode phylo --fasta examples/aligned_orthologs.fasta --chains 4 --steps 10000
```

**Fold a custom Hydrophobic-Polar (HP) sequence with 8 temperature replicas:**
```bash
python cli.py --mode protein --hp-seq HHPHPHPHPHHHHPHPPPHH --replicas 8 --steps 5000
```

---

## 🐍 Python API Usage

Every engine is modular and can be imported independently into your own bioinformatics pipelines or Jupyter notebooks:

```python
from monte_carlo import (
    OffLatticeProteinREMC,
    StochasticGeneNetworkMC,
    GibbsMotifSampler,
    BayesianPhylogeneticMCMC,
    MCDiagnostics
)

# 1. Replica Exchange Protein Folding
folding = OffLatticeProteinREMC("HPHPPHHPHPPHPHHPPHPH", seed=42)
remc_result = folding.run_remc(n_replicas=6, t_min=0.15, t_max=2.0, n_steps=3000)
print(f"Minimum Energy: {remc_result.lowest_energy:.3f}")

# 2. Stochastic Gene Expression (Exact Gillespie SSA)
network = StochasticGeneNetworkMC(k_on=0.2, k_off=0.3, k_tx=4.0, seed=42)
ssa_result = network.run_gillespie_ssa(t_max=500.0)
print("mRNA Fano Factor:", ssa_result.fano_factors["mRNA"])

# 3. Bayesian K2P Phylogenetics with Convergence Diagnostics
s1, s2 = BayesianPhylogeneticMCMC.simulate_aligned_pair(seq_len=1200, true_d=0.42, true_kappa=2.8)
mcmc = BayesianPhylogeneticMCMC(s1, s2, seed=42)
post = mcmc.run_mcmc(n_chains=4, n_steps=5000, burn_in=1000)
print(f"d = {post.d_chains.mean():.3f}, R-hat = {post.rhat_d:.4f}, ESS = {post.ess_d:.0f}")
```

---

## 🧪 Running the Test Suite

Unit tests verify mathematical invariants (stochastic matrix normalization, zero-divergence analytical limits, detailed balance, and MCMC convergence):

```bash
pytest -v
```

---

## 📄 Citation

If you use **Monte Carlo** in academic coursework, benchmarks, or research, please cite it using the repository metadata (`CITATION.cff`).

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
