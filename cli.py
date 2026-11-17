#!/usr/bin/env python3
"""
Command-Line Interface (CLI) and FASTA I/O wrapper for Monte Carlo.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from monte_carlo import (
    MonteCarloSuite,
    OffLatticeProteinREMC,
    StochasticGeneNetworkMC,
    GibbsMotifSampler,
    BayesianPhylogeneticMCMC,
)


def parse_fasta(filepath: Path) -> List[Tuple[str, str]]:
    """Lightweight, dependency-free FASTA file parser returning (header, sequence) pairs."""
    records: List[Tuple[str, str]] = []
    header = None
    seq_chunks: List[str] = []

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.upper())

    if header is not None:
        records.append((header, "".join(seq_chunks)))

    if not records:
        raise ValueError(f"No valid FASTA records found in {filepath}")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monte-carlo",
        description="Monte Carlo: Multi-Scale Bioinformatics Simulation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "protein", "ssa", "motif", "phylo"],
        default="all",
        help="Select which Monte Carlo simulation engine to execute.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Master random number generator seed.")
    parser.add_argument("--fasta", type=Path, default=None, help="Path to input FASTA file (for motif or phylo modes).")
    parser.add_argument("--hp-seq", type=str, default="HPHPPHHPHPPHPHHPPHPH", help="HP sequence for protein folding.")
    parser.add_argument("--replicas", type=int, default=6, help="Number of temperature replicas for REMC.")
    parser.add_argument("--steps", type=int, default=3000, help="Number of MC steps/sweeps per chain.")
    parser.add_argument("--chains", type=int, default=4, help="Number of independent Markov chains (phylo mode).")
    parser.add_argument("--motif-width", type=int, default=8, help="Motif width W in bp (motif mode).")
    parser.add_argument("--iterations", type=int, default=500, help="Gibbs sampler iterations (motif mode).")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive matplotlib window.")
    parser.add_argument("--save-fig", type=Path, default=None, help="Save diagnostic figure to file (e.g. dashboard.png).")
    return parser


def main():
    args = build_parser().parse_args()

    if args.no_plot or args.save_fig is not None:
        matplotlib.use("Agg")

    if args.mode == "all":
        suite = MonteCarloSuite(seed=args.seed)
        suite.run_all(plot_dashboard=not args.no_plot or args.save_fig is not None)
        if args.save_fig is not None:
            args.save_fig.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.save_fig, dpi=220, bbox_inches="tight")
            print(f"[+] Saved diagnostic dashboard to: {args.save_fig}")

    elif args.mode == "protein":
        engine = OffLatticeProteinREMC(args.hp_seq, seed=args.seed)
        res = engine.run_remc(n_replicas=args.replicas, n_steps=args.steps)
        print(f"[Protein REMC] Sequence      : {res.sequence}")
        print(f"[Protein REMC] Lowest Energy : {res.lowest_energy:.4f}")
        print(f"[Protein REMC] Final R_g     : {res.rg_trajectories[0, -1]:.4f}")
        print(f"[Protein REMC] Swap Rate     : {res.swap_acceptance_rate * 100:.1f}%")

    elif args.mode == "ssa":
        net = StochasticGeneNetworkMC(seed=args.seed)
        ssa = net.run_gillespie_ssa(t_max=300.0)
        print(f"[Gillespie SSA] Total Reaction Events : {len(ssa.times):,}")
        for sp, ff in ssa.fano_factors.items():
            print(f"[Gillespie SSA] Fano Factor ({sp:7s}) : {ff:.3f}")

    elif args.mode == "motif":
        if args.fasta is not None:
            records = parse_fasta(args.fasta)
            min_len = min(len(seq) for _, seq in records)
            seqs = [seq[:min_len] for _, seq in records]
            print(f"[Gibbs Motif] Loaded {len(seqs)} sequences (truncated to uniform length {min_len} bp).")
        else:
            seqs, _ = GibbsMotifSampler.generate_synthetic_dataset(seed=args.seed)
            print("[Gibbs Motif] No --fasta provided; using synthetic planted TATA-box dataset.")

        sampler = GibbsMotifSampler(seqs, motif_width=args.motif_width, seed=args.seed)
        res = sampler.run(n_iterations=args.iterations, n_restarts=3)
        print(f"[Gibbs Motif] Consensus Motif     : {res.consensus_motif}")
        print(f"[Gibbs Motif] Information Content : {res.information_content_bits:.2f} bits")

    elif args.mode == "phylo":
        if args.fasta is not None:
            records = parse_fasta(args.fasta)
            if len(records) < 2:
                raise ValueError("FASTA file must contain at least 2 aligned sequences for phylo mode.")
            s1, s2 = records[0][1], records[1][1]
            print(f"[Phylo MCMC] Comparing '{records[0][0]}' vs '{records[1][0]}' ({len(s1)} bp)")
        else:
            s1, s2 = BayesianPhylogeneticMCMC.simulate_aligned_pair(seed=args.seed)
            print("[Phylo MCMC] No --fasta provided; using simulated K2P alignment (d=0.35, kappa=3.2).")

        burn = max(100, args.steps // 4)
        mcmc = BayesianPhylogeneticMCMC(s1, s2, seed=args.seed)
        res = mcmc.run_mcmc(n_chains=args.chains, n_steps=args.steps, burn_in=burn)
        print(f"[Phylo MCMC] Distance (d)  : {np.mean(res.d_chains):.4f} (95% HPD: {res.d_hpd[0]:.4f} - {res.d_hpd[1]:.4f})")
        print(f"[Phylo MCMC] Ratio (kappa) : {np.mean(res.kappa_chains):.4f} (95% HPD: {res.kappa_hpd[0]:.4f} - {res.kappa_hpd[1]:.4f})")
        print(f"[Phylo MCMC] Convergence   : R-hat(d)={res.rhat_d:.4f}, ESS(d)={res.ess_d:.0f}")


if __name__ == "__main__":
    main()