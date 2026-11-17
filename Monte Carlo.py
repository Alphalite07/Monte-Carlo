#!/usr/bin/env python3
"""
MONTE CARLO: Multi-Scale Monte Carlo Simulation Suite for Bioinformatics
========================================================================
Covers:
  1. Bayesian MCMC Diagnostics (Gelman-Rubin R-hat, FFT-ESS, Autocorrelation)
  2. Structural Biophysics (3D Off-Lattice Protein Folding via Replica Exchange MC)
  3. Systems Biology (Stochastic Gene Expression via Gillespie SSA & Tau-Leaping)
  4. Sequence Genomics (De Novo Motif Discovery via Collapsed Gibbs Sampling)
  5. Molecular Evolution (Bayesian Phylogenetic Parameter Estimation via MCMC)
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# MODULE 1: STATISTICAL & CONVERGENCE DIAGNOSTICS
# =============================================================================

class MCDiagnostics:
    """Statistical convergence and efficiency diagnostics for Markov chains."""

    @staticmethod
    def autocorrelation_fft(chain: np.ndarray) -> np.ndarray:
        """Compute 1D normalized autocorrelation function using Fast Fourier Transform."""
        x = np.asarray(chain, dtype=np.float64)
        n = len(x)
        variance = np.var(x)
        if variance < 1e-15:
            return np.ones(n)
        x_centered = x - np.mean(x)
        n_fft = 1 << (2 * n - 1).bit_length()
        f = np.fft.fft(x_centered, n=n_fft)
        acf = np.fft.ifft(f * np.conjugate(f))[:n].real
        acf /= acf[0]
        return acf

    @staticmethod
    def effective_sample_size(chains: np.ndarray) -> float:
        """
        Compute Effective Sample Size (ESS) using Geyer's Initial Positive Sequence.
        Accepts shape (n_samples,) or (n_chains, n_samples).
        """
        chains = np.atleast_2d(chains)
        n_chains, n_samples = chains.shape
        total_ess = 0.0

        for m in range(n_chains):
            acf = MCDiagnostics.autocorrelation_fft(chains[m])
            rho_sum = 0.0
            for k in range(1, n_samples - 1, 2):
                pair_sum = acf[k] + acf[k + 1]
                if pair_sum < 0:
                    break
                rho_sum += pair_sum
            tau_int = max(1.0, 1.0 + 2.0 * rho_sum)
            total_ess += n_samples / tau_int

        return float(total_ess)

    @staticmethod
    def gelman_rubin_rhat(chains: np.ndarray) -> float:
        """
        Compute split-chain Gelman-Rubin R-hat convergence statistic.
        Input shape: (n_chains, n_samples). R-hat < 1.05 indicates convergence.
        """
        chains = np.atleast_2d(chains)
        m, n = chains.shape
        if m < 2 or n < 4:
            return np.nan

        half = n // 2
        split_chains = np.vstack([chains[:, :half], chains[:, half:2 * half]])
        m_split, n_split = split_chains.shape

        chain_means = np.mean(split_chains, axis=1)
        chain_vars = np.var(split_chains, axis=1, ddof=1)

        W = np.mean(chain_vars)
        if W < 1e-15:
            return 1.0
        B = n_split * np.var(chain_means, ddof=1)

        var_hat_plus = ((n_split - 1.0) / n_split) * W + (1.0 / n_split) * B
        r_hat = np.sqrt(var_hat_plus / W)
        return float(r_hat)

    @staticmethod
    def hpd_interval(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute Highest Posterior Density (HPD) credible interval."""
        x = np.sort(np.asarray(samples).ravel())
        n = len(x)
        n_included = int(np.floor((1.0 - alpha) * n))
        intervals = x[n_included:] - x[:n - n_included]
        min_idx = int(np.argmin(intervals))
        return float(x[min_idx]), float(x[min_idx + n_included])


# =============================================================================
# MODULE 2: STRUCTURAL BIOPHYSICS (3D OFF-LATTICE PROTEIN FOLDING - REMC)
# =============================================================================

@dataclass
class ProteinFoldingResult:
    sequence: str
    temperatures: np.ndarray
    energy_trajectories: np.ndarray
    rg_trajectories: np.ndarray
    lowest_energy: float
    best_coords: np.ndarray
    heat_capacity: np.ndarray
    swap_acceptance_rate: float


class OffLatticeProteinREMC:
    """
    3D Off-Lattice Coarse-Grained Protein Folding via Replica Exchange Monte Carlo
    (Parallel Tempering).
    """

    def __init__(self, hp_sequence: str, seed: int = 42):
        self.sequence = hp_sequence.upper()
        self.n_residues = len(self.sequence)
        self.rng = np.random.default_rng(seed)

        self.is_h = np.array([1 if res == 'H' else 0 for res in self.sequence], dtype=np.int32)
        self.pair_coeff = np.full((self.n_residues, self.n_residues), -0.2, dtype=np.float64)
        hh_mask = np.outer(self.is_h, self.is_h) == 1
        self.pair_coeff[hh_mask] = 1.0

        self.r0 = 1.0
        self.k_bond = 30.0
        self.sigma = 1.0
        self.eps = 1.5

    def _initialize_extended_chain(self) -> np.ndarray:
        coords = np.zeros((self.n_residues, 3), dtype=np.float64)
        angle = np.deg2rad(35.0)
        for i in range(1, self.n_residues):
            dx = self.r0 * np.cos(angle)
            dy = self.r0 * np.sin(angle) * (1 if i % 2 == 0 else -1)
            dz = 0.05 * self.rng.standard_normal()
            coords[i] = coords[i - 1] + np.array([dx, dy, dz])
        return coords

    def compute_energy(self, coords: np.ndarray) -> float:
        bond_vecs = np.diff(coords, axis=0)
        bond_lengths = np.linalg.norm(bond_vecs, axis=1)
        e_bond = 0.5 * self.k_bond * np.sum((bond_lengths - self.r0) ** 2)

        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        idx = np.arange(self.n_residues)
        mask = (idx[None, :] - idx[:, None]) >= 2

        inv_r2 = (self.sigma ** 2) / np.maximum(dist_sq[mask], 0.25)
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2

        c_ij = self.pair_coeff[mask]
        e_lj = np.sum(4.0 * self.eps * (inv_r12 - c_ij * inv_r6))

        return float(e_bond + e_lj)

    @staticmethod
    def radius_of_gyration(coords: np.ndarray) -> float:
        com = np.mean(coords, axis=0)
        return float(np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=1))))

    def _propose_move(self, coords: np.ndarray, step_scale: float = 0.18) -> np.ndarray:
        new_coords = coords.copy()
        if self.rng.random() < 0.70 or self.n_residues < 4:
            idx = self.rng.integers(0, self.n_residues)
            new_coords[idx] += self.rng.normal(0.0, step_scale, size=3)
        else:
            pivot_idx = self.rng.integers(1, self.n_residues - 1)
            axis = self.rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-12
            theta = self.rng.uniform(-0.35, 0.35) * step_scale / 0.18

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            K = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0]
            ])
            R = np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)

            origin = new_coords[pivot_idx].copy()
            tail = new_coords[pivot_idx + 1:] - origin
            new_coords[pivot_idx + 1:] = (tail @ R.T) + origin

        return new_coords

    def run_remc(
        self,
        n_replicas: int = 6,
        t_min: float = 0.15,
        t_max: float = 2.0,
        n_steps: int = 3000,
        swap_interval: int = 25
    ) -> ProteinFoldingResult:
        temperatures = np.geomspace(t_min, t_max, n_replicas)
        betas = 1.0 / temperatures

        replicas = [self._initialize_extended_chain() for _ in range(n_replicas)]
        energies = np.array([self.compute_energy(c) for c in replicas], dtype=np.float64)

        energy_traj = np.zeros((n_replicas, n_steps))
        rg_traj = np.zeros((n_replicas, n_steps))

        best_energy = float(np.min(energies))
        best_coords = replicas[int(np.argmin(energies))].copy()

        swap_attempts = 0
        swap_accepts = 0

        for step in range(n_steps):
            for r in range(n_replicas):
                scale = 0.12 * np.sqrt(temperatures[r] / t_min)
                prop_coords = self._propose_move(replicas[r], step_scale=min(scale, 0.35))
                prop_energy = self.compute_energy(prop_coords)
                dE = prop_energy - energies[r]

                if dE <= 0.0 or self.rng.random() < np.exp(-betas[r] * dE):
                    replicas[r] = prop_coords
                    energies[r] = prop_energy
                    if prop_energy < best_energy:
                        best_energy = prop_energy
                        best_coords = prop_coords.copy()

                energy_traj[r, step] = energies[r]
                rg_traj[r, step] = self.radius_of_gyration(replicas[r])

            if (step + 1) % swap_interval == 0:
                parity = (step // swap_interval) % 2
                for r in range(parity, n_replicas - 1, 2):
                    swap_attempts += 1
                    delta = (betas[r] - betas[r + 1]) * (energies[r + 1] - energies[r])
                    if delta >= 0.0 or self.rng.random() < np.exp(delta):
                        replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]
                        energies[r], energies[r + 1] = energies[r + 1], energies[r]
                        swap_accepts += 1

        burn = n_steps // 2
        e_post = energy_traj[:, burn:]
        heat_capacity = np.var(e_post, axis=1) / (temperatures ** 2)

        return ProteinFoldingResult(
            sequence=self.sequence,
            temperatures=temperatures,
            energy_trajectories=energy_traj,
            rg_trajectories=rg_traj,
            lowest_energy=best_energy,
            best_coords=best_coords,
            heat_capacity=heat_capacity,
            swap_acceptance_rate=swap_accepts / max(1, swap_attempts)
        )


# =============================================================================
# MODULE 3: SYSTEMS BIOLOGY (GILLESPIE SSA & ADAPTIVE TAU-LEAPING)
# =============================================================================

@dataclass
class StochasticKineticsResult:
    times: np.ndarray
    states: np.ndarray
    species_names: List[str]
    method: str
    fano_factors: Dict[str, float]


class StochasticGeneNetworkMC:
    """Stochastic simulation of a Telegraph Autoregulatory Gene Expression Circuit."""

    def __init__(
        self,
        k_on: float = 0.15,
        k_off: float = 0.25,
        feedback_alpha: float = 0.01,
        k_tx: float = 3.5,
        k_tl: float = 2.0,
        d_m: float = 0.30,
        d_p: float = 0.08,
        seed: int = 42
    ):
        self.params = (k_on, k_off, feedback_alpha, k_tx, k_tl, d_m, d_p)
        self.species_names = ["G_off", "G_on", "mRNA", "Protein"]
        self.rng = np.random.default_rng(seed)

        self.stoich = np.array([
            [-1, +1,  0,  0],
            [+1, -1,  0,  0],
            [ 0,  0, +1,  0],
            [ 0,  0,  0, +1],
            [ 0,  0, -1,  0],
            [ 0,  0,  0, -1],
        ], dtype=np.int64)

    def propensities(self, state: np.ndarray) -> np.ndarray:
        g_off, g_on, mrna, prot = state
        k_on, k_off, alpha, k_tx, k_tl, d_m, d_p = self.params
        return np.array([
            k_on * g_off,
            k_off * (1.0 + alpha * prot) * g_on,
            k_tx * g_on,
            k_tl * mrna,
            d_m * mrna,
            d_p * prot
        ], dtype=np.float64)

    def run_gillespie_ssa(
        self,
        t_max: float = 300.0,
        initial_state: Optional[np.ndarray] = None
    ) -> StochasticKineticsResult:
        state = np.array([1, 0, 0, 0], dtype=np.int64) if initial_state is None else initial_state.copy()
        t = 0.0
        times = [0.0]
        states = [state.copy()]

        while t < t_max:
            a = self.propensities(state)
            a0 = np.sum(a)
            if a0 <= 1e-15:
                break

            u1, u2 = self.rng.random(), self.rng.random()
            tau = -np.log(u1) / a0
            t += tau

            cum_a = np.cumsum(a)
            mu = int(np.searchsorted(cum_a, u2 * a0))
            mu = min(mu, len(a) - 1)

            state = state + self.stoich[mu]
            times.append(t)
            states.append(state.copy())

        times_arr = np.array(times)
        states_arr = np.array(states)
        fano = self._compute_fano(states_arr)
        return StochasticKineticsResult(times_arr, states_arr, self.species_names, "Exact Gillespie SSA", fano)

    def run_tau_leaping(
        self,
        t_max: float = 300.0,
        tau_step: float = 0.25,
        initial_state: Optional[np.ndarray] = None
    ) -> StochasticKineticsResult:
        state = np.array([1, 0, 0, 0], dtype=np.int64) if initial_state is None else initial_state.copy()
        t = 0.0
        times = [0.0]
        states = [state.copy()]

        while t < t_max:
            a = self.propensities(state)
            a_promoter = a[0] + a[1]

            dt_switch = -np.log(self.rng.random()) / max(a_promoter, 1e-12)
            dt = min(tau_step, dt_switch, t_max - t)

            k_events = self.rng.poisson(a[2:] * dt)
            delta = k_events @ self.stoich[2:]
            state = np.maximum(0, state + delta)

            if dt == dt_switch and a_promoter > 1e-12:
                if state[0] == 1:
                    state[0], state[1] = 0, 1
                else:
                    state[0], state[1] = 1, 0

            t += dt
            times.append(t)
            states.append(state.copy())

        times_arr = np.array(times)
        states_arr = np.array(states)
        fano = self._compute_fano(states_arr)
        return StochasticKineticsResult(times_arr, states_arr, self.species_names, "Hybrid Tau-Leaping", fano)

    def _compute_fano(self, states: np.ndarray) -> Dict[str, float]:
        half = len(states) // 2
        fano = {}
        for idx, name in enumerate(self.species_names):
            x = states[half:, idx]
            mean_val = float(np.mean(x))
            fano[name] = float(np.var(x) / mean_val) if mean_val > 1e-9 else 0.0
        return fano


# =============================================================================
# MODULE 4: GENOMICS (DE NOVO MOTIF DISCOVERY VIA COLLAPSED GIBBS SAMPLING)
# =============================================================================

@dataclass
class MotifDiscoveryResult:
    pwm: np.ndarray
    information_content_bits: float
    consensus_motif: str
    best_positions: np.ndarray
    score_history: np.ndarray


class GibbsMotifSampler:
    """Collapsed Gibbs Sampler with Simulated Annealing for de novo TFBS Motif Discovery."""

    ALPHABET = "ACGT"
    CHAR_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __init__(self, sequences: List[str], motif_width: int = 8, pseudocount: float = 0.5, seed: int = 42):
        self.sequences = [s.upper() for s in sequences]
        self.n_seqs = len(self.sequences)
        self.seq_len = len(self.sequences[0])
        self.w = motif_width
        self.alpha = pseudocount
        self.rng = np.random.default_rng(seed)

        self.encoded = np.array(
            [[self.CHAR_TO_INT[ch] for ch in seq] for seq in self.sequences],
            dtype=np.int32
        )

        counts = np.bincount(self.encoded.ravel(), minlength=4) + self.alpha
        self.bg_probs = counts / np.sum(counts)

    @classmethod
    def generate_synthetic_dataset(
        cls,
        n_seqs: int = 30,
        seq_len: int = 100,
        planted_motif: str = "TATAWAWR",
        mutation_rate: float = 0.12,
        seed: int = 42
    ) -> Tuple[List[str], np.ndarray]:
        rng = np.random.default_rng(seed)
        bases = np.array(list("ACGT"))
        iupac = {
            'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
            'W': 'AT', 'R': 'AG', 'Y': 'CT', 'S': 'GC'
        }
        w = len(planted_motif)
        sequences = []
        true_positions = np.zeros(n_seqs, dtype=np.int32)

        for i in range(n_seqs):
            bg = rng.choice(bases, size=seq_len, p=[0.28, 0.22, 0.22, 0.28])
            pos = int(rng.integers(0, seq_len - w + 1))
            true_positions[i] = pos

            for j, code in enumerate(planted_motif):
                if rng.random() < mutation_rate:
                    bg[pos + j] = rng.choice(bases)
                else:
                    choices = list(iupac.get(code, code))
                    bg[pos + j] = rng.choice(choices)

            sequences.append("".join(bg))

        return sequences, true_positions

    def _compute_pwm_and_ic(self, positions: np.ndarray, exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, float]:
        counts = np.full((4, self.w), self.alpha, dtype=np.float64)
        for i in range(self.n_seqs):
            if i == exclude_idx:
                continue
            start = positions[i]
            subseq = self.encoded[i, start:start + self.w]
            counts[subseq, np.arange(self.w)] += 1.0

        pwm = counts / np.sum(counts, axis=0, keepdims=True)
        kl_matrix = pwm * np.log2(pwm / self.bg_probs[:, None])
        ic_bits = float(np.sum(kl_matrix))
        return pwm, ic_bits

    def run(self, n_iterations: int = 600, n_restarts: int = 3) -> MotifDiscoveryResult:
        best_overall_ic = -np.inf
        best_overall_pos = None
        best_score_history = None

        max_start = self.seq_len - self.w
        window_idx = np.arange(max_start + 1)[:, None] + np.arange(self.w)[None, :]

        for _ in range(n_restarts):
            positions = self.rng.integers(0, max_start + 1, size=self.n_seqs)
            score_history = np.zeros(n_iterations)

            for it in range(n_iterations):
                beta = min(1.5, 0.5 + 1.0 * (it / n_iterations))

                for i in range(self.n_seqs):
                    pwm, _ = self._compute_pwm_and_ic(positions, exclude_idx=i)
                    log_odds = np.log(pwm / self.bg_probs[:, None])

                    seq_windows = self.encoded[i][window_idx]
                    window_scores = np.sum(log_odds[seq_windows, np.arange(self.w)], axis=1)

                    scaled = beta * (window_scores - np.max(window_scores))
                    probs = np.exp(scaled)
                    probs /= np.sum(probs)
                    positions[i] = self.rng.choice(max_start + 1, p=probs)

                if (it + 1) % 25 == 0:
                    _, current_ic = self._compute_pwm_and_ic(positions)
                    for shift in (-1, 1):
                        shifted = np.clip(positions + shift, 0, max_start)
                        _, shifted_ic = self._compute_pwm_and_ic(shifted)
                        if shifted_ic > current_ic:
                            positions = shifted
                            current_ic = shifted_ic

                _, ic_bits = self._compute_pwm_and_ic(positions)
                score_history[it] = ic_bits

                if ic_bits > best_overall_ic:
                    best_overall_ic = ic_bits
                    best_overall_pos = positions.copy()
                    best_score_history = score_history.copy()

        final_pwm, final_ic = self._compute_pwm_and_ic(best_overall_pos)
        consensus = "".join(self.ALPHABET[idx] for idx in np.argmax(final_pwm, axis=0))

        return MotifDiscoveryResult(
            pwm=final_pwm,
            information_content_bits=final_ic,
            consensus_motif=consensus,
            best_positions=best_overall_pos,
            score_history=best_score_history
        )


# =============================================================================
# MODULE 5: MOLECULAR EVOLUTION (BAYESIAN PHYLOGENETIC MCMC - KIMURA 2P)
# =============================================================================

@dataclass
class PhylogeneticMCMCResult:
    d_chains: np.ndarray
    kappa_chains: np.ndarray
    log_post_chains: np.ndarray
    rhat_d: float
    rhat_kappa: float
    ess_d: float
    ess_kappa: float
    d_hpd: Tuple[float, float]
    kappa_hpd: Tuple[float, float]


class BayesianPhylogeneticMCMC:
    """Bayesian Multi-Chain Metropolis-Hastings MCMC under the Kimura 2-Parameter (K2P) model."""

    PURINES = {'A', 'G'}
    PYRIMIDINES = {'C', 'T'}

    def __init__(self, seq1: str, seq2: str, seed: int = 42):
        assert len(seq1) == len(seq2), "Aligned sequences must have equal length."
        self.seq_len = len(seq1)
        self.rng = np.random.default_rng(seed)

        self.n_match = 0
        self.n_ts = 0
        self.n_tv = 0

        for b1, b2 in zip(seq1.upper(), seq2.upper()):
            if b1 == b2:
                self.n_match += 1
            elif (b1 in self.PURINES and b2 in self.PURINES) or (b1 in self.PYRIMIDINES and b2 in self.PYRIMIDINES):
                self.n_ts += 1
            else:
                self.n_tv += 1

    @classmethod
    def simulate_aligned_pair(
        cls,
        seq_len: int = 800,
        true_d: float = 0.35,
        true_kappa: float = 3.2,
        seed: int = 42
    ) -> Tuple[str, str]:
        rng = np.random.default_rng(seed)
        bases = np.array(['A', 'C', 'G', 'T'])
        ancestral = rng.choice(bases, size=seq_len)

        p_ts = 0.25 + 0.25 * np.exp(-4.0 * true_d / (true_kappa + 2.0)) - 0.5 * np.exp(-2.0 * (true_kappa + 1.0) * true_d / (true_kappa + 2.0))
        p_tv = 0.5 - 0.5 * np.exp(-4.0 * true_d / (true_kappa + 2.0))
        p_same = 1.0 - p_ts - p_tv

        ts_map = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
        tv_map = {'A': ('C', 'T'), 'G': ('C', 'T'), 'C': ('A', 'G'), 'T': ('A', 'G')}

        descendant = []
        for b in ancestral:
            r = rng.random()
            if r < p_same:
                descendant.append(b)
            elif r < p_same + p_ts:
                descendant.append(ts_map[b])
            else:
                descendant.append(rng.choice(tv_map[b]))

        return "".join(ancestral), "".join(descendant)

    def log_posterior(self, d: float, kappa: float) -> float:
        if d <= 1e-7 or kappa <= 1e-7 or d > 10.0 or kappa > 50.0:
            return -np.inf

        exp1 = np.exp(-4.0 * d / (kappa + 2.0))
        exp2 = np.exp(-2.0 * (kappa + 1.0) * d / (kappa + 2.0))

        p_ts = 0.25 + 0.25 * exp1 - 0.5 * exp2
        p_tv = 0.5 - 0.5 * exp1
        p_same = 1.0 - p_ts - p_tv

        if p_ts <= 1e-15 or p_tv <= 1e-15 or p_same <= 1e-15:
            return -np.inf

        log_lik = (
            self.n_match * np.log(p_same)
            + self.n_ts * np.log(p_ts)
            + self.n_tv * np.log(p_tv)
        )

        log_prior_d = -1.0 * d
        log_prior_kappa = (2.0 - 1.0) * np.log(kappa) - (kappa / 2.0)

        return float(log_lik + log_prior_d + log_prior_kappa)

    def run_mcmc(
        self,
        n_chains: int = 4,
        n_steps: int = 4000,
        burn_in: int = 1000
    ) -> PhylogeneticMCMCResult:
        d_chains = np.zeros((n_chains, n_steps - burn_in))
        k_chains = np.zeros((n_chains, n_steps - burn_in))
        lp_chains = np.zeros((n_chains, n_steps - burn_in))

        for c in range(n_chains):
            curr_d = float(self.rng.uniform(0.05, 1.2))
            curr_k = float(self.rng.uniform(0.5, 6.0))
            curr_lp = self.log_posterior(curr_d, curr_k)

            step_d, step_k = 0.04, 0.25

            for s in range(n_steps):
                prop_d = curr_d * np.exp(self.rng.normal(0.0, step_d))
                prop_k = curr_k * np.exp(self.rng.normal(0.0, step_k))
                prop_lp = self.log_posterior(prop_d, prop_k)

                log_hastings = (prop_lp - curr_lp) + np.log(prop_d / curr_d) + np.log(prop_k / curr_k)

                if log_hastings >= 0.0 or np.log(self.rng.random()) < log_hastings:
                    curr_d, curr_k, curr_lp = prop_d, prop_k, prop_lp

                if s >= burn_in:
                    idx = s - burn_in
                    d_chains[c, idx] = curr_d
                    k_chains[c, idx] = curr_k
                    lp_chains[c, idx] = curr_lp

        return PhylogeneticMCMCResult(
            d_chains=d_chains,
            kappa_chains=k_chains,
            log_post_chains=lp_chains,
            rhat_d=MCDiagnostics.gelman_rubin_rhat(d_chains),
            rhat_kappa=MCDiagnostics.gelman_rubin_rhat(k_chains),
            ess_d=MCDiagnostics.effective_sample_size(d_chains),
            ess_kappa=MCDiagnostics.effective_sample_size(k_chains),
            d_hpd=MCDiagnostics.hpd_interval(d_chains),
            kappa_hpd=MCDiagnostics.hpd_interval(k_chains)
        )


# =============================================================================
# MODULE 6: ORCHESTRATOR & VISUALIZATION DASHBOARD
# =============================================================================

class MonteCarloSuite:
    """Master orchestrator that executes all bioinformatics MC engines and renders diagnostics."""

    def __init__(self, seed: int = 2026):
        self.seed = seed

    def run_all(self, plot_dashboard: bool = True):
        t0 = time.perf_counter()
        print("=" * 78)
        print("   MONTE CARLO: MULTI-SCALE BIOINFORMATICS SIMULATION SUITE")
        print("=" * 78)

        # 1. Structural Biophysics: 3D Protein Folding via Replica Exchange MC
        print("\n[1/4] Running 3D Off-Lattice Protein Folding (Replica Exchange MC)...")
        hp_seq = "HPHPPHHPHPPHPHHPPHPH"
        protein_engine = OffLatticeProteinREMC(hp_seq, seed=self.seed)
        remc_res = protein_engine.run_remc(n_replicas=6, n_steps=2500, swap_interval=20)
        print(f"      -> Sequence         : {remc_res.sequence} (N={len(hp_seq)})")
        print(f"      -> Lowest Energy E* : {remc_res.lowest_energy:.3f} kcal/mol (reduced)")
        print(f"      -> Native R_g       : {remc_res.rg_trajectories[0, -1]:.3f}")
        print(f"      -> REMC Swap Rate   : {remc_res.swap_acceptance_rate * 100:.1f}%")

        # 2. Systems Biology: Gillespie SSA & Tau-Leaping Gene Network
        print("\n[2/4] Running Stochastic Gene Regulatory Network (Gillespie SSA & Tau-Leaping)...")
        gene_net = StochasticGeneNetworkMC(seed=self.seed)
        ssa_res = gene_net.run_gillespie_ssa(t_max=250.0)
        tau_res = gene_net.run_tau_leaping(t_max=250.0, tau_step=0.25)
        print(f"      -> Exact SSA Events : {len(ssa_res.times):,} reaction firings")
        print(f"      -> mRNA Fano Factor : {ssa_res.fano_factors['mRNA']:.2f} (Super-Poissonian Bursting)")
        print(f"      -> Prot Fano Factor : {ssa_res.fano_factors['Protein']:.2f}")

        # 3. Sequence Genomics: Collapsed Gibbs Sampler for Motif Discovery
        print("\n[3/4] Running De Novo TFBS Motif Discovery (Collapsed Gibbs Sampler)...")
        planted = "TATAAAWG"
        seqs, true_pos = GibbsMotifSampler.generate_synthetic_dataset(
            n_seqs=28, seq_len=90, planted_motif=planted, mutation_rate=0.10, seed=self.seed
        )
        motif_engine = GibbsMotifSampler(seqs, motif_width=len(planted), seed=self.seed)
        motif_res = motif_engine.run(n_iterations=400, n_restarts=2)
        site_acc = np.mean(np.abs(motif_res.best_positions - true_pos) <= 1) * 100.0
        print(f"      -> Planted Motif    : {planted}")
        print(f"      -> Discovered Motif : {motif_res.consensus_motif}")
        print(f"      -> Info Content     : {motif_res.information_content_bits:.2f} bits")
        print(f"      -> Site Recovery    : {site_acc:.1f}% within +/-1 bp")

        # 4. Molecular Evolution: Bayesian Phylogenetic MCMC
        print("\n[4/4] Running Bayesian Phylogenetic MCMC (Kimura 2-Parameter Model)...")
        true_d, true_kappa = 0.38, 3.40
        s1, s2 = BayesianPhylogeneticMCMC.simulate_aligned_pair(
            seq_len=1000, true_d=true_d, true_kappa=true_kappa, seed=self.seed
        )
        phylo_engine = BayesianPhylogeneticMCMC(s1, s2, seed=self.seed)
        phylo_res = phylo_engine.run_mcmc(n_chains=4, n_steps=3500, burn_in=1000)
        d_mean = float(np.mean(phylo_res.d_chains))
        k_mean = float(np.mean(phylo_res.kappa_chains))
        print(f"      -> Distance (d)     : Post. Mean = {d_mean:.3f} (True = {true_d:.3f}), "
              f"95% HPD = [{phylo_res.d_hpd[0]:.3f}, {phylo_res.d_hpd[1]:.3f}]")
        print(f"      -> Ratio (kappa)    : Post. Mean = {k_mean:.3f} (True = {true_kappa:.3f}), "
              f"95% HPD = [{phylo_res.kappa_hpd[0]:.3f}, {phylo_res.kappa_hpd[1]:.3f}]")
        print(f"      -> Convergence      : R-hat(d) = {phylo_res.rhat_d:.4f}, "
              f"R-hat(kappa) = {phylo_res.rhat_kappa:.4f} (Target < 1.05)")
        print(f"      -> Effective Samples: ESS(d) = {phylo_res.ess_d:.0f}, "
              f"ESS(kappa) = {phylo_res.ess_kappa:.0f}")

        elapsed = time.perf_counter() - t0
        print("\n" + "=" * 78)
        print(f"   MONTE CARLO SUITE COMPLETED IN {elapsed:.2f} SECONDS.")
        print("=" * 78)

        if plot_dashboard:
            self._render_dashboard(remc_res, ssa_res, tau_res, motif_res, phylo_res, true_d, true_kappa)

        return {
            "protein_remc": remc_res,
            "gene_ssa": ssa_res,
            "gene_tau": tau_res,
            "motif_gibbs": motif_res,
            "phylo_mcmc": phylo_res,
        }

    @staticmethod
    def _render_dashboard(
        remc: ProteinFoldingResult,
        ssa: StochasticKineticsResult,
        tau: StochasticKineticsResult,
        motif: MotifDiscoveryResult,
        phylo: PhylogeneticMCMCResult,
        true_d: float,
        true_kappa: float
    ):
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Monte Carlo: Multi-Scale Bioinformatics Diagnostic Dashboard",
                     fontsize=14, fontweight="bold")

        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        coords = remc.best_coords
        ax1.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="slategray", lw=2.0, alpha=0.8)
        for i, res in enumerate(remc.sequence):
            color = "crimson" if res == "H" else "dodgerblue"
            ax1.scatter(coords[i, 0], coords[i, 1], coords[i, 2], color=color, s=75, edgecolors="k")
        ax1.set_title(f"1. 3D REMC Minimum-Energy Fold\n(E* = {remc.lowest_energy:.2f}, Red=H, Blue=P)")
        ax1.set_xlabel("X (Å)")
        ax1.set_ylabel("Y (Å)")
        ax1.set_zlabel("Z (Å)")

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(remc.energy_trajectories[0], label=f"T_cold = {remc.temperatures[0]:.2f}", alpha=0.85, lw=1.0)
        ax2.plot(remc.energy_trajectories[-1], label=f"T_hot = {remc.temperatures[-1]:.2f}", alpha=0.65, lw=1.0)
        ax2.set_title("2. Parallel Tempering Energy Mixing")
        ax2.set_xlabel("MC Sweep")
        ax2.set_ylabel("Potential Energy E(R)")
        ax2.legend(loc="upper right")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.step(ssa.times, ssa.states[:, 2], where="post", label="mRNA (Exact SSA)", color="darkorange", alpha=0.85)
        ax3.step(ssa.times, ssa.states[:, 3], where="post", label="Protein (Exact SSA)", color="teal", lw=1.5)
        ax3.plot(tau.times, tau.states[:, 3], label="Protein (Tau-Leaping)", color="navy", ls="--", alpha=0.7)
        ax3.set_title(f"3. Stochastic Gene Network (mRNA Fano={ssa.fano_factors['mRNA']:.2f})")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Molecular Copy Number")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(alpha=0.3)

        ax4 = fig.add_subplot(2, 3, 4)
        im = ax4.imshow(motif.pwm, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_yticklabels(["A", "C", "G", "T"])
        ax4.set_xticks(range(motif.pwm.shape[1]))
        ax4.set_xticklabels([f"P{i+1}\n({c})" for i, c in enumerate(motif.consensus_motif)])
        ax4.set_title(f"4. Gibbs Sampler Discovered PWM ({motif.information_content_bits:.1f} bits)")
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(2, 3, 5)
        d_flat = phylo.d_chains.ravel()
        k_flat = phylo.kappa_chains.ravel()
        ax5.scatter(d_flat[::4], k_flat[::4], s=6, alpha=0.25, color="purple", label="MCMC Posterior")
        ax5.axvline(true_d, color="red", ls="--", lw=1.5, label=f"True d={true_d}")
        ax5.axhline(true_kappa, color="red", ls=":", lw=1.5, label=f"True κ={true_kappa}")
        ax5.set_title(f"5. Bayesian K2P Posterior (R-hat={phylo.rhat_d:.3f})")
        ax5.set_xlabel("Evolutionary Distance (d)")
        ax5.set_ylabel("Ts/Tv Rate Ratio (κ)")
        ax5.legend(loc="upper right", fontsize=8)
        ax5.grid(alpha=0.3)

        ax6 = fig.add_subplot(2, 3, 6)
        max_lag = 80
        for c in range(phylo.d_chains.shape[0]):
            acf = MCDiagnostics.autocorrelation_fft(phylo.d_chains[c])[:max_lag]
            ax6.plot(range(max_lag), acf, lw=1.5, alpha=0.8, label=f"Chain {c+1}")
        ax6.axhline(0.0, color="black", lw=0.8, ls="--")
        ax6.set_title(f"6. MCMC Autocorrelation Decay (ESS={phylo.ess_d:.0f})")
        ax6.set_xlabel("Lag k")
        ax6.set_ylabel("Autocorrelation ρ(k)")
        ax6.legend(loc="upper right", fontsize=8)
        ax6.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    suite = MonteCarloSuite(seed=2026)
    results = suite.run_all(plot_dashboard=True)