"""
Unit and mathematical invariant tests for Monte Carlo.
Run with: pytest -v
"""

import numpy as np

from monte_carlo import (
    MCDiagnostics,
    OffLatticeProteinREMC,
    StochasticGeneNetworkMC,
    GibbsMotifSampler,
    BayesianPhylogeneticMCMC,
)


def test_diagnostics_stationary_convergence():
    """Independent chains drawn from the same distribution must yield R-hat ~ 1.0 and high ESS."""
    rng = np.random.default_rng(123)
    chains = rng.normal(loc=5.0, scale=1.5, size=(4, 1000))
    rhat = MCDiagnostics.gelman_rubin_rhat(chains)
    ess = MCDiagnostics.effective_sample_size(chains)
    low, high = MCDiagnostics.hpd_interval(chains, alpha=0.05)

    assert 0.98 <= rhat <= 1.05
    assert ess > 2500
    assert 1.5 < low < 2.5
    assert 7.5 < high < 8.5


def test_protein_remc_energy_and_rg():
    """Verify REMC reduces initial extended chain energy and maintains finite coordinates."""
    seq = "HPHPPHHPHPPH"
    engine = OffLatticeProteinREMC(seq, seed=42)
    init_coords = engine._initialize_extended_chain()
    init_energy = engine.compute_energy(init_coords)

    res = engine.run_remc(n_replicas=4, n_steps=300, swap_interval=15)
    assert np.isfinite(res.lowest_energy)
    assert res.lowest_energy <= init_energy
    assert res.best_coords.shape == (len(seq), 3)
    assert 0.0 <= res.swap_acceptance_rate <= 1.0


def test_stochastic_gene_network_invariants():
    """Verify promoter conservation (G_off + G_on == 1) and non-negative molecular counts."""
    net = StochasticGeneNetworkMC(seed=42)
    ssa = net.run_gillespie_ssa(t_max=50.0)
    tau = net.run_tau_leaping(t_max=50.0, tau_step=0.2)

    assert np.all(ssa.states[:, 0] + ssa.states[:, 1] == 1)
    assert np.all(tau.states[:, 0] + tau.states[:, 1] == 1)
    assert np.all(ssa.states >= 0)
    assert np.all(tau.states >= 0)


def test_gibbs_motif_pwm_normalization():
    """Verify PWM columns sum to 1.0 and planted high-IC motif is accurately recovered."""
    seqs, true_pos = GibbsMotifSampler.generate_synthetic_dataset(
        n_seqs=20, seq_len=60, planted_motif="TATAAATA", mutation_rate=0.02, seed=101
    )
    sampler = GibbsMotifSampler(seqs, motif_width=8, seed=101)
    res = sampler.run(n_iterations=150, n_restarts=1)

    np.testing.assert_allclose(np.sum(res.pwm, axis=0), np.ones(8), atol=1e-6)
    assert res.information_content_bits > 7.0
    accuracy = np.mean(np.abs(res.best_positions - true_pos) <= 1)
    assert accuracy >= 0.75


def test_bayesian_phylo_mcmc_parameter_recovery():
    """Verify K2P Bayesian MCMC recovers true simulated evolutionary distance within 95% HPD."""
    true_d, true_k = 0.30, 3.0
    s1, s2 = BayesianPhylogeneticMCMC.simulate_aligned_pair(
        seq_len=1500, true_d=true_d, true_kappa=true_k, seed=99
    )
    mcmc = BayesianPhylogeneticMCMC(s1, s2, seed=99)
    res = mcmc.run_mcmc(n_chains=3, n_steps=1500, burn_in=500)

    assert res.rhat_d < 1.08
    assert res.d_hpd[0] <= true_d <= res.d_hpd[1]
    assert res.kappa_hpd[0] <= true_k <= res.kappa_hpd[1]