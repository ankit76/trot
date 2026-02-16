import os

import numpy as np

from ad_afqmc import config

config.setup_jax()
from jax import numpy as jnp

from ad_afqmc import utils, wavefunctions

seed = 102
np.random.seed(seed)
norb, nelec, nchol = 10, (5, 5), 5
rhf = wavefunctions.rhf(norb, nelec)
wave_data = {}
wave_data["mo_coeff"] = jnp.eye(norb)[:, : nelec[0]]
wave_data["rdm1"] = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
walker = jnp.array(np.random.rand(norb, nelec[0])) + 1.0j * jnp.array(
    np.random.rand(norb, nelec[0])
)
ham_data = {}
ham_data["h0"] = np.random.rand(
    1,
)[0]
ham_data["rot_h1"] = jnp.array(np.random.rand(nelec[0], norb))
ham_data["rot_chol"] = jnp.array(np.random.rand(nchol, nelec[0], norb))
ham_data["h1"] = jnp.array([np.random.rand(norb, norb)] * 2)
ham_data["chol"] = jnp.array(np.random.rand(nchol, norb, norb))
ham_data["ene0"] = 0.0


ham_data["normal_ordering_term"] = -0.5 * jnp.einsum(
    "gik,gjk->ij",
    ham_data["chol"].reshape(-1, norb, norb),
    ham_data["chol"].reshape(-1, norb, norb),
    optimize="optimal",
)

multislater = wavefunctions.multislater(norb, nelec, max_excitation=6)
path = os.path.dirname(os.path.abspath(__file__))
Acre, Ades, Bcre, Bdes, coeff, ref_det = utils.get_excitations(
    fname=path + "/dets.bin", max_excitation=6, ndets=10
)  # reads dets.bin
ref_det = jnp.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 2])
wave_data_multislater = {
    "Acre": Acre,
    "Ades": Ades,
    "Bcre": Bcre,
    "Bdes": Bdes,
    "coeff": coeff,
    "ref_det": ref_det,
    "orbital_rotation": jnp.eye(norb),
}

nelec_sp = (3, 2)
uhf = wavefunctions.uhf(norb, nelec_sp)
wave_data_u = {}
wave_data_u["mo_coeff"] = [
    jnp.array(np.random.rand(norb, nelec_sp[0])),
    jnp.array(np.random.rand(norb, nelec_sp[1])),
]
wave_data_u["rdm1"] = jnp.array(
    [
        jnp.array(wave_data_u["mo_coeff"][0] @ wave_data_u["mo_coeff"][0].T),
        jnp.array(wave_data_u["mo_coeff"][1] @ wave_data_u["mo_coeff"][1].T),
    ]
)
walker_up, walker_dn = jnp.array(np.random.rand(norb, nelec_sp[0])) + 1.0j * jnp.array(
    np.random.rand(norb, nelec_sp[0])
), jnp.array(np.random.rand(norb, nelec_sp[1])) + 1.0j * jnp.array(
    np.random.rand(norb, nelec_sp[1])
)
ham_data_u = {}
ham_data_u["h0"] = ham_data["h0"]
ham_data_u["rot_h1"] = [
    ham_data["rot_h1"][: nelec_sp[0], :],
    ham_data["rot_h1"][: nelec_sp[1], :],
]
ham_data_u["rot_chol"] = [
    ham_data["rot_chol"][:, : nelec_sp[0], :],
    ham_data["rot_chol"][:, : nelec_sp[1], :],
]
ham_data_u["h1"] = ham_data["h1"]  # jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data_u["chol"] = ham_data["chol"]
ham_data_u["ene0"] = ham_data["ene0"]

ndets = 5
noci = wavefunctions.noci(norb, nelec_sp, ndets)
dets = [
    jnp.array(np.random.rand(ndets, norb, nelec_sp[0])),
    jnp.array(np.random.rand(ndets, norb, nelec_sp[1])),
]
ci_coeffs = jnp.array(np.random.randn(ndets))
wave_data_noci = {}
wave_data_noci["ci_coeffs_dets"] = [ci_coeffs, dets]
wave_data_noci["rdm1"] = wave_data_u["rdm1"]
ham_data_noci = {}
ham_data_noci["h0"] = ham_data["h0"]
ham_data_noci["rot_h1"] = [
    jnp.array(np.random.rand(ndets, nelec_sp[0], norb)),
    jnp.array(np.random.rand(ndets, nelec_sp[1], norb)),
]
ham_data_noci["rot_chol"] = [
    jnp.array(np.random.rand(ndets, nchol, nelec_sp[0], norb)),
    jnp.array(np.random.rand(ndets, nchol, nelec_sp[1], norb)),
]


nelec_sp = (3, 2)
ghf = wavefunctions.ghf(norb, nelec_sp)
wave_data_g = {}
wave_data_g["mo_coeff"] = jnp.array(np.random.rand(2 * norb, nelec_sp[0] + nelec_sp[1]))
wave_data_g["rdm1"] = wave_data_u["rdm1"]
ham_data_g = {}
ham_data_g["h0"] = ham_data["h0"]
ham_data_g["rot_h1"] = jnp.array(np.random.rand(nelec_sp[0] + nelec_sp[1], 2 * norb))
ham_data_g["rot_chol"] = jnp.array(
    np.random.rand(nchol, nelec_sp[0] + nelec_sp[1], 2 * norb)
)
ham_data_g["h1"] = jnp.array([ham_data["h1"], ham_data["h1"]])
ham_data_g["chol"] = ham_data["chol"]
ham_data_g["ene0"] = ham_data["ene0"]

uhf_cpmc = wavefunctions.uhf_cpmc(norb, nelec_sp)

neighbors = tuple((i, (i + 1) % norb) for i in range(norb))
uhf_cpmc_nn = wavefunctions.uhf_cpmc_nn(norb, nelec_sp, neighbors=neighbors)

def test_rhf_overlap():
    overlap = rhf._calc_overlap_restricted(walker, wave_data)
    assert np.allclose(jnp.real(overlap), -0.10794844182417201)


def test_rhf_green():
    green = rhf._calc_green(walker, wave_data)
    assert green.shape == (nelec[0], norb)
    assert np.allclose(jnp.real(jnp.sum(green)), 12.181348093111438)


def test_rhf_force_bias():
    force_bias = rhf._calc_force_bias_restricted(walker, ham_data, wave_data)
    assert force_bias.shape == (nchol,)
    assert np.allclose(jnp.real(jnp.sum(force_bias)), 66.13455680423321)


def test_rhf_energy():
    energy = rhf._calc_energy_restricted(walker, ham_data, wave_data)
    assert np.allclose(jnp.real(energy), 217.79874063608622)


def test_rhf_optimize_orbs():
    wave_data_opt = rhf.optimize(ham_data, wave_data)
    orbs = wave_data_opt["mo_coeff"]
    assert orbs.shape == (norb, nelec[0])
    # assert np.allclose(jnp.sum(orbs), 2.9662577668717933)


def test_uhf_overlap():
    overlap = uhf._calc_overlap_unrestricted(walker_up, walker_dn, wave_data_u)
    assert np.allclose(jnp.real(overlap), -0.4029825074695857)


def test_uhf_green():
    green = uhf._calc_green(walker_up, walker_dn, wave_data_u)
    assert green[0].shape == (nelec_sp[0], norb)
    assert green[1].shape == (nelec_sp[1], norb)
    assert np.allclose(
        jnp.real(jnp.sum(green[0]) + jnp.sum(green[1])), 3.6324117896217394
    )


def test_uhf_force_bias():
    force_bias = uhf._calc_force_bias_unrestricted(walker_up, walker_dn, ham_data_u, wave_data_u)
    assert force_bias.shape == (nchol,)
    assert np.allclose(jnp.real(jnp.sum(force_bias)), 10.441272099672341)


def test_uhf_energy():
    energy = uhf._calc_energy_unrestricted(
        walker_up,
        walker_dn,
        ham_data_u,
        wave_data_u,
    )
    assert np.allclose(jnp.real(energy), -1.7203463308366032)


def test_uhf_optimize_orbs():
    wave_data_opt = uhf.optimize(ham_data_u, wave_data_u)
    orbs = wave_data_opt["mo_coeff"]
    assert orbs[0].shape == (norb, nelec_sp[0])
    assert orbs[1].shape == (norb, nelec_sp[1])
    # assert np.allclose(jnp.sum(orbs[0]) + jnp.sum(orbs[1]), 7.402931219898609)


def test_ghf_overlap():
    overlap = ghf._calc_overlap_unrestricted(walker_up, walker_dn, wave_data_g)
    assert np.allclose(jnp.real(overlap), -0.7645032356687913)


def test_ghf_green():
    green = ghf._calc_green(walker_up, walker_dn, wave_data_g)
    assert green.shape == (nelec_sp[0] + nelec_sp[1], 2 * norb)


def test_ghf_force_bias():
    force_bias = ghf._calc_force_bias_unrestricted(walker_up, walker_dn, ham_data_g, wave_data_g)
    assert force_bias.shape == (nchol,)


def test_ghf_energy():
    energy = ghf._calc_energy_unrestricted(
        walker_up,
        walker_dn,
        ham_data_g,
        wave_data_g,
    )
    assert np.allclose(jnp.real(energy), 47.91857449460195)


def test_noci_overlap():
    overlap = noci._calc_overlap_unrestricted(walker_up, walker_dn, wave_data_noci)
    # print(overlap)    
    # assert np.allclose(jnp.real(overlap), -0.4029825074695857)


def test_noci_green():
    green_0, green_1, _ = noci._calc_green(walker_up, walker_dn, wave_data_noci)
    assert green_0.shape == (ndets, nelec_sp[0], norb)
    assert green_1.shape == (ndets, nelec_sp[1], norb)
    # assert np.allclose(jnp.real(jnp.sum(green[0])+jnp.sum(green[1])), 3.6324117896217394)


def test_noci_force_bias():
    force_bias = noci._calc_force_bias_unrestricted(
        walker_up, walker_dn, ham_data_noci, wave_data_noci
    )
    assert force_bias.shape == (nchol,)
    # assert np.allclose(jnp.real(jnp.sum(force_bias)), 10.441272099672341)


def test_noci_energy():
    energy = noci._calc_energy_unrestricted(
        walker_up,
        walker_dn,
        ham_data_noci,
        wave_data_noci,
    )
    # assert np.allclose(jnp.real(energy), -1.7203463308366032)


def test_noci_get_rdm1():
    rdm1 = noci.get_rdm1(wave_data_noci)
    assert rdm1.shape == (2, norb, norb)


def test_uhf_cpmc():
    u = 4.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )
    green = uhf_cpmc._calc_green_full_unrestricted(walker_up, walker_dn, wave_data_u)
    assert green[0].shape == (norb, norb)
    assert green[1].shape == (norb, norb)

    update_indices = jnp.array([[0, 3], [1, 3]])
    wick_ratio = uhf_cpmc._calc_overlap_ratio(
        green,
        update_indices,
        hs_constant[0] - 1,
    )
    overlap = uhf_cpmc._calc_overlap_unrestricted(walker_up, walker_dn, wave_data_u)
    new_walker_up = walker_up.at[update_indices[0, 1], :].mul(hs_constant[0, 0])
    new_walker_dn = walker_dn.at[update_indices[1, 1], :].mul(hs_constant[0, 1])
    ratio = uhf_cpmc._calc_overlap_unrestricted(new_walker_up, new_walker_dn, wave_data_u) / overlap
    assert np.allclose(ratio, wick_ratio)

    new_green = uhf_cpmc._calc_green_full_unrestricted(new_walker_up, new_walker_dn, wave_data_u)
    new_green_wick = uhf_cpmc._update_green(
        green,
        ratio,
        update_indices,
        hs_constant[0] - 1,
    )
    assert np.allclose(new_green, new_green_wick)


def test_uhf_cpmc_nn():
    u = 4.0
    u_1 = 1.0
    dt = 0.005
    gamma = jnp.arccosh(jnp.exp(dt * u / 2))
    const = jnp.exp(-dt * u / 2)
    hs_constant = const * jnp.array(
        [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
    )
    gamma_1 = jnp.arccosh(jnp.exp(dt * u_1 / 2))
    const_1 = jnp.exp(-dt * u_1 / 2)
    hs_constant_nn = const_1 * jnp.array(
        [
            [jnp.exp(gamma_1), jnp.exp(-gamma_1)],
            [jnp.exp(-gamma_1), jnp.exp(gamma_1)],
        ]
    )
    green = uhf_cpmc_nn._calc_green_full_unrestricted(walker_up, walker_dn, wave_data_u)
    assert green[0].shape == (norb, norb)
    assert green[1].shape == (norb, norb)

    # onsite interactions
    update_indices = jnp.array([[0, 3], [1, 3]])
    wick_ratio = uhf_cpmc_nn._calc_overlap_ratio(
        green,
        update_indices,
        hs_constant[0] - 1,
    )
    overlap = uhf_cpmc_nn._calc_overlap_unrestricted(walker_up, walker_dn, wave_data_u)
    new_walker_up = walker_up.at[update_indices[0, 1], :].mul(hs_constant[0, 0])
    new_walker_dn = walker_dn.at[update_indices[1, 1], :].mul(hs_constant[0, 1])
    ratio = uhf_cpmc_nn._calc_overlap_unrestricted(new_walker_up, new_walker_dn, wave_data_u) / overlap
    assert np.allclose(ratio, wick_ratio)
    new_green = uhf_cpmc._calc_green_full_unrestricted(new_walker_up, new_walker_dn, wave_data_u)
    new_green_wick = uhf_cpmc._update_green(
        green,
        ratio,
        update_indices,
        hs_constant[0] - 1,
    )
    assert np.allclose(new_green, new_green_wick)

    # nearest-neighbor interactions
    update_indices = jnp.array([[0, 3], [1, 4]])
    wick_ratio_nn = uhf_cpmc_nn._calc_overlap_ratio(
        new_green,
        update_indices,
        hs_constant_nn[0] - 1,
    )
    overlap_nn = uhf_cpmc_nn._calc_overlap_unrestricted(new_walker_up, new_walker_dn, wave_data_u)
    new_walker_nn_up = new_walker_up.at[update_indices[0, 1], :].mul(hs_constant_nn[0, 0])
    new_walker_nn_dn = new_walker_dn.at[update_indices[1, 1], :].mul(hs_constant_nn[0, 1])
    ratio_nn = uhf_cpmc_nn._calc_overlap_unrestricted(
        new_walker_nn_up, new_walker_nn_dn, wave_data_u
    ) / overlap_nn
    assert np.allclose(ratio_nn, wick_ratio_nn)
    new_green_nn = uhf_cpmc_nn._calc_green_full_unrestricted(new_walker_nn_up, new_walker_nn_dn, wave_data_u)
    new_green_wick_nn = uhf_cpmc_nn._update_green(
        new_green,
        ratio_nn,
        update_indices,
        hs_constant_nn[0] - 1,
    )
    assert np.allclose(new_green_nn, new_green_wick_nn)


def test_multislater():
    # slow test due to compilation time for fb and energy
    overlap = multislater._calc_overlap_restricted(walker, wave_data_multislater)
    assert np.allclose(jnp.real(overlap), -0.10468995287669804)

    green = multislater._calc_green_restricted(walker, wave_data_multislater)
    assert green.shape == (nelec[0], norb)
    assert np.allclose(jnp.real(jnp.sum(green)), 12.181348093111438)

    force_bias = multislater._calc_force_bias_restricted(
        walker, ham_data, wave_data_multislater
    )
    assert force_bias.shape == (nchol,)
    assert np.allclose(jnp.real(jnp.sum(force_bias)), 63.76752581694084)

    energy = multislater._calc_energy_restricted(
        walker, ham_data, wave_data_multislater
    )
    assert np.allclose(jnp.real(energy), 204.41045113133066)


def test_cisd():
    norb, nocc, nchol = 10, 3, 20
    nelec = (nocc, nocc)
    ci1 = jnp.array(np.random.randn(nocc, norb - nocc))
    walker = jnp.array(np.random.randn(norb, nocc)) + 1.0j * jnp.array(
        np.random.randn(norb, nocc)
    )
    trial = wavefunctions.cisd(
        norb,
        nelec,
        _mixed_real_dtype_testing=jnp.float64,
        _mixed_complex_dtype_testing=jnp.complex128,
    )
    trial_hm = wavefunctions.cisd(
        norb,
        nelec,
        memory_mode="high",
        _mixed_real_dtype_testing=jnp.float64,
        _mixed_complex_dtype_testing=jnp.complex128,
    )
    trial_auto = wavefunctions.CISD(norb, nelec)
    ci2 = jnp.array(np.random.randn(nocc, norb - nocc, nocc, norb - nocc))
    ci2 = (ci2 + ci2.transpose(2, 3, 0, 1)) / 2.0
    wave_data = {"ci1": ci1, "ci2": ci2}
    h0 = jnp.array(np.random.randn(1))
    h1 = jnp.array(np.random.randn(norb, norb))
    h1 = (h1 + h1.T) / 2.0
    chol = jnp.array(np.random.randn(nchol, norb, norb)) / jnp.sqrt(norb)
    chol = (chol + chol.transpose(0, 2, 1)) / 2.0
    ham_data = {"h0": h0, "h1": jnp.array([h1, h1]), "chol": chol}
    ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
    ham_data = trial_auto._build_measurement_intermediates(ham_data, wave_data)
    ene_auto = trial_auto._calc_energy_restricted(walker, ham_data, wave_data)
    ene_manual_lm = trial._calc_energy_restricted(walker, ham_data, wave_data)
    ene_manual_hm = trial_hm._calc_energy_restricted(walker, ham_data, wave_data)
    assert np.allclose(ene_auto, ene_manual_lm, atol=1.0e-4)
    assert np.allclose(ene_manual_lm, ene_manual_hm, atol=1.0e-6)
    assert np.allclose(
        trial._calc_force_bias_restricted(walker, ham_data, wave_data),
        trial_auto._calc_force_bias_restricted(walker, ham_data, wave_data),
        atol=1.0e-5,
    )


def test_ucisd():
    norb, nocc_a, nocc_b, nchol = 10, 3, 5, 20
    nelec = (nocc_a, nocc_b)
    ci1_a = jnp.array(np.random.randn(nocc_a, norb - nocc_a)) / (
        (norb - nocc_a) * nocc_a
    )
    ci1_b = jnp.array(np.random.randn(nocc_b, norb - nocc_b)) / (
        (norb - nocc_b) * nocc_b
    )
    walker_up = jnp.array(
        np.random.randn(norb, nocc_a) + 1.0j * np.random.randn(norb, nocc_a)
    )
    walker_dn = jnp.array(
        np.random.randn(norb, nocc_b) + 1.0j * np.random.randn(norb, nocc_b)
    )
    trial = wavefunctions.ucisd(
        norb,
        nelec,
        _mixed_real_dtype_testing=jnp.float64,
        _mixed_complex_dtype_testing=jnp.complex128,
    )
    trial_hm = wavefunctions.ucisd(
        norb,
        nelec,
        memory_mode="high",
        _mixed_real_dtype_testing=jnp.float64,
        _mixed_complex_dtype_testing=jnp.complex128,
    )
    trial_auto = wavefunctions.UCISD(norb, nelec)
    ci2_aa = jnp.array(np.random.randn(nocc_a, norb - nocc_a, nocc_a, norb - nocc_a))
    ci2_aa = (ci2_aa + ci2_aa.transpose(2, 3, 0, 1)) / 2.0
    ci2_aa = (ci2_aa - ci2_aa.transpose(0, 3, 2, 1)) / 2.0
    ci2_bb = jnp.array(np.random.randn(nocc_b, norb - nocc_b, nocc_b, norb - nocc_b))
    ci2_bb = (ci2_bb + ci2_bb.transpose(2, 3, 0, 1)) / 2.0
    ci2_bb = (ci2_bb - ci2_bb.transpose(0, 3, 2, 1)) / 2.0
    ci2_ab = jnp.array(np.random.randn(nocc_a, norb - nocc_a, nocc_b, norb - nocc_b))
    mo_coeff_b = jnp.linalg.qr(jnp.array(np.random.randn(norb, norb)))[0]
    mo_coeff = jnp.array([np.eye(norb, norb), mo_coeff_b])
    wave_data = {
        "ci1A": ci1_a,
        "ci1B": ci1_b,
        "ci2AA": ci2_aa,
        "ci2BB": ci2_bb,
        "ci2AB": ci2_ab,
        "mo_coeff": mo_coeff,
    }
    h0 = jnp.array(np.random.randn(1))
    h1 = jnp.array(np.random.randn(norb, norb))
    h1 = (h1 + h1.T) / 2.0 / h1.size
    chol = jnp.array(np.random.randn(nchol, norb, norb))
    chol = (chol + chol.transpose(0, 2, 1)) / 2.0 / jnp.sqrt(norb)
    ham_data = {"h0": h0, "h1": jnp.array([h1, h1]), "chol": chol}
    ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
    ham_data = trial_auto._build_measurement_intermediates(ham_data, wave_data)
    ene_lm = trial._calc_energy_unrestricted(walker_up, walker_dn, ham_data, wave_data)
    ene_hm = trial_hm._calc_energy_unrestricted(walker_up, walker_dn, ham_data, wave_data)
    ene_auto = trial_auto._calc_energy_unrestricted(walker_up, walker_dn, ham_data, wave_data)
    assert np.allclose(ene_auto, ene_lm, atol=1.0e-4)
    assert np.allclose(ene_lm, ene_hm, atol=1.0e-6)
    assert np.allclose(
        trial._calc_force_bias_unrestricted(walker_up, walker_dn, ham_data, wave_data),
        trial_auto._calc_force_bias_unrestricted(walker_up, walker_dn, ham_data, wave_data),
        atol=1.0e-5,
    )


def test_cisd_eom_t():
    norb, nocc, nchol = 10, 3, 20
    nelec = (nocc, nocc)
    ci1 = jnp.array(np.random.randn(nocc, norb - nocc))
    r1 = jnp.array(np.random.randn(nocc, norb - nocc))
    walker = jnp.array(np.random.randn(norb, nocc)) + 1.0j * jnp.array(
        np.random.randn(norb, nocc)
    )
    trial = wavefunctions.cisd_eom_t(
        norb, nelec, mixed_complex_dtype=jnp.complex128, mixed_real_dtype=jnp.float64
    )
    trial_auto = wavefunctions.cisd_eom_t_auto(norb, nelec)
    ci2 = jnp.array(np.random.randn(nocc, norb - nocc, nocc, norb - nocc))
    ci2 = (ci2 + ci2.transpose(2, 3, 0, 1)) / 2.0
    r2 = jnp.array(np.random.randn(nocc, norb - nocc, nocc, norb - nocc))
    r2 = (r2 + r2.transpose(2, 3, 0, 1)) / 2.0
    wave_data = {"ci1": ci1, "ci2": ci2, "r1": r1, "r2": r2}
    h0 = jnp.array(np.random.randn(1))
    h1 = jnp.array(np.random.randn(norb, norb))
    h1 = (h1 + h1.T) / 2.0
    chol = jnp.array(np.random.randn(nchol, norb, norb)) / jnp.sqrt(norb)
    chol = (chol + chol.transpose(0, 2, 1)) / 2.0
    ham_data = {"h0": h0, "h1": jnp.array([h1, h1]), "chol": chol}
    ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
    ham_data = trial_auto._build_measurement_intermediates(ham_data, wave_data)
    assert np.allclose(
        trial._calc_energy_restricted(walker, ham_data, wave_data),
        trial_auto._calc_energy_restricted(walker, ham_data, wave_data),
        atol=1.0e-4,
    )
    assert np.allclose(
        trial._calc_force_bias_restricted(walker, ham_data, wave_data),
        trial_auto._calc_force_bias_restricted(walker, ham_data, wave_data),
        atol=1.0e-5,
    )


def test_cisd_eom():
    norb, nocc, nchol = 10, 3, 20
    nelec = (nocc, nocc)
    ci1 = jnp.array(np.random.randn(nocc, norb - nocc))
    r1 = jnp.array(np.random.randn(nocc, norb - nocc))
    walker = jnp.array(np.random.randn(norb, nocc)) + 1.0j * jnp.array(
        np.random.randn(norb, nocc)
    )
    trial = wavefunctions.cisd_eom(
        norb, nelec, mixed_complex_dtype=jnp.complex128, mixed_real_dtype=jnp.float64
    )
    trial_auto = wavefunctions.cisd_eom_auto(norb, nelec)
    ci2 = jnp.array(np.random.randn(nocc, norb - nocc, nocc, norb - nocc))
    ci2 = (ci2 + ci2.transpose(2, 3, 0, 1)) / 2.0
    r2 = jnp.array(np.random.randn(nocc, norb - nocc, nocc, norb - nocc))
    r2 = (r2 + r2.transpose(2, 3, 0, 1)) / 2.0
    wave_data = {"ci1": ci1, "ci2": ci2, "r1": r1, "r2": r2}
    h0 = jnp.array(np.random.randn(1))
    h1 = jnp.array(np.random.randn(norb, norb))
    h1 = (h1 + h1.T) / 2.0
    chol = jnp.array(np.random.randn(nchol, norb, norb)) / jnp.sqrt(norb)
    chol = (chol + chol.transpose(0, 2, 1)) / 2.0
    ham_data = {"h0": h0, "h1": jnp.array([h1, h1]), "chol": chol}
    ham_data = trial._build_measurement_intermediates(ham_data, wave_data)
    ham_data = trial_auto._build_measurement_intermediates(ham_data, wave_data)
    assert np.allclose(
        trial._calc_energy_restricted(walker, ham_data, wave_data),
        trial_auto._calc_energy_restricted(walker, ham_data, wave_data),
        atol=1.0e-4,
    )
    assert np.allclose(
        trial._calc_force_bias_restricted(walker, ham_data, wave_data),
        trial_auto._calc_force_bias_restricted(walker, ham_data, wave_data),
        atol=1.0e-5,
    )


if __name__ == "__main__":
    test_rhf_overlap()
    test_rhf_green()
    test_rhf_force_bias()
    test_rhf_energy()
    test_rhf_optimize_orbs()
    test_uhf_overlap()
    test_uhf_green()
    test_uhf_force_bias()
    test_uhf_energy()
    test_uhf_optimize_orbs()
    test_ghf_overlap()
    test_ghf_green()
    test_ghf_force_bias()
    test_ghf_energy()
    test_noci_overlap()
    test_noci_green()
    test_noci_force_bias()
    test_noci_energy()
    test_noci_get_rdm1()
    test_uhf_cpmc()
    test_uhf_cpmc_nn()
    test_multislater()
    test_cisd()
    test_ucisd()
    test_cisd_eom_t()
    test_cisd_eom()
