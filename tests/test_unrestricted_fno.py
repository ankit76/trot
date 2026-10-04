"""Compact spin-specific trial spaces in the unchanged full Hamiltonian."""

from importlib import import_module

from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.trial import ucisd, ucisd_k, ucisd_k_modes
from trot.trial import ptuccsd_thouless, ptuccsd_modes
from trot.meas import ucisd as ci_meas
from trot.meas import ptuccsd_thouless as pt_meas
from trot.meas import ptuccsd_modes as pt_mode_meas


def _inputs(nva=3, nvb=2):
    rng = np.random.default_rng(5083)
    norb, noa, nob = 7, 3, 2
    cb, _ = np.linalg.qr(np.eye(norb) + 0.08 * rng.normal(size=(norb, norb)))

    def ss(no, nv):
        a = 0.02 * rng.normal(size=(no, nv, no, nv))
        return 0.25 * (
            a - a.transpose(2, 1, 0, 3) - a.transpose(0, 3, 2, 1) + a.transpose(2, 3, 0, 1)
        )

    data = dict(
        mo_coeff_a=np.eye(norb),
        mo_coeff_b=cb,
        ci1a=0.03 * rng.normal(size=(noa, nva)),
        ci1b=0.03 * rng.normal(size=(nob, nvb)),
        ci2aa=ss(noa, nva),
        ci2bb=ss(nob, nvb),
        ci2ab=0.02 * rng.normal(size=(noa, nva, nob, nvb)),
    )
    padded = {**data}
    for name in ("ci1a", "ci1b", "ci2aa", "ci2ab", "ci2bb"):
        a = data[name]
        va, vb = norb - noa - nva, norb - nob - nvb
        pad = {
            "ci1a": ((0, 0), (0, va)),
            "ci1b": ((0, 0), (0, vb)),
            "ci2aa": ((0, 0), (0, va), (0, 0), (0, va)),
            "ci2ab": ((0, 0), (0, va), (0, 0), (0, vb)),
            "ci2bb": ((0, 0), (0, vb), (0, 0), (0, vb)),
        }[name]
        padded[name] = np.pad(a, pad)
    w = jnp.asarray(
        np.eye(norb, noa)
        + 0.07 * (rng.normal(size=(norb, noa)) + 1j * rng.normal(size=(norb, noa)))
    )
    h = rng.normal(size=(norb, norb))
    l = rng.normal(size=(5, norb, norb))
    ham = HamChol(
        h0=jnp.asarray(0.3),
        h1=jnp.asarray((h + h.T) / 2),
        chol=jnp.asarray((l + l.transpose(0, 2, 1)) / 2),
        basis="restricted",
    )
    return data, padded, w, ham, System(norb, (noa, nob), "restricted")


def _cfg(cls, memory_mode="high"):
    return cls(
        memory_mode=memory_mode,
        chol_batch_size=2,
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )


def _assert(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize(
    "representation,memory_mode",
    [(r, "high") for r in ["ucisd", "ucisd_k", "ucisd_k_modes", "ucisd_modes"]]
    + [("ucisd", "low")],
)
def test_compact_ci_matches_zero_padded(representation, memory_mode):
    data, padded, w, ham, sys = _inputs()
    full = ucisd.make_ucisd_trial_data(padded, sys)
    mod = import_module("trot.trial." + representation)
    factory = getattr(
        mod,
        {
            "ucisd": "make_ucisd_trial_data",
            "ucisd_k": "make_ucisd_k_trial_data",
            "ucisd_k_modes": "make_ucisd_k_mode_trial_data",
            "ucisd_modes": "make_ucisd_mode_trial_data",
        }[representation],
    )
    kw = {"mixed_precision": False} if "modes" in representation else {}
    data = {**data, "c1a": data["ci1a"], "c1b": data["ci1b"]}
    if representation == "ucisd_k_modes":
        ev, vec = np.linalg.eigh(
            ucisd_k.build_ucisd_kernel(data["ci2aa"], data["ci2ab"], data["ci2bb"])
        )
        data.update(eigenvalues=ev, modes=vec.T)
    elif representation == "ucisd_modes":
        aa = data["ci2aa"].reshape(9, 9)
        ab = data["ci2ab"].reshape(9, 4)
        bb = data["ci2bb"].reshape(4, 4)
        ea, va = np.linalg.eigh(aa)
        eb, vb = np.linalg.eigh(bb)
        u, s, v = np.linalg.svd(ab, full_matrices=False)
        data.update(
            eigenvalues_aa=ea,
            modes_aa=va.T.reshape(9, 3, 3),
            eigenvalues_bb=eb,
            modes_bb=vb.T.reshape(4, 2, 2),
            singular_values_ab=s,
            left_modes_ab=u.T.reshape(4, 3, 3),
            right_modes_ab=v.reshape(4, 2, 2),
        )
    trial = factory(data, sys, **kw)
    assert trial.norb == 7 and trial.nvir == (3, 2)
    meas = import_module("trot.meas." + representation)
    cfg = _cfg(ci_meas.UcisdMeasCfg, memory_mode)
    ctx = meas.build_meas_ctx(ham, trial, cfg=cfg)
    fctx = ci_meas.build_meas_ctx(ham, full, cfg)
    _assert(jax.jit(mod.overlap_r)(w, trial), ucisd.overlap_r(w, full))
    _assert(mod.get_rdm1(trial), ucisd.get_rdm1(full))
    for name in ["force_bias_kernel_rw_rh", "energy_kernel_rw_rh"]:
        _assert(
            jax.jit(getattr(meas, name))(w, ham, ctx, trial),
            getattr(ci_meas, name)(w, ham, fctx, full),
        )


@pytest.mark.parametrize("walker_kind", ["unrestricted", "generalized"])
def test_dense_ci_other_walkers(walker_kind):
    data, padded, w, ham, sys = _inputs()
    trial = ucisd.make_ucisd_trial_data(data, sys)
    full = ucisd.make_ucisd_trial_data(padded, sys)
    if walker_kind == "unrestricted":
        walker = (w, w[:, :2])
        suffix = "uw_rh"
        overlap = ucisd.overlap_u
    else:
        walker = jnp.block([[w, 0.02 * w[:, :2]], [0.03 * w, w[:, :2]]])
        suffix = "gw_rh"
        overlap = ucisd.overlap_g
    _assert(jax.jit(overlap)(walker, trial), overlap(walker, full))
    cfg = _cfg(ci_meas.UcisdMeasCfg)
    ctx = ci_meas.build_meas_ctx(ham, trial, cfg=cfg)
    fctx = ci_meas.build_meas_ctx(ham, full, cfg)
    for what in ("energy", "force_bias"):
        fn = getattr(ci_meas, what + "_kernel_" + suffix)
        _assert(jax.jit(fn)(walker, ham, ctx, trial), fn(walker, ham, fctx, full))


def _pt_data(data):
    return dict(
        mo_coeff_b=data["mo_coeff_b"],
        t1a=data["ci1a"],
        t1b=data["ci1b"],
        t2aa=data["ci2aa"],
        t2ab=data["ci2ab"],
        t2bb=data["ci2bb"],
        t2_layout="iajb",
    )


@pytest.mark.parametrize("memory_mode", ["high", "low"])
@pytest.mark.parametrize("nva,nvb", [(3, 2), (2, 0)])
def test_compact_pt_dense_and_modes_match_zero_padded(memory_mode, nva, nvb):
    data, padded, w, ham, sys = _inputs(nva, nvb)
    trial = ptuccsd_thouless.make_ptuccsd_thouless_trial_data(_pt_data(data), sys)
    full = ptuccsd_thouless.make_ptuccsd_thouless_trial_data(_pt_data(padded), sys)
    mode = ptuccsd_modes.make_ptuccsd_thouless_mode_trial_data(
        _pt_data(data), sys, mixed_precision=False
    )
    assert trial.nvir == mode.nvir == (nva, nvb)
    assert mode.modes.shape[1] == 3 * nva + 2 * nvb
    assert trial.mo_t_a.shape == full.mo_t_a.shape == (7, 3)
    cfg = _cfg(pt_meas.PtuccsdThoulessMeasCfg, memory_mode)
    mcfg = _cfg(pt_mode_meas.PtuccsdModeMeasCfg, memory_mode)
    fctx = pt_meas.build_ptuccsd_thouless_meas_ctx(ham, full, cfg)
    expected = pt_meas.components_ptuccsd_thouless_rw_rh(w, ham, fctx, full)
    for t, m, tm, c in [
        (
            trial,
            pt_meas,
            ptuccsd_thouless,
            pt_meas.build_ptuccsd_thouless_meas_ctx(ham, trial, cfg),
        ),
        (
            mode,
            pt_mode_meas,
            ptuccsd_modes,
            pt_mode_meas.build_ptuccsd_mode_meas_ctx(ham, mode, mcfg),
        ),
    ]:
        _assert(jax.jit(tm.overlap_r)(w, t), ptuccsd_thouless.overlap_r(w, full))
        _assert(tm.get_rdm1(t), ptuccsd_thouless.get_rdm1(full))
        fn = (
            m.components_ptuccsd_thouless_rw_rh if m is pt_meas else m.components_ptuccsd_mode_rw_rh
        )
        _assert(jax.jit(fn)(w, ham, c, t), expected)
        _assert(
            jax.jit(m.force_bias_kernel_rw_rh)(w, ham, c, t),
            pt_meas.force_bias_kernel_rw_rh(w, ham, fctx, full),
        )
    cached = {
        k: getattr(mode, k)
        for k in ("mo_t_a", "mo_t_b", "mo_coeff_b", "eigenvalues", "modes", "nvir_t_outer")
    }
    restored = ptuccsd_modes.make_ptuccsd_thouless_mode_trial_data(
        cached, sys, mixed_precision=False
    )
    _assert(jax.jit(ptuccsd_modes.overlap_r)(w, restored), ptuccsd_thouless.overlap_r(w, full))


@pytest.fixture(scope="module")
def pyscf_fno_cc():
    from pyscf import gto, scf, mp, cc, lib

    lib.num_threads(1)
    mol = gto.M(atom="N 0 0 0; N 0 0 2.4", unit="Bohr", basis="cc-pvdz", verbose=5)
    mf = scf.UHF(mol).run()
    ump2 = mp.UMP2(mf, frozen=2).run()
    frozen, orbitals = ump2.make_fno(nvir_act=(5, 4))
    fno = cc.UCCSD(mf, frozen=frozen, mo_coeff=orbitals).run()
    assert fno.converged
    return fno


def test_pyscf_fno_staging_and_mode_cache(pyscf_fno_cc, tmp_path):
    from trot import staging
    from trot.cisd_workflow import CisdModeConfig, prepare_cisd_modes

    fno = pyscf_fno_cc
    path = tmp_path / "fno.h5"
    staged = staging.stage(fno, norb_frozen_core=2, cache=path)
    assert staged.ham.norb == fno.mo_coeff[0].shape[1] - 2
    assert staged.ham.nelec == (5, 5)
    assert staged.trial.data["ci1a"].shape == (5, 5)
    assert staged.trial.data["ci1b"].shape == (5, 4)
    assert staged.ham.h1.shape == (26, 26)
    assert staged.trial.data["mo_coeff_b"].shape == (26, 26)
    loaded = staging.load(path)
    for spin in range(2):
        np.testing.assert_array_equal(loaded.trial.frozen[spin], fno.frozen[spin])
    cfg = CisdModeConfig(threshold=0.0)
    prepared = prepare_cisd_modes(path, cfg, verbose=False)
    reloaded = staging.load(path, derived_trial_key=prepared.cache_key)
    sys = System(26, (5, 5), "restricted")
    mode = ucisd_k_modes.make_ucisd_k_mode_trial_data(
        reloaded.trial.data, sys, mixed_precision=False
    )
    assert mode.pair_dim == (25, 20)
    assert mode.nvir == (5, 4)
    pt_input = staging._stage_pt2uccsd_input(staging.StagedMfOrCc(fno, 2))
    pt = ptuccsd_modes.make_ptuccsd_thouless_mode_trial_data(
        pt_input.data, sys, mixed_precision=False
    )
    assert pt.nvir == (5, 4) and pt.nvir_t_outer == (16, 17)
    assert pt.mo_t_a.shape == pt.mo_t_b.shape == (26, 5)


def test_staging_rejects_unsupported_occupied_freezing(pyscf_fno_cc):
    from trot.staging import StagedCc

    with pytest.raises(NotImplementedError, match="norb_frozen_core"):
        StagedCc(pyscf_fno_cc, None)


@pytest.mark.parametrize("head", [2, 5])
def test_compact_pt_pair_sampling_matches_padded(head):
    data, padded, w, ham, sys = _inputs()
    cfg = _cfg(pt_mode_meas.PtuccsdModeMeasCfg)
    sampling = pt_mode_meas.PtuccsdModePairSamplingCfg(
        chol_head_size=head,
        pair_sample_size=16,
        head_chol_batch_size=2,
        tail_probability_uniform_mix=1.0,
        walker_guide_policy="abs_weight",
    )
    walkers = jnp.stack([w, w * (1 + 0.01j), w * (1 - 0.01j)])
    weights = jnp.asarray([1 + 0.2j, 0.8 - 0.1j, 1.2 + 0.1j])
    results = []
    for d in (data, padded):
        trial = ptuccsd_modes.make_ptuccsd_thouless_mode_trial_data(
            _pt_data(d), sys, mixed_precision=False
        )
        ctx = pt_mode_meas.build_ptuccsd_mode_meas_ctx(
            ham, trial, cfg, n_mode_chunks=3, component_sampling=sampling
        )
        result = jax.jit(pt_mode_meas.pair_sampled_ptuccsd_block_components, static_argnums=3)(
            walkers, weights, jax.random.PRNGKey(501), 2, ham, ctx, trial
        )
        results.append(result)
    _assert(results[0].numerator, results[1].numerator)
    _assert(results[0].weight, results[1].weight)


@pytest.mark.parametrize("head", [2, 5])
def test_compact_ci_pair_sampling_matches_padded(head):
    from trot.meas import ucisd_k_modes as meas

    data, padded, w, ham, sys = _inputs()
    cfg = _cfg(ci_meas.UcisdMeasCfg)
    sampling = meas.UcisdKModePairSamplingCfg(
        chol_head_size=head,
        pair_sample_size=16,
        head_chol_batch_size=2,
        tail_probability_uniform_mix=1.0,
        walker_guide_policy="weight",
    )
    walkers = jnp.stack([w, w * (1 + 0.01j), w * (1 - 0.01j)])
    weights = jnp.asarray([1.0, 0.8, 1.2])
    results = []
    for d in (data, padded):
        values, vectors = np.linalg.eigh(
            ucisd_k.build_ucisd_kernel(d["ci2aa"], d["ci2ab"], d["ci2bb"])
        )
        trial = ucisd_k_modes.make_ucisd_k_mode_trial_data(
            dict(d, c1a=d["ci1a"], c1b=d["ci1b"], eigenvalues=values, modes=vectors.T),
            sys,
            mixed_precision=False,
        )
        ctx = meas.build_meas_ctx(ham, trial, cfg=cfg, n_mode_chunks=3, energy_sampling=sampling)
        result = jax.jit(meas.pair_sampled_block_energy, static_argnums=4)(
            walkers,
            weights,
            jnp.ones(3),
            jax.random.PRNGKey(503),
            2,
            ham,
            ctx,
            trial,
            jnp.asarray(0.0),
            jnp.asarray(1.0e10),
        )
        results.append(result)
    _assert(results[0], results[1])


@pytest.mark.parametrize("method", ["ci", "pt"])
def test_compact_mixed_precision_matches_padded(method):
    data, padded, w, ham, sys = _inputs()
    values = []
    for d in (data, padded):
        if method == "ci":
            trial = ucisd.make_ucisd_trial_data(d, sys)
            meas = ci_meas
            ctx = meas.build_meas_ctx(ham, trial)
        else:
            trial = ptuccsd_thouless.make_ptuccsd_thouless_trial_data(_pt_data(d), sys)
            meas = pt_meas
            ctx = meas.build_ptuccsd_thouless_meas_ctx(ham, trial)
        values.append(jax.jit(meas.energy_kernel_rw_rh)(w, ham, ctx, trial))
    np.testing.assert_allclose(*values, rtol=2e-6, atol=2e-6)
