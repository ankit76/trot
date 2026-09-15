from trot import config

config.configure_once()

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.core.system import System_uh
from trot.ham.chol import HamChol
from trot.ham.chol_u import HamCholU, from_ham_chol
from trot.meas.pt2ccsd import Pt2ccsdMeasCfg
from trot.meas.pt2ccsd import build_meas_ctx as build_meas_ctx_r
from trot.meas.pt2ccsd import energy_kernel_rw_rh
from trot.meas.upt2ccsd import (
    build_meas_ctx,
    energy_kernel_uw_uh_bar,
    energy_kernel_uw_uh_chunk,
    energy_kernel_uw_uh_sto,
)
from trot.mixed import available_mixed_recipes, get_mixed_recipe
from trot.prop.blocks import block_mixed
from trot.staging import build_ham_uchol, stage, stage_pt2ccsd_trial, stage_upt2ccsd_trial
from trot.stat_utils import pt2ccsd_blocking
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot.trial.upt2ccsd import make_upt2ccsd_trial_data

# ---------------------------------------------------------------------------
# Module-level fixtures — built once for the whole test file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def o2_system():
    """Triplet O2 in sto-6g with two frozen cores: open shell, nocc_a != nocc_b.

    The hamiltonian is cut tightly so that the tau = 0 energy check is limited by
    round-off rather than by the cholesky truncation.
    """
    mol = gto.M(atom="O 0 0 0; O 0 0 1.20577", basis="sto6g", spin=2, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    mo1 = mf.stability()[0]
    mf.kernel(dm0=mf.make_rdm1(mo1, mf.mo_occ))

    mycc = cc.UCCSD(mf, frozen=2)
    mycc.kernel()

    with contextlib.redirect_stdout(io.StringIO()):
        ham_in = build_ham_uchol(mycc, chol_cut=1e-8, norb_frozen_core=2)
    ham_data = HamCholU(
        h0=jnp.asarray(ham_in.h0),
        h1_a=jnp.asarray(ham_in.h1_a),
        h1_b=jnp.asarray(ham_in.h1_b),
        chol_a=jnp.asarray(ham_in.chol_a),
        chol_b=jnp.asarray(ham_in.chol_b),
    )
    sys = System_uh(norb=ham_in.norb, nelec=ham_in.nelec)
    trial_data = make_upt2ccsd_trial_data(stage_upt2ccsd_trial(mycc).data, sys)

    return dict(mf=mf, mycc=mycc, sys=sys, ham_data=ham_data, trial_data=trial_data)


@pytest.fixture(scope="module")
def closed_shell_system():
    """A closed-shell H6 chain solved with RHF/CCSD, and the same solution as UHF/UCCSD."""
    mol = gto.M(atom="; ".join(f"H 0 0 {1.6 * i}" for i in range(6)), basis="631g", unit="b", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    rcc = cc.CCSD(mf)
    rcc.kernel()
    ucc = cc.addons.convert_to_uccsd(rcc)
    return dict(mf=mf, rcc=rcc, ucc=ucc)


def _walkers(s, scale=0.2, seed=0):
    """The UHF guide determinant, and a complex perturbation of it."""
    rng = np.random.default_rng(seed)
    norb_a, norb_b = s["sys"].norb
    na, nb = s["sys"].nelec
    w0 = (jnp.eye(norb_a)[:, :na] + 0j, jnp.eye(norb_b)[:, :nb] + 0j)

    def noisy(w):
        return w + scale * jnp.asarray(rng.normal(size=w.shape) + 1j * rng.normal(size=w.shape))

    return {"tau=0": w0, "perturbed": (noisy(w0[0]), noisy(w0[1]))}


def _energy(ham_data, out):
    t2, e0, e1 = out
    return ham_data.h0 + e0 + e1 - t2 * e0


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "measure_type, kernel, extra",
    [
        ("chunk", energy_kernel_uw_uh_chunk, {}),
        ("bar", energy_kernel_uw_uh_bar, {}),
        ("sto_chol", energy_kernel_uw_uh_sto, {"n_chol_head": "full"}),
    ],
)
def test_initial_energy_matches_uccsd(o2_system, measure_type, kernel, extra):
    """At tau = 0 the walker is the UHF guide, and every kernel must give UCCSD."""
    s = o2_system
    ctx = build_meas_ctx(
        s["ham_data"], s["trial_data"], Pt2ccsdMeasCfg(measure_type=measure_type, **extra)
    )
    out = kernel(_walkers(s)["tau=0"], s["ham_data"], ctx, s["trial_data"])
    assert float(_energy(s["ham_data"], out).real) == pytest.approx(float(s["mycc"].e_tot), abs=1e-6)


@pytest.mark.parametrize("nchol_chunk", [1, 8, 1000])
def test_kernels_agree(o2_system, nchol_chunk):
    """chunk, bar and a fully deterministic sto_chol are one estimator, for any chunk size
    (including one that overshoots nchol and is clamped)."""
    s = o2_system
    ham, trial = s["ham_data"], s["trial_data"]

    def ctx(measure_type, **kw):
        cfg = Pt2ccsdMeasCfg(measure_type=measure_type, nchol_chunk=nchol_chunk, **kw)
        return build_meas_ctx(ham, trial, cfg)

    ctx_chunk, ctx_bar = ctx("chunk"), ctx("bar")
    ctx_full = ctx("sto_chol", n_chol_head="full")
    for w in _walkers(s).values():
        chunk = np.asarray(energy_kernel_uw_uh_chunk(w, ham, ctx_chunk, trial))
        bar = np.asarray(energy_kernel_uw_uh_bar(w, ham, ctx_bar, trial))
        full = np.asarray(energy_kernel_uw_uh_sto(w, ham, ctx_full, trial))  # no key
        np.testing.assert_allclose(bar, chunk, rtol=0, atol=1e-10)
        np.testing.assert_allclose(full, bar, rtol=0, atol=1e-12)


def test_closed_shell_reduces_to_restricted(closed_shell_system):
    """
    A closed-shell RCCSD written as UCCSD, measured on the walker (w, w) against the
    hamiltonian duplicated across spins, is the restricted estimator on w.
    """
    s = closed_shell_system
    with contextlib.redirect_stdout(io.StringIO()):
        ham_in = stage(s["mf"]).ham
    ham_r = HamChol(
        jnp.asarray(ham_in.h0), jnp.asarray(ham_in.h1), jnp.asarray(ham_in.chol), basis=ham_in.basis
    )
    ham_u = from_ham_chol(ham_r)

    staged_r = stage_pt2ccsd_trial(s["rcc"])
    trial_r = Pt2ccsdTrial(mo_t=jnp.asarray(staged_r.data["mo_t"]), t2=jnp.asarray(staged_r.data["t2"]))
    nocc = trial_r.nocc
    sys_u = System_uh(norb=trial_r.norb, nelec=(nocc, nocc))
    trial_u = make_upt2ccsd_trial_data(stage_upt2ccsd_trial(s["ucc"]).data, sys_u)

    ctx_r = build_meas_ctx_r(ham_r, trial_r, Pt2ccsdMeasCfg())
    ctx_u = build_meas_ctx(ham_u, trial_u, Pt2ccsdMeasCfg(measure_type="bar"))

    rng = np.random.default_rng(1)
    w = jnp.eye(trial_r.norb)[:, :nocc] + 0.2 * jnp.asarray(
        rng.normal(size=(trial_r.norb, nocc)) + 1j * rng.normal(size=(trial_r.norb, nocc))
    )
    ref = np.asarray(energy_kernel_rw_rh(w, ham_r, ctx_r, trial_r))
    got = np.asarray(energy_kernel_uw_uh_bar((w, w), ham_u, ctx_u, trial_u))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)


def test_sto_chol_is_unbiased(o2_system):
    """
    Averaging one walker's sampled energy over many keys converges to the exact value.
    t2 and e0 come out exact every time, since e2_0 is never sampled.
    """
    s = o2_system
    ham, trial = s["ham_data"], s["trial_data"]
    w = _walkers(s)["perturbed"]

    exact = np.asarray(
        energy_kernel_uw_uh_bar(w, ham, build_meas_ctx(ham, trial, Pt2ccsdMeasCfg(measure_type="bar")), trial)
    )
    cfg = Pt2ccsdMeasCfg(measure_type="sto_chol", nchol_chunk=8, n_chol_head=4, n_chol_samples=16)
    ctx = build_meas_ctx(ham, trial, cfg)

    n_keys = 400
    f = jax.jit(jax.vmap(lambda key: energy_kernel_uw_uh_sto(w, ham, ctx, trial, key)))
    out = np.asarray(f(jax.random.split(jax.random.PRNGKey(5), n_keys)))

    np.testing.assert_allclose(out[:, :2], np.broadcast_to(exact[:2], (n_keys, 2)), rtol=0, atol=1e-12)
    mean, sem = out[:, 2].mean(), out[:, 2].std() / np.sqrt(n_keys)
    assert sem > 0.0
    assert abs(mean - exact[2]) < 4 * sem


# ---------------------------------------------------------------------------
# AfqmcMixed
# ---------------------------------------------------------------------------

_PARAMS = dict(dt=0.005, n_walkers=4, n_prop_steps=2, n_blocks=20, n_eql_blocks=1, seed=3)


def _run(mycc, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        af = AfqmcMixed(mycc, **_PARAMS, **kwargs)
        e, err = af.kernel()
    return af, e, err, buf.getvalue()


def test_afqmc_mixed_deterministic_kernels_give_one_run(o2_system):
    """
    The guide trajectory does not depend on the trial, so the three deterministic
    unrestricted kernels must give the same run, and the flags must name the
    unrestricted pieces.
    """
    mycc = o2_system["mycc"]
    af, e_chunk, err_chunk, out = _run(mycc)  # trial inferred from the UCCSD object
    assert af.trial == "upt2ccsd" and af.guide == "uhf"
    assert "meas/upt2ccsd.py:energy_kernel_uw_uh_chunk" in out
    assert np.isfinite(e_chunk) and err_chunk > 0.0

    _, e_bar, err_bar, _ = _run(mycc, trial="upt2ccsd_bar", nchol_chunk=8)
    _, e_full, err_full, out_full = _run(
        mycc, trial="upt2ccsd_sto_chol", trial_kwargs={"n_chol_head": "full"}
    )
    assert e_bar == pytest.approx(e_chunk, abs=1e-10)
    assert err_bar == pytest.approx(err_chunk, abs=1e-10)
    assert e_full == pytest.approx(e_bar, abs=1e-10)
    assert err_full == pytest.approx(err_bar, abs=1e-10)
    assert "n_chol_samples_used         = 0" in out_full


def test_afqmc_mixed_sto_chol_runs(o2_system):
    af, e, err, out = _run(o2_system["mycc"], trial="upt2ccsd_sto_chol")
    assert np.isfinite(e) and err > 0.0
    assert "n_chol_head_used" in out and "n_chol_samples_used" in out
    assert af.guide_e_tot is not None


def test_afqmc_mixed_max_memory(o2_system):
    """max_memory sizes the unrestricted chunking, and the context scans what it planned."""
    with contextlib.redirect_stdout(io.StringIO()):
        af = AfqmcMixed(o2_system["mycc"], trial="upt2ccsd_bar", max_memory=0.5, **_PARAMS)
        job = af.build_job()
    plan = job.chunk_plan
    assert plan is not None and plan.bytes_used <= plan.budget_bytes
    assert job.mix_meas_ctx().nchol_chunk == plan.nchol_chunk


# ---------------------------------------------------------------------------
# Recipe registry and API checks
# ---------------------------------------------------------------------------


def test_upt2ccsd_recipes():
    for trial in ("upt2ccsd", "upt2ccsd_bar", "upt2ccsd_sto_chol"):
        rec = get_mixed_recipe(trial)
        assert rec.guide == "uhf"
        assert rec.walker_kind == "unrestricted"
        assert rec.ham_basis == "uchol"
        assert rec.plan_chunking is not None
        assert rec.mixed_block_fn is block_mixed
        assert rec.blocking_fn is pt2ccsd_blocking
        assert ("uhf", trial) in available_mixed_recipes()
    # the restricted recipes are untouched
    assert get_mixed_recipe("pt2ccsd").ham_basis == "restricted"


def test_afqmc_mixed_rejects_mismatched_cc(o2_system, closed_shell_system):
    ucc, rcc = o2_system["mycc"], closed_shell_system["rcc"]
    assert AfqmcMixed(rcc).trial == "pt2ccsd"
    with pytest.raises(ValueError, match="needs a restricted CCSD"):
        AfqmcMixed(ucc, trial="pt2ccsd_bar")
    with pytest.raises(ValueError, match="needs a UCCSD"):
        AfqmcMixed(rcc, trial="upt2ccsd_bar")
    with pytest.raises(ValueError, match="basis_a / basis_b"):
        AfqmcMixed(rcc, basis_a=np.eye(2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
