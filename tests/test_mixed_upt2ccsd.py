"""
Mixed guide/trial AFQMC with the plain pt2CCSD estimators on the unrestricted (uchol)
hamiltonian: the chunked and the bar kernels against the restricted kernel in the closed
shell limit and against each other, the tau = 0 UCCSD energies with a frozen core, and
AfqmcMixed end to end with the UHF and UCISD guides.
"""

from trot import config

config.configure_once()

import contextlib
import io
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.core.system import System_uh
from trot.ham.chol import HamChol
from trot.ham.chol_u import HamCholU, from_ham_chol
from trot.meas.pt2ccsd import combine_first_order_energy, energy_kernel_rw_rh, make_pt2ccsd_meas_ops
from trot.meas.pt2ccsd_chunking import make_chunk_meas_cfg
from trot.meas.upt2ccsd_bar_uh import build_meas_ctx as build_bar_ctx
from trot.meas.upt2ccsd_bar_uh import energy_kernel_uw_uh_bar
from trot.meas.upt2ccsd_uh import build_meas_ctx, energy_kernel_uw_uh_chunk, plan_chunking_for_run_u
from trot.mixed import available_mixed_recipes, get_mixed_recipe, stage_pt2ccsd_trial
from trot.staging import StagedMfOrCc, _stage_ham_input
from trot.staging_u import build_ham_uchol, stage_upt2ccsd_trial_uh
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot.trial.upt2ccsd_uh import Upt2ccsdTrial, make_upt2ccsd_trial_data, overlap_u

jax.config.update("jax_enable_x64", True)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def test_registry():
    pairs = available_mixed_recipes()
    for pair in (
        ("uhf", "upt2ccsd"),
        ("ucisd", "upt2ccsd"),
        ("uhf", "upt2ccsd_bar"),
        ("ucisd", "upt2ccsd_bar"),
    ):
        assert pair in pairs
    rec = get_mixed_recipe("upt2ccsd_bar", guide="ucisd")
    assert rec.walker_kind == "unrestricted" and rec.ham_basis == "uchol"
    assert get_mixed_recipe("upt2ccsd").guide == "uhf"
    with pytest.raises(ValueError, match="hamiltonians"):
        get_mixed_recipe("upt2ccsd", guide="cisd")


# ---------------------------------------------------------------------------
# closed shell: the unrestricted kernels reduce to the restricted one
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def closed_shell():
    """A closed-shell H6 chain solved with RHF/CCSD, and the same solution as UCCSD."""
    mol = gto.M(
        atom="; ".join(f"H 0 0 {1.6 * i}" for i in range(6)), basis="631g", unit="b", verbose=0
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    rcc = cc.CCSD(mf)
    rcc.conv_tol = 1e-10
    rcc.kernel()
    ucc = cc.addons.convert_to_uccsd(rcc)
    ham_in = _quiet(_stage_ham_input, StagedMfOrCc(mf, 0), chol_cut=1e-6, verbose=False)
    ham_r = HamChol(
        jnp.asarray(ham_in.h0), jnp.asarray(ham_in.h1), jnp.asarray(ham_in.chol), basis="restricted"
    )
    ham_u = from_ham_chol(ham_r)
    staged_r = stage_pt2ccsd_trial(rcc)
    trial_r = Pt2ccsdTrial(
        mo_t=jnp.asarray(staged_r.data["mo_t"]), t2=jnp.asarray(staged_r.data["t2"])
    )
    nocc = trial_r.nocc
    sys_u = System_uh(norb=(trial_r.norb, trial_r.norb), nelec=(nocc, nocc))
    trial_u = make_upt2ccsd_trial_data(stage_upt2ccsd_trial_uh(ucc).data, sys_u)
    return dict(
        mf=mf,
        rcc=rcc,
        ucc=ucc,
        ham_r=ham_r,
        ham_u=ham_u,
        trial_r=trial_r,
        trial_u=trial_u,
        sys_u=sys_u,
    )


@pytest.mark.parametrize("nchol_chunk", [1, 5, 1000])
def test_closed_shell_reduces_to_restricted(closed_shell, nchol_chunk):
    """
    A closed-shell RCCSD written as UCCSD, measured on the walker (w, w) against the
    hamiltonian duplicated across spins, is the restricted estimator on w.
    """
    s = closed_shell
    ham_r, ham_u, trial_r, trial_u = s["ham_r"], s["ham_u"], s["trial_r"], s["trial_u"]
    from trot.core.system import System

    sys_r = System(norb=trial_r.norb, nelec=(trial_r.nocc, trial_r.nocc), walker_kind="restricted")
    ctx_r = make_pt2ccsd_meas_ops(sys_r, mixed_precision=False, testing=True).build_meas_ctx(
        ham_r, trial_r
    )
    cfg = make_chunk_meas_cfg(mixed_precision=False, testing=True, nchol_chunk=nchol_chunk)
    ctx_c = build_meas_ctx(ham_u, trial_u, cfg)
    ctx_b = build_bar_ctx(ham_u, trial_u, cfg)

    rng = np.random.default_rng(1)
    for _ in range(3):
        w = jnp.eye(trial_r.norb)[:, : trial_r.nocc] + 0.2 * jnp.asarray(
            rng.normal(size=(trial_r.norb, trial_r.nocc))
            + 1j * rng.normal(size=(trial_r.norb, trial_r.nocc))
        )
        ref = np.asarray(energy_kernel_rw_rh(w, ham_r, ctx_r, trial_r))
        got_c = np.asarray(energy_kernel_uw_uh_chunk((w, w), ham_u, ctx_c, trial_u))
        got_b = np.asarray(energy_kernel_uw_uh_bar((w, w), ham_u, ctx_b, trial_u))
        np.testing.assert_allclose(got_c, ref, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(got_b, ref, rtol=1e-10, atol=1e-11)
        # the reference overlap is the restricted one, det^2
        o_r = complex(jnp.linalg.det(trial_r.mo_t.T @ w) ** 2)
        assert complex(overlap_u((w, w), trial_u)) == pytest.approx(o_r, rel=1e-12)


# ---------------------------------------------------------------------------
# open shell with a frozen core: tau = 0 and chunk == bar
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def o2():
    """Triplet O2 in sto-6g with two frozen cores: open shell, nocc_a != nocc_b."""
    mol = gto.M(atom="O 0 0 0; O 0 0 1.20577", basis="sto6g", spin=2, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mo1 = mf.stability()[0]
    mf.kernel(dm0=mf.make_rdm1(mo1, mf.mo_occ))
    mycc = cc.UCCSD(mf, frozen=2)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    ham_in = _quiet(build_ham_uchol, mycc, chol_cut=1e-8)
    ham = HamCholU(
        h0=jnp.asarray(ham_in.h0),
        h1_a=jnp.asarray(ham_in.h1_a),
        h1_b=jnp.asarray(ham_in.h1_b),
        chol_a=jnp.asarray(ham_in.chol_a),
        chol_b=jnp.asarray(ham_in.chol_b),
    )
    sys = System_uh(norb=ham_in.norb, nelec=ham_in.nelec)
    trial = make_upt2ccsd_trial_data(stage_upt2ccsd_trial_uh(mycc).data, sys)
    return dict(mf=mf, mycc=mycc, ham=ham, sys=sys, trial=trial)


def _o2_walkers(s, seed=0):
    rng = np.random.default_rng(seed)
    norb_a, norb_b = s["sys"].norb
    na, nb = s["sys"].nelec
    w0 = (jnp.eye(norb_a)[:, :na] + 0j, jnp.eye(norb_b)[:, :nb] + 0j)

    def noisy(w):
        return w + 0.2 * jnp.asarray(rng.normal(size=w.shape) + 1j * rng.normal(size=w.shape))

    return {"tau=0": w0, "perturbed": (noisy(w0[0]), noisy(w0[1]))}


@pytest.mark.parametrize("kernel_name", ["chunk", "bar"])
def test_tau0_kernel_energy_is_uccsd(o2, kernel_name):
    s = o2
    cfg = make_chunk_meas_cfg(mixed_precision=False, testing=True, nchol_chunk=None)
    if kernel_name == "chunk":
        ctx, kernel = build_meas_ctx(s["ham"], s["trial"], cfg), energy_kernel_uw_uh_chunk
    else:
        ctx, kernel = build_bar_ctx(s["ham"], s["trial"], cfg), energy_kernel_uw_uh_bar
    w = _o2_walkers(s)["tau=0"]
    e = complex(combine_first_order_energy(s["ham"].h0, kernel(w, s["ham"], ctx, s["trial"])))
    assert e.real == pytest.approx(s["mycc"].e_tot, abs=1e-6)
    assert complex(overlap_u(w, s["trial"])) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("nchol_chunk", [1, 7, 1000])
def test_chunk_and_bar_kernels_agree(o2, nchol_chunk):
    s = o2
    cfg = make_chunk_meas_cfg(mixed_precision=False, testing=True, nchol_chunk=nchol_chunk)
    ctx_c = build_meas_ctx(s["ham"], s["trial"], cfg)
    ctx_b = build_bar_ctx(s["ham"], s["trial"], cfg)
    ref = None
    for w in _o2_walkers(s, seed=3).values():
        c = np.asarray(energy_kernel_uw_uh_chunk(w, s["ham"], ctx_c, s["trial"]))
        b = np.asarray(energy_kernel_uw_uh_bar(w, s["ham"], ctx_b, s["trial"]))
        np.testing.assert_allclose(b, c, rtol=1e-10, atol=1e-11)
        ref = c if ref is None else ref
    plan = plan_chunking_for_run_u(s["sys"], s["ham"], s["trial"], n_walkers=4, budget_bytes=10**8)
    assert plan.n_chunks == 1 and plan.nchol_chunk >= 1


def test_trial_data_validation(o2):
    s = o2
    data = stage_upt2ccsd_trial_uh(s["mycc"]).data
    assert isinstance(s["trial"], Upt2ccsdTrial) and s["trial"].norb == s["sys"].norb
    with pytest.raises(ValueError, match="amplitudes span"):
        make_upt2ccsd_trial_data(
            data, System_uh(norb=(s["sys"].norb_a + 1, s["sys"].norb_b), nelec=s["sys"].nelec)
        )


# ---------------------------------------------------------------------------
# AfqmcMixed end to end
# ---------------------------------------------------------------------------


def _nh2_uccsd(frozen=None):
    mol = gto.M(
        atom="""
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        """,
        basis="sto-6g",
        spin=1,
        verbose=0,
    )
    mf = scf.UHF(mol).newton()
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.UCCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def nh2_ucc():
    return _nh2_uccsd(frozen=1)


_PARAMS: dict[str, Any] = dict(
    dt=0.005, n_walkers=6, n_prop_steps=4, n_blocks=16, n_eql_blocks=3, seed=11
)


def _tau0(af) -> float:
    from trot.driver_mixed import _init_trial_energy

    job = _quiet(af.build_job)
    state = job.prop_ops.init_prop_state(
        sys=job.sys,
        ham_data=job.ham_data,
        trial_ops=job.trial_ops,
        trial_data=job.trial_data,
        meas_ops=job.meas_ops,
        params=job.params,
    )
    e, _ = _init_trial_energy(
        state,
        job.ham_data,
        job.mix_trial_data,
        job.mix_trial_meas_ops,
        job.mix_meas_ctx(),
        job.params,
        job.recipe.components,
        job.recipe.energy_fn,
    )
    return float(np.real(e))


@pytest.mark.parametrize("trial", ["upt2ccsd", "upt2ccsd_bar"])
@pytest.mark.parametrize("guide", ["uhf", "ucisd"])
def test_tau0_energy_is_uccsd(nh2_ucc, guide, trial):
    af = AfqmcMixed(
        nh2_ucc, guide=guide, trial=trial, chol_cut=1e-8, mixed_precision=False, **_PARAMS
    )
    assert _tau0(af) == pytest.approx(nh2_ucc.e_tot, abs=1e-6)
    job = af.job
    assert isinstance(job.ham_data, HamCholU) and job.staged.ham.frozen == (1, 1)
    assert job.staged.trial.kind == guide


@pytest.mark.parametrize("guide", ["uhf", "ucisd"])
def test_bar_and_chunk_trials_give_the_same_run(nh2_ucc, guide):
    chunk = AfqmcMixed(
        nh2_ucc, guide=guide, trial="upt2ccsd", chol_cut=1e-6, mixed_precision=False, **_PARAMS
    )
    e_c, err_c = _quiet(chunk.kernel)
    bar = AfqmcMixed(
        nh2_ucc, guide=guide, trial="upt2ccsd_bar", chol_cut=1e-6, mixed_precision=False, **_PARAMS
    )
    e_b, err_b = _quiet(bar.kernel)
    assert np.isfinite(e_c) and np.isfinite(e_b)
    assert e_b == pytest.approx(e_c, abs=1e-8)
    assert err_b == pytest.approx(err_c, abs=1e-8)
    assert bar.guide_e_tot == pytest.approx(chunk.guide_e_tot, abs=1e-10)
    res = bar.qmc_result
    assert set(res.trial_block_components) == {"theta", "electronic_0", "h_t"}
    assert res.trial_analysis.error_method == "blocking"


def test_mixed_precision_default_and_chunk_plan(nh2_ucc):
    af = AfqmcMixed(nh2_ucc, trial="upt2ccsd_bar", max_memory=2000, **_PARAMS)
    assert af.mixed_precision is True and af.guide == "uhf"
    job = _quiet(af.build_job)
    assert job.chunk_plan is not None
    assert job.chunk_plan.nchol_chunk == job.mix_meas_ctx().nchol_chunk
    e, err = _quiet(af.kernel)
    assert np.isfinite(e) and np.isfinite(err)


if __name__ == "__main__":
    pytest.main([__file__])
