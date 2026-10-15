"""
Mixed guide/trial AFQMC with the plain pt2CCSD estimators on the restricted hamiltonian:
the recipes, the bar kernel against the branch's unchunked kernel, the cholesky chunk
plan, the tau = 0 energies, and AfqmcMixed end to end with the RHF and CISD guides.
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
from trot.core.system import System
from trot.meas.pt2ccsd import combine_first_order_energy, energy_kernel_rw_rh, make_pt2ccsd_meas_ops
from trot.meas.pt2ccsd_bar import (
    build_meas_ctx,
    energy_kernel_rw_rh_bar,
    make_pt2ccsd_bar_meas_ops,
    plan_chunking_for_run,
)
from trot.meas.pt2ccsd_chunking import (
    Pt2ccsdMemoryModel,
    make_chunk_meas_cfg,
    plan_pt2ccsd_chunking,
    resolve_nchol_chunk,
)
from trot.mixed import available_mixed_recipes, get_mixed_recipe
from trot.testing import make_random_ham_chol
from trot.trial.pt2ccsd import Pt2ccsdTrial

jax.config.update("jax_enable_x64", True)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_and_pairing():
    pairs = available_mixed_recipes()
    for pair in (
        ("rhf", "pt2ccsd"),
        ("cisd", "pt2ccsd"),
        ("rhf", "pt2ccsd_bar"),
        ("cisd", "pt2ccsd_bar"),
    ):
        assert pair in pairs
    assert get_mixed_recipe("pt2ccsd").guide == "rhf"
    rec = get_mixed_recipe("pt2ccsd_bar", guide="cisd")
    assert rec.walker_kind == "restricted" and rec.ham_basis == "restricted"
    assert rec.components == ("theta", "electronic_0", "h_t")
    assert rec.energy_fn is combine_first_order_energy
    with pytest.raises(ValueError, match="hamiltonians"):
        get_mixed_recipe("upt2ccsd", guide="rhf")
    with pytest.raises(ValueError, match="walker kind"):
        get_mixed_recipe("pt2ccsd", guide="uhf")
    with pytest.raises(ValueError, match="unknown mixed trial"):
        get_mixed_recipe("no_such_trial")


# ---------------------------------------------------------------------------
# the bar kernel on a random hamiltonian
# ---------------------------------------------------------------------------

_NORB, _NOCC, _NCHOL = 7, 3, 9


@pytest.fixture(scope="module")
def random_case():
    key = jax.random.PRNGKey(5)
    k_ham, k_t1, k_t2, k_w = jax.random.split(key, 4)
    ham = make_random_ham_chol(k_ham, norb=_NORB, n_chol=_NCHOL)
    nvir = _NORB - _NOCC
    t1 = 0.05 * jax.random.normal(k_t1, (_NOCC, nvir))
    t2 = 0.02 * jax.random.normal(k_t2, (_NOCC, nvir, _NOCC, nvir))
    t2 = 0.5 * (t2 + t2.transpose(2, 3, 0, 1))  # t_iajb = t_jbia
    exp_t1 = jnp.eye(_NORB).at[:_NOCC, _NOCC:].set(t1)
    mo_t = exp_t1.T[:, :_NOCC]
    trial = Pt2ccsdTrial(mo_t=mo_t, t2=t2)
    walkers = jnp.eye(_NORB)[:, :_NOCC] + 0.3 * (
        jax.random.normal(k_w, (4, _NORB, _NOCC))
        + 0.5j * jax.random.normal(k_t1, (4, _NORB, _NOCC))
    )
    sys = System(norb=_NORB, nelec=(_NOCC, _NOCC), walker_kind="restricted")
    return dict(ham=ham, trial=trial, walkers=walkers, sys=sys)


@pytest.mark.parametrize("nchol_chunk", [1, 4, 9, 100])
def test_bar_kernel_matches_unchunked_kernel(random_case, nchol_chunk):
    ham, trial, walkers, sys = (random_case[k] for k in ("ham", "trial", "walkers", "sys"))
    meas_r = make_pt2ccsd_meas_ops(sys, mixed_precision=False, testing=True)
    ctx_r = meas_r.build_meas_ctx(ham, trial)
    cfg = make_chunk_meas_cfg(mixed_precision=False, testing=True, nchol_chunk=nchol_chunk)
    ctx_b = build_meas_ctx(ham, trial, cfg)
    assert ctx_b.nchol_chunk == resolve_nchol_chunk(_NCHOL, nchol_chunk)
    for w in walkers:
        ref = np.asarray(energy_kernel_rw_rh(w, ham, ctx_r, trial))
        got = np.asarray(energy_kernel_rw_rh_bar(w, ham, ctx_b, trial))
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)
        e_ref = complex(combine_first_order_energy(ham.h0, jnp.asarray(ref)))
        e_got = complex(combine_first_order_energy(ham.h0, jnp.asarray(got)))
        assert e_got == pytest.approx(e_ref, rel=1e-10)


def test_bar_meas_ops(random_case):
    sys = random_case["sys"]
    ops = make_pt2ccsd_bar_meas_ops(sys, mixed_precision=False, nchol_chunk=3)
    assert ops.has_kernel("energy") and not ops.has_kernel("force_bias")
    with pytest.raises(ValueError, match="restricted walkers"):
        make_pt2ccsd_bar_meas_ops(System(norb=_NORB, nelec=(2, 1), walker_kind="unrestricted"))


# ---------------------------------------------------------------------------
# the chunk plan
# ---------------------------------------------------------------------------


def test_chunk_plan_budget_splits():
    model = Pt2ccsdMemoryModel(resident=1000, per_walker=10, per_walker_chol=5)
    # a loose budget keeps every walker in flight and the whole cholesky set in one step
    plan = plan_pt2ccsd_chunking(model, n_walkers=8, nchol=20, budget_bytes=10_000)
    assert plan.n_chunks == 1 and plan.nchol_chunk == 20
    # a tighter one shrinks the cholesky chunk first, with an even division
    plan = plan_pt2ccsd_chunking(
        model, n_walkers=8, nchol=20, budget_bytes=1000 + 8 * 10 + 8 * 5 * 7
    )
    assert plan.n_chunks == 1 and 1 <= plan.nchol_chunk <= 7
    assert plan.bytes_used <= plan.budget_bytes
    # when one vector per step does not fit, the walkers give way and the chunk is re-derived
    plan = plan_pt2ccsd_chunking(
        model, n_walkers=8, nchol=20, budget_bytes=1000 + 4 * 10 + 4 * 5 * 3
    )
    assert plan.n_chunks > 1 and plan.nchol_chunk >= 1 and "re-derived" in plan.note
    assert plan.bytes_used <= plan.budget_bytes
    # a caller's nchol_chunk is kept and only n_chunks is derived
    plan = plan_pt2ccsd_chunking(
        model, n_walkers=8, nchol=20, budget_bytes=1000 + 2 * 10 + 2 * 5 * 10, nchol_chunk=10
    )
    assert plan.nchol_chunk == 10 and plan.n_chunks == 4
    # n_chunks is a floor
    plan = plan_pt2ccsd_chunking(model, n_walkers=8, nchol=20, budget_bytes=10_000, n_chunks=2)
    assert plan.n_chunks == 2
    with pytest.raises(ValueError, match="resident"):
        plan_pt2ccsd_chunking(model, n_walkers=8, nchol=20, budget_bytes=500)


def test_plan_for_run_uses_the_bar_model(random_case):
    ham, trial, sys = random_case["ham"], random_case["trial"], random_case["sys"]
    plan = plan_chunking_for_run(sys, ham, trial, n_walkers=4, budget_bytes=10**7)
    assert plan.nchol_chunk == _NCHOL and plan.n_chunks == 1
    assert plan.model.resident > 0 and plan.model.per_walker_chol > 0


# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------


def _h4_ccsd():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8", basis="sto-6g", unit="b", verbose=0
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return mycc


def _h2o_ccsd(frozen=None):
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def h4():
    return _h4_ccsd()


@pytest.fixture(scope="module")
def h2o_frozen():
    return _h2o_ccsd(frozen=1)


_PARAMS: dict[str, Any] = dict(
    dt=0.005, n_walkers=6, n_prop_steps=4, n_blocks=16, n_eql_blocks=3, seed=11
)


def _tau0(af) -> float:
    """The trial energy of the initial walkers, from the driver's tau = 0 row."""
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


@pytest.mark.parametrize("trial", ["pt2ccsd", "pt2ccsd_bar"])
@pytest.mark.parametrize("guide", ["rhf", "cisd"])
def test_tau0_energy_is_ccsd(h4, guide, trial):
    """On the initial walkers the pt2CCSD estimator reproduces the CCSD energy."""
    af = AfqmcMixed(h4, guide=guide, trial=trial, chol_cut=1e-8, mixed_precision=False, **_PARAMS)
    assert _tau0(af) == pytest.approx(h4.e_tot, abs=1e-6)


def test_tau0_energy_with_frozen_core(h2o_frozen):
    af = AfqmcMixed(
        h2o_frozen, trial="pt2ccsd_bar", chol_cut=1e-8, mixed_precision=False, **_PARAMS
    )
    assert _tau0(af) == pytest.approx(h2o_frozen.e_tot, abs=1e-6)
    assert af.norb_frozen_core == 1
    with pytest.raises(ValueError, match="contradicts cc.frozen"):
        AfqmcMixed(h2o_frozen, norb_frozen_core=2)


@pytest.mark.parametrize("guide", ["rhf", "cisd"])
def test_bar_and_plain_trials_give_the_same_run(h4, guide):
    """Same guide, same seed: the two estimators differ only by roundoff."""
    plain = AfqmcMixed(
        h4, guide=guide, trial="pt2ccsd", chol_cut=1e-6, mixed_precision=False, **_PARAMS
    )
    e_p, err_p = _quiet(plain.kernel)
    bar = AfqmcMixed(
        h4, guide=guide, trial="pt2ccsd_bar", chol_cut=1e-6, mixed_precision=False, **_PARAMS
    )
    e_b, err_b = _quiet(bar.kernel)
    assert np.isfinite(e_p) and np.isfinite(e_b)
    assert e_b == pytest.approx(e_p, abs=1e-8)
    assert err_b == pytest.approx(err_p, abs=1e-8)
    # the guide side is untouched by the choice of trial
    assert bar.guide_e_tot == pytest.approx(plain.guide_e_tot, abs=1e-10)
    res = bar.qmc_result
    assert set(res.trial_block_components) == {"theta", "electronic_0", "h_t"}
    assert res.trial_block_weights.shape[0] == _PARAMS["n_blocks"]
    assert res.trial_analysis.error_method == "blocking"


def test_tau_eql_sets_the_equilibration_blocks(h4):
    """tau_eql fixes the equilibration length whatever dt and n_prop_steps are."""
    from trot.prop.types import QmcParams

    def n_eql(af):
        params = _quiet(af.build_job).params
        assert isinstance(params, QmcParams)
        return params.n_eql_blocks

    af = AfqmcMixed(
        h4, trial="pt2ccsd", tau_eql=1.0, dt=0.01, n_prop_steps=10, n_walkers=4, n_blocks=4
    )
    assert af.n_eql_blocks_for_tau() == 10 and n_eql(af) == 10
    # a coarser block: the count follows, rounding up so that tau_eql is reached
    af = AfqmcMixed(
        h4, trial="pt2ccsd", tau_eql=1.0, dt=0.005, n_prop_steps=30, n_walkers=4, n_blocks=4
    )
    assert af.n_eql_blocks_for_tau() == 7 and n_eql(af) == 7
    # explicit params are honoured too: the block count follows their dt and n_prop_steps
    af = AfqmcMixed(h4, trial="pt2ccsd", tau_eql=1.0, n_walkers=4, n_blocks=4)
    af.params = QmcParams(dt=0.02, n_prop_steps=5, n_walkers=4, n_blocks=4, n_eql_blocks=99, seed=1)
    assert n_eql(af) == 10
    with pytest.raises(ValueError, match="either tau_eql or n_eql_blocks"):
        AfqmcMixed(h4, tau_eql=1.0, n_eql_blocks=3)
    # without tau_eql, n_eql_blocks is what the user set
    af = AfqmcMixed(h4, trial="pt2ccsd", n_eql_blocks=3, n_walkers=4, n_blocks=4)
    assert af.n_eql_blocks_for_tau() is None and n_eql(af) == 3


def test_chunk_plan_is_attached_and_mixed_precision_default(h4):
    af = AfqmcMixed(h4, trial="pt2ccsd_bar", max_memory=2000, **_PARAMS)
    assert af.mixed_precision is True
    job = _quiet(af.build_job)
    assert (
        job.chunk_plan is not None and job.chunk_plan.nchol_chunk == job.mix_meas_ctx().nchol_chunk
    )
    # the plain trial scans one vector at a time and has no plan
    with pytest.raises(ValueError, match="no memory model"):
        _quiet(AfqmcMixed(h4, trial="pt2ccsd", max_memory=2000, **_PARAMS).build_job)


def _guide_chol_dtype(job):
    """The dtype the guide propagator casts the cholesky vectors to."""
    ctx = job.prop_ops.build_prop_ctx(
        job.ham_data, job.trial_ops.get_rdm1(job.trial_data), job.params
    )
    return ctx.chol_flat.dtype


def test_guide_and_trial_precision_knobs(h4):
    """mixed_precision sets both sides; guide_mixed_precision / trial_mixed_precision override one."""
    from trot.meas.pt2ccsd_bar import get_pt2ccsd_bar_meas_cfg

    def dtypes(**kw):
        af = AfqmcMixed(h4, trial="pt2ccsd_bar", **kw, **_PARAMS)
        job = _quiet(af.build_job)
        trial = get_pt2ccsd_bar_meas_cfg(job.mix_trial_meas_ops).mixed_real_dtype
        return af, _guide_chol_dtype(job), trial

    af, g, t = dtypes(mixed_precision=True)
    assert af.guide_mixed_precision and af.trial_mixed_precision
    assert g == jnp.float32 and t == jnp.float32
    af, g, t = dtypes(mixed_precision=False)
    assert not af.guide_mixed_precision and not af.trial_mixed_precision
    assert g == jnp.float64 and t == jnp.float64
    af, g, t = dtypes(mixed_precision=True, guide_mixed_precision=False)
    assert not af.guide_mixed_precision and af.trial_mixed_precision
    assert g == jnp.float64 and t == jnp.float32
    af, g, t = dtypes(mixed_precision=False, trial_mixed_precision=True)
    assert not af.guide_mixed_precision and af.trial_mixed_precision
    assert g == jnp.float64 and t == jnp.float32
    # the chunk plan is sized with the trial's precision
    af, _, _ = dtypes(mixed_precision=True, trial_mixed_precision=False, max_memory=2000)
    job = _quiet(af.build_job)
    assert job.chunk_plan.model.per_walker_chol == job.chunk_plan.model.per_walker_chol  # exists
    cfg = get_pt2ccsd_bar_meas_cfg(job.mix_trial_meas_ops)
    assert cfg.mixed_real_dtype == jnp.float64


# ---------------------------------------------------------------------------
# the branch's own regression run, through the mixed API
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h8_ccsd():
    """8 well separated H2 dimers, the system of tests/test_pt2ccsd.py."""
    a, d, na, nc = 2, 100, 2, 8
    atoms = ""
    for n in range(nc * na):
        shift = ((n - n % na) // na) * (d - a)
        atoms += f"H {n * a + shift:.5f} 0.00000 0.00000 \n"
    mol = gto.M(atom=atoms, basis="sto6g", unit="b", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    return mycc


_H8_REFERENCES = {
    1: -8.771210912150202,
    2: -8.769501242545516,
    3: -8.769769708305297,
    4: -8.771395845249032,
}


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_h8_reference_energies(h8_ccsd, seed):
    """
    The trajectory of tests/test_pt2ccsd.py (RHF guide, pt2ccsd trial, dt=0.005, one
    walker, one step per block, 50 blocks) through AfqmcMixed: the same walk and the same
    weighted component ratios give the reference energy. The reference error there is a
    delta-method estimate, which this driver does not use, so only the mean is compared.
    """
    af = AfqmcMixed(
        h8_ccsd,
        trial="pt2ccsd",
        guide="rhf",
        mixed_precision=False,
        dt=0.005,
        n_walkers=1,
        n_prop_steps=1,
        n_blocks=50,
        n_eql_blocks=1,
        seed=seed,
    )
    e, err = _quiet(af.kernel)
    assert e == pytest.approx(_H8_REFERENCES[seed], abs=1e-6)
    assert np.isfinite(err)


if __name__ == "__main__":
    pytest.main([__file__])
