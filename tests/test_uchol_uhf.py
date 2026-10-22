"""
Unrestricted (uchol) hamiltonian, phase one: the hamiltonian, the propagator and the UHF
trial on it.

Checks, from the kernels up to the driver:
  - a HamCholU built by duplicating a restricted HamChol across spins reproduces the
    existing restricted-hamiltonian / unrestricted-walker path to roundoff: propagation
    context, one propagation step, force bias and energy kernels;
  - an RHF converted with mf.to_uhf() gives the same AFQMC run as Afqmc with
    unrestricted walkers (same seed), since the beta basis then equals the alpha one;
  - the energy at tau = 0 equals the UHF energy, with and without a frozen core,
    including a core frozen in one spin only;
  - a genuine UHF (NH2) agrees with the alpha-basis run within statistical error;
  - staged inputs round trip through the uchol h5 layout.
"""

from trot import config

config.configure_once()

import contextlib
import io
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, scf

from trot.afqmc import Afqmc, AfqmcUh
from trot.core.system import System, System_uh
from trot.ham.chol_u import HamCholU, from_ham_chol
from trot.meas.uhf import (
    build_meas_ctx,
    energy_kernel_uw_rh,
    force_bias_kernel_uw_rh,
    make_uhf_meas_ops,
)
from trot.meas.uhf_uh import (
    build_meas_ctx_uh,
    energy_kernel_uw_uh,
    force_bias_kernel_uw_uh,
    make_uhf_meas_ops_uh,
)
from trot.prop.afqmc import afqmc_step
from trot.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops
from trot.prop.chol_afqmc_ops_u import _build_prop_ctx_u, make_trotter_ops_u
from trot.prop.types import PropState, QmcParams
from trot.testing import make_random_ham_chol, rand_orthonormal_cols
from trot.trial.uhf import UhfTrial, get_rdm1
from trot.trial.uhf_uh import get_rdm1_uh
from trot.walkers import init_walkers, init_walkers_uh

jax.config.update("jax_enable_x64", True)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# hamiltonian type
# ---------------------------------------------------------------------------


def test_ham_chol_u_invariants():
    key = jax.random.PRNGKey(0)
    ham = make_random_ham_chol(key, norb=4, n_chol=5)
    ham_u = from_ham_chol(ham)
    assert ham_u.basis == "uchol" and ham_u.nchol == 5 and ham_u.norb == (4, 4)

    # different orbital counts per spin are allowed
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (5, 3, 3))
    b = jax.random.normal(k2, (5, 2, 2))
    ham_ab = HamCholU(h0=jnp.array(0.0), h1_a=jnp.eye(3), h1_b=jnp.eye(2), chol_a=a, chol_b=b)
    assert ham_ab.norb == (3, 2) and ham_ab.norb_a == 3 and ham_ab.norb_b == 2

    # the auxiliary field index must be shared
    with pytest.raises(ValueError, match="share the auxiliary field index"):
        HamCholU(h0=jnp.array(0.0), h1_a=jnp.eye(3), h1_b=jnp.eye(2), chol_a=a, chol_b=b[:4])
    # h1 and chol must agree per spin
    with pytest.raises(ValueError, match="inconsistent"):
        HamCholU(h0=jnp.array(0.0), h1_a=jnp.eye(2), h1_b=jnp.eye(2), chol_a=a, chol_b=b)
    # the type survives a pytree round trip
    leaves, treedef = jax.tree_util.tree_flatten(ham_ab)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.norb == (3, 2) and back.nchol == 5


def test_system_uh():
    sys = System_uh(norb=(6, 4), nelec=(3, 2))
    assert sys.norb_a == 6 and sys.norb_b == 4 and sys.ne == 5
    assert System_uh(norb=cast(Any, 5), nelec=(2, 2)).norb == (5, 5)
    with pytest.raises(ValueError, match="unrestricted"):
        System_uh(norb=(4, 4), nelec=(2, 2), walker_kind="restricted")
    with pytest.raises(ValueError, match="does not fit"):
        System_uh(norb=(4, 3), nelec=(2, 4))


# ---------------------------------------------------------------------------
# reduction to the restricted-hamiltonian path on a random hamiltonian
# ---------------------------------------------------------------------------

_NORB, _NUP, _NDN, _NCHOL = 6, 3, 2, 7


@pytest.fixture(scope="module")
def random_pair():
    """A restricted HamChol, its uchol duplicate, a UHF trial and random walkers."""
    key = jax.random.PRNGKey(7)
    k_ham, k_ca, k_cb, k_wa, k_wb = jax.random.split(key, 5)
    ham = make_random_ham_chol(k_ham, norb=_NORB, n_chol=_NCHOL)
    ham_u = from_ham_chol(ham)
    trial = UhfTrial(
        rand_orthonormal_cols(k_ca, _NORB, _NUP, jnp.float64),
        rand_orthonormal_cols(k_cb, _NORB, _NDN, jnp.float64),
    )
    n_walkers = 4
    wa = jax.random.normal(k_wa, (n_walkers, _NORB, _NUP)) + 0.2j * jax.random.normal(
        k_wb, (n_walkers, _NORB, _NUP)
    )
    wb = jax.random.normal(k_wb, (n_walkers, _NORB, _NDN)) + 0.2j * jax.random.normal(
        k_wa, (n_walkers, _NORB, _NDN)
    )
    sys_r = System(norb=_NORB, nelec=(_NUP, _NDN), walker_kind="unrestricted")
    sys_u = System_uh(norb=(_NORB, _NORB), nelec=(_NUP, _NDN))
    return dict(ham=ham, ham_u=ham_u, trial=trial, walkers=(wa, wb), sys_r=sys_r, sys_u=sys_u)


def test_prop_ctx_reduces_to_restricted(random_pair):
    ham, ham_u, trial = random_pair["ham"], random_pair["ham_u"], random_pair["trial"]
    ctx_r = _build_prop_ctx(ham, get_rdm1(trial), 0.01)
    ctx_u = _build_prop_ctx_u(ham_u, get_rdm1_uh(trial), 0.01)
    np.testing.assert_allclose(np.asarray(ctx_u.mf_shifts), np.asarray(ctx_r.mf_shifts), atol=1e-12)
    np.testing.assert_allclose(np.asarray(ctx_u.h0_prop), np.asarray(ctx_r.h0_prop), atol=1e-12)
    for e in (ctx_u.exp_h1_half_a, ctx_u.exp_h1_half_b):
        np.testing.assert_allclose(np.asarray(e), np.asarray(ctx_r.exp_h1_half), atol=1e-12)
    np.testing.assert_allclose(np.asarray(ctx_u.chol_flat_a), np.asarray(ctx_r.chol_flat))
    assert ctx_u.n_fields == ctx_r.chol_flat.shape[0]
    # what afqmc_step reads
    assert ctx_u.chol_flat.shape[0] == _NCHOL and ctx_u.chol_packed is False


def test_kernels_reduce_to_restricted(random_pair):
    ham, ham_u, trial = random_pair["ham"], random_pair["ham_u"], random_pair["trial"]
    wa, wb = random_pair["walkers"]
    ctx_r = build_meas_ctx(ham, trial)
    ctx_u = build_meas_ctx_uh(ham_u, trial)
    for i in range(wa.shape[0]):
        w = (wa[i], wb[i])
        fb_r = force_bias_kernel_uw_rh(w, ham, ctx_r, trial)
        fb_u = force_bias_kernel_uw_uh(w, ham_u, ctx_u, trial)
        np.testing.assert_allclose(np.asarray(fb_u), np.asarray(fb_r), atol=1e-12)
        e_r = energy_kernel_uw_rh(w, ham, ctx_r, trial)
        e_u = energy_kernel_uw_uh(w, ham_u, ctx_u, trial)
        np.testing.assert_allclose(np.asarray(e_u), np.asarray(e_r), atol=1e-10)
        # chunking the cholesky axis does not change the energy (7 vectors in chunks of 3)
        e_c = energy_kernel_uw_uh(w, ham_u, ctx_u, trial, nchol_chunk=3)
        np.testing.assert_allclose(np.asarray(e_c), np.asarray(e_r), atol=1e-10)


def test_step_reduces_to_restricted(random_pair):
    ham, ham_u, trial = random_pair["ham"], random_pair["ham_u"], random_pair["trial"]
    wa, wb = random_pair["walkers"]
    sys_r, sys_u = random_pair["sys_r"], random_pair["sys_u"]
    params = QmcParams(dt=0.01, n_walkers=wa.shape[0], seed=3)

    meas_r = make_uhf_meas_ops(sys_r)
    meas_u = make_uhf_meas_ops_uh(sys_u)
    mctx_r = meas_r.build_meas_ctx(ham, trial)
    mctx_u = meas_u.build_meas_ctx(ham_u, trial)
    pctx_r = _build_prop_ctx(ham, get_rdm1(trial), params.dt)
    pctx_u = _build_prop_ctx_u(ham_u, get_rdm1_uh(trial), params.dt)

    walkers = (wa, wb)
    overlaps = jax.vmap(meas_r.overlap, in_axes=(0, None))(walkers, trial)
    state = PropState(
        walkers=walkers,
        weights=jnp.ones(wa.shape[0]),
        overlaps=overlaps,
        rng_key=jax.random.PRNGKey(11),
        pop_control_ene_shift=jnp.array(-1.0),
        e_estimate=jnp.array(-1.0),
        node_encounters=jnp.asarray(0),
    )
    out_r = afqmc_step(
        state,
        params=params,
        ham_data=ham,
        trial_data=trial,
        meas_ops=meas_r,
        trotter_ops=make_trotter_ops("restricted", "unrestricted"),
        prop_ctx=pctx_r,
        meas_ctx=mctx_r,
    )
    out_u = afqmc_step(
        state,
        params=params,
        ham_data=ham_u,
        trial_data=trial,
        meas_ops=meas_u,
        trotter_ops=cast(Any, make_trotter_ops_u("uchol", "unrestricted")),
        prop_ctx=cast(Any, pctx_u),
        meas_ctx=mctx_u,
    )
    for a, b in zip(out_u.walkers, out_r.walkers):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-11)
    np.testing.assert_allclose(np.asarray(out_u.weights), np.asarray(out_r.weights), atol=1e-11)
    np.testing.assert_allclose(np.asarray(out_u.overlaps), np.asarray(out_r.overlaps), atol=1e-11)


def test_init_walkers_uh_matches_init_walkers(random_pair):
    trial, sys_r, sys_u = random_pair["trial"], random_pair["sys_r"], random_pair["sys_u"]
    w_r = init_walkers(sys_r, get_rdm1(trial), 3)
    w_u = init_walkers_uh(sys_u, get_rdm1_uh(trial), 3)
    for a, b in zip(w_u, w_r):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)
    # and different orbital counts per spin come out with their own shapes
    dm_a = jnp.eye(5)[:, :3] @ jnp.eye(5)[:3, :]
    dm_b = jnp.eye(3)[:, :2] @ jnp.eye(3)[:2, :]
    wa, wb = init_walkers_uh(System_uh(norb=(5, 3), nelec=(3, 2)), (dm_a, dm_b), 2)
    assert wa.shape == (2, 5, 3) and wb.shape == (2, 3, 2)


# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------


def _h2o_rhf():
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
    return mf


def _nh2_uhf():
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
    return mf


@pytest.fixture(scope="module")
def h2o():
    return _h2o_rhf()


@pytest.fixture(scope="module")
def nh2():
    return _nh2_uhf()


_PARAMS: dict[str, Any] = dict(
    n_eql_blocks=4, n_blocks=20, seed=1234, n_walkers=5, error_method="blocking"
)


def _tau0_energy(af) -> float:
    job = _quiet(af.build_job)
    state, _ = _quiet(job.prepare_runtime)
    return float(np.real(state.e_estimate))


def test_closed_shell_to_uhf_reproduces_alpha_basis_run(h2o):
    """
    RHF orbitals converted with to_uhf: the beta basis equals the alpha basis, so the
    unrestricted hamiltonian is the restricted one duplicated and the run must match
    Afqmc with unrestricted walkers, same seed, to roundoff accumulated over the run.
    """
    mf_u = h2o.to_uhf()

    ref = Afqmc(mf_u, chol_cut=1e-6, **_PARAMS)
    ref.walker_kind = "unrestricted"
    ref.mixed_precision = False
    e_ref, err_ref = _quiet(ref.kernel)

    af = AfqmcUh(mf_u, chol_cut=1e-6, **_PARAMS)
    af.mixed_precision = False
    e_u, err_u = _quiet(af.kernel)

    job = af.job
    assert isinstance(job.sys, System_uh) and job.sys.norb == (h2o.mol.nao, h2o.mol.nao)
    assert isinstance(job.ham_data, HamCholU)
    assert e_u == pytest.approx(e_ref, abs=1e-8)
    assert err_u == pytest.approx(err_ref, abs=1e-8)


def test_tau0_energy_is_the_uhf_energy(nh2):
    """The trial energy on the initial walkers is E_UHF up to the cholesky truncation."""
    af = AfqmcUh(nh2, chol_cut=1e-8, **_PARAMS)
    assert _tau0_energy(af) == pytest.approx(nh2.e_tot, abs=1e-6)


@pytest.mark.parametrize("frozen", [1, (1, 1), (1, 0), (0, 1)])
def test_tau0_energy_with_frozen_core(nh2, frozen):
    """
    A frozen core changes h0 and h1 but not the energy of the reference determinant,
    also when only one spin's core is frozen; a wrong core potential shows up here.
    """
    af = AfqmcUh(nh2, norb_frozen_core=frozen, chol_cut=1e-8, **_PARAMS)
    assert _tau0_energy(af) == pytest.approx(nh2.e_tot, abs=1e-6)
    staged = af.stage()
    n_a, n_b = (frozen, frozen) if isinstance(frozen, int) else frozen
    assert staged.ham.norb == (nh2.mol.nao - n_a, nh2.mol.nao - n_b)
    assert staged.ham.nelec == (nh2.nelec[0] - n_a, nh2.nelec[1] - n_b)


def test_frozen_core_reduces_to_restricted(h2o):
    """With identical bases the per spin frozen core gives staging's h0 and h1 exactly."""
    from trot.staging import StagedMfOrCc, _stage_ham_input

    mf_u = h2o.to_uhf()
    ham_r = _stage_ham_input(StagedMfOrCc(mf_u, 1), chol_cut=1e-6, verbose=False)
    ham_u: Any = AfqmcUh(mf_u, norb_frozen_core=1, chol_cut=1e-6).stage().ham
    assert ham_u.h0 == pytest.approx(ham_r.h0, abs=1e-12)
    np.testing.assert_allclose(ham_u.h1_a, ham_r.h1, atol=1e-12)
    np.testing.assert_allclose(ham_u.h1_b, ham_r.h1, atol=1e-12)
    np.testing.assert_allclose(ham_u.chol_a, ham_r.chol, atol=1e-12)


def test_nh2_agrees_with_alpha_basis_run(nh2):
    """
    A genuine UHF: alpha and beta bases differ, so the walk is a different representation
    of the same operators. The energies agree within the statistical error of the short
    runs (the alpha-basis regression value is the one in tests/test_uhf.py).
    """
    e_alpha, err_alpha = -55.43066756011652, 0.00761980459817991
    af = AfqmcUh(nh2, chol_cut=1e-6, **_PARAMS)
    af.mixed_precision = False
    e_u, err_u = _quiet(af.kernel)
    assert np.isfinite(e_u) and np.isfinite(err_u)
    assert abs(e_u - e_alpha) < 3.0 * np.hypot(err_u, err_alpha)


def test_staged_io_round_trip(nh2, tmp_path):
    path = tmp_path / "nh2_uchol.h5"
    af = AfqmcUh(nh2, norb_frozen_core=(1, 1), chol_cut=1e-6, **_PARAMS)
    af.mixed_precision = False
    _quiet(af.save_staged, path)
    e_direct, err_direct = _quiet(af.kernel)

    af2 = AfqmcUh.from_staged(path, **_PARAMS)
    af2.mixed_precision = False
    assert af2.norb_frozen_core == (1, 1)
    staged = af2.stage()
    assert staged.ham.basis == "uchol" and staged.ham.norb == af.stage().ham.norb
    e_loaded, err_loaded = _quiet(af2.kernel)
    assert e_loaded == pytest.approx(e_direct, abs=1e-10)
    assert err_loaded == pytest.approx(err_direct, abs=1e-10)


def test_walker_kind_is_forced(nh2):
    af = AfqmcUh(nh2, chol_cut=1e-6, **_PARAMS)
    af.walker_kind = "restricted"
    with pytest.raises(ValueError, match="unrestricted"):
        _quiet(af.build_job)


if __name__ == "__main__":
    pytest.main([__file__])
