"""Numerical and compiler checks for setup on local Cholesky shards."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.cisd import CisdMeasCfg
from trot.meas.cisd_modes import (
    CisdModePairSamplingCfg,
    _build_lci1,
    _build_reference_chol_scores,
    build_meas_ctx,
    energy_kernel_rw_rh,
    initial_energy_kernel_rw_rh,
    make_cisd_mode_meas_ops,
)
from trot.prop.afqmc import init_prop_state
from trot.prop.chol_afqmc_ops import (
    _build_prop_ctx, _mf_shifts, _sum_chol_squares, _weighted_chol_sum,
)
from trot.prop.types import QmcParams
from trot.sharding import cholesky_model_mesh, replicate, shard_model_axis
from trot.trial.cisd_modes import CisdModeTrial, make_cisd_mode_trial_ops

jax.config.update("jax_enable_x64", True)


def _mesh(n_data=1):
    if jax.local_device_count() < 2 * n_data:
        pytest.skip("Requires multiple logical CPU devices or GPUs.")
    devices = np.array(jax.local_devices()[: 2 * n_data]).reshape(n_data, 2)
    return Mesh(devices, ("data", "model"))


def _assert_no_gather(compiled):
    # The regression is full-input replication induced by global-index loops.
    assert " all-gather(" not in compiled.as_text()


@pytest.mark.parametrize("n_chol", [0, 2, 512, 514, 1030])
@pytest.mark.parametrize("complex_chol", [False, True])
def test_sharded_chol_squares_reduce_only_matrix(n_chol, complex_chol):
    mesh = _mesh()
    rng = np.random.default_rng(919)
    chol = rng.normal(size=(n_chol, 4, 4))
    if complex_chol:
        chol = chol + 1j * rng.normal(size=chol.shape)
    sharded = shard_model_axis(chol, mesh)
    compiled = _sum_chol_squares.lower(sharded, mesh=mesh).compile()
    _assert_no_gather(compiled)
    expected = np.einsum("gik,gkj->ij", chol, chol)
    np.testing.assert_allclose(compiled(sharded), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("complex_chol", [False, True])
def test_mean_field_setup_keeps_cholesky_inputs_sharded(complex_chol):
    mesh = _mesh()
    rng = np.random.default_rng(918)
    chol = rng.normal(size=(14, 4, 4))
    if complex_chol:
        chol = chol + 1j * rng.normal(size=chol.shape)
    density = np.diag([2.0, 1.0, 0.0, 0.0])
    sharded = shard_model_axis(chol, mesh)
    ham = HamChol(replicate(0.0, mesh), replicate(np.zeros((4, 4)), mesh), sharded)
    rdm1 = replicate(density, mesh)
    shifts_fn = _mf_shifts.lower(ham, rdm1).compile()
    _assert_no_gather(shifts_fn)
    shifts = shifts_fn(ham, rdm1)
    assert shifts.sharding.spec == P("model")
    expected_shifts = 1j * np.array([np.trace(matrix @ density) for matrix in chol])
    np.testing.assert_allclose(shifts, expected_shifts, rtol=2e-12, atol=2e-12)

    weights = (1j * shifts).real
    potential_fn = _weighted_chol_sum.lower(weights, sharded).compile()
    _assert_no_gather(potential_fn)
    potential = potential_fn(weights, sharded)
    expected = sum((w * matrix for w, matrix in zip((1j * expected_shifts).real, chol)),
                   np.zeros((4, 4)))
    np.testing.assert_allclose(potential, expected, rtol=2e-12, atol=2e-12)


def _inputs(mesh, n_chol, mixed):
    rng = np.random.default_rng(920)
    trial = CisdModeTrial(
        ci1=jnp.asarray(0.02 * rng.normal(size=(2, 3))),
        eigenvalues=jnp.asarray(0.01 * rng.normal(size=6)),
        modes=jnp.asarray(rng.normal(size=(6, 2, 3)), dtype=jnp.float32 if mixed else jnp.float64),
        nocc_t_core=1, nvir_t_outer=2,
    )
    n = trial.norb
    chol = 0.02 * rng.normal(size=(n_chol, n, n))
    chol = (chol + chol.swapaxes(1, 2)) / 2
    h1 = rng.normal(size=(n, n))
    ham = HamChol(jnp.asarray(1.25), jnp.asarray(h1 + h1.T), jnp.asarray(chol))
    walker = jnp.asarray(np.eye(n, trial.nocc_full) + 0.02 * (
        rng.normal(size=(n, trial.nocc_full)) + 1j * rng.normal(size=(n, trial.nocc_full))
    ))
    single = jax.tree.map(lambda x: jax.device_put(x, jax.devices()[0]), (ham, trial, walker))
    ham_s = HamChol(replicate(np.asarray(ham.h0), mesh), replicate(np.asarray(ham.h1), mesh),
                    shard_model_axis(chol, mesh))
    trial_s, walker_s = jax.tree.map(lambda x: replicate(np.asarray(x), mesh), (trial, walker))
    cfg = CisdMeasCfg(
        memory_mode="high", mixed_real_dtype=jnp.float32 if mixed else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed else jnp.complex128,
    )
    return single, (ham_s, trial_s, walker_s), cfg


@pytest.mark.parametrize("n_chol", [2, 512, 514, 1030])
@pytest.mark.parametrize("mixed", [False, True])
def test_cisd_setup_stays_sharded_and_matches_single_device(n_chol, mixed):
    mesh = _mesh()
    (ham, trial, walker), (ham_s, trial_s, walker_s), cfg = _inputs(mesh, n_chol, mixed)
    ctx = build_meas_ctx(ham, trial, cfg=cfg, n_mode_chunks=2)
    ctx_s = build_meas_ctx(ham_s, trial_s, cfg=cfg, n_mode_chunks=2)
    assert ctx.setup_mesh is None
    assert ctx_s.setup_mesh == mesh
    assert ctx_s.rot_chol.sharding.spec == P("model")
    assert ctx_s.lci1.sharding.spec == P("model")
    np.testing.assert_allclose(ctx_s.lci1, ctx.lci1, rtol=2e-12, atol=2e-12)
    _assert_no_gather(_build_lci1.lower(
        ham_s.chol, trial_s.ci1, vir_start=trial.vir_act_slice.start,
        vir_stop=trial.vir_act_slice.stop, mesh=mesh,
    ).compile())

    compiled = jax.jit(initial_energy_kernel_rw_rh).lower(walker_s, ham_s, ctx_s, trial_s).compile()
    _assert_no_gather(compiled)
    tolerance = 2e-6 if mixed else 2e-12
    expected = jax.jit(energy_kernel_rw_rh)(walker, ham, ctx, trial)
    np.testing.assert_allclose(compiled(walker_s, ham_s, ctx_s, trial_s), expected,
                               rtol=tolerance, atol=tolerance)

    scores_fn = _build_reference_chol_scores.lower(ham_s, ctx_s, trial_s, chol_batch_size=4).compile()
    _assert_no_gather(scores_fn)
    scores = scores_fn(ham_s, ctx_s, trial_s)
    expected_scores = _build_reference_chol_scores(ham, ctx, trial, chol_batch_size=4)
    assert scores.sharding.spec == P("model")
    np.testing.assert_allclose(scores, expected_scores, rtol=tolerance, atol=tolerance)


def test_setup_and_initialization_with_data_model_mesh_and_padding():
    mesh = _mesh(n_data=2)
    (ham, trial, walker), (ham_s, trial_s, walker_s), _ = _inputs(mesh, 515, False)
    assert ham_s.chol.shape[0] == 516
    assert cholesky_model_mesh(ham_s.chol) == mesh
    sys = System(norb=trial.norb, nelec=(trial.nocc_full,) * 2, walker_kind="restricted")
    trial_ops = make_cisd_mode_trial_ops(sys)
    meas_ops = make_cisd_mode_meas_ops(
        sys, mixed_precision=False, n_mode_chunks=2,
        energy_sampling=CisdModePairSamplingCfg(
            chol_head_size=2, pair_sample_size=8, guide_chol_batch_size=4,
        ),
    )
    ctx = meas_ops.build_meas_ctx(ham, trial)
    ctx_s = meas_ops.build_meas_ctx(ham_s, trial_s)
    prop = _build_prop_ctx(ham, trial_ops.get_rdm1(trial), 0.005, packed_cholesky=True)
    prop_s = _build_prop_ctx(ham_s, trial_ops.get_rdm1(trial_s), 0.005, packed_cholesky=True)
    np.testing.assert_allclose(prop_s.exp_h1_half, prop.exp_h1_half, rtol=2e-12, atol=2e-12)
    assert prop_s.chol_flat.sharding.spec == P("model")
    state = init_prop_state(
        sys=sys, ham_data=ham_s, trial_data=trial_s, trial_ops=trial_ops,
        meas_ops=meas_ops, meas_ctx=ctx_s, params=QmcParams(n_walkers=4, n_chunks=2),
        initial_walkers=jnp.stack([walker_s] * 4), mesh=mesh,
    )
    expected = jnp.real(jax.jit(energy_kernel_rw_rh)(walker, ham, ctx, trial))
    np.testing.assert_allclose(state.e_estimate, expected, rtol=2e-12, atol=2e-12)
    assert state.walkers.sharding.spec == P("data")
    np.testing.assert_allclose(ctx_s.reference_chol_scores[:-1], ctx.reference_chol_scores,
                               rtol=2e-12, atol=2e-12)
    assert float(ctx_s.reference_chol_scores[-1]) == 1e-300
