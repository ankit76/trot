"""Compact restricted FNO PT trials versus full-space padded references."""

from dataclasses import replace
from types import SimpleNamespace

from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas import ptccsd_modes as meas
from trot.meas.ptccsd_thouless import (
    build_ptccsd_thouless_meas_ctx,
    energy_pt_rw_rh as dense_energy,
    force_bias_pt_rw_rh as dense_force_bias,
)
from trot.meas.pair_sampling import with_local_cholesky_sampling
from trot.sharding import (
    plan_cholesky_layout,
    replicate,
    shard_cholesky_layout,
    shard_first_axis,
)
from trot.trial.ptccsd_modes import (
    PtccsdThoulessModeTrial,
    decompose_t2_modes,
    get_rdm1_thouless,
    make_ptccsd_thouless_mode_trial_data,
    overlap_ptccsd_thouless_r,
)
from trot.trial.ptccsd_thouless import (
    PtccsdThoulessTrial,
    overlap_ptccsd_thouless_r as dense_overlap,
)


@pytest.fixture(scope="module")
def fno_case():
    rng = np.random.default_rng(20261020)
    no, nv, outer, nx = 2, 2, 3, 5
    norb = no + nv + outer
    t1 = 0.08 * rng.normal(size=(no, nv))
    t2 = 0.04 * rng.normal(size=(no, nv, no, nv))
    t2 = (t2 + t2.transpose(2, 3, 0, 1)) / 2
    values, modes = decompose_t2_modes(t2)
    compact = make_ptccsd_thouless_mode_trial_data(
        dict(t1=t1, eigenvalues=values, modes=modes, nvir_t_outer=outer),
        mixed_precision=False,
    )
    padded = PtccsdThoulessModeTrial(
        compact.mo_t, compact.eigenvalues, jnp.pad(compact.modes, ((0, 0), (0, 0), (0, outer)))
    )
    dense = PtccsdThoulessTrial(
        compact.mo_t,
        jnp.asarray(np.pad(t2, ((0, 0), (0, outer), (0, 0), (0, outer)))),
    )
    h1 = rng.normal(size=(norb, norb))
    chol = rng.normal(size=(nx, norb, norb))
    ham = HamChol(
        jnp.asarray(0.2), jnp.asarray((h1 + h1.T) / 2),
        jnp.asarray((chol + chol.transpose(0, 2, 1)) / 2),
    )
    walkers = jnp.asarray(
        np.eye(norb, no)[None]
        + 0.1 * rng.normal(size=(4, norb, no))
        + 0.07j * rng.normal(size=(4, norb, no))
    )
    return SimpleNamespace(
        compact=compact, padded=padded, dense=dense, ham=ham, walkers=walkers,
        sys=System(norb=norb, nelec=(no, no), walker_kind="restricted"),
        t1=t1, t2=t2,
    )


@pytest.mark.parametrize("memory_mode", ["high", "low"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("complex_gauge", [False, True])
@pytest.mark.parametrize("outer_reference", [False, True])
def test_compact_energy_overlap_and_force_bias_match_padded_oracles(
    fno_case, memory_mode, mixed, complex_gauge, outer_reference, monkeypatch,
):
    case = fno_case
    mo = case.compact.mo_t
    if outer_reference:
        # Compact doubles do not require the reference to vanish outside FNOs.
        mo = mo.astype(jnp.complex128)
        mo = mo.at[case.compact.nocc + case.compact.nvir :, :].set(0.02 + 0.01j)
    if complex_gauge:
        mo = mo @ jnp.asarray([[1.1 + 0.2j, -0.1 + 0.05j],
                               [0.08 - 0.04j, 0.9 - 0.15j]])
    dtype = jnp.float32 if mixed else jnp.float64
    compact = replace(case.compact, mo_t=mo, modes=case.compact.modes.astype(dtype))
    padded = replace(case.padded, mo_t=mo, modes=case.padded.modes.astype(dtype))
    dense = replace(case.dense, mo_t=mo)
    ops = meas.make_ptccsd_thouless_mode_meas_ops(
        case.sys, memory_mode=memory_mode, mixed_precision=mixed, n_mode_chunks=3,
    )
    ctx = ops.build_meas_ctx(case.ham, compact)
    dense_ctx = build_ptccsd_thouless_meas_ctx(case.ham, dense)
    evaluate = lambda fn, trial, context: jax.jit(
        jax.vmap(fn, in_axes=(0, None, None, None))
    )(case.walkers, case.ham, context, trial)
    tolerance = 2e-5 if mixed else 3e-11
    gl_name = "_energy_gl_scalar" if memory_mode == "low" else "_energy_gl_batched"
    gl = getattr(meas, gl_name)

    def compact_gl(green, chol, cfg):
        # Resource regression: do not compute the excluded output columns.
        assert chol.shape[-2:] == (compact.nvir, compact.norb)
        return gl(green, chol, cfg)

    with monkeypatch.context() as patch:
        patch.setattr(meas, gl_name, compact_gl)
        actual = evaluate(meas.components_pt_thouless_rw_rh, compact, ctx)
    # The full-Green oracle still constructs the padded orbital-square correction.
    expected = evaluate(meas._components_pt_thouless_rw_rh_full, padded, ctx)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    actual_energy = case.ham.h0 + actual[:, 1] + actual[:, 2] - actual[:, 0] * actual[:, 1]
    np.testing.assert_allclose(
        actual_energy, evaluate(dense_energy, dense, dense_ctx), rtol=tolerance, atol=tolerance,
    )
    np.testing.assert_allclose(
        evaluate(meas.force_bias_pt_thouless_rw_rh, compact, ctx),
        evaluate(dense_force_bias, dense, dense_ctx), rtol=tolerance, atol=tolerance,
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(overlap_ptccsd_thouless_r, in_axes=(0, None)))(case.walkers, compact),
        jax.vmap(dense_overlap, in_axes=(0, None))(case.walkers, dense),
        rtol=tolerance, atol=tolerance,
    )


@pytest.mark.parametrize("source", ["t1", "full_t1", "mo_t", "raw_t2", "legacy"])
def test_loader_preserves_full_space_and_compact_modes(fno_case, source):
    case = fno_case
    data = dict(eigenvalues=case.compact.eigenvalues, modes=case.compact.modes)
    if source in ("t1", "raw_t2"):
        data.update(t1=case.t1, nvir_t_outer=3)
    elif source == "full_t1":
        data["t1"] = np.pad(case.t1, ((0, 0), (0, 3)))
    elif source == "mo_t":
        data["mo_t"] = case.compact.mo_t
    else:
        data.update(t1=np.pad(case.t1, ((0, 0), (0, 3))), modes=case.padded.modes)
    if source == "raw_t2":
        data.pop("eigenvalues")
        data.pop("modes")
        data["t2"] = case.t2
    trial = make_ptccsd_thouless_mode_trial_data(data, mixed_precision=False)
    assert trial.norb == case.sys.norb
    assert trial.nvir_t_outer == (0 if source == "legacy" else 3)
    assert trial.modes.shape[-1] == (5 if source == "legacy" else 2)
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.nvir_t_outer == trial.nvir_t_outer
    np.testing.assert_allclose(get_rdm1_thouless(rebuilt), get_rdm1_thouless(case.padded))


def test_loader_rejects_incompatible_outer_space(fno_case):
    with pytest.raises(ValueError, match="nonnegative"):
        replace(fno_case.compact, nvir_t_outer=-1)
    with pytest.raises(ValueError, match="mo_t must have shape"):
        replace(fno_case.compact, nvir_t_outer=2)
    with pytest.raises(ValueError, match="retained or full virtual space"):
        make_ptccsd_thouless_mode_trial_data(dict(
            t1=np.zeros((2, 3)), eigenvalues=fno_case.compact.eigenvalues,
            modes=fno_case.compact.modes, nvir_t_outer=3,
        ))


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_compact_sampled_head_and_tail_match_padded(fno_case, memory_mode):
    case = fno_case
    cfg = meas.PtccsdModePairSamplingCfg(
        chol_head_size=2, pair_sample_size=32, head_chol_batch_size=2,
        track_half_sample_diagnostic=True,
    )
    ops = meas.make_ptccsd_thouless_mode_estimator_ops(
        case.sys, component_sampling=cfg, memory_mode=memory_mode, mixed_precision=False,
    )
    weights = jnp.asarray([1.0 + 0.2j, 0.7 - 0.1j, 1.3 + 0.4j, 0.9 - 0.3j])
    evaluate = jax.jit(meas.pair_sampled_ptccsd_block_components, static_argnums=3)
    results = []
    for trial in (case.compact, case.padded):
        ctx = ops.build_estimator_ctx(case.ham, trial)
        # Fix the proposal so the same key samples identical pairs.
        ctx = meas.configure_ptccsd_mode_pair_sampling(
            ctx, sampling=cfg, guide_scores=jnp.arange(1., 6.),
        )
        results.append(evaluate(
            case.walkers, weights, jax.random.PRNGKey(410), 2, case.ham, ctx, trial,
        ))
    for compact, padded in zip(*map(jax.tree_util.tree_leaves, results)):
        np.testing.assert_allclose(compact, padded, rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize("n_data,n_model", [(1, 4), (2, 2)])
def test_compact_local_full_head_matches_padded_oracle(fno_case, n_data, n_model):
    if jax.local_device_count() < n_data * n_model:
        pytest.skip("Requires four logical CPU devices or GPUs.")
    from jax.sharding import Mesh

    case = fno_case
    mesh = Mesh(np.array(jax.local_devices()[:4]).reshape(n_data, n_model), ("data", "model"))
    nx = case.ham.chol.shape[0]
    layout = plan_cholesky_layout(nx, n_model, np.arange(nx), [], [])
    ham = HamChol(
        replicate(case.ham.h0, mesh), replicate(case.ham.h1, mesh),
        shard_cholesky_layout(np.asarray(case.ham.chol), mesh, layout),
    )
    trial = jax.tree.map(lambda a: replicate(a, mesh), case.compact)
    cfg = meas.PtccsdModePairSamplingCfg(nx, 16, head_chol_batch_size=2)
    ctx = meas.build_ptccsd_thouless_mode_meas_ctx(ham, trial)
    ctx = replace(ctx, component_sampling=cfg)
    ctx = with_local_cholesky_sampling(ctx, layout, mesh)
    weights = jnp.asarray([1.0 + 0.2j, 0.7 - 0.1j, 1.3 + 0.4j, 0.9 - 0.3j])
    result = jax.jit(meas.pair_sampled_ptccsd_block_components, static_argnums=3)(
        shard_first_axis(case.walkers, mesh), shard_first_axis(weights, mesh),
        jax.random.PRNGKey(410), 2, ham, ctx, trial,
    )
    oracle_ctx = meas.build_ptccsd_thouless_mode_meas_ctx(case.ham, case.padded)
    components = jax.vmap(
        meas._components_pt_thouless_rw_rh_full, in_axes=(0, None, None, None),
    )(case.walkers, case.ham, oracle_ctx, case.padded)
    np.testing.assert_allclose(result.weight, weights.sum(), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        result.numerator, (weights[:, None] * components).sum(axis=0), rtol=3e-11, atol=3e-11,
    )


def test_c60_common_storage_uses_fno_columns_and_occupied_rank():
    # Abstract shapes only: do not allocate the C60 arrays.
    nw, no, nv, norb, rank, nx = 400, 120, 62, 1740, 873, 4860
    shape = jax.ShapeDtypeStruct
    trial = PtccsdThoulessModeTrial(
        shape((norb, no), jnp.float64), shape((rank,), jnp.float64),
        shape((rank, no, nv), jnp.float32), nvir_t_outer=norb - no - nv,
    )
    ham = HamChol(
        shape((), jnp.float64), shape((norb, norb), jnp.float64),
        shape((nx, norb, norb), jnp.float64),
    )
    common = jax.eval_shape(
        jax.vmap(meas._ptccsd_thouless_mode_energy_common, in_axes=(0, None, None, None)),
        shape((nw, norb, no), jnp.complex128), ham, None, trial,
    )
    assert common.half_green.shape == (nw, no, norb)
    assert common.greenp_coeff.shape == (nw, no + nv, nv)
    assert common.t2_left_coeff.shape == (nw, no + nv, no)
    stored_bytes = sum(a.size * a.dtype.itemsize for a in jax.tree.leaves(common))
    previous_bytes = nw * (no * norb + norb * nv + norb * no + 3) * 16
    assert stored_bytes < previous_bytes / 2
