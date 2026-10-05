"""Storage reuse and bounded force-bias workspace, without production runs."""
from dataclasses import replace
from functools import partial

from trot import config
config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from trot.core.ops import k_force_bias
from trot.meas.chol_contract import contract_cholesky
from trot.meas import ptuccsd_thouless, ptuccsd_modes, ucisd, ucisd_k_modes
from trot.sharding import replicate, shard_model_axis
from trot.trial.ptuccsd_modes import make_ptuccsd_thouless_mode_trial_data
from trot.trial.ucisd_k_modes import factorize_ucisd_k_blocks, make_ucisd_k_mode_trial_data
from tests.test_unrestricted_chol_batches import _case


def _cfg(mixed=True, mesh=None):
    return ucisd.UcisdMeasCfg(
        mixed_real_dtype=jnp.float32 if mixed else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed else jnp.complex128,
        chol_mesh=mesh,
    )


@pytest.mark.parametrize('nchol', [0, 1, 64, 67, 128])
@pytest.mark.parametrize('mixed', [False, True])
def test_batched_contraction_matches_complex_dot(nchol, mixed):
    rng = np.random.default_rng(20711)
    chol = jnp.asarray(rng.normal(size=(nchol, 7, 7)))
    matrices = jnp.asarray(rng.normal(size=(3, 7, 7)) + 1j*rng.normal(size=(3, 7, 7)))
    cfg = _cfg(mixed)
    actual = jax.jit(jax.vmap(partial(contract_cholesky, cfg=cfg), in_axes=(None, 0)))(chol, matrices)
    expected = jnp.einsum('gij,wij->wg', chol.astype(cfg.mixed_real_dtype),
                          matrices.astype(cfg.mixed_complex_dtype))
    assert actual.dtype == cfg.mixed_complex_dtype
    np.testing.assert_allclose(actual, expected, atol=8e-6 if mixed else 1e-12, rtol=2e-6 if mixed else 1e-12)


@pytest.mark.parametrize('kind', ['ucisd', 'ptuccsd'])
@pytest.mark.parametrize('walker_kind', ['restricted', 'unrestricted'])
@pytest.mark.parametrize('mixed', [False, True])
def test_dense_force_bias_matches_unbatched_contractions(monkeypatch, kind, walker_kind, mixed):
    sys, ham, trial, walkers = _case(kind, walker_kind, nchol=67)
    module = ucisd if kind == 'ucisd' else ptuccsd_thouless
    factory = module.make_ucisd_meas_ops if kind == 'ucisd' else module.make_ptuccsd_thouless_meas_ops
    ops = factory(sys, mixed_precision=mixed)
    ctx = ops.build_meas_ctx(ham, trial)
    kernel = ops.require_kernel(k_force_bias)
    actual = jax.jit(jax.vmap(kernel, in_axes=(0, None, None, None)))(walkers, ham, ctx, trial)
    def old_contract(chol, matrix, cfg):
        return jnp.einsum('gij,ij->g', chol.astype(cfg.mixed_real_dtype), matrix.astype(cfg.mixed_complex_dtype))
    monkeypatch.setattr(module, 'contract_cholesky', old_contract)
    expected = jax.vmap(kernel, in_axes=(0, None, None, None))(walkers, ham, ctx, trial)
    np.testing.assert_allclose(actual, expected, atol=3e-7 if mixed else 1e-12, rtol=2e-6 if mixed else 1e-12)


def _trial_pair(compressed):
    sys, ham, guide, walkers = _case('ucisd', nchol=67)
    _, _, estimator, _ = _case('ptuccsd', nchol=67)
    # Separate equal arrays, as with independently read guide/estimator caches.
    estimator = replace(estimator, mo_coeff_b=jnp.array(np.asarray(guide.mo_coeff_b)))
    if not compressed:
        return (sys, ham, guide, estimator, walkers,
                ucisd.make_ucisd_meas_ops(sys),
                ptuccsd_thouless.make_ptuccsd_thouless_estimator_ops(sys, memory_mode='low'))
    modes = factorize_ucisd_k_blocks(guide.c2aa, guide.c2ab, guide.c2bb, threshold=0., solver='dense')
    guide_data = dict(mo_coeff_a=guide.mo_coeff_a, mo_coeff_b=guide.mo_coeff_b,
        c1a=guide.c1a, c1b=guide.c1b, eigenvalues=modes.eigenvalues, modes=modes.modes)
    guide = make_ucisd_k_mode_trial_data(guide_data, sys, mixed_precision=True)
    estimator_data = dict(t1a=estimator.mo_t_a[sys.nup:].T,
        t1b=estimator.mo_t_b[sys.ndn:].T, mo_coeff_b=estimator.mo_coeff_b,
        t2aa=estimator.t2aa, t2ab=estimator.t2ab, t2bb=estimator.t2bb, t2_layout='iajb')
    estimator = make_ptuccsd_thouless_mode_trial_data(estimator_data, sys,
        mode_threshold=0., mixed_precision=True)
    return (sys, ham, guide, estimator, walkers,
            ucisd_k_modes.make_ucisd_k_mode_meas_ops(sys),
            ptuccsd_modes.make_ptuccsd_mode_estimator_ops(sys,
                component_sampling=ptuccsd_modes.PtuccsdModePairSamplingCfg(chol_head_size=4, pair_sample_size=8)))


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('same_basis', [False, True])
def test_estimator_reuses_only_matching_beta_basis(monkeypatch, compressed, same_basis):
    _, ham, guide, estimator, walkers, guide_ops, estimator_ops = _trial_pair(compressed)
    if not same_basis:
        estimator = replace(estimator, mo_coeff_b=estimator.mo_coeff_b[:, ::-1])
    guide_ctx = guide_ops.build_meas_ctx(ham, guide)
    original = estimator_ops.build_estimator_ctx(ham, estimator)
    calls = []
    transform = ptuccsd_thouless.transform_cholesky
    def tracked_transform(chol, **kwargs):
        calls.append(kwargs)
        return transform(chol, **kwargs)
    monkeypatch.setattr(ptuccsd_thouless, 'transform_cholesky', tracked_transform)
    shared = estimator_ops.build_estimator_ctx_from_guide(ham, estimator, guide, guide_ctx)
    guide_base = getattr(guide_ctx, 'base', guide_ctx)
    base = getattr(shared, 'base', shared)
    assert (base.chol_b is guide_base.chol_b) == same_basis
    assert sum(c.get('right') is not None for c in calls) == (0 if same_basis else 1)
    assert base.chol_b.dtype == ham.chol.dtype
    kernel = jax.jit(jax.vmap(estimator_ops.components, in_axes=(0, None, None, None)))
    actual = kernel(walkers, ham, shared, estimator)
    expected = kernel(walkers, ham, original, estimator)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    if compressed:
        np.testing.assert_array_equal(shared.chol_head_indices, original.chol_head_indices)
        np.testing.assert_allclose(shared.chol_tail_prob, original.chol_tail_prob)


def test_force_bias_batches_keep_cholesky_shards_local():
    if jax.local_device_count() < 2:
        pytest.skip('Requires two logical CPU devices or GPUs.')
    mesh = Mesh(np.asarray(jax.local_devices()[:2]), ('model',))
    cfg = _cfg(mesh=mesh)
    rng = np.random.default_rng(20712)
    chol = rng.normal(size=(134, 7, 7))
    matrices = rng.normal(size=(3, 7, 7)) + 1j*rng.normal(size=(3, 7, 7))
    args = (shard_model_axis(chol, mesh), replicate(matrices, mesh))
    fn = jax.jit(jax.vmap(partial(contract_cholesky, cfg=cfg), in_axes=(None, 0)))
    compiled = fn.lower(*args).compile()
    assert ' all-gather(' not in compiled.as_text()
    result = compiled(*args)
    assert result.sharding.spec == P(None, 'model')
    expected = np.einsum('gij,wij->wg', chol.astype('float32'), matrices.astype('complex64'))
    np.testing.assert_allclose(result, expected, atol=8e-6, rtol=2e-6)


def test_force_bias_workspace_does_not_include_full_cholesky_copy():
    fn = jax.jit(jax.vmap(partial(contract_cholesky, cfg=_cfg()), in_axes=(None, 0)))
    matrix = jax.ShapeDtypeStruct((3, 32, 32), jnp.complex128)
    workspaces = []
    for nchol in (67, 1027):
        compiled = fn.lower(jax.ShapeDtypeStruct((nchol, 32, 32), jnp.float64), matrix).compile()
        workspaces.append(compiled.memory_analysis().temp_size_in_bytes)
    assert workspaces[1] < 1027*32*32*4 / 2
    assert workspaces[1] <= workspaces[0] + 64*32*32*8


@pytest.mark.parametrize('use_runtime', [False, True])
@pytest.mark.parametrize('supplied_ctx', [False, True])
def test_mixed_driver_reuses_guide_context_at_initial_build(use_runtime, supplied_ctx):
    from tests.test_mixed_estimator import _make_case, _identity_sr
    from trot.driver import run_mixed_estimator_qmc
    from trot.prop.blocks import block_mixed_estimator
    from trot.runtime_layout import QmcRuntime
    sys, params, ham, state, guide_ops, meas_ops, prop_ops, estimator_ops = _make_case()
    guide_data = jnp.array(0.)
    estimator_data = jnp.array(1.)
    guide_ctx = jnp.array(2.)
    estimator_ctx = jnp.array(3.)
    calls = []
    def build(ham_arg, estimator_arg, guide_arg, ctx_arg):
        assert ham_arg is ham and estimator_arg is estimator_data
        assert guide_arg is guide_data and ctx_arg is guide_ctx
        calls.append(True)
        return estimator_ctx
    estimator_ops = replace(estimator_ops, build_estimator_ctx_from_guide=build)
    context_args = dict(ham_data=ham, guide_prop_ctx=jnp.array(0.), guide_meas_ctx=guide_ctx,
                        estimator_ctx=estimator_ctx if supplied_ctx else None)
    if use_runtime:
        context_args = dict(runtime=QmcRuntime(ham, jnp.array(0.), guide_ctx,
                                             estimator_ctx if supplied_ctx else None))
    result = run_mixed_estimator_qmc(sys=sys, params=params, guide_data=guide_data,
        guide_ops=guide_ops, guide_prop_ops=prop_ops, guide_meas_ops=meas_ops,
        estimator_data=estimator_data, estimator_ops=estimator_ops, state=state,
        mixed_block_fn=partial(block_mixed_estimator, sr_fn=_identity_sr), **context_args)
    assert len(calls) == (0 if supplied_ctx else 1)
    np.testing.assert_allclose(result.guide_mean_energy, 5.75)


def test_compressed_force_bias_matches_unbatched_contractions(monkeypatch):
    from trot.meas import ucisd_modes
    _, ham, guide, _, walkers, guide_ops, _ = _trial_pair(True)
    ctx = guide_ops.build_meas_ctx(ham, guide)
    kernel = guide_ops.require_kernel(k_force_bias)
    actual = jax.jit(jax.vmap(kernel, in_axes=(0, None, None, None)))(walkers, ham, ctx, guide)
    def old_contract(chol, matrix, cfg):
        return jnp.einsum('gij,ij->g', chol.astype(cfg.mixed_real_dtype), matrix.astype(cfg.mixed_complex_dtype))
    monkeypatch.setattr(ucisd_modes, 'contract_cholesky', old_contract)
    expected = jax.vmap(kernel, in_axes=(0, None, None, None))(walkers, ham, ctx, guide)
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=2e-6)


def test_shared_context_is_one_compiled_argument():
    from tests.test_mixed_estimator import _make_case
    from trot.driver import _make_run_mixed_estimator_blocks_with_auto_chunks
    from trot.prop.blocks import BlockObs
    sys, params, ham, state, guide_ops, meas_ops, prop_ops, estimator_ops = _make_case()
    shared = jnp.arange(1024., dtype=jnp.float64).reshape(32, 32)
    compiled_args = dict(ham_data=ham, guide_data=jnp.array(0.), guide_meas_ctx=shared,
                        guide_prop_ctx=jnp.array(0.), estimator_data=jnp.array(1.),
                        estimator_ctx=shared)
    # Use a scalar state-dependent multiplier with a compatible row count.
    def block(state, *, guide_meas_ctx, estimator_ctx, **kwargs):
        value = jnp.sum(guide_meas_ctx[:, 0]) * state.weights[0]
        value += jnp.sum(estimator_ctx[0, :]) * state.weights[1]
        return state, BlockObs(scalars={'value': value}, observables={})
    def compile_case(ctx):
        args = dict(compiled_args, estimator_ctx=ctx)
        _, scanner = _make_run_mixed_estimator_blocks_with_auto_chunks(
            mixed_block_fn=block, sys=sys, params=params, guide_ops=guide_ops,
            guide_meas_ops=meas_ops, guide_prop_ops=prop_ops, estimator_ops=estimator_ops,
            state=state, **args)
        executable = scanner.lower(state, n_blocks=2, **args).compile()
        return executable.memory_analysis().argument_size_in_bytes, executable(state, **args)[1]
    shared_bytes, result = compile_case(shared)
    separate_bytes, expected = compile_case(jnp.array(np.asarray(shared)))
    assert separate_bytes - shared_bytes == shared.nbytes
    np.testing.assert_allclose(result['value'], expected['value'])
