"""Synthetic local PT strata, abstract tracing, and GPU-only kernel checks."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from trot import walkers as wk
from trot.core.ops import d_pt_component_sampling_noise_real, d_pt_component_sampling_noise_imag
from trot.ham.chol import HamChol
from trot.meas import ptccsd_modes as rm, ptuccsd_modes as um
from trot.meas.pair_sampling import (
    local_cholesky_component_head, local_cholesky_tail, with_local_cholesky_sampling,
)
from trot.meas.pt2ccsd import project_first_order_energy_terms
from trot.sharding import plan_cholesky_layout, replicate, shard_cholesky_layout, shard_first_axis
from trot.trial.ptccsd_modes import PtccsdThoulessModeTrial
from trot.trial.ptuccsd_modes import PtuccsdThoulessModeTrial
from tests.test_cholesky_pair_sampling import assert_small_collectives, model_mesh

jax.config.update("jax_enable_x64", True)


@partial(jax.tree_util.register_dataclass,
         data_fields=["chol_head_indices", "chol_tail_indices", "chol_tail_prob", "model_sampling"],
         meta_fields=["component_sampling"])
@dataclass(frozen=True)
class TableContext:
    chol_head_indices: object
    chol_tail_indices: object
    chol_tail_prob: object
    component_sampling: object
    model_sampling: object = None


class TableCommon(NamedTuple):
    theta: object
    walker_indices: object


def table_specs(ctx):
    return jax.tree.map(lambda _: P(), ctx)


def table_head(common, indices, h, ctx, trial, *, n_chunks):
    return h.chol[indices][:, common.walker_indices].transpose(1, 0, 2)


def table_pairs(common, iw, ig, h, ctx, trial, *, n_chunks):
    return h.chol[ig, common.walker_indices[iw]]


def table_inputs(n, head, n_data=1, n_model=4):
    mesh = model_mesh(n_model, n_data)
    rng = np.random.default_rng(371)
    table = rng.normal(size=(n, 8, 2)) + 1j * rng.normal(size=(n, 8, 2))
    tail = np.setdiff1d(np.arange(n), head)
    p = np.arange(1, len(tail)+1, dtype=float)
    p = p/p.sum() if p.size else p
    layout = plan_cholesky_layout(n, n_model, head, tail, p)
    sampling = rm.PtccsdModePairSamplingCfg(len(head), 17, head_chol_batch_size=2,
        walker_guide_policy="head_rms", track_half_sample_diagnostic=True)
    ctx = with_local_cholesky_sampling(TableContext(None, None, None, sampling), layout, mesh)
    zero = replicate(np.asarray(0.), mesh)
    h = HamChol(zero, replicate(np.zeros((8, 8)), mesh), shard_cholesky_layout(table, mesh, layout))
    common = TableCommon(shard_first_axis(np.zeros(8, dtype=complex), mesh),
                         shard_first_axis(np.arange(8), mesh))
    return table, layout, common, h, ctx, zero


@pytest.mark.parametrize("n,head", [(9, [0, 4, 7]), (3, [1, 2]), (7, []), (7, list(range(7)))])
@pytest.mark.parametrize("n_data,n_model", [(1, 4), (2, 2), (4, 1)])
def test_component_head_moments_and_small_collectives(n, head, n_data, n_model):
    table, layout, common, h, ctx, t = table_inputs(n, head, n_data, n_model)
    theta = replicate(np.asarray(.3+.7j), ctx.model_sampling.mesh)
    f = jax.jit(lambda common, theta, h, c, t: local_cholesky_component_head(
        common, theta, h, c, t, terms_fn=table_head, project_fn=project_first_order_energy_terms,
        context_specs_fn=table_specs, n_chunks=3))
    compiled = f.lower(common, theta, h, ctx, t).compile()
    actual = compiled(common, theta, h, ctx, t)
    projected = table[head, :, 1] + (1-np.asarray(theta))*table[head, :, 0]
    expected = (table[head].sum(axis=0), (projected.real**2).sum(axis=0),
                (projected.imag**2).sum(axis=0), (projected.real*projected.imag).sum(axis=0))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=2e-13)
    assert_small_collectives(compiled, 40)


@pytest.mark.parametrize("n,head,weights_kind", [(9, [0, 4, 7], "complex"),
    (3, [1, 2], "complex"), (7, [], "cancel"), (7, [0], "zero")])
@pytest.mark.parametrize("n_data,n_model", [(1, 4), (2, 2), (4, 1)])
def test_complex_tail_mean_covariance_and_half_sample_noise(n, head, weights_kind, n_data, n_model):
    table, layout, common, h, ctx, t = table_inputs(n, head, n_data, n_model)
    pi = np.array([1+.2j, -.3+.4j, .7-1j, .8+.6j, -.2-.7j, 2+.1j, 0, 1j])
    if weights_kind == "cancel":
        pi = np.array([1, -1, 1j, -1j, 2, -2, 2j, -2j])
    q = np.arange(1, 9, dtype=float); q /= q.sum()
    if weights_kind == "zero":
        pi[:] = 0
    if n == 3:
        pi[:4] = 0  # Empty walker strata, alongside empty Cholesky strata.
        q[:4] = 0
        q /= q.sum()
    args = (common, shard_first_axis(pi, ctx.model_sampling.mesh), shard_first_axis(q, ctx.model_sampling.mesh),
            jax.random.PRNGKey(92), h, ctx, t)
    f = jax.jit(lambda common, pi, q, key, h, c, t: jnp.stack(local_cholesky_tail(
        common, pi, q, key, h, c, t, pair_fn=table_pairs,
        context_specs_fn=table_specs, n_chunks=3, components=True)))
    compiled = f.lower(*args).compile()
    assert_small_collectives(compiled, 8)

    def draw_many(args, keys):
        return jax.lax.scan(lambda _, key: (None, f(*args[:3], key, *args[4:])), None, keys)[1]

    samples = np.asarray(jax.jit(draw_many)(args, jax.random.split(jax.random.PRNGKey(38), 8192)))
    exact = np.zeros(2, complex)
    covariance = np.zeros((4, 4))
    for data_id, walkers in enumerate(np.split(np.arange(8), n_data)):
        walkers = walkers[q[walkers] > 0]
        if walkers.size == 0:
            continue
        qd = q[walkers]/q[walkers].sum()
        for model_id, pd in enumerate(layout.local_tail_prob):
            if pd.sum() == 0:
                continue
            d = data_id*n_model+model_id
            ids = layout.permutation.reshape(n_model, -1)[model_id, pd > 0]
            prob = (pd[pd > 0]/pd.sum())[:, None] * qd[None, :]
            values = pi[None, walkers, None]*table[ids][:, walkers] / prob[:, :, None]
            mean = (prob[:, :, None]*values).sum(axis=(0, 1))
            exact += mean
            real_values = np.concatenate((values.real, values.imag), axis=-1).reshape(-1, 4)
            real_mean = np.r_[mean.real, mean.imag]
            centered = real_values - real_mean
            covariance += (centered.T * prob.ravel()) @ centered / (5 if d == 0 else 4)
    values = np.concatenate((samples.real, samples.imag), axis=-1)
    if weights_kind == "zero":
        np.testing.assert_array_equal(samples, 0)
        return
    errors = np.sqrt(np.diag(covariance)/len(samples))
    assert np.all(abs(values[:, 0].mean(axis=0)-np.r_[exact.real, exact.imag]) < 6*errors)
    assert np.all(abs(values[:, 1].mean(axis=0)) < 6*errors)
    for stream in (0, 1):
        observed = np.cov(values[:, stream], rowvar=False)
        tolerance = .08*np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        assert np.all(abs(observed-covariance) < tolerance)


def host_inputs(kind, mixed):
    rng = np.random.default_rng(943)
    n, noa, nob = 5, 2, (2 if kind == "rcc" else 1)
    t1a, t1b = .06*rng.normal(size=(noa, n-noa)), .06*rng.normal(size=(nob, n-nob))
    moa, mob = np.vstack((np.eye(noa), t1a.T)), np.vstack((np.eye(nob), t1b.T))
    if kind == "rcc":
        raw = .02*rng.normal(size=(6, 6)); raw = (raw+raw.T)/2
        t2 = raw.reshape(2, 3, 2, 3)
        kernel = (2*t2-t2.transpose(0, 3, 2, 1)).reshape(6, 6)
    else:
        def same_spin(no, nv):
            a = .02*rng.normal(size=(no, nv, no, nv))
            return (a-a.transpose(2, 1, 0, 3)-a.transpose(0, 3, 2, 1)+a.transpose(2, 3, 0, 1))
        ab = .02*rng.normal(size=(6, 4))
        kernel = np.block([[same_spin(2, 3).reshape(6, 6), ab],
                           [ab.T, same_spin(1, 4).reshape(4, 4)]])
    vals, vecs = np.linalg.eigh(kernel)
    keep = np.argsort(abs(vals))[-5:]
    modes = vecs[:, keep].T.astype(np.float32 if mixed else np.float64)
    if kind == "rcc":
        trial = PtccsdThoulessModeTrial(moa, vals[keep], modes.reshape(5, 2, 3))
    else:
        rotation, _ = np.linalg.qr(np.eye(n)+.12*rng.normal(size=(n, n)))
        trial = PtuccsdThoulessModeTrial(moa, mob, rotation, vals[keep], modes)
    chol = .05*rng.normal(size=(7, n, n)); chol = (chol+chol.transpose(0, 2, 1))/2
    h = HamChol(np.asarray(.2), np.diag(np.linspace(-1., 1., n)), chol)
    walkers = np.eye(n, noa)[None]+.04*(rng.normal(size=(12, n, noa))+1j*rng.normal(size=(12, n, noa)))
    weights = np.arange(1, 13)*np.exp(1j*np.linspace(-.7, .9, 12)); weights[3] = 0
    return h, trial, walkers, weights


def test_complex_joint_strata_with_one_pair_each_and_no_diagnostic():
    table, _, common, h, ctx, t = table_inputs(2, [], n_data=2, n_model=2)
    ctx = replace(ctx, component_sampling=replace(ctx.component_sampling,
        pair_sample_size=4, track_half_sample_diagnostic=False))
    mesh = ctx.model_sampling.mesh
    pi = np.zeros(8, complex); pi[[1, 5]] = [.2+.7j, -.8+.2j]
    q = np.zeros(8); q[[1, 5]] = [.07, .93]
    f = jax.jit(lambda common, pi, q, h, ctx, t: local_cholesky_tail(
        common, pi, q, jax.random.PRNGKey(4), h, ctx, t,
        pair_fn=table_pairs, context_specs_fn=table_specs, n_chunks=3, components=True))
    estimate, noise = f(common, shard_first_axis(pi, mesh), shard_first_axis(q, mesh), h, ctx, t)
    np.testing.assert_allclose(estimate, (table.sum(axis=0)*pi[:, None]).sum(axis=0), atol=1e-14)
    np.testing.assert_array_equal(noise, 0)


def prepare(kind, mixed, n_model, head_size, *, memory_mode="high", n_data=1):
    mesh = model_mesh(n_model, n_data)
    module = rm if kind == "rcc" else um
    h, t, w, weights = host_inputs(kind, mixed)
    head = np.array([1, 4, 0, 6, 3, 5, 2])[:head_size]
    tail = np.setdiff1d(np.arange(7), head)
    p = np.arange(1, tail.size+1, dtype=float); p = p/p.sum() if p.size else p
    layout = plan_cholesky_layout(7, n_model, head, tail, p)
    h = HamChol(replicate(h.h0, mesh), replicate(h.h1, mesh), shard_cholesky_layout(h.chol, mesh, layout))
    t = jax.tree.map(lambda a: replicate(a, mesh), t)
    w, weights = jax.tree.map(lambda a: shard_first_axis(a, mesh), (w, weights))
    cls = rm.PtccsdModeMeasCfg if kind == "rcc" else um.PtuccsdModeMeasCfg
    cfg = cls(memory_mode=memory_mode, mixed_real_dtype=jnp.float32 if mixed else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed else jnp.complex128)
    build = rm.build_ptccsd_thouless_mode_meas_ctx if kind == "rcc" else um.build_ptuccsd_mode_meas_ctx
    context_fn = lambda h, t: build(h, t, cfg=cfg, n_mode_chunks=3,
                                  **({"memory_mode": memory_mode} if kind == "rcc" else {}))
    # Setup has top-level-only compiler options. Build these tiny contexts
    # before abstract-tracing the production kernel below.
    ctx = context_fn(h, t)
    sampling_cls = rm.PtccsdModePairSamplingCfg if kind == "rcc" else um.PtuccsdModePairSamplingCfg
    ctx = replace(ctx, component_sampling=sampling_cls(head_size, 17, head_chol_batch_size=2,
        walker_guide_policy="head_rms", track_half_sample_diagnostic=True))
    ctx = with_local_cholesky_sampling(ctx, layout, mesh)
    return module, mesh, layout, h, ctx, t, w, weights


def functions(kind):
    if kind == "rcc":
        return rm._ptccsd_thouless_mode_energy_common, rm.pair_sampled_ptccsd_block_components
    return um._ptuccsd_mode_energy_common_rw_rh, um.pair_sampled_ptuccsd_block_components


@pytest.mark.parametrize("kind", ["rcc", "ucc"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("n_data,n_model", [(1, 1), (1, 4), (2, 2)])
@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_abstract_pt_local_trace_and_frozen_guards(kind, mixed, n_data, n_model, memory_mode):
    module, mesh, layout, h, ctx, t, w, weights = prepare(
        kind, mixed, n_model, 3, memory_mode=memory_mode, n_data=n_data)
    _, block = functions(kind)
    for chunks in (1, 5):
        result = jax.eval_shape(lambda w, wt, key, h, c, t: block(w, wt, key, chunks, h, c, t),
                                w, weights, np.array([0, 82], dtype=np.uint32), h, ctx, t)
        assert result.numerator.shape == (3,) and result.numerator.dtype == np.complex128
        assert result.weight.dtype == np.complex128
    configure = module.configure_ptccsd_mode_pair_sampling if kind == "rcc" else module.configure_ptuccsd_mode_pair_sampling
    with pytest.raises(ValueError, match="frozen guide"):
        configure(ctx, ctx.component_sampling, np.ones(h.chol.shape[0]))
    retune = module.retune_ptccsd_mode_pair_sampling if kind == "rcc" else module.retune_ptuccsd_mode_pair_sampling
    with pytest.raises(ValueError, match="no automatic retuning"):
        retune(None, None, None, None, h, ctx, t, guide_data=None, guide_meas_ops=None,
               guide_meas_ctx=None, advance_blocks=None, tuning_cfg=None)


@pytest.mark.parametrize("kind", ["rcc", "ucc"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("n_data,n_model,head_size", [
    (1, 1, 3), (1, 4, 3), (1, 4, 0), (1, 4, 7), (1, 4, 6),
    (2, 2, 3), (2, 2, 0), (2, 2, 7), (2, 2, 6)])
def test_gpu_pt_local_matches_explicit_complex_component_sum(kind, mixed, n_data, n_model, head_size):
    if jax.default_backend() != "gpu":
        pytest.skip("Numerical AFQMC kernels require GPUs")
    module, mesh, layout, h, ctx, t, w, weights = prepare(kind, mixed, n_model, head_size, n_data=n_data)
    common_fn, block = functions(kind)
    common = jax.jit(wk.vmap_chunked(common_fn, 1, in_axes=(0, None, None, None)))(w, h, ctx, t)
    terms = np.asarray(jax.jit(lambda common, h, c, t: module._local_cholesky_head_terms(
        common, jnp.arange(h.chol.shape[0]), h, c, t, n_chunks=1))(common, h, ctx, t))
    head = terms[:, layout.head_indices]
    common_host = jax.tree.map(np.asarray, common)
    exact_components = np.stack((common_host.theta,
        common_host.electronic_0_base+head[:, :, 0].sum(axis=1),
        common_host.h_t_base+head[:, :, 1].sum(axis=1)), axis=1)
    key = jax.random.PRNGKey(82)
    tol = 1e-5 if mixed else 2e-12
    functions_by_chunks = {chunks: jax.jit(lambda w, wt, key, h, c, t, chunks=chunks:
        block(w, wt, key, chunks, h, c, t)) for chunks in (1, 5)}
    for policy in ("head_rms", "abs_weight"):
        c = replace(ctx, component_sampling=replace(ctx.component_sampling, walker_guide_policy=policy))
        weight_sets = [np.asarray(weights), np.zeros(12, complex), np.tile([1., -1., 1j, -1j], 3)]
        for wt in weight_sets:
            den = wt.sum(); safe_den = den if den != 0 else 1.
            theta_ref = np.sum(wt*common_host.theta)/safe_den if den != 0 else 0.
            effective = head[:, :, 1]+(1-theta_ref)*head[:, :, 0]
            abs_total = abs(wt).sum()
            q0 = abs(wt)/abs_total if abs_total else np.full(12, 1/12)
            projected = (wt/safe_den)[:, None]*effective
            scores = np.sqrt((projected.real**2).sum(axis=1)) if den != 0 else np.zeros(12)
            q = .1*q0+.9*(scores/scores.sum() if scores.sum() else q0) if policy == "head_rms" else q0
            expected = np.sum(wt[:, None]*exact_components, axis=0)
            noise = np.zeros(2, complex)
            if abs_total and layout.tail_indices.size:
                count, rem = divmod(17, n_data*n_model); capacity = count+bool(rem)
                width = layout.local_tail_prob.shape[1]
                for data_id, walkers in enumerate(np.split(np.arange(12), n_data)):
                    if q[walkers].sum() == 0:
                        continue
                    qd = q[walkers]/q[walkers].sum() if n_data > 1 else q
                    for model_id, pd in enumerate(layout.local_tail_prob):
                        if pd.sum() == 0:
                            continue
                        d = data_id*n_model+model_id
                        pd = pd/pd.sum()
                        kw, kg = jax.random.split(jax.random.fold_in(key, d))
                        uw, ug = (np.asarray(jax.random.uniform(k, (capacity,), dtype=jnp.float64)) for k in (kw, kg))
                        iw = np.searchsorted(np.cumsum(qd), uw*np.sum(qd), side="right")
                        ig = np.searchsorted(np.cumsum(pd), ug*np.sum(pd), side="right")
                        global_iw = walkers[iw]
                        values = wt[global_iw, None]*terms[global_iw, model_id*width+ig]/(qd[iw, None]*pd[ig, None])
                        values = values[:count+(d < rem)]
                        expected[1:] += values.mean(axis=0)
                        first = len(values)//2; second = len(values)-first
                        noise += np.sqrt(first*second)/len(values)*(values[:first].mean(axis=0)-values[first:].mean(axis=0))
            noise = noise/safe_den
            projected_noise = noise[1]+(1-expected[0]/safe_den)*noise[0]
            for fn in functions_by_chunks.values():
                result = fn(w, shard_first_axis(wt, mesh), key, h, c, t)
                np.testing.assert_allclose(result.weight, den, rtol=tol, atol=tol)
                np.testing.assert_allclose(result.numerator, expected, rtol=tol, atol=tol)
                np.testing.assert_allclose(result.diagnostics[d_pt_component_sampling_noise_real], projected_noise.real, rtol=tol, atol=tol)
                np.testing.assert_allclose(result.diagnostics[d_pt_component_sampling_noise_imag], projected_noise.imag, rtol=tol, atol=tol)
