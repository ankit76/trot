"""Runtime ownership, failure recovery, and small numerical AFQMC checks."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from trot import driver
from trot.core.ops import BlockEnergyRetuneResult, BlockComponentRetuneResult
from trot.ham.chol import HamChol
from trot.meas import cisd_modes as cm, ucisd_k_modes as um
from trot.meas import ptccsd_modes as rm, ptuccsd_modes as pm
from trot.meas.cisd import CisdMeasCfg
from trot.meas.ucisd import UcisdMeasCtx, UcisdMeasCfg
from trot.meas.ptuccsd_thouless import PtuccsdThoulessMeasCtx, PtuccsdThoulessMeasCfg
from trot.pair_sampling_runtime import prepare_local_cholesky_sampling
from trot.prop.chol_afqmc_ops import CholAfqmcCtx
from trot.prop.types import QmcParams, PropState
from trot.runtime_layout import QmcRuntime
from trot.sharding import (plan_cholesky_layout, remap_cholesky_layout,
    redistribute_cholesky, replicate, shard_model_axis)
from tests.test_cholesky_pair_sampling import model_mesh

jax.config.update("jax_enable_x64", True)


def host_copy(tree):
    return jax.tree.map(lambda a: np.array(a, copy=True), tree)


@pytest.mark.parametrize("replicated", [False, True])
def test_redistribution_preserves_rows_dtype_and_padding(replicated):
    mesh = model_mesh()
    host = (np.arange(12*2*3).reshape(12, 2, 3)+1j).astype(np.complex64)
    host[-3:] = 0
    data = replicate(host, mesh) if replicated else shard_model_axis(host, mesh)
    head = np.array([10, 3, 6, 0, 2])
    tail = np.setdiff1d(np.arange(12), head)
    prob = np.arange(1, tail.size+1, dtype=float); prob /= prob.sum()
    layout = plan_cholesky_layout(12, 4, head, tail, prob)
    actual = redistribute_cholesky(data, mesh, layout, batch_size=2)
    np.testing.assert_array_equal(actual, host[layout.permutation])
    np.testing.assert_array_equal(data, host)  # immutable inputs still valid
    assert actual.dtype == np.complex64 and actual.sharding.spec[0] == "model"


def test_second_sampler_keeps_own_head_and_proposal_with_padding():
    layout = plan_cholesky_layout(9, 4, np.array([0, 4, 7]),
        np.array([1, 2, 3, 5, 6, 8]), np.ones(6)/6)
    head = np.array([1, 7, 3, 5, 8])
    tail = np.array([0, 2, 4, 6]); prob = np.array([.1, .2, .3, .4])
    own = remap_cholesky_layout(layout, head, tail, prob)
    np.testing.assert_array_equal(own.permutation, layout.permutation)
    np.testing.assert_array_equal(own.permutation[own.head_indices], head)
    np.testing.assert_array_equal(own.permutation[own.tail_indices], tail)
    np.testing.assert_array_equal(own.local_tail_prob.ravel()[own.tail_indices], prob)
    assert np.all(own.local_tail_prob.ravel()[own.permutation < 0] == 0)
    for d, (indices, valid) in enumerate(zip(own.local_head_indices, own.local_head_valid)):
        expected = own.head_indices[own.head_indices//3 == d] % 3
        np.testing.assert_array_equal(indices[valid], expected)


def synthetic_runtime(kind="cisd", model=4, n_data=1):
    mesh = model_mesh(model, n_data)
    rows = lambda scale=1: shard_model_axis(scale*np.arange(1., 13.)[:, None, None]*np.ones((12, 2, 2)), mesh)
    one = replicate(np.asarray(1.), mesh)
    h1 = replicate(np.array([[3., 5.], [7., 9.]]), mesh)
    ham = HamChol(one, h1, rows())
    prop = CholAfqmcCtx(one, one, h1, shard_model_axis(np.arange(12.)+1j, mesh),
                       one, rows().reshape(12, 4), 2)
    if kind == "cisd":
        cls, sampling_cls = cm.CisdModeMeasCtx, cm.CisdModePairSamplingCfg
        kwargs = dict(rot_chol=rows(2), lci1=rows(3), cfg=CisdMeasCfg(), setup_mesh=mesh)
    elif kind == "ucc":
        cls, sampling_cls = um.UcisdKModeMeasCtx, um.UcisdKModePairSamplingCfg
        base = UcisdMeasCtx(h1, rows(2), h1, h1, rows(3), rows(4),
            rows(3).reshape(12, 4), rows(4).reshape(12, 4), rows(5), rows(6), UcisdMeasCfg())
        kwargs = dict(base=base)
    elif kind == "ptr":
        cls, sampling_cls = rm.PtccsdThoulessModeMeasCtx, rm.PtccsdModePairSamplingCfg
        kwargs = dict(rot_chol=rows(2), cfg=rm.PtccsdModeMeasCfg())
    else:
        cls, sampling_cls = pm.PtuccsdModeMeasCtx, pm.PtuccsdModePairSamplingCfg
        kwargs = dict(base=PtuccsdThoulessMeasCtx(h1, rows(2), rows(3), rows(4), PtuccsdThoulessMeasCfg()))
    head = np.array([1, 7, 3]) if kind.startswith("pt") else np.array([0, 4, 6, 10])
    tail = np.setdiff1d(np.arange(12), head)
    prob = np.arange(1, tail.size+1, dtype=float); prob /= prob.sum()
    sampling = sampling_cls(len(head), 17, track_half_sample_diagnostic=True)
    field = "component_sampling" if kind.startswith("pt") else "energy_sampling"
    ctx = cls(**kwargs, **{field: sampling}, reference_chol_scores=replicate(np.arange(12.), mesh),
        chol_head_indices=replicate(head, mesh), chol_tail_indices=replicate(tail, mesh),
        chol_tail_prob=replicate(prob, mesh), n_mode_chunks=3)
    return mesh, ham, prop, ctx


@pytest.mark.parametrize("kind", ["cisd", "ucc", "ptr", "ptu"])
@pytest.mark.parametrize("n_data,n_model", [(1, 4), (2, 2)])
def test_default_installs_consistent_context_and_propagator_layout(kind, n_data, n_model):
    mesh, h, p, c = synthetic_runtime(kind, n_model, n_data)
    assert QmcParams().local_cholesky_sampling
    runtime = QmcRuntime(h, p, c)
    _, p_before, c_before = host_copy((h, p, c))
    h2, p2, (c2,) = prepare_local_cholesky_sampling(h, p, (c,), runtime=runtime)
    perm = np.asarray(h2.chol[:, 0, 0], dtype=int)-1
    assert c2.model_sampling is not None
    assert c2.model_sampling.mesh.shape == mesh.shape
    assert h2.h1 is h.h1 and p2.exp_h1_half is p.exp_h1_half and p2.h0_prop is p.h0_prop
    np.testing.assert_array_equal(np.asarray(p2.chol_flat).reshape(12, -1), p_before.chol_flat[perm])
    assert p2.chol_flat is h2.chol
    np.testing.assert_array_equal(p2.mf_shifts, p_before.mf_shifts[perm])
    np.testing.assert_array_equal(c2.reference_chol_scores, c_before.reference_chol_scores[perm])
    np.testing.assert_array_equal(perm[np.asarray(c2.chol_head_indices)], c.chol_head_indices)
    np.testing.assert_array_equal(perm[np.asarray(c2.chol_tail_indices)], c.chol_tail_indices)
    np.testing.assert_allclose(c2.chol_tail_prob, c.chol_tail_prob)
    base, base2 = getattr(c_before, "base", c_before), getattr(c2, "base", c2)
    for field in ("chol_b", "rot_chol", "rot_chol_a", "rot_chol_b", "rot_chol_flat_a",
                  "rot_chol_flat_b", "lci1", "lci1_a", "lci1_b"):
        if hasattr(base, field):
            np.testing.assert_array_equal(getattr(base2, field), np.asarray(getattr(base, field))[perm])
    if hasattr(base, "h1_b"):
        assert base2.h1_b is c.base.h1_b
    # An explicitly installed layout is not permuted a second time.
    again = prepare_local_cholesky_sampling(h2, p2, (c2,), runtime=runtime)
    assert again[0] is h2 and again[1] is p2 and again[2][0] is c2


@pytest.mark.parametrize("ptkind", ["ptr", "ptu"])
@pytest.mark.parametrize("guide_policy", ["local", "global", "deterministic"])
@pytest.mark.parametrize("n_data,n_model", [(1, 4), (2, 2)])
def test_mixed_guide_and_pt_keep_distinct_sampling_policies(ptkind, guide_policy, n_data, n_model):
    _, h, p, guide = synthetic_runtime("cisd" if ptkind == "ptr" else "ucc", n_model, n_data)
    mesh, _, _, pt = synthetic_runtime(ptkind, n_model, n_data)
    if guide_policy == "global":
        guide = replace(guide, energy_sampling=replace(guide.energy_sampling, pair_sample_size=3))
    elif guide_policy == "deterministic":
        guide = replace(guide, energy_sampling=None)
    runtime = QmcRuntime(h, p, guide, pt)
    guide_before, pt_before = host_copy((guide, pt))
    h2, p2, (g2, e2) = prepare_local_cholesky_sampling(h, p, (guide, pt), runtime=runtime)
    perm = np.asarray(h2.chol[:, 0, 0], dtype=int)-1
    expected_layout = plan_cholesky_layout(12, n_model, pt.chol_head_indices, pt.chol_tail_indices, pt.chol_tail_prob)
    np.testing.assert_array_equal(perm, expected_layout.permutation)
    assert (g2.model_sampling is not None) == (guide_policy == "local")
    assert e2.model_sampling is not None
    for before, after in ((guide_before, g2), (pt_before, e2)):
        base, base2 = getattr(before, "base", before), getattr(after, "base", after)
        rot = "rot_chol" if hasattr(base, "rot_chol") else "rot_chol_a"
        np.testing.assert_array_equal(getattr(base2, rot), np.asarray(getattr(base, rot))[perm])
        if before is guide_before and guide_policy == "deterministic":
            continue
        np.testing.assert_array_equal(perm[np.asarray(after.chol_head_indices)], before.chol_head_indices)
        np.testing.assert_array_equal(perm[np.asarray(after.chol_tail_indices)], before.chol_tail_indices)
        np.testing.assert_allclose(after.chol_tail_prob, before.chol_tail_prob)


@pytest.mark.parametrize("reason", ["disabled", "deterministic", "one_gpu", "data", "budget", "custom"])
def test_ineligible_runs_keep_existing_objects(reason):
    mesh, h, p, c = synthetic_runtime(model=1 if reason == "one_gpu" else 4)
    if reason == "data":
        mesh = Mesh(np.array(jax.local_devices()[:4]).reshape(4, 1), ("data", "model"),
                    axis_types=(AxisType.Auto, AxisType.Auto))
        h = replace(h, chol=shard_model_axis(np.asarray(h.chol), mesh))
    if reason == "deterministic":
        c = replace(c, energy_sampling=None)
    if reason == "budget":
        c = replace(c, energy_sampling=replace(c.energy_sampling, pair_sample_size=3))
    if reason == "custom":
        p = object()
    runtime = QmcRuntime(h, p, c)
    h2, p2, (c2,) = prepare_local_cholesky_sampling(h, p, (c,), enabled=reason != "disabled", runtime=runtime)
    assert h2 is h and p2 is p and c2 is c


@pytest.mark.parametrize("pairs,installed", [(7, False), (8, True)])
def test_joint_mesh_budget_is_per_device_not_per_model_shard(pairs, installed):
    _, h, p, c = synthetic_runtime(model=2, n_data=2)
    c = replace(c, energy_sampling=replace(c.energy_sampling, pair_sample_size=pairs))
    runtime = QmcRuntime(h, p, c)
    _, _, (actual,) = prepare_local_cholesky_sampling(h, p, (c,), runtime=runtime)
    assert (actual.model_sampling is not None) == installed


def test_borrowed_inputs_do_not_silently_allocate_a_second_layout():
    _, h, p, c = synthetic_runtime()
    h2, p2, (c2,) = prepare_local_cholesky_sampling(h, p, (c,))
    assert h2 is h and p2 is p and c2 is c


@pytest.mark.parametrize("n_data,n_model", [(1, 4), (2, 2)])
@pytest.mark.parametrize("packed", [False, True])
def test_owned_mixed_runtime_releases_arrays_during_transition(monkeypatch, n_data, n_model, packed):
    from trot import pair_sampling_runtime as pr
    mesh, h, p, guide = synthetic_runtime("ucc", n_model, n_data)
    _, _, _, pt = synthetic_runtime("ptu", n_model, n_data)
    pt = replace(pt, base=replace(pt.base, chol_b=guide.base.chol_b))
    if packed:
        p = replace(p, chol_flat=shard_model_axis(
            np.asarray(h.chol)[:, [0, 0, 1], [0, 1, 1]].astype(np.float32), mesh), chol_packed=True)
    else:
        p = replace(p, chol_flat=h.chol)
    before = host_copy((h, p, (guide, pt)))
    runtime = QmcRuntime(h, p, guide, pt)
    copy, visited = pr.redistribute_cholesky, []

    def check_release(array, *args, **kwargs):
        # Retain strong references to every original: ordinary reference-count
        # cleanup cannot satisfy these assertions. No real OOM is needed.
        if visited:
            assert visited[-1].is_deleted(), "Previous input still occupies device storage"
        if array is guide.base.chol_b:
            assert p.chol_flat.is_deleted()
        if array is pt.base.rot_chol_a:
            assert guide.base.chol_b.is_deleted()
            assert guide.base.rot_chol_flat_a.is_deleted()
            assert guide.base.rot_chol_flat_b.is_deleted()
        visited.append(array)
        return copy(array, *args, **kwargs)

    monkeypatch.setattr(pr, "redistribute_cholesky", check_release)
    h2, p2, (g2, e2) = prepare_local_cholesky_sampling(h, p, (guide, pt), runtime=runtime)
    assert sum(a is guide.base.chol_b for a in visited) == 1
    assert g2.base.chol_b is e2.base.chol_b
    perm = np.asarray(h2.chol[:, 0, 0], dtype=int)-1
    np.testing.assert_array_equal(g2.base.chol_b, before[2][0].base.chol_b[perm])
    np.testing.assert_array_equal(p2.chol_flat.reshape(12, -1), before[1].chol_flat.reshape(12, -1)[perm])
    assert p2.chol_packed == packed and p2.chol_flat.dtype == before[1].chol_flat.dtype


def test_distinct_objects_sharing_device_buffers_are_released_together(monkeypatch):
    from trot import pair_sampling_runtime as pr
    _, h, p, c = synthetic_runtime("ucc")
    # Unlike an eager reshape (which may copy), this deliberately shares the
    # actual device buffers across two distinct runtime array objects.
    a = c.base.rot_chol_a
    alias = jax.make_array_from_single_device_arrays(a.shape, a.sharding,
        [s.data for s in a.addressable_shards])
    c = replace(c, base=replace(c.base, rot_chol_b=alias))
    before = host_copy(c)
    runtime = QmcRuntime(h, p, c)
    copy, saw_alias = pr.redistribute_cholesky, []

    def check_alias(array, *args, **kwargs):
        if array is alias:
            assert not a.is_deleted(), "The unprocessed alias lost its device buffer"
            saw_alias.append(True)
        if array is c.base.lci1_a:
            assert a.is_deleted() and alias.is_deleted()
        return copy(array, *args, **kwargs)

    monkeypatch.setattr(pr, "redistribute_cholesky", check_alias)
    h2, _, (c2,) = prepare_local_cholesky_sampling(h, p, (c,), runtime=runtime)
    perm = np.asarray(h2.chol[:, 0, 0], dtype=int)-1
    assert saw_alias
    np.testing.assert_array_equal(c2.base.rot_chol_a, before.base.rot_chol_a[perm])
    np.testing.assert_array_equal(c2.base.rot_chol_b, before.base.rot_chol_b[perm])


def test_alias_of_a_preserved_field_is_not_deleted():
    _, h, p, c = synthetic_runtime()
    # A non-Cholesky field can retain the same buffer; replacing the Cholesky
    # field must not invalidate it. Its scientific role is immaterial here.
    alias = jax.make_array_from_single_device_arrays(c.rot_chol.shape, c.rot_chol.sharding,
        [s.data for s in c.rot_chol.addressable_shards])
    h = replace(h, h1=alias)
    before = np.asarray(h.h1).copy()
    runtime = QmcRuntime(h, p, c)
    h2, _, _ = prepare_local_cholesky_sampling(h, p, (c,), runtime=runtime)
    np.testing.assert_array_equal(h2.h1, before)
    assert not h.h1.is_deleted()


@pytest.mark.parametrize("failure", ["ham", "context", "sampling"])
def test_failed_redistribution_restores_owner_in_original_order(monkeypatch, failure):
    from trot import pair_sampling_runtime as pr
    _, h, p, c = synthetic_runtime()
    before = host_copy((h, p, (c,)))
    adopted = []
    runtime = QmcRuntime(h, p, c, on_update=lambda r: adopted.append(r.ham_data))
    copy = pr.redistribute_cholesky
    def fail_at_context(array, *args, **kwargs):
        if array is (h.chol if failure == "ham" else c.rot_chol) and failure != "sampling":
            raise RuntimeError("simulated allocation failure")
        return copy(array, *args, **kwargs)
    monkeypatch.setattr(pr, "redistribute_cholesky", fail_at_context)
    if failure == "sampling":
        def fail_sampling(*args):
            raise RuntimeError("simulated allocation failure")
        monkeypatch.setattr(pr, "with_local_cholesky_sampling", fail_sampling)
    with pytest.raises(RuntimeError, match="simulated allocation failure"):
        prepare_local_cholesky_sampling(h, p, (c,), runtime=runtime)
    actual = runtime.inputs(ham_data=None, prop_ctx=None, contexts=(None,))
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(before)):
        np.testing.assert_array_equal(a, b)
    assert runtime.meas_ctx.model_sampling is None
    assert adopted[-1] is runtime.ham_data
    assert h.chol.is_deleted() == (failure != "ham")
    # A restored owner can make another attempt without re-staging.
    monkeypatch.undo()
    prepare_local_cholesky_sampling(*actual, runtime=runtime)
    assert runtime.meas_ctx.model_sampling is not None


def test_failed_pt_transition_restores_shared_beta_and_packed_propagation(monkeypatch):
    from trot import pair_sampling_runtime as pr
    mesh, h, p, guide = synthetic_runtime("ucc", 2, 2)
    _, _, _, pt = synthetic_runtime("ptu", 2, 2)
    pt = replace(pt, base=replace(pt.base, chol_b=guide.base.chol_b))
    p = replace(p, chol_flat=shard_model_axis(
        np.asarray(h.chol)[:, [0, 0, 1], [0, 1, 1]].astype(np.float32), mesh), chol_packed=True)
    runtime = QmcRuntime(h, p, guide, pt)
    before = host_copy((h, p, (guide, pt)))
    copy = pr.redistribute_cholesky

    def fail_at_pt(array, *args, **kwargs):
        if array is pt.base.rot_chol_a:
            assert h.chol.is_deleted() and guide.base.chol_b.is_deleted()
            raise RuntimeError("simulated PT allocation failure")
        return copy(array, *args, **kwargs)

    monkeypatch.setattr(pr, "redistribute_cholesky", fail_at_pt)
    with pytest.raises(RuntimeError, match="simulated PT allocation failure"):
        prepare_local_cholesky_sampling(h, p, (guide, pt), runtime=runtime)
    after = runtime.inputs(ham_data=None, prop_ctx=None, contexts=(None, None))
    for a, b in zip(jax.tree.leaves(after), jax.tree.leaves(before)):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    assert runtime.meas_ctx.base.chol_b is runtime.estimator_ctx.base.chol_b
    assert runtime.prop_ctx.chol_packed


def test_unrecoverable_transition_invalidates_owner(monkeypatch):
    from trot import pair_sampling_runtime as pr
    _, h, p, c = synthetic_runtime()
    runtime = QmcRuntime(h, p, c)
    copy = pr.redistribute_cholesky

    def fail_restore(*args, **kwargs):
        raise RuntimeError("device unavailable")

    def fail_at_context(array, *args, **kwargs):
        if array is c.rot_chol:
            monkeypatch.setattr(jax, "device_put", fail_restore)
            raise RuntimeError("simulated allocation failure")
        return copy(array, *args, **kwargs)

    monkeypatch.setattr(pr, "redistribute_cholesky", fail_at_context)
    with pytest.raises(RuntimeError, match="Could not restore"):
        prepare_local_cholesky_sampling(h, p, (c,), runtime=runtime)
    assert runtime.ham_data is None and runtime.prop_ctx is None and runtime.meas_ctx is None
    with pytest.raises(RuntimeError, match="rebuild the runtime"):
        runtime.inputs(ham_data=None, prop_ctx=None, contexts=(None,))


def test_single_estimator_update_drops_an_unused_pt_context():
    runtime = QmcRuntime(None, estimator_ctx=SimpleNamespace())
    runtime.update(None, None, (None,))
    assert runtime.estimator_ctx is None


@pytest.mark.parametrize("retune,enabled", [(True, True), (False, False)])
def test_owned_runtime_can_retune_or_disable_an_installed_sampler(retune, enabled):
    _, h, p, c = synthetic_runtime()
    runtime = QmcRuntime(h, p, c, c)
    prepare_local_cholesky_sampling(h, p, (c, c), runtime=runtime)
    h2, p2, contexts = runtime.inputs(ham_data=None, prop_ctx=None, contexts=(None, None),
                                      retune=retune, local_sampling=enabled)
    assert h2 is runtime.ham_data and p2 is runtime.prop_ctx
    assert all(c.model_sampling is None for c in contexts)
    with pytest.raises(ValueError, match="not both"):
        runtime.inputs(ham_data=h, prop_ctx=None, contexts=(None,))


@pytest.mark.parametrize("mixed,tune_guide,tune_pt", [(False, False, False), (False, True, False),
    (True, False, False), (True, True, False), (True, False, True), (True, True, True)])
def test_job_and_pt_driver_release_old_arrays_before_production(monkeypatch, mixed, tune_guide, tune_pt):
    from trot.setup import Job
    class ProductionReached(Exception):
        pass
    state = PropState(jnp.ones((4, 2, 1)), jnp.ones(4), jnp.ones(4), jax.random.PRNGKey(1),
                      jnp.asarray(-1.), jnp.asarray(-1.), jnp.asarray(0))
    def guide_tune(state, e, w, params, h, c, t, **kw):
        assert c.model_sampling is None
        return BlockEnergyRetuneResult(state, replace(c), 1, 1)
    def pt_tune(state, e, w, params, h, c, t, **kw):
        assert c.model_sampling is None
        return BlockComponentRetuneResult(state, replace(c), 1, 1)
    guide_ops = SimpleNamespace(retune_block_energy=guide_tune if tune_guide else None)
    pt_ops = SimpleNamespace(retune_block_components=pt_tune if tune_pt else None,
        use_for_population_control=True, combine_energy=lambda h0, c: c[0])
    params = QmcParams(n_eql_blocks=1, n_blocks=2)

    def owned_inputs():
        # This factory transfers ownership without leaving strong references in
        # the test frame. Weak references are checked inside production blocks.
        mesh, h, p, c = synthetic_runtime()
        arrays = [h.chol, p.chol_flat, p.mf_shifts, c.rot_chol, c.lci1]
        job = Job(staged=None, sys=None, params=params, ham_data=h, trial_data=None,
            trial_ops=None, meas_ops=guide_ops, prop_ops=None, block_fn=lambda *a, **k: None,
            runtime_layout=None, mesh=mesh)
        job._runtime_state, job._runtime_meas_ctx, job._runtime_prop_ctx = state, c, p
        if mixed:
            _, _, _, e = synthetic_runtime("ptr")
            arrays.append(e.rot_chol)
            _, owner = job.prepare_runtime()
            owner.estimator_ctx = e
        else:
            owner = job
        return owner, [weakref.ref(a) for a in arrays]
    owner, originals = owned_inputs()

    def build(**kw):
        def advance(state, *, n_blocks, **inputs):
            c = inputs["guide_meas_ctx" if mixed else "meas_ctx"]
            if c.model_sampling is not None:
                gc.collect()
                assert all(ref() is None for ref in originals), "Old Cholesky arrays are still owned"
                assert owner.ham_data is inputs["ham_data"]
                if mixed:
                    assert owner.prop_ctx is inputs["guide_prop_ctx"]
                    assert owner.meas_ctx is c and owner.estimator_ctx is inputs["estimator_ctx"]
                else:
                    assert owner._runtime_prop_ctx is inputs["prop_ctx"]
                    assert owner._runtime_meas_ctx is c
                raise ProductionReached
            if mixed:
                return state, dict(guide_energy=-jnp.ones(n_blocks), guide_weight=jnp.ones(n_blocks),
                    estimator_weight=jnp.ones(n_blocks, dtype=complex),
                    estimator_components=-jnp.ones((n_blocks, 1), dtype=complex))
            return state, dict(energy=-jnp.ones(n_blocks), weight=jnp.ones(n_blocks)), ()
        return kw["params"], advance
    monkeypatch.setattr(driver, "_make_run_blocks_with_auto_chunks", build)
    monkeypatch.setattr(driver, "_make_run_mixed_estimator_blocks_with_auto_chunks", build)
    monkeypatch.setattr(driver, "_initial_projected_estimator", lambda *a, **k: (-1., 4.))
    for _ in range(2):  # Reusing the adopted runtime must still allow retuning.
        with pytest.raises(ProductionReached):
            if mixed:
                driver.run_mixed_estimator_qmc(sys=None, params=params, runtime=owner, state=state,
                    guide_data=None, guide_ops=None, guide_prop_ops=None, guide_meas_ops=guide_ops,
                    estimator_data=None, estimator_ops=pt_ops, mixed_block_fn=None)
            else:
                owner.kernel()


@pytest.mark.parametrize("mixed,tune_guide,tune_pt", [(False, False, False), (False, True, False),
    (True, False, False), (True, True, False), (True, False, True), (True, True, True)])
def test_driver_installs_only_after_all_requested_tuning(monkeypatch, mixed, tune_guide, tune_pt):
    # Synthetic driver scalars, not physical AFQMC propagation or energy kernels.
    state = PropState(jnp.ones((4, 2, 1)), jnp.ones(4), jnp.ones(4), jax.random.PRNGKey(1),
                      jnp.asarray(-1.), jnp.asarray(-1.), jnp.asarray(0))
    events = []
    def guide_tune(*args, **kw):
        events.append("guide_tuned")
        return BlockEnergyRetuneResult(state, "guide_final", 1, 1)
    def pt_tune(*args, **kw):
        events.append("pt_tuned")
        return BlockComponentRetuneResult(state, "pt_final", 1, 1)
    guide_ops = SimpleNamespace(retune_block_energy=guide_tune if tune_guide else None)
    estimator_ops = SimpleNamespace(retune_block_components=pt_tune if tune_pt else None,
        use_for_population_control=True, combine_energy=lambda h0, c: c[0])
    class TransitionReached(Exception):
        pass
    def prepare(h, p, contexts, *, enabled, runtime):
        assert enabled
        assert events == (["guide_tuned"] if tune_guide else [])+(["pt_tuned"] if tune_pt else [])
        assert contexts[0] == ("guide_final" if tune_guide else "guide_initial")
        if mixed:
            assert contexts[1] == ("pt_final" if tune_pt else "pt_initial")
        raise TransitionReached
    def build(**kw):
        def advance(state, *, n_blocks, **kw):
            if mixed:
                return state, dict(guide_energy=-jnp.ones(n_blocks), guide_weight=jnp.ones(n_blocks),
                    estimator_weight=jnp.ones(n_blocks, dtype=complex),
                    estimator_components=-jnp.ones((n_blocks, 1), dtype=complex))
            return state, dict(energy=-jnp.ones(n_blocks), weight=jnp.ones(n_blocks)), ()
        return kw["params"], advance
    monkeypatch.setattr(driver, "prepare_local_cholesky_sampling", prepare)
    monkeypatch.setattr(driver, "_make_run_blocks_with_auto_chunks", build)
    monkeypatch.setattr(driver, "_make_run_mixed_estimator_blocks_with_auto_chunks", build)
    monkeypatch.setattr(driver, "_initial_projected_estimator", lambda *a, **k: (-1., 4.))
    common = dict(sys=None, params=QmcParams(n_eql_blocks=1, n_blocks=2),
                  ham_data=SimpleNamespace(h0=0.), state=state)
    with pytest.raises(TransitionReached):
        if mixed:
            driver.run_mixed_estimator_qmc(**common, guide_data=None, guide_ops=None, guide_prop_ops=None,
                guide_meas_ops=guide_ops, estimator_data=None, estimator_ops=estimator_ops, mixed_block_fn=None,
                guide_meas_ctx="guide_initial", guide_prop_ctx="prop", estimator_ctx="pt_initial")
        else:
            driver.run_qmc(**common, trial_data=None, trial_ops=None, prop_ops=None, meas_ops=guide_ops,
                           block_fn=None, meas_ctx="guide_initial", prop_ctx="prop")


@pytest.mark.parametrize("kind", ["cisd", "ucisd", "rcc", "ucc"])
@pytest.mark.parametrize("mixed", [False, True])
def test_runtime_reordering_preserves_energy_force_bias_and_propagation(kind, mixed, monkeypatch):
    from trot import walkers as wk
    from trot import pair_sampling_runtime as pr
    from trot.prop.chol_afqmc_ops import _build_prop_ctx, make_trotter_ops
    from tests.test_cholesky_pair_sampling_kernels import prepare as prepare_cisd
    from tests.test_pt_cholesky_pair_sampling import prepare as prepare_pt

    prepare = prepare_pt if kind in ("rcc", "ucc") else prepare_cisd
    module, mesh, _, h, c, t, w, _ = prepare(kind, mixed, 1, 3)
    c = replace(c, model_sampling=None, chol_head_indices=c.chol_head_indices[::-1],
                chol_tail_prob=c.chol_tail_prob[::-1])
    layout = plan_cholesky_layout(h.nchol, 1, c.chol_head_indices, c.chol_tail_indices, c.chol_tail_prob)
    assert not np.array_equal(layout.permutation, np.arange(h.nchol))
    p = _build_prop_ctx(h, jnp.eye(h.chol.shape[1]), .005,
        chol_flat_precision=jnp.float32 if mixed else jnp.float64, packed_cholesky=mixed)
    perm = layout.permutation
    tol = 1e-5 if mixed else 2e-12
    if kind == "rcc":
        energy = module.components_pt_thouless_rw_rh
        force_bias = module.force_bias_pt_thouless_rw_rh
    elif kind == "ucc":
        energy = module.components_ptuccsd_mode_rw_rh
        force_bias = module.force_bias_kernel_rw_rh
    else:
        energy, force_bias = module.energy_kernel_rw_rh, module.force_bias_kernel_rw_rh
    expected = {}
    kernels = {}
    for chunks in (1, 5):
        energy_fn = jax.jit(wk.vmap_chunked(energy, chunks, in_axes=(0, None, None, None)))
        fb_fn = jax.jit(wk.vmap_chunked(force_bias, chunks, in_axes=(0, None, None, None)))
        kernels[chunks] = energy_fn, fb_fn
        expected[chunks] = (np.asarray(energy_fn(w, h, c, t)), np.asarray(fb_fn(w, h, c, t)))
    # Match physical auxiliary fields, not just RNG seeds, after permutation.
    fields = np.random.default_rng(91).normal(size=(len(w), h.nchol)).astype(complex)
    fields += .03j
    apply = make_trotter_ops("restricted", "restricted", mixed_precision=mixed).apply_trotter
    propagate = jax.jit(jax.vmap(lambda w, x, p: apply(w, x, p, 6), in_axes=(0, 0, None)))
    expected_walkers = np.asarray(propagate(w, replicate(fields, mesh), p))
    expected_shifts = np.asarray(p.mf_shifts)[perm]
    # Exercise the owned transition on one CPU or GPU. The public policy
    # intentionally does nothing on one device (covered separately above).
    monkeypatch.setattr(pr, "cholesky_model_mesh", lambda _: mesh)
    runtime = QmcRuntime(h, p, c)
    h2, p2, (c2,) = prepare_local_cholesky_sampling(h, p, (c,), runtime=runtime)
    assert h.chol.is_deleted()
    for chunks, (energy_fn, fb_fn) in kernels.items():
        e, fb = expected[chunks]
        np.testing.assert_allclose(energy_fn(w, h2, c2, t), e, rtol=tol, atol=tol)
        np.testing.assert_allclose(fb_fn(w, h2, c2, t), fb[:, perm], rtol=tol, atol=tol)
    np.testing.assert_allclose(propagate(w, replicate(fields[:, perm], mesh), p2),
        expected_walkers, rtol=tol, atol=tol)
    np.testing.assert_allclose(p2.mf_shifts, expected_shifts, rtol=0, atol=0)


@pytest.mark.parametrize("mixed", [False, True])
def test_gpu_owned_runtime_releases_device_storage(monkeypatch, mixed):
    if jax.default_backend() != "gpu":
        pytest.skip("GPU allocator accounting requires a GPU")
    from trot import pair_sampling_runtime as pr
    mesh = model_mesh(1)
    # Exercise the production ownership transition on ws; eligibility normally
    # excludes one GPU. This tests allocation, not multi-GPU communication.
    monkeypatch.setattr(pr, "cholesky_model_mesh", lambda _: mesh)
    def inputs():
        _, h, p, c = synthetic_runtime(model=1)
        chol = shard_model_axis(np.arange(12*1024*512, dtype=float).reshape(12, 1024, 512), mesh)
        h = replace(h, chol=chol)
        p = replace(p, chol_flat=chol.reshape(12, -1).astype(jnp.float32 if mixed else jnp.float64))
        return QmcRuntime(h, p, c), [weakref.ref(chol), weakref.ref(p.chol_flat)]
    runtime, old = inputs()
    jax.block_until_ready((runtime.ham_data, runtime.prop_ctx, runtime.meas_ctx))
    gc.collect()
    device = jax.local_devices()[0]
    before = device.memory_stats()["bytes_in_use"]
    prepare_local_cholesky_sampling(runtime.ham_data, runtime.prop_ctx,
                                   (runtime.meas_ctx,), runtime=runtime)
    gc.collect()
    assert all(ref() is None for ref in old)
    after = device.memory_stats()["bytes_in_use"]
    # BFC allocation rounding/fragmentation changes bytes_in_use in 16+ MiB
    # increments here. Count live large buffers directly, deduplicating aliases:
    # one 48 MiB Hamiltonian, plus 24 MiB for mixed propagation only.
    buffers = {}
    for array in jax.live_arrays():
        if array.nbytes >= 8 * 1024**2:
            for shard in array.addressable_shards:
                buffers[shard.data.unsafe_buffer_pointer()] = shard.data.nbytes
    expected_bytes = (72 if mixed else 48) * 1024**2
    assert sum(buffers.values()) == expected_bytes
    if not mixed:
        assert runtime.prop_ctx.chol_flat is runtime.ham_data.chol
    print(f"live large buffers={sum(buffers.values())} bytes; "
          f"allocator before={before}, after={after}, delta={after-before} bytes")
