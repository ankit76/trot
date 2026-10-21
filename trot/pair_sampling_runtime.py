"""Host-side installation of frozen local-Cholesky population estimators."""
from dataclasses import replace

import jax
import numpy as np

from .ham.chol import HamChol
from .meas.cisd import CisdMeasCtx
from .meas.cisd_modes import CisdModeMeasCtx
from .meas.pair_sampling import with_local_cholesky_sampling
from .meas.ptccsd_modes import PtccsdThoulessModeMeasCtx
from .meas.ptuccsd_modes import PtuccsdModeMeasCtx
from .meas.rhf import RhfMeasCtx
from .meas.ucisd import UcisdMeasCtx
from .meas.ucisd_k_modes import UcisdKModeMeasCtx
from .meas.ucisd_modes import UcisdModeMeasCtx
from .meas.uhf import UhfMeasCtx
from .prop.chol_afqmc_ops import CholAfqmcCtx, _prepare_chol_for_vhs
from .sharding import (
    cholesky_model_mesh, plan_cholesky_layout, redistribute_cholesky,
    remap_cholesky_layout, replicate,
)


_SAMPLED_CONTEXTS = (CisdModeMeasCtx, UcisdKModeMeasCtx,
                     PtccsdThoulessModeMeasCtx, PtuccsdModeMeasCtx)
_GUIDE_CONTEXTS = (CisdMeasCtx, UcisdMeasCtx, UcisdModeMeasCtx, RhfMeasCtx, UhfMeasCtx)


def _sampling(ctx):
    return getattr(ctx, "component_sampling", getattr(ctx, "energy_sampling", None))


class _CholeskyReorder:
    """Replace owned arrays progressively; keep borrowed inputs immutable.

    Distinct JAX objects can share device buffers (including flattened views).
    Delete a buffer only after every runtime alias has a ready replacement.
    Host copies are needed only to restore the owner after a failed transition.
    """

    def __init__(self, inputs, mesh, layout, runtime):
        self.inputs, self.mesh, self.layout, self.runtime = inputs, mesh, layout, runtime
        self.reordered = {}
        self.created = {}
        self.arrays = {}
        self.buffers = {}
        self.owners = {}
        if runtime is not None:
            self.arrays = {id(a): a for a in jax.tree.leaves(inputs) if isinstance(a, jax.Array)}
            for key, array in self.arrays.items():
                buffers = {(s.device, s.data.unsafe_buffer_pointer())
                           for s in array.addressable_shards if s.data.size}
                self.buffers[key] = buffers
                for buffer in buffers:
                    self.owners.setdefault(buffer, set()).add(key)

    def __call__(self, array):
        key = id(array)
        if key not in self.reordered:
            self.adopt(array, redistribute_cholesky(array, self.mesh, self.layout))
        result = self.reordered[key]
        shape = (self.layout.permutation.size, *array.shape[1:])
        return result if result.shape == shape else result.reshape(shape)

    def adopt(self, old, new):
        new.block_until_ready()
        self.created[id(new)] = new
        self.reordered.setdefault(id(old), new)
        if self.runtime is not None:
            for key in self.reordered:
                array = self.arrays.get(key)
                if array is not None and not array.is_deleted() and all(
                        self.owners[b] <= self.reordered.keys() for b in self.buffers[key]):
                    array.delete()
        return new

    def restore(self):
        """Restore original ordering on failure, using CPU RAM only on this path.

        Old JAX objects remain consumed; the runtime (and Job update hook)
        receive usable replacements. Free partial outputs before restoring
        device storage, so rollback does not recreate the old/new memory peak.
        """
        if self.runtime is None:
            return
        try:
            host = {}
            for key, array in self.arrays.items():
                if array.is_deleted():
                    new = np.asarray(self.reordered[key]).reshape(
                        self.layout.permutation.size, *array.shape[1:])
                    host[key] = new[self.layout.inverse_permutation]
            for array in self.created.values():
                array.delete()
            restored = {key: jax.device_put(value, self.arrays[key].sharding).block_until_ready()
                        for key, value in host.items()}
            self.runtime.update(*jax.tree.map(lambda a: restored.get(id(a), a), self.inputs))
        except Exception as exc:
            # Never expose an owner containing a mixture of deleted/reordered
            # inputs if even restoration fails (for example, device loss).
            self.runtime.update(None, None, (None,) * len(self.inputs[2]))
            raise RuntimeError("Could not restore the Cholesky runtime; rebuild it before reuse.") from exc


def _reorder_context(ctx, reorder):
    """Explicit field adapters: never guess axes from coincident array sizes."""
    if ctx is None:
        return None
    if isinstance(ctx, (UcisdKModeMeasCtx, UcisdModeMeasCtx, PtuccsdModeMeasCtx)):
        if isinstance(ctx, PtuccsdModeMeasCtx):
            base = replace(ctx.base, **{k: reorder(getattr(ctx.base, k)) for k in
                ("chol_b", "rot_chol_a", "rot_chol_b")})
        else:
            base = _reorder_context(ctx.base, reorder)
        changed = replace(ctx, base=base)
    else:
        if isinstance(ctx, (CisdModeMeasCtx, CisdMeasCtx)):
            fields = ("rot_chol", "lci1")
        elif isinstance(ctx, (RhfMeasCtx, PtccsdThoulessModeMeasCtx)):
            fields = ("rot_chol",)
        elif isinstance(ctx, UcisdMeasCtx):
            fields = ("chol_b", "rot_chol_a", "rot_chol_b", "lci1_a", "lci1_b")
        elif isinstance(ctx, UhfMeasCtx):
            fields = ("rot_chol_a", "rot_chol_b")
        else:
            raise TypeError(f"No Cholesky reordering adapter for {type(ctx).__name__}.")
        changed = replace(ctx, **{k: reorder(getattr(ctx, k)) for k in fields})
        # Rebuild flattened representations from the adopted tensors locally.
        for flat, tensor in (("rot_chol_flat", "rot_chol"),
                             ("rot_chol_flat_a", "rot_chol_a"), ("rot_chol_flat_b", "rot_chol_b")):
            if hasattr(changed, flat):
                array = getattr(changed, tensor)
                value = reorder.adopt(getattr(ctx, flat), array.reshape(array.shape[0], -1))
                changed = replace(changed, **{flat: value})
    if hasattr(changed, "reference_chol_scores") and changed.reference_chol_scores.size:
        changed = replace(changed, reference_chol_scores=reorder(changed.reference_chol_scores))
    return changed


def _install_local_cholesky_sampling(ham_data, prop_ctx, contexts, mesh, eligible, *, runtime=None):
    # In a mixed run the estimator is last. Prioritize the PT proposal; retain
    # the guide's own head/probabilities on that same physical permutation.
    primary = contexts[eligible[-1]]
    layout = plan_cholesky_layout(ham_data.chol.shape[0], mesh.shape["model"],
        primary.chol_head_indices, primary.chol_tail_indices, primary.chol_tail_prob)
    reorder = _CholeskyReorder((ham_data, prop_ctx, contexts), mesh, layout, runtime)
    try:
        ham_new = replace(ham_data, chol=reorder(ham_data.chol), nchol=None)
        # Eager JAX reshape can copy. Retain the same tensor when dtypes match and
        # flatten inside the compiled VHS contraction; keep mixed/packed storage.
        share_chol = not prop_ctx.chol_packed and prop_ctx.chol_flat.dtype == ham_new.chol.dtype
        chol_vhs = ham_new.chol if share_chol else _prepare_chol_for_vhs(
            ham_new.chol, dtype=prop_ctx.chol_flat.dtype, packed_cholesky=prop_ctx.chol_packed)
        reorder.adopt(prop_ctx.chol_flat, chol_vhs)
        prop_new = replace(prop_ctx, chol_flat=chol_vhs,
            mf_shifts=reorder(prop_ctx.mf_shifts))
        new_contexts = []
        for i, ctx in enumerate(contexts):
            changed = _reorder_context(ctx, reorder)
            if isinstance(ctx, _SAMPLED_CONTEXTS) and _sampling(ctx) is not None:
                own_layout = remap_cholesky_layout(layout, ctx.chol_head_indices,
                                                   ctx.chol_tail_indices, ctx.chol_tail_prob)
                if i in eligible:
                    changed = with_local_cholesky_sampling(changed, own_layout, mesh)
                else:
                    changed = replace(changed, chol_head_indices=replicate(own_layout.head_indices, mesh),
                        chol_tail_indices=replicate(own_layout.tail_indices, mesh),
                        chol_tail_prob=replicate(own_layout.tail_prob, mesh))
            new_contexts.append(changed)
    except Exception:
        reorder.restore()
        raise
    print(f"[sampling] local Cholesky sampling on {layout.n_model} model shards "
          f"and {mesh.shape.get('data', 1)} data shards; "
          f"balanced layout from {type(primary).__name__}; "
          f"local samplers={len(eligible)}.", flush=True)
    return ham_new, prop_new, tuple(new_contexts)


def prepare_local_cholesky_sampling(ham_data, prop_ctx, contexts, *, enabled=True, runtime=None):
    """Consume an owned runtime's old Cholesky arrays as replacements are ready.

    After success or an allocation failure, use the owner rather than separate
    references to its previous arrays. Borrowed runtime inputs are not consumed.
    """
    result = _prepare_local_cholesky_sampling(
        ham_data, prop_ctx, contexts, enabled=enabled, runtime=runtime,
    )
    if runtime is not None:
        runtime.update(*result)
    return result


def _prepare_local_cholesky_sampling(ham_data, prop_ctx, contexts, *, enabled, runtime):
    """Install local sampling only for supported, frozen, model-sharded runs.

    Drivers call this before the first block for fixed policies, or after
    retuning and before production for adaptive policies. Every context and
    the propagation arrays share one permutation. Walker state has no stored
    Cholesky axis. Combined data/model meshes condition both proposals locally.
    Single-GPU, data-only, deterministic and opted-out runs retain their
    existing objects. Explicitly installed layouts are preserved.
    """
    unchanged = (ham_data, prop_ctx, contexts)
    if not enabled or not isinstance(ham_data, HamChol) or not ham_data.chol.shape[0]:
        return unchanged
    mesh = cholesky_model_mesh(ham_data.chol)
    if mesh is None or not ham_data.chol.is_fully_addressable:
        return unchanged
    if any(getattr(ctx, "model_sampling", None) is not None for ctx in contexts):
        return unchanged
    eligible = []
    for i, ctx in enumerate(contexts):
        sampling = _sampling(ctx)
        if not isinstance(ctx, _SAMPLED_CONTEXTS) or sampling is None:
            continue
        minimum = 2 if sampling.track_half_sample_diagnostic else 1
        if getattr(sampling, "sample_local_walkers", False):
            continue
        n_strata = mesh.shape["model"] * mesh.shape.get("data", 1)
        if ctx.chol_tail_indices.size and sampling.pair_sample_size < minimum * n_strata:
            print("[sampling] pair budget too small for local Cholesky strata; retaining global sampler.", flush=True)
            continue
        eligible.append(i)
    if not eligible:
        return unchanged
    if runtime is None:
        print("[sampling] retaining global sampling for borrowed runtime inputs; "
              "pass QmcRuntime to enable redistribution without retaining old arrays.", flush=True)
        return unchanged
    if not isinstance(prop_ctx, CholAfqmcCtx) or any(
            ctx is not None and not isinstance(ctx, _SAMPLED_CONTEXTS + _GUIDE_CONTEXTS) for ctx in contexts):
        print("[sampling] custom runtime context has no Cholesky layout adapter; retaining global sampling.", flush=True)
        return unchanged
    return _install_local_cholesky_sampling(ham_data, prop_ctx, contexts, mesh, eligible, runtime=runtime)
