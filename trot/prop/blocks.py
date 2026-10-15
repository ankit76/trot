from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from jax.experimental import io_callback

from .. import walkers as wk
from ..core.levels import LevelPack
from ..core.ops import (
    BlockComponentEstimate,
    BlockEnergyEstimate,
    EstimatorOps,
    MeasOps,
    TrialOps,
    k_energy,
)
from ..core.system import System
from ..walkers import SrFn
from .types import PropOps, PropState, QmcParams, QmcParamsFp


class BlockFn(Protocol):
    def __call__(
        self,
        state: PropState,
        *,
        sys: System,
        params: Any,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ops: Any,
        prop_ctx: Any,
        sr_fn: SrFn = wk.stochastic_reconfiguration,
        observable_names: tuple[str, ...] = (),
    ) -> tuple[PropState, BlockObs]: ...


class MixedBlockFn(Protocol):
    def __call__(
        self,
        state: PropState,
        *,
        sys: System,
        params: QmcParams,
        ham_data: Any,
        guide_data: Any,
        guide_ops: TrialOps,
        guide_meas_ops: MeasOps,
        guide_meas_ctx: Any,
        guide_prop_ops: PropOps,
        guide_prop_ctx: Any,
        trial_data: Any,
        trial_meas_ops: MeasOps,
        trial_meas_ctx: Any,
        observable_names: tuple[str, ...] = (),
        sr_fn: Callable = wk.stochastic_reconfiguration,
    ) -> tuple[PropState, BlockObs]: ...


class MixedEstimatorBlockFn(Protocol):
    def __call__(
        self,
        state: PropState,
        *,
        sys: System,
        params: QmcParams,
        ham_data: Any,
        guide_data: Any,
        guide_ops: TrialOps,
        guide_meas_ops: MeasOps,
        guide_meas_ctx: Any,
        guide_prop_ops: PropOps,
        guide_prop_ctx: Any,
        estimator_data: Any,
        estimator_ops: EstimatorOps,
        estimator_ctx: Any,
        sr_fn: Callable = wk.stochastic_reconfiguration,
    ) -> tuple[PropState, BlockObs]: ...


class BlockObs(NamedTuple):
    scalars: dict[str, jax.Array]
    observables: dict[str, jax.Array]


def dump_prop_state_npz(
    state: PropState,
    path: str | Path,
    *,
    compressed: bool = True,
) -> None:
    """
    Save a PropState to a single ``.npz`` file for offline analysis.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "weights": np.asarray(jax.device_get(state.weights)),
        "overlaps": np.asarray(jax.device_get(state.overlaps)),
        "rng_key": np.asarray(jax.device_get(state.rng_key)),
        "pop_control_ene_shift": np.asarray(jax.device_get(state.pop_control_ene_shift)),
        "e_estimate": np.asarray(jax.device_get(state.e_estimate)),
        "node_encounters": np.asarray(jax.device_get(state.node_encounters)),
    }

    walkers = jax.device_get(state.walkers)
    if isinstance(walkers, tuple):
        for i, walker in enumerate(walkers):
            arrays[f"walkers_{i}"] = np.asarray(walker)
    else:
        arrays["walkers"] = np.asarray(walkers)

    arrays_kw = cast(dict[str, Any], arrays)
    if compressed:
        np.savez_compressed(out, **arrays_kw)
    else:
        np.savez(out, **arrays_kw)


def load_prop_state_npz(path: str | Path) -> PropState:
    """
    Load a PropState previously written by ``dump_prop_state_npz``.
    """
    data = np.load(Path(path).expanduser().resolve())

    if "walkers" in data.files:
        walkers: Any = jnp.asarray(data["walkers"])
    else:
        walker_keys = sorted(k for k in data.files if k.startswith("walkers_"))
        walkers = tuple(jnp.asarray(data[k]) for k in walker_keys)

    return PropState(
        walkers=walkers,
        weights=jnp.asarray(data["weights"]),
        overlaps=jnp.asarray(data["overlaps"]),
        rng_key=jnp.asarray(data["rng_key"]),
        pop_control_ene_shift=jnp.asarray(data["pop_control_ene_shift"]),
        e_estimate=jnp.asarray(data["e_estimate"]),
        node_encounters=jnp.asarray(data["node_encounters"]),
    )


def load_all_prop_states_npz(
    directory: str | Path,
    *,
    prefix: str = "block_state",
) -> list[PropState]:
    path = Path(directory).expanduser().resolve()
    files = sorted(path.glob(f"{prefix}_*.npz"))
    return [load_prop_state_npz(file) for file in files]


def make_block_state_logger(
    directory: str | Path,
    *,
    base_block_fn: BlockFn | None = None,
    prefix: str = "block_state",
    start_index: int = 0,
    compressed: bool = True,
) -> BlockFn:
    """
    Wrap a block function so the PropState at the end of every block is written to disk.
    """
    if base_block_fn is None:
        base_block_fn = block

    out_dir = Path(directory).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = int(start_index)
    result_spec = jax.ShapeDtypeStruct((), jnp.int32)

    def _write_state(state: PropState) -> np.int32:
        nonlocal counter
        path = out_dir / f"{prefix}_{counter:05d}.npz"
        dump_prop_state_npz(state, path, compressed=compressed)
        counter += 1
        return np.int32(0)

    def logging_block_fn(
        state: PropState,
        *,
        sys: System,
        params: Any,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ops: Any,
        prop_ctx: Any,
        sr_fn: SrFn = wk.stochastic_reconfiguration,
        observable_names: tuple[str, ...] = (),
    ) -> tuple[PropState, BlockObs]:
        state, obs = base_block_fn(
            state,
            sys=sys,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ops=prop_ops,
            prop_ctx=prop_ctx,
            sr_fn=sr_fn,
            observable_names=observable_names,
        )
        _ = io_callback(_write_state, result_spec, state, ordered=True)
        return state, obs

    return logging_block_fn


def block(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: Any,
    prop_ops: PropOps,
    prop_ctx: Any,
    sr_fn: Callable = wk.stochastic_reconfiguration,
    observable_names: tuple[str, ...] = (),
) -> tuple[PropState, BlockObs]:
    """
    propagation + measurement
    """
    step = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps_new = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        walkers_new, trial_data
    )
    state = state._replace(walkers=walkers_new, overlaps=overlaps_new)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    energy_diagnostics: dict[str, jax.Array] = {}
    if meas_ops.block_energy is None:
        e_kernel = meas_ops.require_kernel(k_energy)
        e_samples = wk.vmap_chunked(
            e_kernel,
            n_chunks=params.n_chunks,
            in_axes=(0, None, None, None),
        )(state.walkers, ham_data, meas_ctx, trial_data)
        e_samples = jnp.real(e_samples)

        is_nan = ~jnp.isfinite(e_samples)
        e_samples = jnp.where(
            is_nan | (jnp.abs(e_samples - e_ref) > thresh),
            e_ref,
            e_samples,
        )

        weights = jnp.where(is_nan, 0.0, state.weights)
        w_sum = jnp.sum(weights)
        w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
        e_block = jnp.sum(weights * e_samples) / w_sum_safe
        e_block = jnp.where(w_sum == 0, e_ref, e_block)

        # Preserve the established random stream exactly for existing trials.
        key_next, key_sr = jax.random.split(state.rng_key)
    else:
        key_next, key_energy, key_sr = jax.random.split(state.rng_key, 3)
        weights = state.weights
        w_sum = jnp.sum(weights)
        w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
        energy_estimate = meas_ops.block_energy(
            state.walkers,
            weights,
            state.overlaps,
            key_energy,
            params.n_chunks,
            ham_data,
            meas_ctx,
            trial_data,
            e_ref,
            thresh,
        )
        if isinstance(energy_estimate, BlockEnergyEstimate):
            e_block = jnp.real(energy_estimate.energy)
            energy_diagnostics = dict(energy_estimate.diagnostics)
        else:
            e_block = jnp.real(energy_estimate)
        is_bad = (~jnp.isfinite(e_block)) | (jnp.abs(e_block - e_ref) > thresh)
        e_block = jnp.where((w_sum == 0) | is_bad, e_ref, e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
    state = state._replace(
        weights=weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * e_block,
    )

    obs_samples: dict[str, jax.Array] = {}
    for name in observable_names:
        kernel = meas_ops.require_observable(name)
        samples = wk.vmap_chunked(kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
            state.walkers, ham_data, meas_ctx, trial_data
        )
        w_shape = (weights.shape[0],) + (1,) * max(samples.ndim - 1, 0)
        num = jnp.sum(weights.reshape(w_shape) * samples, axis=0)
        zero = jnp.zeros_like(num)
        obs_samples[name] = jnp.where(w_sum == 0, zero, num / w_sum_safe)

    zeta = jax.random.uniform(key_sr)
    w_sr, weights_sr = sr_fn(state.walkers, state.weights, zeta, sys.walker_kind)
    overlaps_sr = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        w_sr, trial_data
    )
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key_next,
    )

    scalars = {"energy": e_block, "weight": w_sum}
    scalars.update(energy_diagnostics)
    obs = BlockObs(scalars=scalars, observables=obs_samples)
    return state, obs


def block_mixed_estimator(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_meas_ctx: Any,
    guide_prop_ops: PropOps,
    guide_prop_ctx: Any,
    estimator_data: Any,
    estimator_ops: EstimatorOps,
    estimator_ctx: Any,
    sr_fn: Callable = wk.stochastic_reconfiguration,
) -> tuple[PropState, BlockObs]:
    """Propagate with one guide and accumulate another estimator's statistics.

    An estimator with ``use_for_population_control=True`` also supplies the
    real block energy used to update ``state.e_estimate``. Its already computed
    block components are reused; the guide energy kernel is not evaluated.
    """

    step = lambda st: guide_prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=guide_data,
        trial_ops=guide_ops,
        meas_ops=guide_meas_ops,
        prop_ctx=guide_prop_ctx,
        meas_ctx=guide_meas_ctx,
    )

    def scan_step(carry: PropState, _x: Any):
        return step(carry), None

    state, _ = lax.scan(scan_step, state, xs=None, length=params.n_prop_steps)
    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    guide_overlaps = wk.vmap_chunked(
        guide_meas_ops.overlap,
        n_chunks=params.n_chunks,
        in_axes=(0, None),
    )(walkers_new, guide_data)
    state = state._replace(walkers=walkers_new, overlaps=guide_overlaps)

    threshold = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    energy_reference = state.e_estimate
    guide_diagnostics: dict[str, jax.Array] = {}
    use_estimator_energy = estimator_ops.use_for_population_control
    if use_estimator_energy:
        # The estimator components below supply the population-control energy.
        # Do not evaluate a separate guide local energy: for sampled PT
        # estimators that would reintroduce the deterministic all-Cholesky
        # contraction that population-level sampling is intended to avoid.
        guide_weights = state.weights
        guide_weight = jnp.sum(guide_weights)
        if estimator_ops.block_components is None:
            key_next, key_sr = jax.random.split(state.rng_key)
            key_estimator = None
        else:
            key_next, key_estimator, key_sr = jax.random.split(state.rng_key, 3)
    elif guide_meas_ops.block_energy is None:
        guide_energy_kernel = guide_meas_ops.require_kernel(k_energy)
        guide_energy_samples = wk.vmap_chunked(
            guide_energy_kernel,
            n_chunks=params.n_chunks,
            in_axes=(0, None, None, None),
        )(state.walkers, ham_data, guide_meas_ctx, guide_data)
        guide_energy_samples = jnp.real(guide_energy_samples)
        invalid_guide = ~jnp.isfinite(guide_energy_samples)
        guide_energy_samples = jnp.where(
            invalid_guide
            | (jnp.abs(guide_energy_samples - energy_reference) > threshold),
            energy_reference,
            guide_energy_samples,
        )
        guide_weights = jnp.where(invalid_guide, 0.0, state.weights)
        guide_weight = jnp.sum(guide_weights)
        guide_weight_safe = jnp.where(guide_weight == 0.0, 1.0, guide_weight)
        guide_energy = jnp.sum(guide_weights * guide_energy_samples) / guide_weight_safe
        guide_energy = jnp.where(guide_weight == 0.0, energy_reference, guide_energy)
        if estimator_ops.block_components is None:
            key_next, key_sr = jax.random.split(state.rng_key)
            key_estimator = None
        else:
            key_next, key_estimator, key_sr = jax.random.split(state.rng_key, 3)
    else:
        if estimator_ops.block_components is None:
            key_next, key_energy, key_sr = jax.random.split(state.rng_key, 3)
            key_estimator = None
        else:
            key_next, key_energy, key_estimator, key_sr = jax.random.split(
                state.rng_key, 4
            )
        guide_weights = state.weights
        guide_weight = jnp.sum(guide_weights)
        estimate = guide_meas_ops.block_energy(
            state.walkers,
            guide_weights,
            state.overlaps,
            key_energy,
            params.n_chunks,
            ham_data,
            guide_meas_ctx,
            guide_data,
            energy_reference,
            threshold,
        )
        if isinstance(estimate, BlockEnergyEstimate):
            guide_energy = jnp.real(estimate.energy)
            guide_diagnostics = {
                f"guide_{name}": value for name, value in estimate.diagnostics.items()
            }
        else:
            guide_energy = jnp.real(estimate)
        invalid_energy = (~jnp.isfinite(guide_energy)) | (
            jnp.abs(guide_energy - energy_reference) > threshold
        )
        guide_energy = jnp.where(
            (guide_weight == 0.0) | invalid_energy,
            energy_reference,
            guide_energy,
        )

    reference_overlaps = wk.vmap_chunked(
        estimator_ops.reference_overlap,
        n_chunks=params.n_chunks,
        in_axes=(0, None),
    )(state.walkers, estimator_data)
    overlap_ratio = reference_overlaps / guide_overlaps
    finite_ratio = jnp.isfinite(overlap_ratio)
    candidate_weights = jnp.where(finite_ratio, guide_weights * overlap_ratio, 0.0)
    estimator_diagnostics: dict[str, jax.Array] = {}
    if estimator_ops.block_components is None:
        component_samples = wk.vmap_chunked(
            estimator_ops.components,
            n_chunks=params.n_chunks,
            in_axes=(0, None, None, None),
        )(state.walkers, ham_data, estimator_ctx, estimator_data)
        if component_samples.ndim != 2 or component_samples.shape[1] != len(
            estimator_ops.component_names
        ):
            raise ValueError(
                "Estimator component kernel must return one vector per walker with "
                f"length {len(estimator_ops.component_names)}; got {component_samples.shape}."
            )
        finite_components = jnp.all(jnp.isfinite(component_samples), axis=1)
        valid_estimator = finite_components & finite_ratio
        estimator_weights = jnp.where(valid_estimator, candidate_weights, 0.0)
        safe_components = jnp.where(valid_estimator[:, None], component_samples, 0.0)
        estimator_weight = jnp.sum(estimator_weights)
        estimator_numerator = jnp.sum(
            estimator_weights[:, None] * safe_components,
            axis=0,
        )
    else:
        assert key_estimator is not None
        estimate = estimator_ops.block_components(
            state.walkers,
            candidate_weights,
            key_estimator,
            params.n_chunks,
            ham_data,
            estimator_ctx,
            estimator_data,
        )
        if not isinstance(estimate, BlockComponentEstimate):
            raise TypeError(
                "Estimator block_components must return a BlockComponentEstimate."
            )
        if estimate.numerator.shape != (len(estimator_ops.component_names),):
            raise ValueError(
                "Estimator block-component numerator must have shape "
                f"{(len(estimator_ops.component_names),)}, got {estimate.numerator.shape}."
            )
        estimator_weight = estimate.weight
        estimator_numerator = estimate.numerator
        estimator_diagnostics = {
            f"estimator_{name}": value for name, value in estimate.diagnostics.items()
        }

    estimator_weight_safe = jnp.where(estimator_weight == 0.0, 1.0, estimator_weight)
    estimator_components = estimator_numerator / estimator_weight_safe
    estimator_components = jnp.where(
        estimator_weight == 0.0,
        jnp.zeros_like(estimator_components),
        estimator_components,
    )

    if use_estimator_energy:
        guide_energy = jnp.real(
            estimator_ops.combine_energy(ham_data.h0, estimator_components)
        )
        invalid_energy = (~jnp.isfinite(guide_energy)) | (
            jnp.abs(guide_energy - energy_reference) > threshold
        )
        guide_energy = jnp.where(
            (guide_weight == 0.0)
            | (estimator_weight == 0.0)
            | invalid_energy,
            energy_reference,
            guide_energy,
        )

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(guide_energy))
    state = state._replace(
        weights=guide_weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * guide_energy,
    )

    zeta = jax.random.uniform(key_sr)
    walkers_sr, weights_sr = sr_fn(
        state.walkers,
        state.weights,
        zeta,
        sys.walker_kind,
    )
    overlaps_sr = wk.vmap_chunked(
        guide_meas_ops.overlap,
        n_chunks=params.n_chunks,
        in_axes=(0, None),
    )(walkers_sr, guide_data)
    state = state._replace(
        walkers=walkers_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key_next,
    )

    scalars = {
        "guide_energy": guide_energy,
        "guide_weight": guide_weight,
        "estimator_weight": estimator_weight,
        "estimator_components": estimator_components,
    }
    scalars.update(guide_diagnostics)
    scalars.update(estimator_diagnostics)
    return state, BlockObs(scalars=scalars, observables={})


def block_fp(
    state: PropState,
    *,
    sys: System,
    params: QmcParamsFp,
    ham_data: Any,
    trial_data: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: Any,
    prop_ops: PropOps,
    prop_ctx: Any,
    sr_fn: Callable = wk.stochastic_reconfiguration,
    observable_names: tuple[str, ...] = (),
) -> tuple[PropState, BlockObs]:
    """
    propagation + measurement
    """
    assert params.n_prop_steps % params.n_qr_blocks == 0

    step_fp = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step_fp(carry)
        return carry, None

    def _qr_blocks(state: PropState, _x: Any):
        state, _ = lax.scan(
            _scan_step, state, xs=None, length=params.n_prop_steps // params.n_qr_blocks
        )
        wk_kind = sys.walker_kind.lower()
        q, norms = wk.orthogonalize(state.walkers, wk_kind)
        weights_new = state.weights * norms.real
        state = state._replace(weights=weights_new, walkers=q)
        return state, None

    # state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)
    state, _ = lax.scan(_qr_blocks, state, xs=None, length=params.n_qr_blocks)

    overlaps_new = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        state.walkers, trial_data
    )
    state = state._replace(overlaps=overlaps_new)
    e_kernel = meas_ops.require_kernel(k_energy)
    e_samples = wk.vmap_chunked(e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
        state.walkers, ham_data, meas_ctx, trial_data
    )

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))

    ene0 = params.ene0
    assert ene0 is not None
    e_samples = jnp.where(jnp.abs(e_samples - ene0) > thresh, ene0, e_samples)
    e_samples = jnp.array(e_samples)

    weights = state.weights
    overlaps = state.overlaps
    w_sum = jnp.sum(weights * overlaps)
    e_block = jnp.sum(weights * overlaps * e_samples) / w_sum
    ov = jnp.sum(overlaps)
    abs_ov = jnp.sum(jnp.abs(overlaps))

    obs = BlockObs(
        scalars={"energy": e_block, "weight": w_sum, "overlap": ov, "abs_overlap": abs_ov},
        observables={},
    )
    return state, obs


@tree_util.register_pytree_node_class
class MlmcMeasCtx(NamedTuple):
    """
    prop_meas_ctx:
      The meas_ctx used by prop_ops.step (force bias etc). Typically "full" (untruncated).

    packs:
      Tuple of LevelPack's, ordered low to high, used only for energy evaluation.

    m_deltas:
      Tuple of fixed subsample sizes for each increment delta_l = E_l - E_{l-1}.
      Must have length len(packs) - 1.
      Each m must be int and <= n_walkers.
    """

    prop_meas_ctx: Any
    packs: tuple[LevelPack, ...]
    m_deltas: tuple[int, ...]

    def tree_flatten(self):
        # children: pytrees with arrays
        children = (self.prop_meas_ctx, self.packs)
        # aux: static python metadata
        aux = (self.m_deltas,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (m_deltas,) = aux
        prop_meas_ctx, packs = children
        return cls(prop_meas_ctx=prop_meas_ctx, packs=packs, m_deltas=m_deltas)


def block_mlmc(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    meas_ctx: MlmcMeasCtx,
    prop_ops: PropOps,
    prop_ctx: Any,
    sr_fun: Callable = wk.stochastic_reconfiguration,
    observable_names: tuple[str, ...] = (),
) -> tuple[PropState, BlockObs]:
    """
    propagation + MLMC measurement

    sum_i w_i E0(i) + sum_{l>=1} N/m_l sum_{i in S_l} w_i [E_l(i) - E_{l-1}(i)] / sum_i w_i
    """
    prop_meas_ctx = meas_ctx.prop_meas_ctx

    step = lambda st: prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ctx=prop_ctx,
        meas_ctx=prop_meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        carry = step(carry)
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps_new = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        walkers_new, trial_data
    )
    state = state._replace(walkers=walkers_new, overlaps=overlaps_new)

    # --- MLMC measurement ---
    packs = meas_ctx.packs
    m_deltas = meas_ctx.m_deltas
    if len(packs) < 1:
        raise ValueError("MlmcMeasCtx.packs must contain at least one level pack.")
    if len(m_deltas) != max(0, len(packs) - 1):
        raise ValueError("MlmcMeasCtx.m_deltas must have length len(packs)-1.")

    e_kernel = meas_ops.require_kernel(k_energy)
    energy_vmapped = wk.vmap_chunked(
        e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )

    e_ref = state.e_estimate

    # baseline level (0): evaluate on all walkers
    p0 = packs[0]
    walkers0 = wk.slice_walkers(state.walkers, sys.walker_kind, p0.norb_keep)
    e0 = energy_vmapped(walkers0, p0.ham_data, p0.meas_ctx, p0.trial_data)
    e0 = jnp.real(e0)
    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    is_nan = ~jnp.isfinite(e0)
    e0 = jnp.where(is_nan | (jnp.abs(e0 - e_ref) > thresh), e_ref, e0)

    weights = jnp.where(is_nan, 0.0, state.weights)
    n_walkers = int(weights.shape[0])
    w_sum = jnp.sum(weights)
    w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)

    num0 = jnp.sum(weights * e0)
    e0_block = jnp.where(w_sum == 0, e_ref, num0 / w_sum_safe)

    # corrections: telescope with fixed-size subsamples
    # independent subsets per level increment
    key_next, key_sel, key_sr = jax.random.split(state.rng_key, 3)
    if len(m_deltas) > 0:
        keys = jax.random.split(key_sel, len(m_deltas))
    else:
        keys = jnp.zeros((0, 2), dtype=key_sel.dtype)  # unused

    num_corr = jnp.array(0.0, dtype=jnp.result_type(num0))
    delta_blocks = []  # for optional diagnostics

    for ell in range(1, len(packs)):
        m = int(m_deltas[ell - 1])
        if m <= 0:
            delta_blocks.append(jnp.array(0.0, dtype=jnp.result_type(num0)))
            continue

        # sample subset indices
        idx = jax.random.choice(keys[ell - 1], n_walkers, shape=(m,), replace=False)

        # evaluate E_ell and E_{ell-1} on same subset
        phi = wk.take_walkers(state.walkers, idx)
        w_sub = weights[idx]

        phi_hi = wk.slice_walkers(phi, sys.walker_kind, packs[ell].norb_keep)
        phi_lo = wk.slice_walkers(phi, sys.walker_kind, packs[ell - 1].norb_keep)

        e_hi = energy_vmapped(
            phi_hi,
            packs[ell].ham_data,
            packs[ell].meas_ctx,
            packs[ell].trial_data,
        )
        e_lo = energy_vmapped(
            phi_lo,
            packs[ell - 1].ham_data,
            packs[ell - 1].meas_ctx,
            packs[ell - 1].trial_data,
        )

        thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
        e0 = jnp.where(~jnp.isfinite(e0) | (jnp.abs(e0 - e_ref) > thresh), e_ref, e0)
        e_hi = jnp.where(~jnp.isfinite(e_hi) | (jnp.abs(e_hi - e_ref) > thresh), e_ref, e_hi)
        e_lo = jnp.where(~jnp.isfinite(e_lo) | (jnp.abs(e_lo - e_ref) > thresh), e_ref, e_lo)

        delta = jnp.array(e_hi) - jnp.array(e_lo)

        # unbiased estimator of sum_i w_i delta_i via Horvitz–Thompson scaling
        scale = jnp.asarray(n_walkers / m, dtype=jnp.result_type(num0))
        num_inc_hat = scale * jnp.sum(w_sub * delta)

        num_corr = num_corr + jnp.real(num_inc_hat)
        delta_blocks.append(jnp.where(w_sum == 0, 0.0, jnp.real(num_inc_hat) / w_sum_safe))

    num_total = num0 + num_corr
    e_mlmc_block = jnp.where(w_sum == 0, e_ref, num_total / w_sum_safe)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_mlmc_block))
    state = state._replace(
        weights=weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * e_mlmc_block,
    )

    zeta = jax.random.uniform(key_sr)
    w_sr, weights_sr = sr_fun(state.walkers, state.weights, zeta, sys.walker_kind)
    overlaps_sr = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
        w_sr, trial_data
    )
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key_next,
    )

    obs = BlockObs(
        scalars={
            "energy": e_mlmc_block,
            "energy_base": e0_block,  # optional: for debugging/monitoring
            "weight": w_sum,
            "mlmc_delta": (
                jnp.asarray(delta_blocks)
                if delta_blocks
                else jnp.zeros((0,), dtype=jnp.result_type(num0))
            ),
        },
        observables={},
    )
    return state, obs
