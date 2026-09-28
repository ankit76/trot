from __future__ import annotations

import dataclasses
import math
import time
from functools import partial
from typing import Any, Callable, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import walkers as wk
from .core.ops import (
    EstimatorOps,
    MeasOps,
    TrialOps,
    d_energy_head_guard_count,
    d_energy_head_guard_weight,
    d_energy_sampling_noise,
    d_energy_walker_guide_ess,
    d_energy_walker_guide_max_correction,
    d_pt_component_sampling_noise_real,
)
from .core.system import System
from .meas.pt2ccsd import get_init_pt2trial_energy
from .pair_sampling_runtime import prepare_local_cholesky_sampling
from .prop.blocks import BlockFn, MixedBlockFn, MixedEstimatorBlockFn
from .prop.types import PropOps, PropState, QmcParams, QmcParamsBase, QmcParamsFp
from .runtime_layout import QmcRuntime
from .stat_utils import (
    blocking_analysis_components,
    blocking_analysis_ratio,
    clean_pt2ccsd,
    component_estimator_outlier_mask,
    gamma_analysis_components,
    gamma_analysis_ratio,
    jackknife_ratios,
    pt2ccsd_blocking,
    rebin_observable,
    reject_outliers,
)
from .walkers import stochastic_reconfiguration

print = partial(print, flush=True)

# Leave headroom for live buffers and execution overhead outside the compiled
# memory estimate; 95% admitted a 5Z batch that failed at its first execution.
_AUTO_CHUNK_MEMORY_FRACTION = 0.85
_COMPILER_MEMORY_ERROR_MARKERS = (
    "resource_exhausted",
    "resource exhausted",
    "out of memory",
    "failed to allocate",
    # GPU GEMM autotuning can report this after every candidate kernel fails
    # to produce a reference result because its buffers could not be allocated.
    "no reference output found",
    # Some XLA GPU autotuners discard every candidate after allocation
    # failures and only expose this generic final error to Python.
    "no valid config found",
)


class QmcResult(NamedTuple):
    """AFQMC estimates, block histories, observables, and raw diagnostics.

    ``stderr_energy`` is selected by ``error_method``.  Standard importance-
    sampled runs also retain the Gamma-method and blocking estimates and their
    reliability diagnostics in the explicitly named fields below.

    ``block_diagnostics`` contains production-block estimator diagnostics. It
    is intentionally not filtered by the energy outlier mask, so rare sampling
    excursions remain visible to validation code.
    """

    mean_energy: jax.Array | float
    stderr_energy: jax.Array | float
    stderr_gamma: float | None
    stderr_blocking: float | None
    error_method: str
    error_reliable: bool
    gamma_window: int | None
    gamma_window_found: bool
    tau_int: float | None
    effective_sample_size: float | None
    gamma_warnings: tuple[str, ...]
    blocking_B_star: int | None
    blocking_plateau_found: bool
    blocking_selection_reason: str
    block_energies: jax.Array
    block_weights: jax.Array
    block_observables: dict[str, jax.Array]
    observable_means: dict[str, jax.Array]
    observable_stderrs: dict[str, jax.Array]
    block_diagnostics: dict[str, jax.Array]


class EnergyErrorAnalysis(NamedTuple):
    """Primary energy error plus the two underlying analyses."""

    mean: float
    stderr: float
    error_method: str
    reliable: bool
    gamma: dict[str, Any]
    blocking: dict[str, Any]


def _analyze_energy_errors(
    energies: Any,
    weights: Any,
    *,
    error_method: str,
) -> EnergyErrorAnalysis:
    """Evaluate blocking and Gamma errors without silently substituting either."""

    if error_method not in ("gamma", "blocking"):
        raise ValueError(
            "error_method must be either 'gamma' or 'blocking'; " f"received {error_method!r}"
        )

    blocking = blocking_analysis_ratio(energies, weights, print_q=False)
    n_samples = int(np.asarray(energies).size)
    if n_samples < 4:
        gamma = {
            "mu": float(blocking["mu"]),
            "se_gamma": float("nan"),
            "window": None,
            "window_found": False,
            "tau_int": None,
            "effective_sample_size": None,
            "reliable": False,
            "warnings": ("Gamma analysis requires at least four samples",),
        }
    else:
        gamma = gamma_analysis_ratio(energies, weights, print_q=False)
        if not np.isclose(
            float(blocking["mu"]),
            float(gamma["mu"]),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "blocking and Gamma analyses produced different weighted means: "
                f"{blocking['mu']} versus {gamma['mu']}"
            )

    if error_method == "gamma":
        stderr = float(gamma["se_gamma"])
        reliable = bool(gamma["reliable"])
    else:
        blocking_se = blocking["se_star"]
        stderr = float(blocking_se) if blocking_se is not None else float("nan")
        reliable = bool(blocking["plateau_found"]) and np.isfinite(stderr)

    return EnergyErrorAnalysis(
        mean=float(blocking["mu"]),
        stderr=stderr,
        error_method=error_method,
        reliable=reliable,
        gamma=gamma,
        blocking=blocking,
    )


def _analyze_component_estimator_errors(
    h0: Any,
    weights: Any,
    components: Any,
    combine_energy: Callable[[Any, Any], Any],
    *,
    error_method: str,
) -> EnergyErrorAnalysis:
    """Evaluate blocking and Gamma errors for a nonlinear component estimator."""

    if error_method not in ("gamma", "blocking"):
        raise ValueError(
            "error_method must be either 'gamma' or 'blocking'; "
            f"received {error_method!r}"
        )

    blocking = blocking_analysis_components(
        h0,
        weights,
        components,
        combine_energy,
        print_q=False,
    )
    n_samples = int(np.asarray(weights).size)
    if n_samples < 4:
        gamma = {
            "mu": float(blocking["mu"]),
            "mean_components": blocking["mean_components"],
            "se_gamma": float("nan"),
            "window": None,
            "window_found": False,
            "tau_int": None,
            "effective_sample_size": None,
            "reliable": False,
            "warnings": ("Gamma analysis requires at least four samples",),
        }
    else:
        gamma = gamma_analysis_components(
            h0,
            weights,
            components,
            combine_energy,
            print_q=False,
        )
        if not np.isclose(
            float(blocking["mu"]),
            float(gamma["mu"]),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "blocking and Gamma component analyses produced different means: "
                f"{blocking['mu']} versus {gamma['mu']}"
            )

    if error_method == "gamma":
        stderr = float(gamma["se_gamma"])
        reliable = bool(gamma["reliable"])
    else:
        blocking_se = blocking["se_star"]
        stderr = float(blocking_se) if blocking_se is not None else float("nan")
        reliable = bool(blocking["plateau_found"]) and np.isfinite(stderr)

    return EnergyErrorAnalysis(
        mean=float(blocking["mu"]),
        stderr=stderr,
        error_method=error_method,
        reliable=reliable,
        gamma=gamma,
        blocking=blocking,
    )


def _print_energy_error_analysis(analysis: EnergyErrorAnalysis) -> None:
    """Print the compact final summary for both energy-error estimators."""

    gamma = analysis.gamma
    blocking = analysis.blocking
    gamma_label = "Gamma SE [primary]" if analysis.error_method == "gamma" else "Gamma SE"
    blocking_label = (
        "Blocking SE [primary]"
        if analysis.error_method == "blocking"
        else "Blocking SE [diagnostic]"
    )
    gamma_se = float(gamma["se_gamma"])
    blocking_se = blocking["se_star"]
    gamma_se_text = f"{gamma_se:.6e}" if np.isfinite(gamma_se) else "unavailable"
    blocking_se_text = f"{float(blocking_se):.6e}" if blocking_se is not None else "unavailable"
    tau = gamma["tau_int"]
    n_eff = gamma["effective_sample_size"]

    print(f"  mean                    = {analysis.mean:.12f}")
    print(f"  {gamma_label:<25s}= {gamma_se_text}")
    print(f"  Gamma reliable          = {'yes' if gamma['reliable'] else 'no'}")
    print(f"  Gamma window W          = {gamma['window']}")
    print(f"  tau_int                 = {tau if tau is not None else 'unavailable'}")
    print(f"  effective samples       = {n_eff if n_eff is not None else 'unavailable'}")
    print(f"  {blocking_label:<25s}= {blocking_se_text}")
    print(f"  Blocking B              = {blocking['B_star']}")
    print(
        "  Blocking selection      = "
        f"{blocking['selection_reason']} "
        f"(plateau_found={blocking['plateau_found']})"
    )
    print(
        f"  Reported method         = {analysis.error_method} "
        f"(reliable={'yes' if analysis.reliable else 'no'})"
    )
    for warning in gamma["warnings"]:
        print(f"  Gamma warning           = {warning}")


class MixedQmcResult(NamedTuple):
    # currently only support pt2CCSD
    guide_mean_energy: jax.Array
    guide_stderr_energy: jax.Array
    guide_block_energies: jax.Array
    guide_block_weights: jax.Array
    trial_mean_energy: jax.Array
    trial_stderr_energy: jax.Array
    trial_block_weights: jax.Array
    trial_block_t2s: jax.Array
    trial_block_e0s: jax.Array
    trial_block_e1s: jax.Array


class MixedEstimatorQmcResult(NamedTuple):
    """Guide and projected-estimator results from one AFQMC trajectory.

    For backward compatibility, the ``guide_*energy*`` fields hold the energy
    series used for population control. When
    ``EstimatorOps.use_for_population_control`` is enabled, that series is the
    combined estimator energy rather than a separately evaluated guide energy.
    Projected-estimator block arrays are always raw. The proxy energies and
    keep mask record the robust cleanup used for the reported estimator result.
    """

    guide_mean_energy: float
    guide_stderr_energy: float
    estimator_mean_energy: float
    estimator_stderr_energy: float
    estimator_mean_components: jax.Array
    estimator_component_names: tuple[str, ...]
    guide_block_energies: jax.Array
    guide_block_weights: jax.Array
    estimator_block_weights: jax.Array
    estimator_block_components: jax.Array
    estimator_block_proxy_energies: jax.Array
    estimator_block_keep_mask: jax.Array
    guide_block_diagnostics: dict[str, jax.Array]
    final_state: PropState
    guide_analysis: EnergyErrorAnalysis
    estimator_analysis: dict[str, Any]


def _weighted_block_mean(values: jax.Array, weights: jax.Array) -> jax.Array:
    w_sum = jnp.sum(weights)
    w_shape = (weights.shape[0],) + (1,) * max(values.ndim - 1, 0)
    num = jnp.sum(weights.reshape(w_shape) * values, axis=0)
    zero = jnp.zeros_like(num)
    return jnp.where(w_sum == 0, zero, num / w_sum)


def _initial_projected_estimator(
    state: PropState,
    *,
    ham_data: Any,
    estimator_data: Any,
    estimator_ops: EstimatorOps,
    estimator_ctx: Any,
) -> tuple[float, float]:
    """Evaluate the projected estimator on the representative initial walker.

    AFQMC initialization broadcasts one determinant across the walker
    population. Evaluating that representative walker therefore gives the
    exact block-zero energy while avoiding an all-walker deterministic PT
    contraction when production uses population-level component sampling.
    """

    walker_0 = wk.take_walkers(state.walkers, jnp.asarray([0]))
    components_0 = wk.vmap_chunked(
        estimator_ops.components,
        n_chunks=1,
        in_axes=(0, None, None, None),
    )(walker_0, ham_data, estimator_ctx, estimator_data)[0]
    energy_0 = estimator_ops.combine_energy(ham_data.h0, components_0)

    reference_overlap_0 = wk.vmap_chunked(
        estimator_ops.reference_overlap,
        n_chunks=1,
        in_axes=(0, None),
    )(walker_0, estimator_data)[0]
    estimator_weight_0 = jnp.sum(state.weights) * reference_overlap_0 / state.overlaps[0]

    return (
        float(np.real(np.asarray(energy_0).reshape(()))),
        float(np.real(np.asarray(estimator_weight_0).reshape(()))),
    )


def make_run_blocks(
    *,
    block_fn: BlockFn,
    sys: System,
    params: QmcParamsBase,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    prop_ops: PropOps,
    observable_names: tuple[str, ...] = (),
) -> Callable:
    """
    Build a jitted run_blocks.
    We keep ham_data, trial_data, meas_ctx, prop_ctx as arguments to
    improve compilation, as these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_blocks(
        state0,
        *,
        ham_data,
        trial_data,
        meas_ctx,
        prop_ctx,
        n_blocks: int,
    ):
        def one_block(state, _):
            state, obs = block_fn(
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
                observable_names=observable_names,
            )
            obs_tuple = tuple(obs.observables[name] for name in observable_names)
            return state, (obs.scalars, obs_tuple)

        stateN, (scalars, obs) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, scalars, obs

    return run_blocks


def _auto_chunk_probe_n_blocks(params: QmcParams) -> int:
    """Largest static block batch used by the standard progress loops."""

    eql_chunk = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 1
    sampling_chunk = params.n_blocks // 10 if params.n_blocks >= 10 else 1
    return max(1, eql_chunk, sampling_chunk)


def _state_devices(state: PropState) -> tuple[jax.Device, ...]:  # pyright: ignore
    devices: set[jax.Device] = set()  # pyright: ignore
    for leaf in jax.tree_util.tree_leaves(state):
        leaf_devices = getattr(leaf, "devices", None)
        if callable(leaf_devices):
            devices.update(cast(Any, leaf_devices)())
    if not devices:
        devices.update(jax.devices())  # pyright: ignore
    return tuple(devices)


def _device_memory_limit_bytes(state: PropState) -> int | None:
    """Return the smallest reported allocator limit across state devices."""

    limits: list[int] = []
    for device in _state_devices(state):
        try:
            stats = cast(dict[str, int] | None, device.memory_stats())
        except (RuntimeError, NotImplementedError):
            stats = None
        if not stats:
            return None

        # ``bytes_limit`` is reported by current GPU backends. Keep the
        # fallbacks because memory-stat keys are explicitly backend-dependent.
        value = next(
            (stats[key] for key in ("bytes_limit", "memory_limit", "total_memory") if key in stats),
            None,
        )
        if value is None or int(value) <= 0:
            return None
        limits.append(int(value))
    return min(limits) if limits else None


def _compiled_memory_bytes(compiled: Any) -> int | None:
    """Total compiler-estimated bytes, avoiding aliased double-counting."""

    stats = compiled.memory_analysis()
    if stats is None:
        return None

    names = (
        "temp_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
    )
    values = tuple(getattr(stats, name, None) for name in names)
    if any(value is None for value in values):
        return None
    alias = getattr(stats, "alias_size_in_bytes", 0)
    return max(0, sum(int(cast(Any, value)) for value in values) - int(alias))


def _next_auto_n_chunks(
    current: int,
    estimated_bytes: int,
    budget_bytes: int,
    n_walkers: int,
) -> int:
    """Increase chunks using the observed memory excess as a lower bound."""

    scaled = math.ceil(current * estimated_bytes / budget_bytes)
    return min(n_walkers, max(current + 1, scaled))


def _format_mib(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.1f} MiB"


def _is_compiler_memory_error(
    exc: jax.errors.JaxRuntimeError,  # pyright: ignore[reportInvalidTypeForm]
) -> bool:
    """Recognize recoverable compiler/autotuner memory failures."""

    message = str(exc).lower()
    return any(marker in message for marker in _COMPILER_MEMORY_ERROR_MARKERS)


def _make_run_blocks_with_auto_chunks(
    *,
    block_fn: BlockFn,
    sys: System,
    params: QmcParams,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    prop_ops: PropOps,
    state: PropState,
    ham_data: Any,
    trial_data: Any,
    meas_ctx: Any,
    prop_ctx: Any,
    observable_names: tuple[str, ...],
) -> tuple[QmcParams, Any]:
    """Build ``run_blocks`` and optionally select a safe walker chunk count.

    The accepted candidate is the same jitted callable returned to the QMC
    loop, so its ahead-of-time compilation is reused during execution.
    """

    def build(candidate_params: QmcParams) -> Any:
        return make_run_blocks(
            block_fn=block_fn,
            sys=sys,
            params=candidate_params,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            observable_names=observable_names,
        )

    if not params.auto_n_chunks:
        return params, build(params)
    if params.n_chunks <= 0:
        raise ValueError("QmcParams.n_chunks must be positive.")

    memory_limit = _device_memory_limit_bytes(state)
    if memory_limit is None:
        print(
            "[chunks] automatic selection unavailable: the backend did not "
            "report a device memory limit; using "
            f"n_chunks={params.n_chunks}."
        )
        return params, build(params)

    budget = int(_AUTO_CHUNK_MEMORY_FRACTION * memory_limit)
    if budget <= 0:
        raise RuntimeError("Automatic chunk memory budget is not positive.")

    n_walkers = max(
        1, wk.n_local_walkers(state.walkers) if state is not None else int(params.n_walkers)
    )
    candidate = min(int(params.n_chunks), n_walkers)
    probe_n_blocks = _auto_chunk_probe_n_blocks(params)
    print(
        "[chunks] selecting n_chunks automatically: "
        f"allocator_limit={_format_mib(memory_limit)}, "
        f"budget={_format_mib(budget)}, probe_blocks={probe_n_blocks}, "
        f"walkers_per_data_shard={n_walkers}."
    )

    while True:
        candidate_params = dataclasses.replace(params, n_chunks=candidate)
        run_blocks = build(candidate_params)
        start = time.perf_counter()
        try:
            compiled = run_blocks.lower(
                state,
                ham_data=ham_data,
                trial_data=trial_data,
                meas_ctx=meas_ctx,
                prop_ctx=prop_ctx,
                n_blocks=probe_n_blocks,
            ).compile()
        except jax.errors.JaxRuntimeError as exc:
            if not _is_compiler_memory_error(exc):
                raise
            compile_seconds = time.perf_counter() - start
            if candidate >= n_walkers:
                raise MemoryError(
                    "The one-walker chunk failed to compile because the GPU "
                    "compiler or autotuner ran out of memory. Increase internal "
                    "estimator chunking or use a device with more memory."
                ) from exc
            next_candidate = min(n_walkers, 2 * candidate)
            print(
                f"[chunks] n_chunks={candidate}: compiler/autotuner memory failure "
                f"after {compile_seconds:.1f} s; retrying with "
                f"n_chunks={next_candidate}."
            )
            candidate = next_candidate
            continue
        compile_seconds = time.perf_counter() - start
        estimated_bytes = _compiled_memory_bytes(compiled)
        if estimated_bytes is None:
            print(
                "[chunks] compiler memory analysis unavailable; using "
                f"n_chunks={candidate} (compiled in {compile_seconds:.1f} s)."
            )
            return candidate_params, run_blocks

        print(
            f"[chunks] n_chunks={candidate}: compiler estimate "
            f"{_format_mib(estimated_bytes)} "
            f"(compiled in {compile_seconds:.1f} s)."
        )
        if estimated_bytes <= budget:
            print(f"[chunks] selected n_chunks={candidate}.")
            return candidate_params, run_blocks

        if candidate >= n_walkers:
            raise MemoryError(
                "The compiler estimates that a one-walker chunk requires "
                f"{_format_mib(estimated_bytes)}, exceeding the automatic "
                f"budget of {_format_mib(budget)}. Increase internal estimator "
                "chunking or use a device with more memory."
            )
        candidate = _next_auto_n_chunks(
            candidate,
            estimated_bytes,
            budget,
            n_walkers,
        )


def make_run_mixed_blocks(
    *,
    mixed_block_fn: MixedBlockFn,
    sys: System,
    params: QmcParams,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_prop_ops: PropOps,
    trial_meas_ops: MeasOps,
    observable_names: tuple[str, ...] = (),
) -> Callable:
    """
    Build a jitted run_blocks for mixed sampling (Trial =! Guide).
    We keep ham_data, trial_data, meas_ctx, prop_ctx as arguments to
    improve compilation, as these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_mixed_blocks(
        state0,
        *,
        ham_data,
        guide_data,
        guide_meas_ctx,
        guide_prop_ctx,
        trial_data,
        trial_meas_ctx,
        n_blocks: int,
    ):
        def one_block(state, _):
            state, obs = mixed_block_fn(
                state,
                sys=sys,
                params=params,
                ham_data=ham_data,
                guide_data=guide_data,
                guide_ops=guide_ops,
                guide_meas_ops=guide_meas_ops,
                guide_meas_ctx=guide_meas_ctx,
                guide_prop_ops=guide_prop_ops,
                guide_prop_ctx=guide_prop_ctx,
                trial_data=trial_data,
                trial_meas_ops=trial_meas_ops,
                trial_meas_ctx=trial_meas_ctx,
                observable_names=observable_names,
            )
            obs_tuple = tuple(obs.observables[name] for name in observable_names)
            return state, (obs.scalars, obs_tuple)

        stateN, (scalars, obs) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, scalars, obs

    return run_mixed_blocks


def make_run_mixed_estimator_blocks(
    *,
    mixed_block_fn: MixedEstimatorBlockFn,
    sys: System,
    params: QmcParams,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_prop_ops: PropOps,
    estimator_ops: EstimatorOps,
) -> Callable:
    """Build a jitted block scanner for guide-independent estimators."""

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_mixed_estimator_blocks(
        state0,
        *,
        ham_data,
        guide_data,
        guide_meas_ctx,
        guide_prop_ctx,
        estimator_data,
        estimator_ctx,
        n_blocks: int,
    ):
        def one_block(state, _):
            state, obs = mixed_block_fn(
                state,
                sys=sys,
                params=params,
                ham_data=ham_data,
                guide_data=guide_data,
                guide_ops=guide_ops,
                guide_meas_ops=guide_meas_ops,
                guide_meas_ctx=guide_meas_ctx,
                guide_prop_ops=guide_prop_ops,
                guide_prop_ctx=guide_prop_ctx,
                estimator_data=estimator_data,
                estimator_ops=estimator_ops,
                estimator_ctx=estimator_ctx,
            )
            return state, obs.scalars

        return lax.scan(one_block, state0, xs=None, length=n_blocks)

    return run_mixed_estimator_blocks


def _make_run_mixed_estimator_blocks_with_auto_chunks(
    *,
    mixed_block_fn: MixedEstimatorBlockFn,
    sys: System,
    params: QmcParams,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_prop_ops: PropOps,
    estimator_ops: EstimatorOps,
    state: PropState,
    ham_data: Any,
    guide_data: Any,
    guide_meas_ctx: Any,
    guide_prop_ctx: Any,
    estimator_data: Any,
    estimator_ctx: Any,
) -> tuple[QmcParams, Callable]:
    """Build the mixed scanner and optionally select walker chunking."""

    def build(candidate_params: QmcParams) -> Callable:
        return make_run_mixed_estimator_blocks(
            mixed_block_fn=mixed_block_fn,
            sys=sys,
            params=candidate_params,
            guide_ops=guide_ops,
            guide_meas_ops=guide_meas_ops,
            guide_prop_ops=guide_prop_ops,
            estimator_ops=estimator_ops,
        )

    if not params.auto_n_chunks:
        return params, build(params)
    if params.n_chunks <= 0:
        raise ValueError("QmcParams.n_chunks must be positive.")

    memory_limit = _device_memory_limit_bytes(state)
    if memory_limit is None:
        print(
            "[chunks] automatic selection unavailable: the backend did not "
            "report a device memory limit; using "
            f"n_chunks={params.n_chunks}."
        )
        return params, build(params)

    budget = int(_AUTO_CHUNK_MEMORY_FRACTION * memory_limit)
    if budget <= 0:
        raise RuntimeError("Automatic chunk memory budget is not positive.")

    n_walkers = max(
        1, wk.n_local_walkers(state.walkers) if state is not None else int(params.n_walkers)
    )
    candidate = min(int(params.n_chunks), n_walkers)
    probe_n_blocks = _auto_chunk_probe_n_blocks(params)
    print(
        "[chunks] selecting n_chunks automatically for mixed estimation: "
        f"allocator_limit={_format_mib(memory_limit)}, "
        f"budget={_format_mib(budget)}, probe_blocks={probe_n_blocks}, "
        f"walkers_per_data_shard={n_walkers}."
    )

    while True:
        candidate_params = dataclasses.replace(params, n_chunks=candidate)
        run_blocks = build(candidate_params)
        start = time.perf_counter()
        try:
            compiled = run_blocks.lower(
                state,
                ham_data=ham_data,
                guide_data=guide_data,
                guide_meas_ctx=guide_meas_ctx,
                guide_prop_ctx=guide_prop_ctx,
                estimator_data=estimator_data,
                estimator_ctx=estimator_ctx,
                n_blocks=probe_n_blocks,
            ).compile()
        except jax.errors.JaxRuntimeError as exc:
            if not _is_compiler_memory_error(exc):
                raise
            compile_seconds = time.perf_counter() - start
            if candidate >= n_walkers:
                raise MemoryError(
                    "The one-walker mixed-estimator chunk failed to compile "
                    "because the compiler or autotuner ran out of memory."
                ) from exc
            next_candidate = min(n_walkers, 2 * candidate)
            print(
                f"[chunks] n_chunks={candidate}: compiler/autotuner memory failure "
                f"after {compile_seconds:.1f} s; retrying with "
                f"n_chunks={next_candidate}."
            )
            candidate = next_candidate
            continue

        compile_seconds = time.perf_counter() - start
        estimated_bytes = _compiled_memory_bytes(compiled)
        if estimated_bytes is None:
            print(
                "[chunks] compiler memory analysis unavailable; using "
                f"n_chunks={candidate} (compiled in {compile_seconds:.1f} s)."
            )
            return candidate_params, run_blocks

        print(
            f"[chunks] n_chunks={candidate}: compiler estimate "
            f"{_format_mib(estimated_bytes)} "
            f"(compiled in {compile_seconds:.1f} s)."
        )
        if estimated_bytes <= budget:
            print(f"[chunks] selected n_chunks={candidate}.")
            return candidate_params, run_blocks
        if candidate >= n_walkers:
            raise MemoryError(
                "The compiler estimates that a one-walker mixed-estimator "
                f"chunk requires {_format_mib(estimated_bytes)}, exceeding "
                f"the automatic budget of {_format_mib(budget)}."
            )
        candidate = _next_auto_n_chunks(
            candidate,
            estimated_bytes,
            budget,
            n_walkers,
        )


def run_qmc(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any = None,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    prop_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
    observable_names: tuple[str, ...] = (),
    runtime: QmcRuntime | None = None,
) -> QmcResult:
    """
    equilibration blocks then sampling blocks.

    ``runtime`` transfers ownership of Hamiltonian/context replacements; normal
    Jobs supply it automatically. Separate array arguments are borrowed, so
    automatic Cholesky redistribution is disabled for that calling convention.

    Returns:
      QmcResult with energy statistics plus block-level observable estimates.
    """
    for name in observable_names:
        meas_ops.require_observable(name)

    if runtime is not None:
        ham_data, prop_ctx, (meas_ctx,) = runtime.inputs(
            ham_data=ham_data, prop_ctx=prop_ctx, contexts=(meas_ctx,),
            retune=meas_ops.retune_block_energy is not None,
            local_sampling=params.local_cholesky_sampling,
        )

    # build ctx
    if prop_ctx is None:
        prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            meas_ctx=meas_ctx,
            mesh=mesh,
        )

    def prepare_sampling():
        nonlocal ham_data, prop_ctx, meas_ctx
        ham_data, prop_ctx, (meas_ctx,) = prepare_local_cholesky_sampling(
            ham_data, prop_ctx, (meas_ctx,),
            enabled=params.local_cholesky_sampling, runtime=runtime,
        )

    if meas_ops.retune_block_energy is None:
        prepare_sampling()

    if mesh is None or mesh.size == 1:
        block_fn_sr = block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        block_fn_sr = partial(block_fn, sr_fn=sr_sharded)

    params, run_blocks = _make_run_blocks_with_auto_chunks(
        block_fn=block_fn_sr,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        state=state,
        ham_data=ham_data,
        trial_data=trial_data,
        meas_ctx=meas_ctx,
        prop_ctx=prop_ctx,
        observable_names=observable_names,
    )

    t0 = time.perf_counter()
    t_mark = t0

    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    block_e_eq = []
    block_w_eq = []
    block_obs_eq = {name: [] for name in observable_names}
    block_e_eq.append(state.e_estimate)
    block_w_eq.append(jnp.sum(state.weights))
    print("\nEquilibration:\n")
    if print_every:
        print(f"{'':4s}{'block':>9s}  {'E_blk':>14s}  {'W':>12s}   {'nodes':>10s}  {'t[s]':>8s}")
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{float(state.e_estimate):14.10f}  "
        f"{float(jnp.sum(state.weights)):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, scalars_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_eq.extend(scalars_chunk["energy"].tolist())
        block_w_eq.extend(scalars_chunk["weight"].tolist())
        for i, name in enumerate(observable_names):
            block_obs_eq[name].append(obs_chunk[i])
        e_chunk = scalars_chunk["energy"]
        w_chunk = scalars_chunk["weight"]
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{int(state.node_encounters):10d}  "
            f"{elapsed:8.1f}"
        )
    if meas_ops.retune_block_energy is not None:
        print("\nRetuning block-energy sampling after equilibration:")

        def advance_equilibration_blocks(state, *, n_blocks: int):
            return run_blocks(
                state,
                ham_data=ham_data,
                trial_data=trial_data,
                meas_ctx=meas_ctx,
                prop_ctx=prop_ctx,
                n_blocks=n_blocks,
            )

        retuned = meas_ops.retune_block_energy(
            state,
            jnp.asarray(block_e_eq[1:]),
            jnp.asarray(block_w_eq[1:]),
            params,
            ham_data,
            meas_ctx,
            trial_data,
            advance_blocks=advance_equilibration_blocks,
            target_error=target_error,
        )
        if retuned.initial_n_chunks <= 0:
            raise ValueError("retuned initial_n_chunks must be positive.")
        if retuned.settling_blocks < 0:
            raise ValueError("retuned settling_blocks must be nonnegative.")

        state = cast(PropState, retuned.state)
        meas_ctx = retuned.meas_ctx
        production_params = dataclasses.replace(
            params,
            n_chunks=retuned.initial_n_chunks,
            n_eql_blocks=retuned.settling_blocks,
        )
        settling_blocks = retuned.settling_blocks
        del retuned  # Do not retain the pre-redistribution measurement arrays.
        prepare_sampling()
        params, run_blocks = _make_run_blocks_with_auto_chunks(
            block_fn=block_fn_sr,
            sys=sys,
            params=production_params,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            state=state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            observable_names=observable_names,
        )

        if settling_blocks > 0:
            print(f"\nPost-tuning settling: {settling_blocks} blocks")
            settling_chunk = max(1, settling_blocks // 5)
            for start in range(0, settling_blocks, settling_chunk):
                n = min(settling_chunk, settling_blocks - start)
                start_batch = time.perf_counter()
                state, scalars_chunk, obs_chunk = run_blocks(
                    state,
                    ham_data=ham_data,
                    trial_data=trial_data,
                    meas_ctx=meas_ctx,
                    prop_ctx=prop_ctx,
                    n_blocks=n,
                )
                block_e_eq.extend(scalars_chunk["energy"].tolist())
                block_w_eq.extend(scalars_chunk["weight"].tolist())
                for i, name in enumerate(observable_names):
                    block_obs_eq[name].append(obs_chunk[i])
                print(
                    f"[settle {start + n:4d}/{settling_blocks}]  "
                    f"E={float(jnp.mean(scalars_chunk['energy'])):14.10f}  "
                    f"dt={(time.perf_counter() - start_batch) / n:.3f} s/block"
                )

    block_e_eq = jnp.asarray(block_e_eq)
    block_w_eq = jnp.asarray(block_w_eq)
    block_obs_eq = {
        name: (jnp.concatenate(block_obs_eq[name], axis=0) if len(block_obs_eq[name]) > 0 else None)
        for name in observable_names
    }

    # sampling
    t_mark = time.perf_counter()
    print("\nSampling:\n")
    if target_error is None:
        target_error = 0.0
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    block_e_s = []
    block_w_s = []
    block_obs_s = {name: [] for name in observable_names}
    block_diagnostics_s: dict[str, list[jax.Array]] = {}
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'E_avg':>14s}  {'E_err':>10s}  {'E_block':>14s}  "
            f"{'W':>12s}    {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, scalars_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        e_chunk = scalars_chunk["energy"]
        w_chunk = scalars_chunk["weight"]
        block_e_s.extend(e_chunk.tolist())
        block_w_s.extend(w_chunk.tolist())
        for name, values in scalars_chunk.items():
            if name not in ("energy", "weight"):
                block_diagnostics_s.setdefault(name, []).append(values)
        for i, name in enumerate(observable_names):
            block_obs_s[name].append(obs_chunk[i])
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        error_analysis = _analyze_energy_errors(
            jnp.asarray(block_e_s),
            jnp.asarray(block_w_s),
            error_method=params.error_method,
        )
        mu = error_analysis.mean
        se = error_analysis.stderr
        nodes = int(state.node_encounters)
        diagnostic_text = ""
        noise_chunks = block_diagnostics_s.get(d_energy_sampling_noise, [])
        if noise_chunks:
            noise_values = jnp.concatenate(noise_chunks)
            noise_rms_mha = 1000.0 * jnp.sqrt(jnp.mean(noise_values**2))
            diagnostic_text = f"  tail_rms={float(noise_rms_mha):.3f} mHa"
        se_text = f"{se:10.3e}" if np.isfinite(se) else f"{'unavailable':>10s}"
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{mu:14.10f}  "
            f"{se_text}  "
            f"{float(e_chunk_avg):16.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
            f"{diagnostic_text}"
        )
        if (
            np.isfinite(se)
            and (error_analysis.reliable or error_analysis.error_method == "blocking")
            and se <= target_error
            and target_error > 0.0
        ):
            print(f"\nTarget error {target_error:.3e} reached at block {start + n}.")
            break
    block_e_s = jnp.asarray(block_e_s)
    block_w_s = jnp.asarray(block_w_s)
    block_obs_s = {
        name: (jnp.concatenate(block_obs_s[name], axis=0) if len(block_obs_s[name]) > 0 else None)
        for name in observable_names
    }
    block_diagnostics = {
        name: jnp.concatenate(chunks) for name, chunks in block_diagnostics_s.items()
    }
    noise_values = block_diagnostics.get(d_energy_sampling_noise)
    if noise_values is not None and noise_values.size > 0:
        noise_rms_mha = 1000.0 * jnp.sqrt(jnp.mean(noise_values**2))
        noise_mean_mha = 1000.0 * jnp.mean(noise_values)
        print(
            "\nEnergy-sampling diagnostic: "
            f"half-difference RMS={float(noise_rms_mha):.3f} mHa, "
            f"mean={float(noise_mean_mha):.3f} mHa, "
            f"blocks={noise_values.size}."
        )
    guard_counts = block_diagnostics.get(d_energy_head_guard_count)
    guard_weights = block_diagnostics.get(d_energy_head_guard_weight)
    if guard_counts is not None and guard_counts.size > 0:
        guarded_events = jnp.sum(guard_counts)
        guarded_blocks = jnp.count_nonzero(guard_counts)
        mean_guarded_weight = (
            jnp.mean(guard_weights) if guard_weights is not None else jnp.asarray(float("nan"))
        )
        max_guarded_weight = (
            jnp.max(guard_weights) if guard_weights is not None else jnp.asarray(float("nan"))
        )
        print(
            "Head-deviation guard diagnostic: "
            f"rejected_walker_events={int(guarded_events)}, "
            f"affected_blocks={int(guarded_blocks)}/{guard_counts.size}, "
            f"mean_rejected_weight={float(mean_guarded_weight):.3e}, "
            f"max_rejected_weight={float(max_guarded_weight):.3e}."
        )
    walker_guide_ess = block_diagnostics.get(d_energy_walker_guide_ess)
    walker_guide_max_correction = block_diagnostics.get(d_energy_walker_guide_max_correction)
    if (
        walker_guide_ess is not None
        and walker_guide_max_correction is not None
        and walker_guide_ess.size > 0
    ):
        print(
            "Walker-importance diagnostic: "
            f"mean_proposal_ess={float(jnp.mean(walker_guide_ess)):.1f}, "
            f"minimum_proposal_ess={float(jnp.min(walker_guide_ess)):.1f}, "
            f"maximum_ht_correction="
            f"{float(jnp.max(walker_guide_max_correction)):.3e}."
        )

    data_clean, keep_mask = reject_outliers(jnp.column_stack((block_e_s, block_w_s)), obs=0)
    print(f"\nRejected {block_e_s.shape[0] - data_clean.shape[0]} outlier blocks.")
    block_e_s = jnp.asarray(data_clean[:, 0])
    block_w_s = jnp.asarray(data_clean[:, 1])
    keep_mask = jnp.asarray(keep_mask)
    block_obs_s = {
        name: (arr[keep_mask] if arr is not None else None) for name, arr in block_obs_s.items()
    }
    print("\nFinal statistical analysis:")
    error_analysis = _analyze_energy_errors(
        block_e_s,
        block_w_s,
        error_method=params.error_method,
    )
    _print_energy_error_analysis(error_analysis)
    mean, err = error_analysis.mean, error_analysis.stderr
    blocking_stats = error_analysis.blocking
    gamma_stats = error_analysis.gamma

    block_e_all = jnp.concatenate([block_e_eq, block_e_s])
    block_w_all = jnp.concatenate([block_w_eq, block_w_s])
    block_obs_all: dict[str, jax.Array] = {}
    for name in observable_names:
        arr_eq = block_obs_eq[name]
        arr_s = block_obs_s[name]
        if arr_eq is None and arr_s is None:
            block_obs_all[name] = jnp.zeros((0,))
            continue
        if arr_eq is None:
            arr_eq = arr_s[:0]
        if arr_s is None:
            arr_s = arr_eq[:0]
        block_obs_all[name] = jnp.concatenate([arr_eq, arr_s], axis=0)

    obs_means: dict[str, jax.Array] = {}
    obs_stderrs: dict[str, jax.Array] = {}
    b_star = blocking_stats.get("B_star")
    for name in observable_names:
        arr = block_obs_s[name]
        if arr is None:
            obs_means[name] = jnp.zeros((0,))
            obs_stderrs[name] = jnp.zeros((0,))
            continue
        obs_means[name] = _weighted_block_mean(arr, block_w_s)
        if b_star is not None and b_star >= 1:
            num, denom = rebin_observable(np.asarray(arr), np.asarray(block_w_s), b_star)
            if num.shape[0] >= 2:
                _, se = jackknife_ratios(num, denom)
                obs_stderrs[name] = jnp.asarray(se)
            else:
                obs_stderrs[name] = jnp.full(arr.shape[1:], jnp.nan)
        else:
            obs_stderrs[name] = jnp.full(arr.shape[1:], jnp.nan)

    return QmcResult(
        mean_energy=mean,
        stderr_energy=err,
        stderr_gamma=float(gamma_stats["se_gamma"]),
        stderr_blocking=(
            float(blocking_stats["se_star"]) if blocking_stats["se_star"] is not None else None
        ),
        error_method=error_analysis.error_method,
        error_reliable=error_analysis.reliable,
        gamma_window=gamma_stats["window"],
        gamma_window_found=bool(gamma_stats["window_found"]),
        tau_int=gamma_stats["tau_int"],
        effective_sample_size=gamma_stats["effective_sample_size"],
        gamma_warnings=tuple(gamma_stats["warnings"]),
        blocking_B_star=blocking_stats["B_star"],
        blocking_plateau_found=bool(blocking_stats["plateau_found"]),
        blocking_selection_reason=str(blocking_stats["selection_reason"]),
        block_energies=block_e_all,
        block_weights=block_w_all,
        block_observables=block_obs_all,
        observable_means=obs_means,
        observable_stderrs=obs_stderrs,
        block_diagnostics=block_diagnostics,
    )


def run_mixed_estimator_qmc(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any = None,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_prop_ops: PropOps,
    guide_meas_ops: MeasOps,
    estimator_data: Any,
    estimator_ops: EstimatorOps,
    mixed_block_fn: MixedEstimatorBlockFn,
    state: PropState | None = None,
    guide_meas_ctx: Any | None = None,
    guide_prop_ctx: Any | None = None,
    estimator_ctx: Any | None = None,
    target_error: float | None = None,
    estimator_outlier_zeta: float | None = 20.0,
    mesh: Mesh | None = None,
    runtime: QmcRuntime | None = None,
) -> MixedEstimatorQmcResult:
    """Run one guide trajectory and evaluate arbitrary projected components.

    Propagation always uses ``guide_*``. Population control normally uses the
    guide energy, but an estimator may explicitly supply the combined block
    energy instead via ``EstimatorOps.use_for_population_control``. The
    estimator supplies a reference overlap and sufficient statistics; each
    block is reweighted by

    ``guide_weight * reference_overlap / guide_overlap``.

    This keeps estimator algebra out of the driver and supports nonlinear
    energy combinations while retaining component covariance in both the
    blocking and Gamma analyses. ``params.error_method`` selects the reported
    projected-estimator uncertainty, just as it does for the guide energy. By
    default, the reported projected estimate also applies the historical
    PT2-CCSD block cleanup at ``zeta=20``. Raw block arrays are retained in the
    result, together with the proxy energies and keep mask.

    Use ``runtime=QmcRuntime(...)`` (or ``job.prepare_runtime()``) to let the
    driver replace Cholesky-indexed inputs without retaining the old layout.
    Separate Hamiltonian/context arguments remain borrowed and use the global
    sampler unless a local layout was explicitly installed beforehand.
    """

    if params.n_blocks <= 0:
        raise ValueError("QmcParams.n_blocks must be positive.")
    if params.n_eql_blocks < 0:
        raise ValueError("QmcParams.n_eql_blocks must be nonnegative.")

    if runtime is not None:
        ham_data, guide_prop_ctx, (guide_meas_ctx, estimator_ctx) = runtime.inputs(
            ham_data=ham_data, prop_ctx=guide_prop_ctx,
            contexts=(guide_meas_ctx, estimator_ctx),
            retune=(guide_meas_ops.retune_block_energy is not None
                    or estimator_ops.retune_block_components is not None),
            local_sampling=params.local_cholesky_sampling,
        )

    if guide_prop_ctx is None:
        guide_prop_ctx = guide_prop_ops.build_prop_ctx(
            ham_data,
            guide_ops.get_rdm1(guide_data),
            params,
        )
    if guide_meas_ctx is None:
        guide_meas_ctx = guide_meas_ops.build_meas_ctx(ham_data, guide_data)
    if estimator_ctx is None:
        estimator_ctx = estimator_ops.build_estimator_ctx(ham_data, estimator_data)
    if state is None:
        state = guide_prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=guide_ops,
            trial_data=guide_data,
            meas_ops=guide_meas_ops,
            params=params,
            meas_ctx=guide_meas_ctx,
            mesh=mesh,
        )

    def prepare_sampling():
        nonlocal ham_data, guide_prop_ctx, guide_meas_ctx, estimator_ctx
        ham_data, guide_prop_ctx, (guide_meas_ctx, estimator_ctx) = prepare_local_cholesky_sampling(
            ham_data, guide_prop_ctx, (guide_meas_ctx, estimator_ctx),
            enabled=params.local_cholesky_sampling, runtime=runtime,
        )

    if guide_meas_ops.retune_block_energy is None and estimator_ops.retune_block_components is None:
        prepare_sampling()

    if mesh is None or mesh.size == 1:
        mixed_block_fn_sr = mixed_block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        mixed_block_fn_sr = partial(mixed_block_fn, sr_fn=sr_sharded)

    def build_blocks(
        params_i: QmcParams,
        state_i: PropState,
        guide_meas_ctx_i: Any,
        estimator_ctx_i: Any,
    ) -> tuple[QmcParams, Callable]:
        return _make_run_mixed_estimator_blocks_with_auto_chunks(
            mixed_block_fn=mixed_block_fn_sr,
            sys=sys,
            params=params_i,
            guide_ops=guide_ops,
            guide_meas_ops=guide_meas_ops,
            guide_prop_ops=guide_prop_ops,
            estimator_ops=estimator_ops,
            state=state_i,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx_i,
            guide_prop_ctx=guide_prop_ctx,
            estimator_data=estimator_data,
            estimator_ctx=estimator_ctx_i,
        )

    params, run_blocks = build_blocks(params, state, guide_meas_ctx, estimator_ctx)

    def advance(state_i: PropState, n_blocks: int):
        return run_blocks(
            state_i,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            estimator_data=estimator_data,
            estimator_ctx=estimator_ctx,
            n_blocks=n_blocks,
        )

    def combined_estimator_energy(
        weights: jax.Array,
        components: jax.Array,
    ) -> float:
        mean_components = _weighted_block_mean(components, weights)
        energy = estimator_ops.combine_energy(ham_data.h0, mean_components)
        return float(np.real(np.asarray(energy).reshape(())))

    shift_energy_label = (
        "Shift" if estimator_ops.use_for_population_control else "Guide"
    )
    run_started = time.perf_counter()
    print("\nMixed-estimator equilibration:\n")
    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  "
            f"{f'{shift_energy_label}_E_blk':>14s}  "
            f"{'Guide_W_blk':>12s}  "
            f"{'Estimator_E_blk':>16s}  "
            f"{'Estimator_W_blk':>15s}  "
            f"{'nodes':>10s}  "
            f"{'t[s]':>8s}"
        )
    estimator_energy_0, estimator_weight_0 = _initial_projected_estimator(
        state,
        ham_data=ham_data,
        estimator_data=estimator_data,
        estimator_ops=estimator_ops,
        estimator_ctx=estimator_ctx,
    )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{float(jnp.real(state.e_estimate)):14.10f}  "
        f"{float(jnp.real(jnp.sum(state.weights))):12.6e}  "
        f"{estimator_energy_0:16.10f}  "
        f"{estimator_weight_0:15.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    if params.n_eql_blocks > 0:
        equilibration_energy_chunks = []
        equilibration_weight_chunks = []
        equilibration_estimator_component_chunks = []
        equilibration_estimator_weight_chunks = []
        equilibration_chunk = print_every if print_every > 0 else 1
        for start in range(0, params.n_eql_blocks, equilibration_chunk):
            n = min(equilibration_chunk, params.n_eql_blocks - start)
            state, scalars_chunk = advance(state, n)
            guide_energies_chunk = scalars_chunk["guide_energy"]
            guide_weights_chunk = scalars_chunk["guide_weight"]
            estimator_weights_chunk = scalars_chunk["estimator_weight"]
            estimator_components_chunk = scalars_chunk["estimator_components"]
            equilibration_energy_chunks.append(guide_energies_chunk)
            equilibration_weight_chunks.append(guide_weights_chunk)
            equilibration_estimator_component_chunks.append(estimator_components_chunk)
            equilibration_estimator_weight_chunks.append(estimator_weights_chunk)
            guide_energy_chunk = _weighted_block_mean(
                guide_energies_chunk,
                guide_weights_chunk,
            )
            estimator_energy_chunk = combined_estimator_energy(
                estimator_weights_chunk,
                estimator_components_chunk,
            )
            print(
                f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
                f"{float(guide_energy_chunk):14.10f}  "
                f"{float(jnp.mean(guide_weights_chunk)):12.6e}  "
                f"{estimator_energy_chunk:16.10f}  "
                f"{float(jnp.real(jnp.mean(estimator_weights_chunk))):15.6e}  "
                f"{int(state.node_encounters):10d}  "
                f"{time.perf_counter() - run_started:8.1f}"
            )
        equilibration_energies = jnp.concatenate(equilibration_energy_chunks)
        equilibration_weights = jnp.concatenate(equilibration_weight_chunks)
        equilibration_estimator_components = jnp.concatenate(
            equilibration_estimator_component_chunks
        )
        equilibration_estimator_weights = jnp.concatenate(
            equilibration_estimator_weight_chunks
        )
    else:
        equilibration_energies = jnp.zeros((0,), dtype=jnp.result_type(state.e_estimate))
        equilibration_weights = jnp.zeros((0,), dtype=jnp.result_type(state.weights))
        equilibration_estimator_components = jnp.zeros(
            (0, len(estimator_ops.component_names)),
            dtype=jnp.complex128,
        )
        equilibration_estimator_weights = jnp.zeros((0,), dtype=jnp.complex128)
        print("  no equilibration blocks requested.")

    if guide_meas_ops.retune_block_energy is not None:
        print("\nRetuning guide block-energy sampling after equilibration:")

        def advance_for_retune(state_i, *, n_blocks: int):
            state_n, scalars_n = advance(state_i, n_blocks)
            guide_scalars = {
                "energy": scalars_n["guide_energy"],
                "weight": scalars_n["guide_weight"],
            }
            return state_n, guide_scalars, ()

        retuned = guide_meas_ops.retune_block_energy(
            state,
            equilibration_energies,
            equilibration_weights,
            params,
            ham_data,
            guide_meas_ctx,
            guide_data,
            advance_blocks=advance_for_retune,
            target_error=target_error,
        )
        if retuned.initial_n_chunks <= 0:
            raise ValueError("retuned initial_n_chunks must be positive.")
        if retuned.settling_blocks < 0:
            raise ValueError("retuned settling_blocks must be nonnegative.")
        state = cast(PropState, retuned.state)
        guide_meas_ctx = retuned.meas_ctx
        params = dataclasses.replace(
            params,
            n_chunks=retuned.initial_n_chunks,
            n_eql_blocks=retuned.settling_blocks,
        )
        settling_blocks = retuned.settling_blocks
        del retuned
        if estimator_ops.retune_block_components is None:
            prepare_sampling()
        params, run_blocks = build_blocks(params, state, guide_meas_ctx, estimator_ctx)
        if settling_blocks > 0:
            print(f"\nPost-tuning settling: {settling_blocks} blocks")
            settling_chunk = max(1, settling_blocks // 5)
            for start in range(0, settling_blocks, settling_chunk):
                n = min(settling_chunk, settling_blocks - start)
                batch_started = time.perf_counter()
                state, scalars_chunk = advance(state, n)
                guide_energy_chunk = _weighted_block_mean(
                    scalars_chunk["guide_energy"],
                    scalars_chunk["guide_weight"],
                )
                estimator_energy_chunk = combined_estimator_energy(
                    scalars_chunk["estimator_weight"],
                    scalars_chunk["estimator_components"],
                )
                print(
                    f"[settle {start + n:4d}/{settling_blocks}]  "
                    f"{shift_energy_label}_E={float(guide_energy_chunk):14.10f}  "
                    f"Estimator_E={estimator_energy_chunk:14.10f}  "
                    f"dt={(time.perf_counter() - batch_started) / n:.3f} s/block"
                )

    if estimator_ops.retune_block_components is not None:
        print("\nRetuning projected-estimator component sampling after equilibration:")

        def advance_for_estimator_retune(state_i, *, n_blocks: int):
            state_n, scalars_n = advance(state_i, n_blocks)
            return state_n, scalars_n, ()

        estimator_retuned = estimator_ops.retune_block_components(
            state,
            equilibration_estimator_components,
            equilibration_estimator_weights,
            params,
            ham_data,
            estimator_ctx,
            estimator_data,
            guide_data=guide_data,
            guide_meas_ops=guide_meas_ops,
            guide_meas_ctx=guide_meas_ctx,
            advance_blocks=advance_for_estimator_retune,
            target_error=target_error,
        )
        if estimator_retuned.initial_n_chunks <= 0:
            raise ValueError("retuned estimator initial_n_chunks must be positive.")
        if estimator_retuned.settling_blocks < 0:
            raise ValueError("retuned estimator settling_blocks must be nonnegative.")
        state = cast(PropState, estimator_retuned.state)
        estimator_ctx = estimator_retuned.estimator_ctx
        params = dataclasses.replace(
            params,
            n_chunks=estimator_retuned.initial_n_chunks,
            n_eql_blocks=estimator_retuned.settling_blocks,
        )
        settling_blocks = estimator_retuned.settling_blocks
        del estimator_retuned
        prepare_sampling()
        params, run_blocks = build_blocks(params, state, guide_meas_ctx, estimator_ctx)
        if settling_blocks > 0:
            print(
                "\nPost-estimator-tuning settling: "
                f"{settling_blocks} blocks"
            )
            settling_chunk = max(1, settling_blocks // 5)
            for start in range(0, settling_blocks, settling_chunk):
                n = min(settling_chunk, settling_blocks - start)
                batch_started = time.perf_counter()
                state, scalars_chunk = advance(state, n)
                guide_energy_chunk = _weighted_block_mean(
                    scalars_chunk["guide_energy"],
                    scalars_chunk["guide_weight"],
                )
                estimator_energy_chunk = combined_estimator_energy(
                    scalars_chunk["estimator_weight"],
                    scalars_chunk["estimator_components"],
                )
                print(
                    f"[estimator settle "
                    f"{start + n:4d}/{settling_blocks}]  "
                    f"{shift_energy_label}_E={float(guide_energy_chunk):14.10f}  "
                    f"Estimator_E={estimator_energy_chunk:14.10f}  "
                    f"dt={(time.perf_counter() - batch_started) / n:.3f} s/block"
                )

    print("\nMixed-estimator sampling:\n")
    sampling_started = time.perf_counter()
    sampling_mark = sampling_started
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    guide_energy_chunks = []
    guide_weight_chunks = []
    estimator_weight_chunks = []
    estimator_component_chunks = []
    guide_diagnostic_chunks: dict[str, list[jax.Array]] = {}
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  "
            f"{f'{shift_energy_label}_E_avg':>14s}  "
            f"{f'{shift_energy_label}_E_err':>11s}  "
            f"{'Guide_W':>12s}  "
            f"{'Estimator_E_avg':>16s}  "
            f"{'Estimator_E_err':>15s}  "
            f"{'nodes':>10s}  "
            f"{'dt[s/bl]':>10s}  "
            f"{'t[s]':>8s}"
        )

    sampling_chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, sampling_chunk):
        n = min(sampling_chunk, params.n_blocks - start)
        state, scalars_chunk = advance(state, n)
        guide_energy_chunks.append(scalars_chunk["guide_energy"])
        guide_weight_chunks.append(scalars_chunk["guide_weight"])
        estimator_weight_chunks.append(scalars_chunk["estimator_weight"])
        estimator_component_chunks.append(scalars_chunk["estimator_components"])
        for name, values in scalars_chunk.items():
            if name not in (
                "guide_energy",
                "guide_weight",
                "estimator_weight",
                "estimator_components",
            ):
                guide_diagnostic_chunks.setdefault(name, []).append(values)

        guide_block_energies_so_far = jnp.concatenate(guide_energy_chunks)
        guide_block_weights_so_far = jnp.concatenate(guide_weight_chunks)
        estimator_block_weights_so_far = jnp.concatenate(estimator_weight_chunks)
        estimator_block_components_so_far = jnp.concatenate(estimator_component_chunks)
        guide_progress = _analyze_energy_errors(
            guide_block_energies_so_far,
            guide_block_weights_so_far,
            error_method=params.error_method,
        )
        _, estimator_progress_keep = component_estimator_outlier_mask(
            ham_data.h0,
            estimator_block_weights_so_far,
            estimator_block_components_so_far,
            estimator_ops.combine_energy,
            zeta=estimator_outlier_zeta,
        )
        estimator_progress = _analyze_component_estimator_errors(
            ham_data.h0,
            estimator_block_weights_so_far[estimator_progress_keep],
            estimator_block_components_so_far[estimator_progress_keep],
            estimator_ops.combine_energy,
            error_method=params.error_method,
        )
        estimator_progress_stderr = estimator_progress.stderr
        guide_stderr_text = (
            f"{guide_progress.stderr:11.3e}"
            if np.isfinite(guide_progress.stderr)
            else f"{'unavailable':>11s}"
        )
        estimator_stderr_text = (
            f"{float(estimator_progress_stderr):15.3e}"
            if estimator_progress_stderr is not None
            and np.isfinite(estimator_progress_stderr)
            else f"{'unavailable':>15s}"
        )
        batch_finished = time.perf_counter()
        diagnostic_text = ""
        noise_chunks = guide_diagnostic_chunks.get(
            f"guide_{d_energy_sampling_noise}",
            [],
        )
        if not noise_chunks:
            noise_chunks = guide_diagnostic_chunks.get(
                f"estimator_{d_pt_component_sampling_noise_real}",
                [],
            )
        if noise_chunks:
            noise_values = jnp.concatenate(noise_chunks)
            noise_rms_mha = 1000.0 * jnp.sqrt(jnp.mean(noise_values**2))
            diagnostic_text = f"  tail_rms={float(noise_rms_mha):.3f} mHa"
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{guide_progress.mean:14.10f}  "
            f"{guide_stderr_text}  "
            f"{float(jnp.mean(scalars_chunk['guide_weight'])):12.6e}  "
            f"{estimator_progress.mean:16.10f}  "
            f"{estimator_stderr_text}  "
            f"{int(state.node_encounters):10d}  "
            f"{(batch_finished - sampling_mark) / n:10.3f}  "
            f"{batch_finished - run_started:8.1f}"
            f"{diagnostic_text}"
        )
        sampling_mark = batch_finished

    elapsed = time.perf_counter() - sampling_started
    guide_block_energies = jnp.concatenate(guide_energy_chunks)
    guide_block_weights = jnp.concatenate(guide_weight_chunks)
    estimator_block_weights = jnp.concatenate(estimator_weight_chunks)
    estimator_block_components = jnp.concatenate(estimator_component_chunks)
    guide_block_diagnostics = {
        name: jnp.concatenate(chunks) for name, chunks in guide_diagnostic_chunks.items()
    }

    guide_analysis = _analyze_energy_errors(
        guide_block_energies,
        guide_block_weights,
        error_method=params.error_method,
    )
    estimator_block_proxy_energies, estimator_block_keep_mask = (
        component_estimator_outlier_mask(
            ham_data.h0,
            estimator_block_weights,
            estimator_block_components,
            estimator_ops.combine_energy,
            zeta=estimator_outlier_zeta,
        )
    )
    n_estimator_blocks = int(estimator_block_keep_mask.size)
    n_estimator_blocks_retained = int(np.count_nonzero(estimator_block_keep_mask))
    n_estimator_blocks_rejected = n_estimator_blocks - n_estimator_blocks_retained
    if n_estimator_blocks_retained == 0:
        raise ValueError("Projected-estimator cleanup rejected every production block.")
    if estimator_outlier_zeta is None:
        print(
            "  projected-estimator robust cleanup disabled; "
            f"retained {n_estimator_blocks_retained}/{n_estimator_blocks} finite blocks."
        )
    else:
        print(
            f"  rejected {n_estimator_blocks_rejected}/{n_estimator_blocks} "
            "projected-estimator blocks "
            f"with zeta={estimator_outlier_zeta:g} median-deviation cleanup."
        )

    estimator_error_analysis = _analyze_component_estimator_errors(
        ham_data.h0,
        estimator_block_weights[estimator_block_keep_mask],
        estimator_block_components[estimator_block_keep_mask],
        estimator_ops.combine_energy,
        error_method=params.error_method,
    )
    estimator_analysis = dict(estimator_error_analysis.blocking)
    estimator_analysis.update(
        {
            "blocking": estimator_error_analysis.blocking,
            "error_method": estimator_error_analysis.error_method,
            "error_reliable": estimator_error_analysis.reliable,
            "gamma": estimator_error_analysis.gamma,
            "stderr": estimator_error_analysis.stderr,
            "stderr_blocking": estimator_error_analysis.blocking["se_star"],
            "stderr_gamma": estimator_error_analysis.gamma["se_gamma"],
            "outlier_zeta": estimator_outlier_zeta,
            "n_raw_blocks": n_estimator_blocks,
            "n_retained_blocks": n_estimator_blocks_retained,
            "n_rejected_blocks": n_estimator_blocks_rejected,
        }
    )
    estimator_stderr = float(estimator_error_analysis.stderr)

    print("\nProjected-estimator statistical analysis:")
    _print_energy_error_analysis(estimator_error_analysis)

    print(
        f"  completed {params.n_blocks} blocks in {elapsed:.1f} s; "
        f"{shift_energy_label.lower()} E={guide_analysis.mean:.10f} +/- "
        f"{guide_analysis.stderr:.3e}; "
        f"estimator E={estimator_error_analysis.mean:.10f} +/- "
        f"{estimator_stderr:.3e}."
    )
    return MixedEstimatorQmcResult(
        guide_mean_energy=guide_analysis.mean,
        guide_stderr_energy=guide_analysis.stderr,
        estimator_mean_energy=estimator_error_analysis.mean,
        estimator_stderr_energy=estimator_stderr,
        estimator_mean_components=jnp.asarray(estimator_analysis["mean_components"]),
        estimator_component_names=estimator_ops.component_names,
        guide_block_energies=guide_block_energies,
        guide_block_weights=guide_block_weights,
        estimator_block_weights=estimator_block_weights,
        estimator_block_components=estimator_block_components,
        estimator_block_proxy_energies=jnp.asarray(estimator_block_proxy_energies),
        estimator_block_keep_mask=jnp.asarray(estimator_block_keep_mask),
        guide_block_diagnostics=guide_block_diagnostics,
        final_state=state,
        guide_analysis=guide_analysis,
        estimator_analysis=estimator_analysis,
    )


def run_mixed_qmc(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_prop_ops: PropOps,
    guide_meas_ops: MeasOps,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    mix_block_fn: MixedBlockFn,
    state: PropState | None = None,
    guide_meas_ctx: Any | None = None,
    trial_meas_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
    observable_names: tuple[str, ...] = (),
) -> MixedQmcResult:
    """
    equilibration blocks then sampling blocks.
    Guide != Trial
    Currently only support pt2CCSD trial without observables

    Returns:
      MixedQmcResult with energy statistics plus block-level observable estimates.
    """

    for name in observable_names:
        guide_meas_ops.require_observable(name)

    # build ctx
    guide_prop_ctx = guide_prop_ops.build_prop_ctx(ham_data, guide_ops.get_rdm1(guide_data), params)
    if guide_meas_ctx is None:
        guide_meas_ctx = guide_meas_ops.build_meas_ctx(ham_data, guide_data)

    if trial_meas_ctx is None:
        trial_meas_ctx = trial_meas_ops.build_meas_ctx(ham_data, trial_data)

    if state is None:
        state = guide_prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=guide_ops,
            trial_data=guide_data,
            meas_ops=guide_meas_ops,
            params=params,
            meas_ctx=guide_meas_ctx,
            mesh=mesh,
        )

    assert trial_meas_ctx is not None
    trial_energy0, trial_weights0 = get_init_pt2trial_energy(
        init_state=state,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_meas_ops=trial_meas_ops,
        trial_meas_ctx=trial_meas_ctx,
        params=params,
    )

    if mesh is None or mesh.size == 1:
        mix_block_fn_sr = mix_block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        mix_block_fn_sr = partial(mix_block_fn, sr_fn=sr_sharded)

    run_blocks = make_run_mixed_blocks(
        mixed_block_fn=mix_block_fn_sr,
        sys=sys,
        params=params,
        guide_ops=guide_ops,
        guide_meas_ops=guide_meas_ops,
        guide_prop_ops=guide_prop_ops,
        trial_meas_ops=trial_meas_ops,
        observable_names=observable_names,
    )

    t0 = time.perf_counter()
    t_mark = t0

    block_time = params.dt * params.n_prop_steps
    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    guide_block_e_eq = []
    guide_block_w_eq = []
    trial_block_w_eq = []
    trial_block_t2_eq = []
    trial_block_e0_eq = []
    trial_block_e1_eq = []
    # guide_block_obs_eq = {name: [] for name in observable_names}
    guide_block_e_eq.append(state.e_estimate)
    guide_block_w_eq.append(jnp.sum(state.weights))
    # trial_block_e_eq.append(trial_energy0)
    # trial_block_w_eq.append(jnp.sum(trial_weights0))
    print("\nEquilibration:\n")
    if print_every:
        print(
            f"{'':4s}"
            f"{'block':>9s}  "
            f"{'1/Tmp':>6s}  "
            f"{'Guide_E_blk':>14s}  "
            f"{'Guide_W_blk':>12s}   "
            f"{'Trial_E_blk':>14s}  "
            f"{'Trial_W_blk':>12s}   "
            f"{'nodes':>10s}  "
            f"{'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{0 * block_time:6.2f}  "
        f"{float(guide_block_e_eq[0].real):14.10f}  "
        f"{float(guide_block_w_eq[0].real):12.6e}  "
        f"{float(trial_energy0.real):14.10f}  "
        f"{float(trial_weights0.real):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, scalars_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            trial_data=trial_data,
            trial_meas_ctx=trial_meas_ctx,
            n_blocks=n,
        )
        # guide
        guide_block_e_eq.extend(scalars_chunk["guide_energy"].tolist())
        guide_block_w_eq.extend(scalars_chunk["guide_weight"].tolist())
        guide_e_chunk = scalars_chunk["guide_energy"]
        guide_w_chunk = scalars_chunk["guide_weight"]
        guide_w_chunk_avg = jnp.mean(guide_w_chunk)
        guide_e_chunk_avg = jnp.mean(guide_e_chunk * guide_w_chunk) / guide_w_chunk_avg
        # trial
        trial_block_w_eq.extend(scalars_chunk["trial_weight"].tolist())
        trial_block_t2_eq.extend(scalars_chunk["trial_t2"].tolist())
        trial_block_e0_eq.extend(scalars_chunk["trial_e0"].tolist())
        trial_block_e1_eq.extend(scalars_chunk["trial_e1"].tolist())
        # for i, name in enumerate(observable_names):
        #     block_obs_eq[name].append(obs_chunk[i])
        trial_w_chunk = scalars_chunk["trial_weight"]
        trial_t2_chunk = scalars_chunk["trial_t2"]
        trial_e0_chunk = scalars_chunk["trial_e0"]
        trial_e1_chunk = scalars_chunk["trial_e1"]
        trial_w_chunk_avg = jnp.mean(trial_w_chunk)
        trial_t2_chunk_avg = jnp.mean(trial_w_chunk * trial_t2_chunk) / trial_w_chunk_avg
        trial_e0_chunk_avg = jnp.mean(trial_w_chunk * trial_e0_chunk) / trial_w_chunk_avg
        trial_e1_chunk_avg = jnp.mean(trial_w_chunk * trial_e1_chunk) / trial_w_chunk_avg
        pt2trial_energy_avg = (
            ham_data.h0
            + trial_e0_chunk_avg
            + trial_e1_chunk_avg
            - trial_t2_chunk_avg * trial_e0_chunk_avg
        )
        elapsed = time.perf_counter() - t0
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{(start + n) * block_time:6.2f}  "
            f"{float(guide_e_chunk_avg):14.10f}  "
            f"{float(guide_w_chunk_avg):12.6e}  "
            f"{float(pt2trial_energy_avg.real):14.10f}  "
            f"{float(trial_w_chunk_avg.real):12.6e}  "
            f"{int(state.node_encounters):10d}  "
            f"{elapsed:8.1f}"
        )

    guide_block_w_eq = jnp.asarray(guide_block_w_eq)
    guide_block_e_eq = jnp.asarray(guide_block_e_eq)

    trial_block_w_eq = jnp.asarray(trial_block_w_eq)
    trial_block_t2_eq = jnp.asarray(trial_block_t2_eq)
    trial_block_e0_eq = jnp.asarray(trial_block_e0_eq)
    trial_block_e1_eq = jnp.asarray(trial_block_e1_eq)

    # block_obs_eq = {
    #     name: (jnp.concatenate(block_obs_eq[name], axis=0) if len(block_obs_eq[name]) > 0 else None)
    #     for name in observable_names
    # }

    # sampling
    print("\nSampling:\n")
    if target_error is None:
        target_error = 0.0
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0

    guide_block_w_sp = []
    guide_block_e_sp = []
    trial_block_w_sp = []
    trial_block_t2_sp = []
    trial_block_e0_sp = []
    trial_block_e1_sp = []

    # block_obs_sp = {name: [] for name in observable_names}
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'Guide_E_avg':>14s}  {'Guide_E_err':>10s}  {'Guide_W':>12s}  "
            f"{'Trial_E_avg':>14s}  {'Trial_E_err':>10s}  {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, scalars_chunk, obs_chunk = run_blocks(
            state,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            trial_data=trial_data,
            trial_meas_ctx=trial_meas_ctx,
            n_blocks=n,
        )
        # guide
        # guide_w_chunk = scalars_chunk["guide_weight"]
        # guide_e_chunk = scalars_chunk["guide_energy"]
        guide_block_w_sp.extend(scalars_chunk["guide_weight"].tolist())
        guide_block_e_sp.extend(scalars_chunk["guide_energy"].tolist())
        # for i, name in enumerate(observable_names):
        #     block_obs_sp[name].append(obs_chunk[i])
        guide_w_avg = jnp.mean(jnp.asarray(guide_block_w_sp))
        # guide_e_avg = jnp.mean(jnp.asarray(guide_block_w_sp) * jnp.asarray(guide_block_e_sp)) / guide_w_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(guide_block_e_sp), jnp.asarray(guide_block_w_sp), print_q=False
        )
        guide_mu = stats["mu"]
        guide_se = stats["se_star"]
        nodes = int(state.node_encounters)
        # trial
        trial_block_w_sp.extend(scalars_chunk["trial_weight"].tolist())
        trial_block_t2_sp.extend(scalars_chunk["trial_t2"].tolist())
        trial_block_e0_sp.extend(scalars_chunk["trial_e0"].tolist())
        trial_block_e1_sp.extend(scalars_chunk["trial_e1"].tolist())
        # trial_w_avg = jnp.mean(jnp.asarray(trial_block_w_sp))
        # trial_t2_avg = jnp.mean(jnp.asarray(trial_block_w_sp) * jnp.asarray(trial_block_t2_sp)) / trial_w_avg
        # trial_e0_avg = jnp.mean(jnp.asarray(trial_block_w_sp) * jnp.asarray(trial_block_e0_sp)) / trial_w_avg
        # trial_e1_avg = jnp.mean(jnp.asarray(trial_block_w_sp) * jnp.asarray(trial_block_e1_sp)) / trial_w_avg
        # trial_e_avg = ham_data.h0 + trial_e0_avg + trial_e1_avg - trial_t2_avg * trial_e0_avg
        trial_e_avg, trial_error = pt2ccsd_blocking(
            ham_data.h0,
            jnp.asarray(trial_block_w_sp),
            jnp.asarray(trial_block_t2_sp),
            jnp.asarray(trial_block_e0_sp),
            jnp.asarray(trial_block_e1_sp),
            printQ=False,
        )

        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{guide_mu:14.10f}  "
            f"{(f'{guide_se:10.3e}' if guide_se is not None else ' ' * 10)}  "
            # f"{float(guide_e_avg):16.10f}  "
            f"{float(guide_w_avg):12.6e}  "
            f"{float(trial_e_avg.real):14.10f}  "
            f"{float(trial_error.real):10.3e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
        if guide_se is not None and guide_se <= target_error and target_error > 0.0:
            print(f"\nTarget error {target_error:.3e} reached at block {start + n}.")
            break

    # Guide
    guide_block_w_sp = jnp.asarray(guide_block_w_sp)
    guide_block_e_sp = jnp.asarray(guide_block_e_sp)
    guide_block_w_all = jnp.concatenate(
        [guide_block_w_eq, guide_block_w_sp]
    )  # return both eql and sampling
    guide_block_e_all = jnp.concatenate([guide_block_e_eq, guide_block_e_sp])
    guide_data_clean, guide_keep_mask = reject_outliers(
        jnp.column_stack((guide_block_e_sp, guide_block_w_sp)), obs=0
    )
    print(
        f"\nRejected {guide_block_e_sp.shape[0] - guide_data_clean.shape[0]} guiding outlier blocks."
    )
    guide_block_e_sp = jnp.asarray(guide_data_clean[:, 0])
    guide_block_w_sp = jnp.asarray(guide_data_clean[:, 1])
    guide_keep_mask = jnp.asarray(guide_keep_mask)

    # Trial
    trial_block_w_sp = jnp.asarray(trial_block_w_sp)
    trial_block_t2_sp = jnp.asarray(trial_block_t2_sp)
    trial_block_e0_sp = jnp.asarray(trial_block_e0_sp)
    trial_block_e1_sp = jnp.asarray(trial_block_e1_sp)
    trial_block_w_all = jnp.concatenate([trial_block_w_eq, trial_block_w_sp])
    trial_block_t2_all = jnp.concatenate([trial_block_t2_eq, trial_block_t2_sp])
    trial_block_e0_all = jnp.concatenate([trial_block_e0_eq, trial_block_e0_sp])
    trial_block_e1_all = jnp.concatenate([trial_block_e1_eq, trial_block_e1_sp])
    # esmitate of pt2ccsd energy sample, biased
    trial_block_e_sp = (
        ham_data.h0 + trial_block_e0_sp + trial_block_e1_sp - trial_block_t2_sp * trial_block_e0_sp
    )
    # trial_data_clean, trial_keep_mask = reject_outliers(
    #     jnp.column_stack((trial_block_e_sp.real,
    #                       trial_block_w_sp,
    #                       trial_block_t2_sp,
    #                       trial_block_e0_sp,
    #                       trial_block_e1_sp)),
    #                       obs=0)
    # trial_block_e_sp = trial_data_clean[:, 0]
    # trial_block_w_sp = trial_data_clean[:, 1]
    # trial_block_t2_sp = trial_data_clean[:, 2]
    # trial_block_e0_sp = trial_data_clean[:, 3]
    # trial_block_e1_sp = trial_data_clean[:, 4]
    print("Clean AFQMC/pt2CCSD Observation...")
    (
        trial_block_w_sp_clean,
        trial_block_t2_sp_clean,
        trial_block_e0_sp_clean,
        trial_block_e1_sp_clean,
    ) = clean_pt2ccsd(
        trial_block_e_sp.real,
        trial_block_w_sp,
        trial_block_t2_sp,
        trial_block_e0_sp,
        trial_block_e1_sp,
        zeta=20,
    )

    print(
        f"\nRejected {len(trial_block_w_sp) - len(trial_block_w_sp_clean)} AFQMC/pt2CCSD outlier blocks."
    )

    print("\nFinal blocking analysis:")
    guide_stats = blocking_analysis_ratio(guide_block_e_sp, guide_block_w_sp, print_q=True)
    guide_e_mean, guide_e_err = guide_stats["mu"], guide_stats["se_star"]

    trial_e_mean, trial_e_err = pt2ccsd_blocking(
        ham_data.h0,
        trial_block_w_sp_clean,
        trial_block_t2_sp_clean,
        trial_block_e0_sp_clean,
        trial_block_e1_sp_clean,
        printQ=True,
    )

    print(f"AFQMC/pt2CCSD energy = {trial_e_mean.real:.6f} +/- {trial_e_err.real:.6f} (1-sigma)")

    return MixedQmcResult(
        guide_mean_energy=guide_e_mean,
        guide_stderr_energy=guide_e_err,
        guide_block_energies=guide_block_e_all,
        guide_block_weights=guide_block_w_all,
        trial_mean_energy=trial_e_mean,
        trial_stderr_energy=trial_e_err,
        trial_block_weights=trial_block_w_all,
        trial_block_t2s=trial_block_t2_all,
        trial_block_e0s=trial_block_e0_all,
        trial_block_e1s=trial_block_e1_all,
        # block_observables=block_obs_all,
        # observable_means=obs_means,
        # observable_stderrs=obs_stderrs,
    )


def run_qmc_energy(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    prop_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> tuple[float, float, jax.Array, jax.Array]:
    out = run_qmc(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trial_ops=trial_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        state=state,
        meas_ctx=meas_ctx,
        prop_ctx=prop_ctx,
        target_error=target_error,
        mesh=mesh,
        observable_names=(),
    )
    return (
        float(out.mean_energy),
        float(out.stderr_energy),
        out.block_energies,
        out.block_weights,
    )


def run_qmc_fp(
    *,
    sys: System,
    params: QmcParamsFp,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    prop_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> QmcResult:
    """
    Returns:
      (mean_energy, stderr, block_energies, block_weights)
    """
    # build ctx
    if prop_ctx is None:
        prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            meas_ctx=meas_ctx,
        )

    block_fn_sr = block_fn

    run_blocks = make_run_blocks(
        block_fn=block_fn_sr,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )

    t0 = time.perf_counter()

    # sampling
    print("\nSampling:\n")
    # print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 1
    print_every = 1
    block_e_all = jnp.zeros((params.n_traj, params.n_blocks + 1)) + 0.0j
    block_w_all = jnp.zeros((params.n_traj, params.n_blocks + 1)) + 0.0j
    total_sign = jnp.ones((params.n_traj, params.n_blocks + 1)) + 0.0j

    chunk = print_every
    for i in range(params.n_traj):
        print("Trajectory count", i + 1)
        print(f"{'tau':^12s}    {'E_avg':^14s}  {'E_err':^13s}  {'sign':>6s}")
        if i > 0:
            params = dataclasses.replace(params, seed=params.seed + i)
            state = prop_ops.init_prop_state(
                sys=sys,
                ham_data=ham_data,
                trial_ops=trial_ops,
                trial_data=trial_data,
                meas_ops=meas_ops,
                params=params,
                meas_ctx=meas_ctx,
            )

        block_e_all = block_e_all.at[i, 0].set(jnp.array(state.e_estimate))
        block_w_all = block_w_all.at[i, 0].set(jnp.sum(state.weights))
        total_sign = total_sign.at[i, 0].set(
            jnp.sum(state.overlaps) / (jnp.sum(jnp.abs(state.overlaps)))
        )

        n = params.n_blocks
        state, scalars_chunk, _obs = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_s = scalars_chunk["energy"]
        block_w_s = scalars_chunk["weight"]
        block_ov_s = scalars_chunk["overlap"]
        block_abs_ov_s = scalars_chunk["abs_overlap"]

        block_e_all = block_e_all.at[i, 1:].set(block_e_s)
        block_w_all = block_w_all.at[i, 1:].set(block_w_s)
        sign = block_ov_s / block_abs_ov_s
        total_sign = total_sign.at[i, 1:].set(sign)
        mean = jnp.sum(block_e_all[: i + 1] * block_w_all[: i + 1], axis=0) / jnp.sum(
            block_w_all[: i + 1], axis=0
        )
        sign = jnp.sum(total_sign[: i + 1] * block_w_all[: i + 1], axis=0) / jnp.sum(
            block_w_all[: i + 1], axis=0
        )
        if i == 0:
            err = jnp.zeros_like(mean)
        else:
            err = jnp.std(block_e_all[: i + 1], axis=0) / jnp.sqrt(i)

        timer = params.dt * params.n_prop_steps * jnp.arange(params.n_blocks + 1)
        for j in range(0, params.n_blocks + 1, chunk):
            print(
                f"{(timer[j]):12.4f}    "
                f"{(mean[j].real):14.10f}  "
                f"{(err[j].real):13.7e}  "
                f"{(sign[j].real):6.2f}"
            )

        # not implemented in free projection yet
        block_obs_all: dict[str, jax.Array] = {}
        obs_means: dict[str, jax.Array] = {}
        obs_stderrs: dict[str, jax.Array] = {}

        elapsed = time.perf_counter() - t0

        print(f"Wall time :{elapsed:12.1f} s\n")

    return QmcResult(
        mean_energy=mean,
        stderr_energy=err,
        stderr_gamma=None,
        stderr_blocking=None,
        error_method="trajectory_standard_error",
        error_reliable=bool(params.n_traj >= 2),
        gamma_window=None,
        gamma_window_found=False,
        tau_int=None,
        effective_sample_size=None,
        gamma_warnings=("Gamma analysis is not defined for free-projection trajectory output",),
        blocking_B_star=None,
        blocking_plateau_found=False,
        blocking_selection_reason="not_applicable",
        block_energies=block_e_all,
        block_weights=block_w_all,
        block_observables=block_obs_all,
        observable_means=obs_means,
        observable_stderrs=obs_stderrs,
        block_diagnostics={},
    )


def run_qmc_energy_fp(
    *,
    sys: System,
    params: QmcParamsFp,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    prop_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    out = run_qmc_fp(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trial_ops=trial_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        state=state,
        meas_ctx=meas_ctx,
        prop_ctx=prop_ctx,
        target_error=target_error,
    )
    return (
        cast(jax.Array, out.mean_energy),
        cast(jax.Array, out.stderr_energy),
        out.block_energies,
        out.block_weights,
    )
