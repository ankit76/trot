from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from jax.sharding import Mesh, PartitionSpec as P

from .. import walkers as wk
from ..core.ops import (
    BlockEnergyAdvanceFn,
    BlockEnergyEstimate,
    BlockEnergyRetuneResult,
    MeasOps,
    d_energy_head_guard_count,
    d_energy_head_guard_weight,
    d_energy_sampling_noise,
    d_energy_walker_guide_ess,
    d_energy_walker_guide_max_correction,
    k_energy,
    k_energy_init,
    k_force_bias,
)
from ..core.system import System
from ..ham.chol import HamChol
from ..sharding import cholesky_model_mesh
from ..trial.cisd_modes import CisdModeTrial, mode_apply, mode_quadratic
from ..trial.cisd_modes import overlap_r as cisd_mode_overlap_r
from .cisd import CisdMeasCfg, _energy_gl_batched_realimag, _force_bias_chol_contract_high_realimag
from .pair_sampling import (
    ModelPairSamplingData, local_cholesky_head, local_cholesky_tail,
    local_common_and_head, local_pair_mesh, local_pair_tail,
)

_CISD_MODE_MEAS_CFG_ATTR = "_cisd_mode_meas_cfg"
_CISD_SETUP_CHOL_BATCH_SIZE = 256
_CISD_INITIAL_ENERGY_CHOL_BATCH_SIZE = 256


def _greens_restricted(walker: jax.Array, nocc: int) -> jax.Array:
    wocc = walker[:nocc, :]
    return jnp.linalg.solve(wocc.T, walker.T)


def _active_green_blocks(
    green: jax.Array,
    trial_data: CisdModeTrial,
    *,
    compact_rows: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green_act = green[trial_data.occ_act_slice, :]
    green_occ = green[trial_data.occ_act_slice, trial_data.vir_act_slice]

    nrows = trial_data.vir_act_slice.stop if compact_rows else trial_data.norb
    greenp = jnp.zeros((nrows, trial_data.nvir), dtype=green.dtype)
    greenp = greenp.at[: trial_data.nocc_full, :].set(green[:, trial_data.vir_act_slice])
    greenp = greenp.at[trial_data.vir_act_slice, :].set(
        -jnp.eye(trial_data.nvir, dtype=green.dtype)
    )
    return green_act, green_occ, greenp


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdModeMeasCtx:
    rot_chol: jax.Array
    lci1: jax.Array
    reference_chol_scores: jax.Array
    chol_head_indices: jax.Array
    chol_tail_indices: jax.Array
    chol_tail_prob: jax.Array
    cfg: CisdMeasCfg
    n_mode_chunks: int
    energy_sampling: CisdModePairSamplingCfg | None
    # Static layout for setup kernels; ordinary measurements use array sharding.
    setup_mesh: Mesh | None = None
    model_sampling: ModelPairSamplingData | None = None

    def tree_flatten(self):
        children = (
            self.rot_chol,
            self.lci1,
            self.reference_chol_scores,
            self.chol_head_indices,
            self.chol_tail_indices,
            self.chol_tail_prob,
            self.model_sampling,
        )
        aux = (self.cfg, self.n_mode_chunks, self.energy_sampling, self.setup_mesh)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, n_mode_chunks, energy_sampling, setup_mesh = aux
        (
            rot_chol,
            lci1,
            reference_chol_scores,
            chol_head_indices,
            chol_tail_indices,
            chol_tail_prob,
            model_sampling,
        ) = children
        return cls(
            rot_chol=rot_chol,
            lci1=lci1,
            reference_chol_scores=reference_chol_scores,
            chol_head_indices=chol_head_indices,
            chol_tail_indices=chol_tail_indices,
            chol_tail_prob=chol_tail_prob,
            model_sampling=model_sampling,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
            energy_sampling=energy_sampling,
            setup_mesh=setup_mesh,
        )


@dataclass(frozen=True)
class CisdModePairSamplingCfg:
    """Walker--Cholesky pair sampling for the block energy.

    ``chol_head_size`` contributions are evaluated exactly for every walker.
    By default they are the original Cholesky prefix. When
    ``rank_head_by_guide`` is true, they are the largest reference- or
    population-guide scores. ``pair_sample_size`` weighted walker--Cholesky
    pairs are drawn from the remaining tail. When ``guard_head_deviations`` is
    enabled, walkers with nonfinite or anomalous exact head energies contribute
    the current reference energy and are excluded from tail sampling.
    ``walker_guide_policy='head_rms'`` additionally proposes accepted walkers
    using the noncancelling RMS magnitude of their exact head terms. The
    ``walker_guide_weight_mix`` fraction retains ordinary weight-proportional
    sampling as a defensive component. ``sample_local_walkers`` opts into
    stratified sampling on each data shard with a replicated Hamiltonian.
    The total pair budget is divided across shards; global weighting and the
    guard center are retained. This option currently requires frozen sampling
    settings (no automatic retuning). Single-device sampling is unchanged.
    """

    chol_head_size: int
    pair_sample_size: int
    rank_head_by_guide: bool = False
    guide_chol_batch_size: int = 16
    head_chol_batch_size: int = 0
    tail_probability_uniform_mix: float = 0.0
    track_half_sample_diagnostic: bool = False
    guard_head_deviations: bool = False
    walker_guide_policy: Literal["weight", "head_rms"] = "weight"
    walker_guide_weight_mix: float = 0.1
    sample_local_walkers: bool = False

    def __post_init__(self) -> None:
        if self.chol_head_size < 0:
            raise ValueError("chol_head_size must be nonnegative.")
        if self.pair_sample_size <= 0:
            raise ValueError("pair_sample_size must be positive.")
        if self.guide_chol_batch_size <= 0:
            raise ValueError("guide_chol_batch_size must be positive.")
        if self.head_chol_batch_size < 0:
            raise ValueError("head_chol_batch_size must be nonnegative.")
        if not 0.0 <= self.tail_probability_uniform_mix <= 1.0:
            raise ValueError("tail_probability_uniform_mix must lie in [0, 1].")
        if self.walker_guide_policy not in ("weight", "head_rms"):
            raise ValueError("walker_guide_policy must be 'weight' or 'head_rms'.")
        if not 0.0 < self.walker_guide_weight_mix <= 1.0:
            raise ValueError("walker_guide_weight_mix must lie in (0, 1].")
        if self.track_half_sample_diagnostic and self.pair_sample_size < 2:
            raise ValueError(
                "track_half_sample_diagnostic requires pair_sample_size to be at least two."
            )


@dataclass(frozen=True)
class CisdModePairTuningCfg:
    """Post-equilibration pair-sampling tuning policy.

    The equilibrated population always supplies the moments used to predict
    the variance of candidate estimators. ``guide_policy`` controls only the
    Cholesky head ranking and tail sampling probabilities. Multiple tuning
    populations are collected sequentially and discarded, with
    ``tuning_population_spacing_blocks`` propagation blocks between them.
    Their stored first and second moments provide leave-one-population-out
    variance validation without additional AFQMC blocks.

    ``target_tail_std_ha`` is the highest-priority target override. Otherwise
    ``target_tail_std_fraction`` multiplies the standard deviation of the
    deterministic calibration-population energies. With only one population,
    the deterministic independent-walker estimate supplies a rough fallback.
    Production length and the driver's stopping target do not affect sampling.
    ``final_error_target_ha`` and ``final_error_sampling_fraction`` are retained
    for input compatibility but no longer set the sampling budget.
    """

    guide_policy: Literal["population_rms", "hf"] = "population_rms"
    final_error_target_ha: float | None = None
    final_error_sampling_fraction: float = 0.2
    target_tail_std_fraction: float = 0.35
    target_tail_std_ha: float | None = None
    safety_factor: float = 1.0
    cross_validation_quantile: float = 1.0
    candidate_sample_sizes: tuple[int, ...] = (
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    )
    minimum_head_fraction: float = 0.0
    maximum_head_fraction: float = 1.0
    head_size_stride: int = 1
    tuning_n_chunks: int = 10
    tuning_chol_batch_size: int = 16
    tuning_population_count: int = 5
    tuning_population_spacing_blocks: int = 2
    production_initial_n_chunks: int = 1
    production_head_chol_batch_size: int = 0
    tail_probability_uniform_mix: float = 0.01
    track_half_sample_diagnostic: bool = True
    guard_head_deviations: bool = False
    walker_guide_policy: Literal["weight", "head_rms"] = "weight"
    walker_guide_weight_mix: float = 0.1
    settling_blocks: int = 5

    def __post_init__(self) -> None:
        if self.guide_policy not in ("population_rms", "hf"):
            raise ValueError("guide_policy must be 'population_rms' or 'hf'.")
        if self.walker_guide_policy not in ("weight", "head_rms"):
            raise ValueError("walker_guide_policy must be 'weight' or 'head_rms'.")
        if not 0.0 < self.walker_guide_weight_mix <= 1.0:
            raise ValueError("walker_guide_weight_mix must lie in (0, 1].")
        if self.final_error_target_ha is not None and self.final_error_target_ha <= 0.0:
            raise ValueError("final_error_target_ha must be positive when provided.")
        if not 0.0 < self.final_error_sampling_fraction <= 1.0:
            raise ValueError("final_error_sampling_fraction must lie in (0, 1].")
        if self.target_tail_std_fraction <= 0.0:
            raise ValueError("target_tail_std_fraction must be positive.")
        if self.target_tail_std_ha is not None and self.target_tail_std_ha <= 0.0:
            raise ValueError("target_tail_std_ha must be positive when provided.")
        if self.safety_factor <= 0.0:
            raise ValueError("safety_factor must be positive.")
        if not 0.0 < self.cross_validation_quantile <= 1.0:
            raise ValueError("cross_validation_quantile must lie in (0, 1].")
        if not self.candidate_sample_sizes or any(
            sample_size <= 0 for sample_size in self.candidate_sample_sizes
        ):
            raise ValueError("candidate_sample_sizes must contain positive integers.")
        if self.track_half_sample_diagnostic and any(
            sample_size < 2 for sample_size in self.candidate_sample_sizes
        ):
            raise ValueError(
                "track_half_sample_diagnostic requires candidate sample sizes of at least two."
            )
        if not 0.0 <= self.minimum_head_fraction <= self.maximum_head_fraction <= 1.0:
            raise ValueError("head fractions must satisfy 0 <= minimum <= maximum <= 1.")
        if self.head_size_stride <= 0:
            raise ValueError("head_size_stride must be positive.")
        if self.tuning_n_chunks <= 0:
            raise ValueError("tuning_n_chunks must be positive.")
        if self.tuning_chol_batch_size <= 0:
            raise ValueError("tuning_chol_batch_size must be positive.")
        if self.tuning_population_count <= 0:
            raise ValueError("tuning_population_count must be positive.")
        if self.tuning_population_spacing_blocks <= 0:
            raise ValueError("tuning_population_spacing_blocks must be positive.")
        if self.production_initial_n_chunks <= 0:
            raise ValueError("production_initial_n_chunks must be positive.")
        if self.production_head_chol_batch_size < 0:
            raise ValueError("production_head_chol_batch_size must be nonnegative.")
        if not 0.0 <= self.tail_probability_uniform_mix <= 1.0:
            raise ValueError("tail_probability_uniform_mix must lie in [0, 1].")
        if self.settling_blocks < 0:
            raise ValueError("settling_blocks must be nonnegative.")


@dataclass(frozen=True)
class CisdModePopulationStats:
    """Streaming sufficient statistics for population pair sampling."""

    term_means: np.ndarray
    term_second_moments: np.ndarray
    rms_scores: np.ndarray
    local_energies: np.ndarray
    exact_block_energy_ha: float
    independent_population_std_ha: float
    wall_seconds: float
    population_term_means: np.ndarray | None = None
    population_term_second_moments: np.ndarray | None = None


@dataclass(frozen=True)
class CisdModePairTuningResult:
    """Selected estimator and its predicted cost/noise."""

    sampling: CisdModePairSamplingCfg
    guide_policy: Literal["population_rms", "hf"]
    chol_head_fraction: float
    in_sample_single_pair_variance_ha2: float
    estimated_single_pair_variance_ha2: float
    cross_validation_fold_count: int
    cross_validation_quantile: float
    estimated_tail_std_ha: float
    guarded_tail_std_ha: float
    target_tail_std_ha: float
    target_tail_std_source: str
    calibration_std_ha: float
    estimated_pair_evaluations: int


class CisdModeEnergyCommon(NamedTuple):
    """Per-walker intermediates shared by all Cholesky contributions."""

    green: jax.Array
    greenp: jax.Array
    overlap: jax.Array
    ci1g1: jax.Array
    ci2_green: jax.Array
    base: jax.Array


def get_cisd_mode_meas_cfg(meas_ops: MeasOps) -> CisdMeasCfg | None:
    cfg = getattr(meas_ops, _CISD_MODE_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, CisdMeasCfg) else None


@partial(
    jax.jit,
    static_argnames=("vir_start", "vir_stop", "batch_size", "mesh"),
    # Setup slice fusions take the full Cholesky tensor as input. Avoid
    # autotuner copies of that input, scoped to this one-time compilation.
    compiler_options={"xla_gpu_autotune_level": 0},
)
def _build_lci1(
    chol: jax.Array,
    ci1: jax.Array,
    *,
    vir_start: int,
    vir_stop: int,
    batch_size: int = _CISD_SETUP_CHOL_BATCH_SIZE,
    mesh: Mesh | None = None,
) -> jax.Array:
    """Contract singles in batches, locally within model shards when mesh is set."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    local_contract = partial(
        _build_lci1_local, vir_start=vir_start, vir_stop=vir_stop, batch_size=batch_size
    )
    if mesh is not None:
        return jax.shard_map(
            local_contract, mesh=mesh, in_specs=(P("model"), P()), out_specs=P("model")
        )(chol, ci1)
    return local_contract(chol, ci1)


def _build_lci1_local(
    chol: jax.Array,
    ci1: jax.Array,
    *,
    vir_start: int,
    vir_stop: int,
    batch_size: int,
) -> jax.Array:
    def contract(block):
        return jnp.einsum(
            "git,pt->gip", block[:, :, vir_start:vir_stop], ci1, optimize="optimal"
        )

    n_chol = chol.shape[0]
    if n_chol <= batch_size:
        return contract(chol)
    n_batches, remainder = divmod(n_chol, batch_size)

    def body(index):
        block = lax.dynamic_slice_in_dim(chol, index * batch_size, batch_size, axis=0)
        return contract(block)

    # Map small indices, not a reshaped/sliced prefix of the full input.
    blocks = lax.map(body, jnp.arange(n_batches, dtype=jnp.int32))
    result = blocks.reshape(n_batches * batch_size, chol.shape[1], ci1.shape[0])
    if remainder:
        result = jnp.concatenate((result, contract(chol[n_batches * batch_size :])), axis=0)
    return result


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: CisdModeTrial,
    *,
    cfg: CisdMeasCfg = CisdMeasCfg(memory_mode="high"),
    n_mode_chunks: int = 1,
    energy_sampling: CisdModePairSamplingCfg | None = None,
) -> CisdModeMeasCtx:
    """Build full-Cholesky measurement intermediates.

    ``n_mode_chunks=1`` evaluates the complete mode axis in one batch. Larger
    values reduce mode-dependent temporary memory by scanning over that many
    partitions; the requested count is capped at the retained mode rank.
    """
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if cfg.memory_mode != "high":
        raise ValueError("Mode-native CISD measurements currently require memory_mode='high'.")

    if ham_data.basis != "restricted":
        raise ValueError("CISD mode MeasOps requires HamChol.basis == 'restricted'.")

    chol = ham_data.chol
    setup_mesh = cholesky_model_mesh(chol)
    n_chol = int(chol.shape[0])
    if energy_sampling is not None and energy_sampling.chol_head_size > n_chol:
        raise ValueError(
            f"chol_head_size must not exceed the number of Cholesky vectors ({n_chol})."
        )
    rot_chol = chol[:, : trial_data.nocc_full, :]
    lci1 = _build_lci1(
        chol,
        trial_data.ci1,
        vir_start=trial_data.vir_act_slice.start,
        vir_stop=trial_data.vir_act_slice.stop,
        mesh=setup_mesh,
    )
    meas_ctx = CisdModeMeasCtx(
        rot_chol=rot_chol,
        lci1=lci1,
        reference_chol_scores=jnp.empty((0,), dtype=jnp.float64),
        chol_head_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_prob=jnp.empty((0,), dtype=jnp.float64),
        cfg=cfg,
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
        energy_sampling=energy_sampling,
        setup_mesh=setup_mesh,
    )
    if energy_sampling is not None:
        guide_scores = _build_reference_chol_scores(
            ham_data,
            meas_ctx,
            trial_data,
            chol_batch_size=energy_sampling.guide_chol_batch_size,
        )
        meas_ctx = replace(meas_ctx, reference_chol_scores=guide_scores)
        meas_ctx = configure_cisd_mode_pair_sampling(
            meas_ctx,
            energy_sampling,
            guide_scores,
        )
        print(
            "[sampling] configured HF-guide CISD-mode pair estimator: "
            f"chol_head_size={energy_sampling.chol_head_size}/{n_chol} "
            f"({energy_sampling.chol_head_size / n_chol:.3%}), "
            f"pair_sample_size={energy_sampling.pair_sample_size}, "
            f"ranked_head={energy_sampling.rank_head_by_guide}, "
            f"head_guard={energy_sampling.guard_head_deviations}, "
            f"walker_guide={energy_sampling.walker_guide_policy}, "
            f"walker_weight_mix={energy_sampling.walker_guide_weight_mix:.3%}."
        )
    return meas_ctx


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(
        green, trial_data, compact_rows=True
    )

    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    ci1gp = jnp.einsum("pt,it->pi", trial_data.ci1, greenp, optimize="optimal")
    gci1gp = jnp.einsum("pj,pi->ij", green_act, ci1gp, optimize="optimal")

    projections, kg = mode_apply(trial_data, green_occ)
    gkg = mode_quadratic(trial_data, green_occ, projections)
    overlap = 1.0 + 2.0 * ci1g + gkg

    cisd_green = -2.0 * (greenp @ kg.T) @ green_act
    correction = _force_bias_chol_contract_high_realimag(
        # The correction has no rows in the excluded virtual space.
        ham_data.chol[:, : greenp.shape[0], :],
        cisd_green - 2.0 * gci1gp,
        meas_ctx.cfg,
    )
    return (2.0 * lg + 4.0 * ci1g * lg + 2.0 * lg * gkg + correction) / overlap


def mode_quadratic_matrices(
    trial_data: Any,
    matrices: jax.Array,
    *,
    n_mode_chunks: int = 1,
) -> jax.Array:
    """Evaluate ``x.T @ K @ x`` for a batch of pair-space matrices.

    Every retained mode is included. ``n_mode_chunks`` only bounds the
    temporary projection matrix and introduces no stochastic approximation.
    """
    pair_shape = (trial_data.nocc, trial_data.nvir)
    if matrices.shape[-2:] != pair_shape:
        raise ValueError(
            f"matrices must end in pair-space shape {pair_shape}, got {matrices.shape}."
        )

    leading_shape = matrices.shape[:-2]
    matrices_flat = matrices.reshape((-1,) + pair_shape)
    rank = trial_data.mode_rank

    def evaluate_chunk(eigenvalues, modes):
        matrices_r = jnp.real(matrices_flat).astype(modes.dtype)
        projections_r = jnp.einsum("spt,rpt->sr", matrices_r, modes, optimize="optimal")
        if jnp.issubdtype(matrices.dtype, jnp.complexfloating):
            matrices_i = jnp.imag(matrices_flat).astype(modes.dtype)
            projections_i = jnp.einsum("spt,rpt->sr", matrices_i, modes, optimize="optimal")
            projections = projections_r.astype(jnp.complex128) + 1.0j * projections_i.astype(
                jnp.complex128
            )
            reduction_dtype = jnp.complex128
        else:
            projections = projections_r.astype(jnp.float64)
            reduction_dtype = jnp.float64
        return jnp.sum(
            eigenvalues.astype(jnp.float64)[None, :] * projections * projections,
            axis=1,
            dtype=reduction_dtype,
        )

    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    n_mode_chunks = min(int(n_mode_chunks), rank)
    if n_mode_chunks == 1:
        values = evaluate_chunk(trial_data.eigenvalues, trial_data.modes)
        return values.reshape(leading_shape)

    base_chunk_size = rank // n_mode_chunks
    n_larger_chunks = rank % n_mode_chunks
    chunk_size = base_chunk_size + int(n_larger_chunks > 0)
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    result_dtype = (
        jnp.complex128 if jnp.issubdtype(matrices.dtype, jnp.complexfloating) else jnp.float64
    )
    # Each device can carry a different partial sum under shard_map.
    zero = jnp.zeros_like(matrices_flat, shape=(matrices_flat.shape[0],), dtype=result_dtype)

    def scan_body(total, chunk_index):
        is_larger = chunk_index < n_larger_chunks
        chunk_length = base_chunk_size + is_larger.astype(jnp.int32)
        start = chunk_index * base_chunk_size + jnp.minimum(chunk_index, n_larger_chunks)
        indices = start + chunk_offsets
        valid = chunk_offsets < chunk_length
        indices = jnp.minimum(indices, rank - 1)
        eigenvalues = jnp.where(valid, trial_data.eigenvalues[indices], 0.0)
        modes = trial_data.modes[indices]
        return total + evaluate_chunk(eigenvalues, modes), None

    values, _ = lax.scan(
        scan_body,
        zero,
        jnp.arange(n_mode_chunks, dtype=jnp.int32),
    )
    return values.reshape(leading_shape)


def _mode_quadratic_matrices(
    trial_data: CisdModeTrial,
    meas_ctx: CisdModeMeasCtx,
    matrices: jax.Array,
) -> jax.Array:
    """Backward-compatible wrapper around the trial-independent mode kernel."""

    return mode_quadratic_matrices(
        trial_data,
        matrices,
        n_mode_chunks=meas_ctx.n_mode_chunks,
    )


def _cisd_mode_energy_common(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> CisdModeEnergyCommon:
    """Build the exact per-walker base and reusable Cholesky intermediates."""
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, green_occ, greenp = _active_green_blocks(green, trial_data)

    h1 = ham_data.h1
    chol = ham_data.chol
    hg = jnp.einsum("pj,pj->", h1[: trial_data.nocc_full, :], green, optimize="optimal")
    e1_0 = 2.0 * hg

    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, green_occ, optimize="optimal")
    ci1_green = (greenp @ trial_data.ci1.T) @ green_act
    e1_1 = 4.0 * ci1g * hg - 2.0 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")

    projections, kg = mode_apply(trial_data, green_occ)
    doubles = mode_quadratic(trial_data, green_occ, projections)
    ci2_green = (greenp @ kg.T) @ green_act
    e1_2 = 2.0 * hg * doubles - 2.0 * jnp.einsum("ij,ij->", h1, ci2_green, optimize="optimal")

    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    e2_0_direct = 2.0 * (lg @ lg)
    lci1g = _force_bias_chol_contract_high_realimag(chol, ci1_green, meas_ctx.cfg)
    e2_1_2 = -2.0 * (lci1g @ lg)
    lci2g = _force_bias_chol_contract_high_realimag(chol, ci2_green, meas_ctx.cfg)
    e2_2_2_1 = -(lci2g @ lg)

    overlap = 1.0 + 2.0 * ci1g + doubles
    ci1g1 = trial_data.ci1 @ green[:, trial_data.vir_act_slice].T
    base = (
        ham_data.h0 + e2_0_direct + (e1_0 + e1_1 + e1_2 + 2.0 * e2_1_2 + 4.0 * e2_2_2_1) / overlap
    )
    return CisdModeEnergyCommon(
        green=green,
        greenp=greenp,
        overlap=overlap,
        ci1g1=ci1g1,
        ci2_green=ci2_green,
        base=base,
    )


def _cisd_mode_chol_terms(
    common: CisdModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    lci1: jax.Array,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    """Return one walker's residual energy contribution for each Cholesky vector."""
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, common.green, optimize="optimal")
    e20 = -jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2), axis=(-1, -2))
    e2131 = jnp.einsum(
        "gpq,gqa,ap->g",
        lg1,
        lg1[:, :, trial_data.occ_act_slice],
        common.ci1g1,
        optimize="optimal",
    )
    lci1g_mat = jnp.einsum("gia,qi->gaq", lci1, common.green, optimize="optimal")
    e2132 = -jnp.einsum(
        "gaq,gqa->g",
        lci1g_mat,
        lg1[:, :, trial_data.occ_act_slice],
        optimize="optimal",
    )

    gl = _energy_gl_batched_realimag(common.green, chol, meas_ctx.cfg)
    lci2_green = jnp.einsum("gpk,ik->gpi", rot_chol, common.ci2_green, optimize="optimal")
    e2222 = 0.5 * jnp.einsum("gpi,gpi->g", gl, lci2_green, optimize="optimal")
    glgp = jnp.einsum("gpi,it->gpt", gl, common.greenp, optimize="optimal")
    glgp = glgp[:, trial_data.occ_act_slice, :]
    e223 = _mode_quadratic_matrices(trial_data, meas_ctx, glgp)
    return e20 + (2.0 * (e2131 + e2132) + 4.0 * e2222 + e223) / common.overlap


def _cisd_mode_chol_terms_for_walkers(
    common: CisdModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    lci1: jax.Array,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    n_chunks: int = 1,
) -> jax.Array:
    """Return residual terms for every walker--Cholesky combination."""
    return wk.vmap_chunked(
        lambda common_i: _cisd_mode_chol_terms(
            common_i,
            chol,
            rot_chol,
            lci1,
            meas_ctx,
            trial_data,
        ),
        n_chunks=n_chunks,
    )(common)


def _cisd_mode_chol_index_terms(
    common: CisdModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    *,
    n_chunks: int,
) -> jax.Array:
    """Return one walker's terms at arbitrary Cholesky indices."""

    return wk.vmap_chunked(
        lambda chol_i: _cisd_mode_chol_terms(
            common,
            ham_data.chol[chol_i][None, ...],
            meas_ctx.rot_chol[chol_i][None, ...],
            meas_ctx.lci1[chol_i][None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        shard_walkers=False,
    )(chol_indices)


def _cisd_mode_chol_index_moments_for_walkers(
    common: CisdModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
    compute_squared_norm: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Return the head sum and noncancelling real squared norm per walker."""

    head_size = int(chol_indices.shape[0])
    if head_size == 0:
        return (
            jnp.zeros_like(common.base, dtype=jnp.complex128),
            jnp.zeros_like(jnp.real(common.base), dtype=jnp.float64),
        )

    batch_size = head_size if chol_batch_size <= 0 else min(chol_batch_size, head_size)
    n_batches = math.ceil(head_size / batch_size)
    padded_size = n_batches * batch_size
    padded_indices = jnp.pad(chol_indices, (0, padded_size - head_size)).reshape(
        n_batches, batch_size
    )
    valid = (jnp.arange(padded_size) < head_size).reshape(n_batches, batch_size)

    if n_batches == 1:
        terms = _cisd_mode_chol_terms_for_walkers(
            common,
            ham_data.chol[chol_indices],
            meas_ctx.rot_chol[chol_indices],
            meas_ctx.lci1[chol_indices],
            meas_ctx,
            trial_data,
            n_chunks=n_walker_chunks,
        )
        total = jnp.sum(terms, axis=1, dtype=jnp.complex128)
        if compute_squared_norm:
            terms_real = jnp.real(terms).astype(jnp.float64)
            squared_norm = jnp.sum(
                terms_real**2,
                axis=1,
                dtype=jnp.float64,
            )
        else:
            squared_norm = jnp.zeros_like(jnp.real(common.base), dtype=jnp.float64)
        return total, squared_norm

    def scan_body(carry, xs):
        total, squared_norm = carry
        indices_i, valid_i = xs
        terms_i = _cisd_mode_chol_terms_for_walkers(
            common,
            ham_data.chol[indices_i],
            meas_ctx.rot_chol[indices_i],
            meas_ctx.lci1[indices_i],
            meas_ctx,
            trial_data,
            n_chunks=n_walker_chunks,
        )
        terms_i = jnp.where(valid_i[None, :], terms_i, 0.0)
        total = total + jnp.sum(terms_i, axis=1, dtype=jnp.complex128)
        if compute_squared_norm:
            terms_real = jnp.real(terms_i).astype(jnp.float64)
            squared_norm = squared_norm + jnp.sum(
                terms_real**2,
                axis=1,
                dtype=jnp.float64,
            )
        return (total, squared_norm), None

    zero_total = jnp.zeros_like(common.base, dtype=jnp.complex128)
    zero_squared_norm = jnp.zeros_like(jnp.real(common.base), dtype=jnp.float64)
    (total, squared_norm), _ = lax.scan(
        scan_body,
        (zero_total, zero_squared_norm),
        (padded_indices, valid),
    )
    return total, squared_norm


def _cisd_mode_chol_index_sum_for_walkers(
    common: CisdModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
) -> jax.Array:
    """Sum arbitrary head terms while bounding the gathered Cholesky batch."""

    total, _ = _cisd_mode_chol_index_moments_for_walkers(
        common,
        chol_indices,
        ham_data,
        meas_ctx,
        trial_data,
        n_walker_chunks=n_walker_chunks,
        chol_batch_size=chol_batch_size,
        compute_squared_norm=False,
    )
    return total


def _cisd_mode_chol_pair_terms(
    common: CisdModeEnergyCommon,
    sample_walker: jax.Array,
    sample_chol: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    n_chunks: int = 1,
) -> jax.Array:
    """Return sampled pair terms while gathering only one microbatch at a time."""
    return wk.vmap_chunked(
        lambda walker_i, chol_i: _cisd_mode_chol_terms(
            tree_util.tree_map(lambda value: value[walker_i], common),
            ham_data.chol[chol_i][None, ...],
            meas_ctx.rot_chol[chol_i][None, ...],
            meas_ctx.lci1[chol_i][None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        in_axes=(0, 0),
        shard_walkers=False,
    )(sample_walker, sample_chol)


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    """Complete deterministic local energy from every stored K mode."""
    common = _cisd_mode_energy_common(walker, ham_data, meas_ctx, trial_data)
    chol_terms = _cisd_mode_chol_terms(
        common,
        ham_data.chol,
        meas_ctx.rot_chol,
        meas_ctx.lci1,
        meas_ctx,
        trial_data,
    )
    return common.base + jnp.sum(chol_terms, dtype=jnp.complex128)


def initial_energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    """Same deterministic energy with bounded Cholesky workspace for initialization."""
    common = _cisd_mode_energy_common(walker, ham_data, meas_ctx, trial_data)
    if ham_data.chol.shape[0] == 0:
        return common.base
    if meas_ctx.setup_mesh is not None:
        def local_sum(common, chol, rot_chol, lci1, trial):
            total = _initial_energy_chol_sum(common, chol, rot_chol, lci1, meas_ctx, trial)
            return lax.psum(total, "model")

        total = jax.shard_map(
            local_sum,
            mesh=meas_ctx.setup_mesh,
            in_specs=(P(), P("model"), P("model"), P("model"), P()),
            out_specs=P(),
        )(common, ham_data.chol, meas_ctx.rot_chol, meas_ctx.lci1, trial_data)
    else:
        total = _initial_energy_chol_sum(
            common, ham_data.chol, meas_ctx.rot_chol, meas_ctx.lci1, meas_ctx, trial_data
        )
    # common.base already contains the global one-body and direct terms.
    # Reduce only residual Cholesky contributions, then add that base once.
    return common.base + total


def _initial_energy_chol_sum(
    common: CisdModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    lci1: jax.Array,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
) -> jax.Array:
    def sum_terms(chol, rot_chol, lci1):
        terms = _cisd_mode_chol_terms(common, chol, rot_chol, lci1, meas_ctx, trial_data)
        return jnp.sum(terms, dtype=jnp.complex128)

    n_chol = chol.shape[0]
    batch_size = _CISD_INITIAL_ENERGY_CHOL_BATCH_SIZE
    if n_chol <= batch_size:
        return sum_terms(chol, rot_chol, lci1)

    def accumulate(index, total):
        start = index * batch_size
        return total + sum_terms(
            lax.dynamic_slice_in_dim(chol, start, batch_size, axis=0),
            lax.dynamic_slice_in_dim(rot_chol, start, batch_size, axis=0),
            lax.dynamic_slice_in_dim(lci1, start, batch_size, axis=0),
        )

    # Reduce each batch to a scalar; never collect full-Cholesky energy tensors.
    n_batches, remainder = divmod(n_chol, batch_size)
    total = lax.fori_loop(
        0, n_batches, accumulate, jnp.zeros_like(chol, shape=(), dtype=jnp.complex128)
    )
    if remainder:
        start = n_batches * batch_size
        total = total + sum_terms(chol[start:], rot_chol[start:], lci1[start:])
    return total


@partial(
    jax.jit,
    static_argnames=("chol_batch_size",),
    # Compile setup as a whole to discard the unused common.base work.
    # Avoid profiling copies of large inputs only for this setup call.
    compiler_options={"xla_gpu_autotune_level": 0},
)
def _build_reference_chol_scores(
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    *,
    chol_batch_size: int,
) -> jax.Array:
    """Build bounded-memory HF-reference scores for every Cholesky vector."""
    n_chol = int(ham_data.chol.shape[0])
    if n_chol == 0:
        return jnp.empty((0,), dtype=jnp.float64)
    reference_walker = jnp.eye(
        trial_data.norb,
        trial_data.nocc_full,
        dtype=jnp.complex128,
    )
    common = _cisd_mode_energy_common(
        reference_walker,
        ham_data,
        meas_ctx,
        trial_data,
    )
    if meas_ctx.setup_mesh is not None:
        def local_scores(common, chol, rot_chol, lci1, trial):
            n_local = chol.shape[0]

            def term(index):
                return _cisd_mode_chol_terms(
                    common, chol[index][None], rot_chol[index][None], lci1[index][None],
                    meas_ctx, trial,
                )[0]

            terms = wk.vmap_chunked(
                term,
                n_chunks=max(1, math.ceil(n_local / chol_batch_size)),
                in_axes=0,
                shard_walkers=False,
            )(jnp.arange(n_local, dtype=jnp.int32))
            return jnp.maximum(jnp.abs(terms).astype(jnp.float64), 1.0e-300)

        return jax.shard_map(
            local_scores,
            mesh=meas_ctx.setup_mesh,
            in_specs=(P(), P("model"), P("model"), P("model"), P()),
            out_specs=P("model"),
        )(common, ham_data.chol, meas_ctx.rot_chol, meas_ctx.lci1, trial_data)
    indices = jnp.arange(n_chol, dtype=jnp.int32)
    n_chunks = max(1, math.ceil(n_chol / chol_batch_size))
    terms = _cisd_mode_chol_index_terms(
        common,
        indices,
        ham_data,
        meas_ctx,
        trial_data,
        n_chunks=n_chunks,
    )
    return jnp.maximum(jnp.abs(terms).astype(jnp.float64), 1.0e-300)


def configure_cisd_mode_pair_sampling(
    meas_ctx: CisdModeMeasCtx,
    sampling: CisdModePairSamplingCfg,
    guide_scores: jax.Array,
) -> CisdModeMeasCtx:
    """Attach a prefix or guide-ranked head and normalized tail probabilities."""

    if meas_ctx.model_sampling is not None:
        raise ValueError("Rebuild the Cholesky layout and contexts before changing the frozen guide.")
    scores = jnp.asarray(guide_scores, dtype=jnp.float64)
    n_chol = int(meas_ctx.rot_chol.shape[0])
    if scores.shape != (n_chol,):
        raise ValueError(f"guide_scores must have shape {(n_chol,)}, got {scores.shape}.")
    if sampling.chol_head_size > n_chol:
        raise ValueError(
            f"chol_head_size must not exceed the number of Cholesky vectors ({n_chol})."
        )

    if sampling.rank_head_by_guide:
        order = jnp.argsort(-scores)
    else:
        order = jnp.arange(n_chol, dtype=jnp.int32)
    head_indices = jnp.sort(order[: sampling.chol_head_size]).astype(jnp.int32)
    tail_indices = jnp.sort(order[sampling.chol_head_size :]).astype(jnp.int32)
    if int(tail_indices.shape[0]) == 0:
        tail_prob = jnp.empty((0,), dtype=jnp.float64)
    else:
        tail_scores = jnp.maximum(scores[tail_indices], 1.0e-300)
        tail_prob = tail_scores / jnp.sum(tail_scores, dtype=jnp.float64)
        uniform_mix = sampling.tail_probability_uniform_mix
        if uniform_mix > 0.0:
            uniform_prob = jnp.full_like(tail_prob, 1.0 / tail_prob.shape[0])
            tail_prob = (1.0 - uniform_mix) * tail_prob + uniform_mix * uniform_prob
    return replace(
        meas_ctx,
        chol_head_indices=head_indices,
        chol_tail_indices=tail_indices,
        chol_tail_prob=tail_prob,
        energy_sampling=sampling,
    )


def _local_cholesky_context_specs(ctx):
    specs = jax.tree.map(lambda _: P(), ctx)
    return replace(specs, rot_chol=P("model"), lci1=P("model"))


def _local_cholesky_head_terms(common, indices, h, ctx, trial, *, n_chunks):
    return _cisd_mode_chol_terms_for_walkers(
        common, h.chol[indices], ctx.rot_chol[indices], ctx.lci1[indices],
        ctx, trial, n_chunks=n_chunks,
    )


def pair_sampled_block_energy(
    walkers: jax.Array,
    weights: jax.Array,
    overlaps: jax.Array,
    rng_key: jax.Array,
    n_chunks: int,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    e_ref: jax.Array,
    energy_clip_threshold: jax.Array,
) -> jax.Array | BlockEnergyEstimate:
    """Exact Cholesky head plus sampled walker--Cholesky tail energy.

    Walkers are sampled according to their normalized phaseless weights and
    tail Cholesky vectors according to the guide stored in ``meas_ctx``. All
    retained K modes are summed exactly for every evaluated pair. When the
    head-deviation guard is enabled, a walker whose exact head energy differs
    from the finite weighted head center by more than
    ``energy_clip_threshold`` contributes ``e_ref`` and is excluded from the
    sampled tail. The surviving tail estimate retains its original population
    weight rather than being renormalized.
    """
    del overlaps
    sampling = meas_ctx.energy_sampling
    if sampling is None:
        raise ValueError("pair_sampled_block_energy requires an energy sampling config.")

    if meas_ctx.model_sampling is not None and sampling.sample_local_walkers:
        raise ValueError("A Cholesky layout already handles local walkers; leave sample_local_walkers=False.")
    local_mesh = local_pair_mesh(walkers, sampling.sample_local_walkers)
    if local_mesh is not None:
        common, local_head, local_squared = local_common_and_head(
            local_mesh, walkers, ham_data, meas_ctx, trial_data,
            common_fn=_cisd_mode_energy_common,
            moments_fn=_cisd_mode_chol_index_moments_for_walkers, n_chunks=n_chunks,
        )
    else:
        common = wk.vmap_chunked(
            _cisd_mode_energy_common,
            n_chunks=n_chunks,
            in_axes=(0, None, None, None),
        )(walkers, ham_data, meas_ctx, trial_data)

    weights_real = jnp.real(weights).astype(jnp.float64)
    weight_sum = jnp.sum(weights_real, dtype=jnp.float64)
    weight_sum_safe = jnp.where(weight_sum == 0.0, 1.0, weight_sum)
    norm_weights = weights_real / weight_sum_safe

    if meas_ctx.model_sampling is not None:
        head_sum, head_squared_norm = local_cholesky_head(
            common, ham_data, meas_ctx, trial_data, terms_fn=_local_cholesky_head_terms,
            context_specs_fn=_local_cholesky_context_specs, n_chunks=n_chunks,
        )
        head_energy = jnp.real(common.base + head_sum)
    elif local_mesh is not None:
        head_energy = jnp.real(common.base + local_head)
        head_squared_norm = local_squared
    elif sampling.chol_head_size > 0 and sampling.walker_guide_policy == "head_rms":
        head_sum, head_squared_norm = _cisd_mode_chol_index_moments_for_walkers(
            common,
            meas_ctx.chol_head_indices,
            ham_data,
            meas_ctx,
            trial_data,
            n_walker_chunks=n_chunks,
            chol_batch_size=sampling.head_chol_batch_size,
        )
        head_energy = jnp.real(common.base + head_sum)
    elif sampling.chol_head_size > 0:
        head_sum = _cisd_mode_chol_index_sum_for_walkers(
            common,
            meas_ctx.chol_head_indices,
            ham_data,
            meas_ctx,
            trial_data,
            n_walker_chunks=n_chunks,
            chol_batch_size=sampling.head_chol_batch_size,
        )
        head_energy = jnp.real(common.base + head_sum)
        head_squared_norm = jnp.zeros_like(head_energy, dtype=jnp.float64)
    else:
        head_energy = jnp.real(common.base)
        head_squared_norm = jnp.zeros_like(head_energy, dtype=jnp.float64)

    finite_head = jnp.isfinite(head_energy)
    if sampling.guard_head_deviations:
        finite_head_weights = jnp.where(finite_head, norm_weights, 0.0)
        finite_head_weight = jnp.sum(finite_head_weights, dtype=jnp.float64)
        finite_head_weight_safe = jnp.where(finite_head_weight == 0.0, 1.0, finite_head_weight)
        head_center = (
            jnp.sum(
                finite_head_weights * jnp.where(finite_head, head_energy, 0.0),
                dtype=jnp.float64,
            )
            / finite_head_weight_safe
        )
        head_center = jnp.where(finite_head_weight == 0.0, jnp.real(e_ref), head_center)
        head_guarded = (~finite_head) | (jnp.abs(head_energy - head_center) > energy_clip_threshold)
    else:
        head_guarded = jnp.zeros_like(finite_head)

    safe_head_energy = jnp.where(finite_head, head_energy, jnp.real(e_ref))
    guarded_head_energy = jnp.where(head_guarded, jnp.real(e_ref), safe_head_energy)
    block_head = jnp.sum(norm_weights * guarded_head_energy, dtype=jnp.float64)
    accepted_weights = jnp.where(head_guarded, 0.0, norm_weights)
    accepted_weight = jnp.sum(accepted_weights, dtype=jnp.float64)
    accepted_weight_safe = jnp.where(accepted_weight == 0.0, 1.0, accepted_weight)
    accepted_probabilities = accepted_weights / accepted_weight_safe
    accepted_probabilities = jnp.where(
        accepted_weight == 0.0,
        jnp.ones_like(accepted_weights) / accepted_weights.shape[0],
        accepted_probabilities,
    )
    diagnostics: dict[str, jax.Array] = {}
    if sampling.guard_head_deviations:
        diagnostics[d_energy_head_guard_count] = jnp.sum(head_guarded, dtype=jnp.int32)
        diagnostics[d_energy_head_guard_weight] = jnp.sum(
            norm_weights * head_guarded,
            dtype=jnp.float64,
        )

    if sampling.walker_guide_policy == "head_rms":
        head_rms_scores = jnp.sqrt(jnp.maximum(head_squared_norm, 0.0))
        head_rms_scores = jnp.where(
            head_guarded | (~jnp.isfinite(head_rms_scores)),
            0.0,
            head_rms_scores,
        )
        guided_weights = accepted_weights * head_rms_scores
        guided_weight_sum = jnp.sum(guided_weights, dtype=jnp.float64)
        guided_weight_sum_safe = jnp.where(guided_weight_sum == 0.0, 1.0, guided_weight_sum)
        guided_probabilities = guided_weights / guided_weight_sum_safe
        guided_probabilities = jnp.where(
            guided_weight_sum == 0.0,
            accepted_probabilities,
            guided_probabilities,
        )
        weight_mix = sampling.walker_guide_weight_mix
        walker_probabilities = (
            weight_mix * accepted_probabilities + (1.0 - weight_mix) * guided_probabilities
        )
        walker_corrections = jnp.where(
            walker_probabilities > 0.0,
            accepted_weights / walker_probabilities,
            0.0,
        )
        diagnostics[d_energy_walker_guide_ess] = 1.0 / jnp.sum(
            walker_probabilities**2,
            dtype=jnp.float64,
        )
        diagnostics[d_energy_walker_guide_max_correction] = jnp.max(walker_corrections)
    else:
        walker_probabilities = accepted_probabilities
        walker_corrections = jnp.full_like(accepted_weights, accepted_weight)

    tail_size = int(meas_ctx.chol_tail_prob.shape[0])
    if tail_size == 0:
        if sampling.track_half_sample_diagnostic:
            diagnostics[d_energy_sampling_noise] = jnp.asarray(0.0, dtype=jnp.float64)
        if diagnostics:
            return BlockEnergyEstimate(energy=block_head, diagnostics=diagnostics)
        return block_head

    if meas_ctx.model_sampling is not None:
        tail_estimate, sampling_noise = local_cholesky_tail(
            common, accepted_weights, walker_probabilities, rng_key,
            ham_data, meas_ctx, trial_data, pair_fn=_cisd_mode_chol_pair_terms,
            context_specs_fn=_local_cholesky_context_specs, n_chunks=n_chunks,
        )
    elif local_mesh is not None:
        tail_estimate, sampling_noise = local_pair_tail(
            local_mesh, common, accepted_weights, walker_probabilities, rng_key,
            ham_data, meas_ctx, trial_data, pair_fn=_cisd_mode_chol_pair_terms, n_chunks=n_chunks,
        )
    if meas_ctx.model_sampling is not None or local_mesh is not None:
        if sampling.track_half_sample_diagnostic:
            diagnostics[d_energy_sampling_noise] = sampling_noise
        energy = block_head + tail_estimate
        return BlockEnergyEstimate(energy=energy, diagnostics=diagnostics) if diagnostics else energy

    key_walker, key_chol = jax.random.split(rng_key)
    sample_walker = jax.random.choice(
        key_walker,
        weights_real.shape[0],
        shape=(sampling.pair_sample_size,),
        replace=True,
        p=walker_probabilities,
    )
    sample_chol_rel = jax.random.choice(
        key_chol,
        tail_size,
        shape=(sampling.pair_sample_size,),
        replace=True,
        p=meas_ctx.chol_tail_prob,
    )
    sample_chol = meas_ctx.chol_tail_indices[sample_chol_rel]
    walker_batch_size = (int(weights_real.shape[0]) + n_chunks - 1) // n_chunks
    pair_n_chunks = (sampling.pair_sample_size + walker_batch_size - 1) // walker_batch_size
    sample_terms = _cisd_mode_chol_pair_terms(
        common,
        sample_walker,
        sample_chol,
        ham_data,
        meas_ctx,
        trial_data,
        n_chunks=pair_n_chunks,
    )
    importance_samples = (
        walker_corrections[sample_walker]
        * jnp.real(sample_terms)
        / meas_ctx.chol_tail_prob[sample_chol_rel]
    )
    tail_estimate = jnp.mean(importance_samples, dtype=jnp.float64)
    energy = block_head + tail_estimate
    if not sampling.track_half_sample_diagnostic:
        if diagnostics:
            return BlockEnergyEstimate(energy=energy, diagnostics=diagnostics)
        return energy

    first_size = sampling.pair_sample_size // 2
    second_size = sampling.pair_sample_size - first_size
    first_mean = jnp.mean(importance_samples[:first_size], dtype=jnp.float64)
    second_mean = jnp.mean(importance_samples[first_size:], dtype=jnp.float64)
    diagnostic_scale = math.sqrt(first_size * second_size) / sampling.pair_sample_size
    sampling_noise = diagnostic_scale * (first_mean - second_mean)
    diagnostics[d_energy_sampling_noise] = sampling_noise
    return BlockEnergyEstimate(
        energy=energy,
        diagnostics=diagnostics,
    )


@jax.jit
def _cisd_mode_population_common_batch(walkers, ham_data, meas_ctx, trial_data):
    return jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham_data, meas_ctx, trial_data)


@jax.jit
def _cisd_mode_population_term_batch(
    common,
    chol_indices,
    ham_data,
    meas_ctx,
    trial_data,
):
    return _cisd_mode_chol_terms_for_walkers(
        common,
        ham_data.chol[chol_indices],
        meas_ctx.rot_chol[chol_indices],
        meas_ctx.lci1[chol_indices],
        meas_ctx,
        trial_data,
        n_chunks=1,
    )


def stream_cisd_mode_population_statistics(
    walkers: jax.Array,
    weights: jax.Array,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    *,
    n_walker_chunks: int = 10,
    chol_batch_size: int = 16,
) -> CisdModePopulationStats:
    """Accumulate population statistics without forming ``Nw x Nchol`` terms."""

    if n_walker_chunks <= 0:
        raise ValueError("n_walker_chunks must be positive.")
    if chol_batch_size <= 0:
        raise ValueError("chol_batch_size must be positive.")

    start_time = time.perf_counter()
    n_walkers = int(walkers.shape[0])
    n_chol = int(ham_data.chol.shape[0])
    walker_batch_size = math.ceil(n_walkers / min(n_walker_chunks, n_walkers))

    weights_np = np.maximum(
        np.real(np.asarray(jax.device_get(weights), dtype=np.complex128)),
        0.0,
    ).astype(np.float64)
    weight_sum = float(np.sum(weights_np, dtype=np.float64))
    norm_weights = (
        weights_np / weight_sum
        if np.isfinite(weight_sum) and weight_sum > 0.0
        else np.full(n_walkers, 1.0 / n_walkers, dtype=np.float64)
    )

    term_means = np.zeros(n_chol, dtype=np.float64)
    term_second_moments = np.zeros(n_chol, dtype=np.float64)
    local_energies = np.empty(n_walkers, dtype=np.float64)

    for walker_start in range(0, n_walkers, walker_batch_size):
        walker_stop = min(walker_start + walker_batch_size, n_walkers)
        valid_walkers = walker_stop - walker_start
        walker_indices = np.minimum(
            walker_start + np.arange(walker_batch_size, dtype=np.int32),
            n_walkers - 1,
        )
        common = _cisd_mode_population_common_batch(
            walkers[jnp.asarray(walker_indices)],
            ham_data,
            meas_ctx,
            trial_data,
        )
        base = np.real(np.asarray(jax.device_get(common.base), dtype=np.complex128))
        term_sum = np.zeros(walker_batch_size, dtype=np.float64)
        batch_weights = np.zeros(walker_batch_size, dtype=np.float64)
        batch_weights[:valid_walkers] = norm_weights[walker_start:walker_stop]

        for chol_start in range(0, n_chol, chol_batch_size):
            chol_stop = min(chol_start + chol_batch_size, n_chol)
            valid_chol = chol_stop - chol_start
            chol_indices = np.minimum(
                chol_start + np.arange(chol_batch_size, dtype=np.int32),
                n_chol - 1,
            )
            terms = _cisd_mode_population_term_batch(
                common,
                jnp.asarray(chol_indices),
                ham_data,
                meas_ctx,
                trial_data,
            )
            terms_np = np.real(np.asarray(jax.device_get(terms), dtype=np.complex128))
            valid_terms = terms_np[:, :valid_chol]
            term_sum += np.sum(valid_terms, axis=1, dtype=np.float64)
            term_means[chol_start:chol_stop] += np.sum(
                batch_weights[:, None] * valid_terms,
                axis=0,
                dtype=np.float64,
            )
            term_second_moments[chol_start:chol_stop] += np.sum(
                batch_weights[:, None] * valid_terms**2,
                axis=0,
                dtype=np.float64,
            )

        local_energies[walker_start:walker_stop] = base[:valid_walkers] + term_sum[:valid_walkers]

    exact_block_energy = float(np.sum(norm_weights * local_energies, dtype=np.float64))
    sum_weight_squared = float(np.sum(norm_weights**2, dtype=np.float64))
    correction = max(1.0e-300, 1.0 - sum_weight_squared)
    individual_variance = float(
        np.sum(norm_weights * (local_energies - exact_block_energy) ** 2, dtype=np.float64)
        / correction
    )
    independent_population_std = math.sqrt(max(0.0, individual_variance * sum_weight_squared))
    rms_scores = np.sqrt(np.maximum(term_second_moments, 0.0))
    return CisdModePopulationStats(
        term_means=term_means,
        term_second_moments=term_second_moments,
        rms_scores=rms_scores,
        local_energies=local_energies,
        exact_block_energy_ha=exact_block_energy,
        independent_population_std_ha=independent_population_std,
        wall_seconds=time.perf_counter() - start_time,
        population_term_means=term_means[None, :],
        population_term_second_moments=term_second_moments[None, :],
    )


def average_cisd_mode_population_statistics(
    population_stats: list[CisdModePopulationStats],
) -> CisdModePopulationStats:
    """Average guide moments while retaining temporal means for variance tuning."""

    if not population_stats:
        raise ValueError("population_stats must be nonempty.")
    n_chol = population_stats[0].term_means.size
    if any(
        stats.term_means.shape != (n_chol,) or stats.term_second_moments.shape != (n_chol,)
        for stats in population_stats
    ):
        raise ValueError("all population statistics must have matching Cholesky shapes.")

    population_means = np.stack(
        [np.asarray(stats.term_means, dtype=np.float64) for stats in population_stats]
    )
    population_second_moments = np.stack(
        [np.asarray(stats.term_second_moments, dtype=np.float64) for stats in population_stats]
    )
    mean_term_means = np.asarray(
        np.mean(population_means, axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    second_moments = np.asarray(
        np.mean(
            population_second_moments,
            axis=0,
            dtype=np.float64,
        ),
        dtype=np.float64,
    )
    last = population_stats[-1]
    return CisdModePopulationStats(
        term_means=mean_term_means,
        term_second_moments=second_moments,
        rms_scores=np.sqrt(np.maximum(second_moments, 0.0)),
        local_energies=last.local_energies,
        exact_block_energy_ha=last.exact_block_energy_ha,
        independent_population_std_ha=float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        [stats.independent_population_std_ha**2 for stats in population_stats],
                        dtype=np.float64,
                    )
                )
            )
        ),
        wall_seconds=float(sum(stats.wall_seconds for stats in population_stats)),
        population_term_means=population_means,
        population_term_second_moments=population_second_moments,
    )


def select_cisd_mode_pair_sampling(
    stats: CisdModePopulationStats,
    cfg: CisdModePairTuningCfg,
    *,
    n_walkers: int,
    reference_guide_scores: np.ndarray | None = None,
    calibration_std_ha: float | None = None,
    calibration_source: str = "independent population standard deviation",
    final_error_target_ha: float | None = None,
    n_blocks: int | None = None,
) -> CisdModePairTuningResult:
    """Choose the least pair work that meets a cross-validated noise target.

    ``final_error_target_ha`` and ``n_blocks`` are accepted for compatibility;
    the tail-noise budget depends only on calibration or an absolute override.
    """

    del final_error_target_ha, n_blocks

    if n_walkers <= 0:
        raise ValueError("n_walkers must be positive.")
    rms_scores = np.asarray(stats.rms_scores, dtype=np.float64)
    means = np.asarray(stats.term_means, dtype=np.float64)
    second_moments = np.asarray(stats.term_second_moments, dtype=np.float64)
    population_means = (
        means[None, :]
        if stats.population_term_means is None
        else np.asarray(stats.population_term_means, dtype=np.float64)
    )
    population_second_moments = (
        second_moments[None, :]
        if stats.population_term_second_moments is None
        else np.asarray(stats.population_term_second_moments, dtype=np.float64)
    )
    if (
        rms_scores.ndim != 1
        or means.shape != rms_scores.shape
        or second_moments.shape != rms_scores.shape
        or population_means.ndim != 2
        or population_means.shape[1:] != rms_scores.shape
        or population_means.shape[0] == 0
        or population_second_moments.shape != population_means.shape
        or rms_scores.size == 0
    ):
        raise ValueError("population statistics must contain matching nonempty Cholesky arrays.")
    if (
        not np.all(np.isfinite(rms_scores))
        or not np.all(np.isfinite(means))
        or not np.all(np.isfinite(second_moments))
        or not np.all(np.isfinite(population_means))
        or not np.all(np.isfinite(population_second_moments))
        or np.any(second_moments < 0.0)
        or np.any(population_second_moments < 0.0)
    ):
        raise ValueError("population Cholesky statistics must be finite.")
    if cfg.guide_policy == "population_rms":
        guide_scores = rms_scores
    else:
        if reference_guide_scores is None:
            raise ValueError("reference_guide_scores are required for guide_policy='hf'.")
        guide_scores = np.asarray(reference_guide_scores, dtype=np.float64)
        if guide_scores.shape != rms_scores.shape:
            raise ValueError(
                f"reference_guide_scores must have shape {rms_scores.shape}, "
                f"got {guide_scores.shape}."
            )
        if not np.all(np.isfinite(guide_scores)) or np.any(guide_scores < 0.0):
            raise ValueError("reference_guide_scores must be finite and nonnegative.")
    guide_scores = np.maximum(guide_scores, 1.0e-300)

    n_chol = int(guide_scores.size)
    calibration_std = float(
        stats.independent_population_std_ha if calibration_std_ha is None else calibration_std_ha
    )
    if cfg.target_tail_std_ha is not None:
        target_tail_std = cfg.target_tail_std_ha
        target_source = "absolute override"
    else:
        if not np.isfinite(calibration_std) or calibration_std < 0.0:
            raise ValueError(
                "A nonnegative finite deterministic calibration standard deviation "
                "is required when target_tail_std_ha is not provided."
            )
        target_tail_std = cfg.target_tail_std_fraction * calibration_std
        target_source = f"{cfg.target_tail_std_fraction:.3f} x {calibration_source}"

    order = np.argsort(-guide_scores, kind="stable")
    ordered_guide_scores = guide_scores[order]
    ordered_second_moments = second_moments[order]
    ordered_population_means = population_means[:, order]
    tail_mean_sums = np.concatenate(
        (
            np.cumsum(ordered_population_means[:, ::-1], axis=1, dtype=np.float64)[:, ::-1],
            np.zeros((population_means.shape[0], 1), dtype=np.float64),
        ),
        axis=1,
    )
    tail_mean_squares = np.asarray(
        np.mean(tail_mean_sums**2, axis=0, dtype=np.float64),
        dtype=np.float64,
    )

    def tail_variance(
        head_size: int,
        ordered_scores: np.ndarray,
        ordered_seconds: np.ndarray,
        tail_mean_square: float,
    ) -> float:
        tail_scores = ordered_scores[head_size:]
        if tail_scores.size == 0:
            return 0.0
        guide_prob = tail_scores / np.sum(tail_scores, dtype=np.float64)
        uniform_mix = cfg.tail_probability_uniform_mix
        probabilities = (1.0 - uniform_mix) * guide_prob + uniform_mix / tail_scores.size
        importance_second_moment = np.sum(
            ordered_seconds[head_size:] / probabilities,
            dtype=np.float64,
        )
        return max(0.0, float(importance_second_moment - tail_mean_square))

    cross_validation_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_populations = int(population_means.shape[0])
    if n_populations > 1:
        second_moment_sum = np.sum(
            population_second_moments,
            axis=0,
            dtype=np.float64,
        )
        for held_out in range(n_populations):
            if cfg.guide_policy == "population_rms":
                training_seconds = (second_moment_sum - population_second_moments[held_out]) / (
                    n_populations - 1
                )
                fold_scores = np.sqrt(np.maximum(training_seconds, 0.0))
            else:
                fold_scores = guide_scores
            fold_scores = np.maximum(fold_scores, 1.0e-300)
            fold_order = np.argsort(-fold_scores, kind="stable")
            ordered_fold_means = population_means[held_out, fold_order]
            fold_tail_means = np.concatenate(
                (
                    np.cumsum(
                        ordered_fold_means[::-1],
                        dtype=np.float64,
                    )[::-1],
                    np.zeros(1, dtype=np.float64),
                )
            )
            cross_validation_folds.append(
                (
                    fold_scores[fold_order],
                    population_second_moments[held_out, fold_order],
                    fold_tail_means**2,
                )
            )

    minimum_head = max(0, min(n_chol, int(round(cfg.minimum_head_fraction * n_chol))))
    maximum_head = max(0, min(n_chol, int(round(cfg.maximum_head_fraction * n_chol))))
    head_sizes = list(range(minimum_head, maximum_head + 1, cfg.head_size_stride))
    if not head_sizes or head_sizes[-1] != maximum_head:
        head_sizes.append(maximum_head)
    sample_sizes = sorted({int(value) for value in cfg.candidate_sample_sizes})

    candidates: list[CisdModePairTuningResult] = []
    for head_size in head_sizes:
        in_sample_variance = tail_variance(
            head_size,
            ordered_guide_scores,
            ordered_second_moments,
            float(tail_mean_squares[head_size]),
        )
        if cross_validation_folds:
            fold_variances = np.asarray(
                [
                    tail_variance(
                        head_size,
                        fold_scores,
                        fold_seconds,
                        float(fold_tail_mean_squares[head_size]),
                    )
                    for fold_scores, fold_seconds, fold_tail_mean_squares in (
                        cross_validation_folds
                    )
                ],
                dtype=np.float64,
            )
            cross_validated_variance = float(
                np.quantile(
                    fold_variances,
                    cfg.cross_validation_quantile,
                    method="higher",
                )
            )
            selection_variance = max(in_sample_variance, cross_validated_variance)
        else:
            selection_variance = in_sample_variance
        for sample_size in sample_sizes:
            tail_std = math.sqrt(selection_variance / sample_size) if head_size < n_chol else 0.0
            guarded_std = cfg.safety_factor * tail_std
            if guarded_std > target_tail_std:
                continue
            pair_evaluations = n_walkers * head_size
            if head_size < n_chol:
                pair_evaluations += sample_size
            candidates.append(
                CisdModePairTuningResult(
                    sampling=CisdModePairSamplingCfg(
                        chol_head_size=head_size,
                        pair_sample_size=sample_size,
                        rank_head_by_guide=True,
                        guide_chol_batch_size=cfg.tuning_chol_batch_size,
                        head_chol_batch_size=cfg.production_head_chol_batch_size,
                        tail_probability_uniform_mix=cfg.tail_probability_uniform_mix,
                        track_half_sample_diagnostic=cfg.track_half_sample_diagnostic,
                        guard_head_deviations=cfg.guard_head_deviations,
                        walker_guide_policy=cfg.walker_guide_policy,
                        walker_guide_weight_mix=cfg.walker_guide_weight_mix,
                    ),
                    guide_policy=cfg.guide_policy,
                    chol_head_fraction=head_size / n_chol,
                    in_sample_single_pair_variance_ha2=in_sample_variance,
                    estimated_single_pair_variance_ha2=selection_variance,
                    cross_validation_fold_count=len(cross_validation_folds),
                    cross_validation_quantile=cfg.cross_validation_quantile,
                    estimated_tail_std_ha=tail_std,
                    guarded_tail_std_ha=guarded_std,
                    target_tail_std_ha=target_tail_std,
                    target_tail_std_source=target_source,
                    calibration_std_ha=calibration_std,
                    estimated_pair_evaluations=pair_evaluations,
                )
            )

    if not candidates:
        raise ValueError(
            "No population pair-sampling candidate meets the requested tail-noise target. "
            "Increase maximum_head_fraction or the candidate sample sizes."
        )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.estimated_pair_evaluations,
            candidate.guarded_tail_std_ha,
            candidate.sampling.chol_head_size,
            candidate.sampling.pair_sample_size,
        ),
    )


def retune_cisd_mode_pair_sampling(
    state,
    equilibration_energies: jax.Array,
    equilibration_weights: jax.Array,
    params,
    ham_data: HamChol,
    meas_ctx: CisdModeMeasCtx,
    trial_data: CisdModeTrial,
    *,
    advance_blocks: BlockEnergyAdvanceFn,
    tuning_cfg: CisdModePairTuningCfg,
    target_error: float | None = None,
) -> BlockEnergyRetuneResult:
    """Tune and install the requested production guide after equilibration."""

    if meas_ctx.model_sampling is not None:
        raise ValueError("Local Cholesky sampling requires frozen settings (no automatic retuning).")
    if meas_ctx.energy_sampling is not None and meas_ctx.energy_sampling.sample_local_walkers:
        raise ValueError("Local walker sampling currently requires frozen settings (no retuning).")
    del equilibration_energies, equilibration_weights
    population_stats = []
    for population_index in range(tuning_cfg.tuning_population_count):
        if population_index > 0:
            spacing = tuning_cfg.tuning_population_spacing_blocks
            print(
                f"[sampling] advancing {spacing} calibration blocks before "
                f"population {population_index + 1}/{tuning_cfg.tuning_population_count}."
            )
            state, _, _ = advance_blocks(state, n_blocks=spacing)
            jax.block_until_ready(state)

        print(
            "[sampling] streaming population statistics: "
            f"population={population_index + 1}/{tuning_cfg.tuning_population_count}, "
            f"walker_chunks={min(tuning_cfg.tuning_n_chunks, int(state.walkers.shape[0]))}, "
            f"chol_batch_size={tuning_cfg.tuning_chol_batch_size}."
        )
        stats_i = stream_cisd_mode_population_statistics(
            state.walkers,
            state.weights,
            ham_data,
            meas_ctx,
            trial_data,
            n_walker_chunks=tuning_cfg.tuning_n_chunks,
            chol_batch_size=tuning_cfg.tuning_chol_batch_size,
        )
        population_stats.append(stats_i)
        state = state._replace(
            e_estimate=jnp.asarray(
                stats_i.exact_block_energy_ha,
                dtype=jnp.result_type(state.e_estimate),
            )
        )
        print(
            f"[sampling] population {population_index + 1} exact energy="
            f"{stats_i.exact_block_energy_ha:.10f} Ha, "
            f"statistics_seconds={stats_i.wall_seconds:.1f}."
        )
    stats = average_cisd_mode_population_statistics(population_stats)
    snapshot_energies = np.asarray(
        [population.exact_block_energy_ha for population in population_stats],
        dtype=np.float64,
    )
    if snapshot_energies.size > 1:
        calibration_std = float(np.std(snapshot_energies, ddof=1))
        calibration_source = "exact calibration-population standard deviation"
    else:
        calibration_std = stats.independent_population_std_ha
        calibration_source = "independent population standard deviation fallback"
    selected = select_cisd_mode_pair_sampling(
        stats,
        tuning_cfg,
        n_walkers=int(state.walkers.shape[0]),
        reference_guide_scores=np.asarray(
            jax.device_get(meas_ctx.reference_chol_scores),
            dtype=np.float64,
        ),
        calibration_std_ha=calibration_std,
        calibration_source=calibration_source,
        final_error_target_ha=target_error,
        n_blocks=(int(params.n_blocks) if params is not None else None),
    )
    if tuning_cfg.guide_policy == "population_rms":
        production_guide_scores = stats.rms_scores
        guide_label = "population-RMS"
    else:
        production_guide_scores = np.asarray(
            jax.device_get(meas_ctx.reference_chol_scores),
            dtype=np.float64,
        )
        guide_label = "HF-reference"
    production_ctx = configure_cisd_mode_pair_sampling(
        meas_ctx,
        selected.sampling,
        jnp.asarray(production_guide_scores, dtype=jnp.float64),
    )
    print(
        "[sampling] population-moment tuning sweep complete: "
        f"populations={tuning_cfg.tuning_population_count}, "
        f"seconds={stats.wall_seconds:.1f}, "
        f"final_exact_snapshot_energy={stats.exact_block_energy_ha:.10f} Ha, "
        f"independent_population_std={stats.independent_population_std_ha:.3e} Ha, "
        f"calibration_std={calibration_std:.3e} Ha."
    )
    print(
        f"[sampling] selected {guide_label} estimator: "
        f"chol_head_size={selected.sampling.chol_head_size}/{stats.rms_scores.size} "
        f"({selected.chol_head_fraction:.3%}), "
        f"pair_sample_size={selected.sampling.pair_sample_size}, "
        f"tail_std={selected.estimated_tail_std_ha:.3e} Ha, "
        f"in_sample_tail_std="
        f"{math.sqrt(selected.in_sample_single_pair_variance_ha2 / selected.sampling.pair_sample_size):.3e} Ha, "
        f"cv_folds={selected.cross_validation_fold_count}, "
        f"cv_quantile={selected.cross_validation_quantile:.3f}, "
        f"guarded_tail_std={selected.guarded_tail_std_ha:.3e} Ha, "
        f"target={selected.target_tail_std_ha:.3e} Ha, "
        f"target_source={selected.target_tail_std_source}, "
        f"uniform_mix={selected.sampling.tail_probability_uniform_mix:.3%}, "
        f"head_guard={selected.sampling.guard_head_deviations}, "
        f"walker_guide={selected.sampling.walker_guide_policy}, "
        f"walker_weight_mix={selected.sampling.walker_guide_weight_mix:.3%}, "
        f"work_proxy={selected.estimated_pair_evaluations} pairs."
    )
    energy_dtype = jnp.result_type(state.e_estimate)
    state = state._replace(
        e_estimate=jnp.asarray(stats.exact_block_energy_ha, dtype=energy_dtype),
    )
    return BlockEnergyRetuneResult(
        state=state,
        meas_ctx=production_ctx,
        initial_n_chunks=tuning_cfg.production_initial_n_chunks,
        settling_blocks=tuning_cfg.settling_blocks,
    )


def make_cisd_mode_meas_ops(
    sys: System,
    *,
    mixed_precision: bool = True,
    n_mode_chunks: int = 1,
    energy_sampling: CisdModePairSamplingCfg | None = None,
    energy_tuning: CisdModePairTuningCfg | None = None,
) -> MeasOps:
    """Build retained-mode CISD measurements.

    The deterministic default evaluates all Cholesky vectors together; AFQMC
    initialization uses a separate batched kernel for the same energy. Passing
    ``energy_sampling`` instead evaluates its Cholesky head exactly and uses
    unbiased weighted walker--Cholesky sampling for the tail. When
    ``energy_tuning`` is also supplied, that estimator is used during
    equilibration, then a bounded-memory deterministic sweep builds the
    population moments needed to choose the production head and sample count.
    When requested, several temporally separated populations are averaged
    without retaining multiple walker populations. The configured tuning guide
    controls the production head ranking and tail probabilities.
    All retained modes remain deterministic in either case. The result is
    exact when all pair-space modes are retained and is the consistent
    truncated-K approximation otherwise.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD mode MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if energy_tuning is not None and energy_sampling is not None and energy_sampling.sample_local_walkers:
        raise ValueError("Local walker sampling currently requires frozen settings (energy_tuning=None).")
    if energy_tuning is not None and energy_sampling is None:
        raise ValueError("energy_tuning requires an equilibration energy_sampling config.")

    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    retune_block_energy = (
        partial(retune_cisd_mode_pair_sampling, tuning_cfg=energy_tuning)
        if energy_tuning is not None
        else None
    )
    meas_ops = MeasOps(
        overlap=cisd_mode_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(
            ham_data,
            trial_data,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
            energy_sampling=energy_sampling,
        ),
        kernels={
            k_force_bias: force_bias_kernel_rw_rh,
            k_energy: energy_kernel_rw_rh,
            k_energy_init: initial_energy_kernel_rw_rh,
        },
        observables={},
        block_energy=pair_sampled_block_energy if energy_sampling is not None else None,
        retune_block_energy=retune_block_energy,
    )
    object.__setattr__(meas_ops, _CISD_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops
