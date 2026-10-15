from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax import lax, tree_util

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
    k_force_bias,
)
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ucisd import UcisdTrial
from ..trial.ucisd_k import UcisdKTrial
from ..trial.ucisd_k_modes import UcisdKModeTrial, get_rdm1
from ..trial.ucisd_k_modes import overlap_r as ucisd_k_mode_overlap_r
from .ucisd import UcisdMeasCfg, UcisdMeasCtx
from .ucisd import build_meas_ctx as build_dense_meas_ctx
from .ucisd_k import (
    UcisdKEnergyCommon,
    _combined_pair_batch,
    _force_bias_kernel_rw_rh_with_apply,
    _ucisd_k_chol_terms,
    _ucisd_k_energy_common,
)
from .cisd_modes import (
    CisdModePairTuningCfg,
    CisdModePopulationStats,
    average_cisd_mode_population_statistics,
    select_cisd_mode_pair_sampling,
)
from .pair_sampling import (
    ModelPairSamplingData, local_cholesky_head, local_cholesky_tail,
    local_common_and_head, local_pair_mesh, local_pair_tail,
)

_UCISD_K_MODE_MEAS_CFG_ATTR = "_ucisd_k_mode_meas_cfg"

# The selection policy depends only on walker--Cholesky first and second
# moments, not on the trial-specific pair kernel. The shared statistics type
# keeps the selection policy exactly synchronized with retained-mode RCISD.
UcisdKModePopulationStats = CisdModePopulationStats


@dataclass(frozen=True)
class UcisdKModePairSamplingCfg:
    """Walker--Cholesky pair sampling for the UCISD K-mode block energy.

    ``chol_head_size`` selected Cholesky contributions are evaluated exactly
    for every restricted walker. ``pair_sample_size`` weighted walker--tail
    pairs estimate the remainder while every retained combined-K mode remains
    deterministic within an evaluated pair.

    ``sample_local_walkers`` opts into the same data-shard stratification as
    CISD modes. It requires a replicated Hamiltonian and frozen sampling
    settings, and preserves the existing path on a single device.
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
class UcisdKModePairTuningCfg(CisdModePairTuningCfg):
    """Post-equilibration automatic tuning for UCISD K-mode pair sampling."""


@dataclass(frozen=True)
class UcisdKModePairTuningResult:
    """Automatically selected UCISD estimator and its predicted cost/noise."""

    sampling: UcisdKModePairSamplingCfg
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


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdKModeMeasCtx:
    """UCISD measurement intermediates, mode chunking, and energy guide."""

    base: UcisdMeasCtx
    reference_chol_scores: jax.Array
    chol_head_indices: jax.Array
    chol_tail_indices: jax.Array
    chol_tail_prob: jax.Array
    n_mode_chunks: int
    energy_sampling: UcisdKModePairSamplingCfg | None
    model_sampling: ModelPairSamplingData | None = None

    def tree_flatten(self):
        children = (
            self.base,
            self.reference_chol_scores,
            self.chol_head_indices,
            self.chol_tail_indices,
            self.chol_tail_prob,
            self.model_sampling,
        )
        return children, (self.n_mode_chunks, self.energy_sampling)

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_mode_chunks, energy_sampling = aux
        (
            base,
            reference_chol_scores,
            chol_head_indices,
            chol_tail_indices,
            chol_tail_prob,
            model_sampling,
        ) = children
        return cls(
            base=base,
            reference_chol_scores=reference_chol_scores,
            chol_head_indices=chol_head_indices,
            chol_tail_indices=chol_tail_indices,
            chol_tail_prob=chol_tail_prob,
            model_sampling=model_sampling,
            n_mode_chunks=n_mode_chunks,
            energy_sampling=energy_sampling,
        )


def get_ucisd_k_mode_meas_cfg(meas_ops: MeasOps) -> UcisdMeasCfg | None:
    cfg = getattr(meas_ops, _UCISD_K_MODE_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, UcisdMeasCfg) else None


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: UcisdKModeTrial,
    *,
    cfg: UcisdMeasCfg = UcisdMeasCfg(memory_mode="high"),
    n_mode_chunks: int = 1,
    energy_sampling: UcisdKModePairSamplingCfg | None = None,
) -> UcisdKModeMeasCtx:
    """Build full-Cholesky intermediates for retained combined-K modes."""
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if cfg.memory_mode != "high":
        raise ValueError("UCISD K-mode measurements currently require memory_mode='high'.")
    if ham_data.basis != "restricted":
        raise ValueError("UCISD K-mode MeasOps requires HamChol.basis == 'restricted'.")

    base = build_dense_meas_ctx(ham_data, cast(UcisdTrial, trial_data), cfg)
    chunks = min(int(n_mode_chunks), trial_data.mode_rank) if trial_data.mode_rank else 1
    n_chol = int(ham_data.chol.shape[0])
    if energy_sampling is not None and energy_sampling.chol_head_size > n_chol:
        raise ValueError(
            f"chol_head_size must not exceed the number of Cholesky vectors ({n_chol})."
        )
    meas_ctx = UcisdKModeMeasCtx(
        base=base,
        reference_chol_scores=jnp.empty((0,), dtype=jnp.float64),
        chol_head_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_prob=jnp.empty((0,), dtype=jnp.float64),
        n_mode_chunks=chunks,
        energy_sampling=energy_sampling,
    )
    if energy_sampling is not None:
        guide_scores = _build_reference_chol_scores(
            ham_data,
            meas_ctx,
            trial_data,
            chol_batch_size=energy_sampling.guide_chol_batch_size,
        )
        meas_ctx = replace(meas_ctx, reference_chol_scores=guide_scores)
        meas_ctx = configure_ucisd_k_mode_pair_sampling(
            meas_ctx,
            energy_sampling,
            guide_scores,
        )
        print(
            "[sampling] configured reference-guide UCISD-K-mode pair estimator: "
            f"chol_head_size={energy_sampling.chol_head_size}/{n_chol} "
            f"({energy_sampling.chol_head_size / n_chol:.3%}), "
            f"pair_sample_size={energy_sampling.pair_sample_size}, "
            f"ranked_head={energy_sampling.rank_head_by_guide}, "
            f"head_guard={energy_sampling.guard_head_deviations}, "
            f"walker_guide={energy_sampling.walker_guide_policy}, "
            f"walker_weight_mix={energy_sampling.walker_guide_weight_mix:.3%}."
        )
    return meas_ctx


def _k_mode_apply_realimag(
    trial_data: UcisdKModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    cfg: UcisdMeasCfg,
) -> tuple[jax.Array, jax.Array]:
    """Apply modes after separately projecting the alpha and beta pair spaces."""
    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrix_a.shape != expected_a or matrix_b.shape != expected_b:
        raise ValueError(
            f"matrix pair must have shapes {expected_a} and {expected_b}, got "
            f"{matrix_a.shape} and {matrix_b.shape}."
        )

    da, _ = trial_data.pair_dim
    modes = trial_data.modes.astype(cfg.mixed_real_dtype)
    modes_a = modes[:, :da]
    modes_b = modes[:, da:]
    values = trial_data.eigenvalues.astype(cfg.mixed_real_dtype)
    vector_a_r = jnp.real(matrix_a).reshape(-1).astype(cfg.mixed_real_dtype)
    vector_b_r = jnp.real(matrix_b).reshape(-1).astype(cfg.mixed_real_dtype)
    projection_a_r = jnp.einsum("rp,p->r", modes_a, vector_a_r, optimize="optimal")
    projection_b_r = jnp.einsum("rp,p->r", modes_b, vector_b_r, optimize="optimal")
    projection_r = (projection_a_r.astype(jnp.float64) + projection_b_r.astype(jnp.float64)).astype(
        cfg.mixed_real_dtype
    )

    vector_a_i = jnp.imag(matrix_a).reshape(-1).astype(cfg.mixed_real_dtype)
    vector_b_i = jnp.imag(matrix_b).reshape(-1).astype(cfg.mixed_real_dtype)
    projection_a_i = jnp.einsum("rp,p->r", modes_a, vector_a_i, optimize="optimal")
    projection_b_i = jnp.einsum("rp,p->r", modes_b, vector_b_i, optimize="optimal")
    projection_i = (projection_a_i.astype(jnp.float64) + projection_b_i.astype(jnp.float64)).astype(
        cfg.mixed_real_dtype
    )

    applied_r = jnp.einsum("r,rp->p", values * projection_r, modes, optimize="optimal")
    applied_i = jnp.einsum("r,rp->p", values * projection_i, modes, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    applied = applied_r.astype(cfg.mixed_complex_dtype)
    applied += imag_unit * applied_i.astype(cfg.mixed_complex_dtype)

    return applied[:da].reshape(expected_a), applied[da:].reshape(expected_b)


def _k_mode_quadratic_batched_realimag(
    trial_data: UcisdKModeTrial,
    matrices_a: jax.Array,
    matrices_b: jax.Array,
    cfg: UcisdMeasCfg,
    n_mode_chunks: int = 1,
) -> jax.Array:
    """Evaluate the three spin-block mode quadratics after split projections."""
    vectors, leading_shape = _combined_pair_batch(
        cast(UcisdKTrial, trial_data),
        matrices_a,
        matrices_b,
    )
    da, _ = trial_data.pair_dim
    vectors_a = vectors[:, :da]
    vectors_b = vectors[:, da:]
    rank = trial_data.mode_rank
    result_dtype = (
        jnp.complex128 if jnp.issubdtype(vectors.dtype, jnp.complexfloating) else jnp.float64
    )
    if rank == 0:
        return jnp.zeros(leading_shape, dtype=result_dtype)

    vectors_a_r = jnp.real(vectors_a).astype(cfg.mixed_real_dtype_testing)
    vectors_b_r = jnp.real(vectors_b).astype(cfg.mixed_real_dtype_testing)
    vectors_a_i = jnp.imag(vectors_a).astype(cfg.mixed_real_dtype_testing)
    vectors_b_i = jnp.imag(vectors_b).astype(cfg.mixed_real_dtype_testing)

    def evaluate_chunk(values_i: jax.Array, modes_i: jax.Array) -> jax.Array:
        modes_i = modes_i.astype(cfg.mixed_real_dtype_testing)
        modes_a_i = modes_i[:, :da]
        modes_b_i = modes_i[:, da:]
        projection_a_r = jnp.einsum(
            "sp,rp->sr",
            vectors_a_r,
            modes_a_i,
            optimize="optimal",
        )
        projection_b_r = jnp.einsum(
            "sp,rp->sr",
            vectors_b_r,
            modes_b_i,
            optimize="optimal",
        )
        values_t = values_i.astype(jnp.float64)[None, :]
        if result_dtype == jnp.complex128:
            projection_a_i = jnp.einsum(
                "sp,rp->sr",
                vectors_a_i,
                modes_a_i,
                optimize="optimal",
            )
            projection_b_i = jnp.einsum(
                "sp,rp->sr",
                vectors_b_i,
                modes_b_i,
                optimize="optimal",
            )
            projection_a = projection_a_r.astype(jnp.complex128)
            projection_a += 1.0j * projection_a_i.astype(jnp.complex128)
            projection_b = projection_b_r.astype(jnp.complex128)
            projection_b += 1.0j * projection_b_i.astype(jnp.complex128)
            contribution = 0.5 * values_t * projection_a * projection_a
            contribution += values_t * projection_a * projection_b
            contribution += 0.5 * values_t * projection_b * projection_b
            return jnp.sum(
                contribution,
                axis=1,
                dtype=jnp.complex128,
            )
        projection_a_t = projection_a_r.astype(jnp.float64)
        projection_b_t = projection_b_r.astype(jnp.float64)
        contribution = 0.5 * values_t * projection_a_t * projection_a_t
        contribution += values_t * projection_a_t * projection_b_t
        contribution += 0.5 * values_t * projection_b_t * projection_b_t
        return jnp.sum(
            contribution,
            axis=1,
            dtype=jnp.float64,
        )

    chunks = min(int(n_mode_chunks), rank)
    if chunks == 1:
        result = evaluate_chunk(trial_data.eigenvalues, trial_data.modes)
        return result.reshape(leading_shape)

    base_chunk_size = rank // chunks
    n_larger_chunks = rank % chunks
    chunk_size = base_chunk_size + int(n_larger_chunks > 0)
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    # Keep the data-axis variation when called inside a local walker map.
    zero = jnp.zeros_like(vectors, shape=(vectors.shape[0],), dtype=result_dtype)

    def scan_body(total, chunk_index):
        is_larger = chunk_index < n_larger_chunks
        chunk_length = base_chunk_size + is_larger.astype(jnp.int32)
        start = chunk_index * base_chunk_size + jnp.minimum(chunk_index, n_larger_chunks)
        indices = start + chunk_offsets
        valid = chunk_offsets < chunk_length
        indices = jnp.minimum(indices, rank - 1)
        values_i = jnp.where(valid, trial_data.eigenvalues[indices], 0.0)
        contribution = evaluate_chunk(values_i, trial_data.modes[indices])
        return total + contribution, None

    result, _ = lax.scan(scan_body, zero, jnp.arange(chunks, dtype=jnp.int32))
    return result.reshape(leading_shape)


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
) -> jax.Array:
    """Deterministic combined-K-mode force bias for a restricted walker."""
    return _force_bias_kernel_rw_rh_with_apply(
        walker,
        ham_data,
        meas_ctx,
        trial_data,
        _k_mode_apply_realimag,
    )


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
) -> jax.Array:
    """Deterministic local energy from retained combined UCISD K modes."""
    common: UcisdKEnergyCommon = _ucisd_k_energy_common(
        walker,
        ham_data,
        meas_ctx,
        trial_data,
        _k_mode_apply_realimag,
    )
    chol_terms = _ucisd_k_chol_terms(
        common,
        ham_data,
        meas_ctx,
        trial_data,
        _k_mode_quadratic_batched_realimag,
    )
    return common.base + jnp.sum(chol_terms, dtype=jnp.complex128)


def _ucisd_k_mode_chol_terms_for_walkers(
    common: UcisdKEnergyCommon,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    *,
    chol_indices: jax.Array | None = None,
    n_chunks: int = 1,
) -> jax.Array:
    """Return residual terms for each walker and selected Cholesky vector."""
    return wk.vmap_chunked(
        lambda common_i: _ucisd_k_chol_terms(
            common_i,
            ham_data,
            meas_ctx,
            trial_data,
            _k_mode_quadratic_batched_realimag,
            chol_indices,
        ),
        n_chunks=n_chunks,
    )(common)


def _ucisd_k_mode_chol_index_terms(
    common: UcisdKEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    *,
    n_chunks: int,
) -> jax.Array:
    """Return one walker's terms at arbitrary Cholesky indices."""
    return wk.vmap_chunked(
        lambda chol_i: _ucisd_k_chol_terms(
            common,
            ham_data,
            meas_ctx,
            trial_data,
            _k_mode_quadratic_batched_realimag,
            chol_i[None],
        )[0],
        n_chunks=n_chunks,
        shard_walkers=False,
    )(chol_indices)


def _ucisd_k_mode_chol_index_moments_for_walkers(
    common: UcisdKEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
    compute_squared_norm: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Return the selected-term sum and real squared norm per walker."""
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
        n_batches,
        batch_size,
    )
    valid = (jnp.arange(padded_size) < head_size).reshape(n_batches, batch_size)

    if n_batches == 1:
        terms = _ucisd_k_mode_chol_terms_for_walkers(
            common,
            ham_data,
            meas_ctx,
            trial_data,
            chol_indices=chol_indices,
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
        terms_i = _ucisd_k_mode_chol_terms_for_walkers(
            common,
            ham_data,
            meas_ctx,
            trial_data,
            chol_indices=indices_i,
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


def _ucisd_k_mode_chol_index_sum_for_walkers(
    common: UcisdKEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
) -> jax.Array:
    """Sum selected Cholesky terms with bounded head batches."""
    total, _ = _ucisd_k_mode_chol_index_moments_for_walkers(
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


def _ucisd_k_mode_chol_pair_terms(
    common: UcisdKEnergyCommon,
    sample_walker: jax.Array,
    sample_chol: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    *,
    n_chunks: int = 1,
) -> jax.Array:
    """Evaluate sampled pairs after gathering walker and spin-Cholesky data."""
    return wk.vmap_chunked(
        lambda walker_i, chol_i: _ucisd_k_chol_terms(
            tree_util.tree_map(lambda value: value[walker_i], common),
            ham_data,
            meas_ctx,
            trial_data,
            _k_mode_quadratic_batched_realimag,
            chol_i[None],
        )[0],
        n_chunks=n_chunks,
        in_axes=(0, 0),
        shard_walkers=False,
    )(sample_walker, sample_chol)


def _build_reference_chol_scores(
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    *,
    chol_batch_size: int,
) -> jax.Array:
    """Build bounded-memory scores at the restricted projected-UHF reference."""
    sys = System(
        norb=trial_data.norb,
        nelec=trial_data.nocc,
        walker_kind="restricted",
    )
    reference_walker = wk.init_walkers(sys, get_rdm1(trial_data), 1)[0]
    common = _ucisd_k_energy_common(
        reference_walker,
        ham_data,
        meas_ctx,
        trial_data,
        _k_mode_apply_realimag,
    )
    n_chol = int(ham_data.chol.shape[0])
    indices = jnp.arange(n_chol, dtype=jnp.int32)
    n_chunks = max(1, math.ceil(n_chol / chol_batch_size))
    terms = _ucisd_k_mode_chol_index_terms(
        common,
        indices,
        ham_data,
        meas_ctx,
        trial_data,
        n_chunks=n_chunks,
    )
    return jnp.maximum(jnp.abs(terms).astype(jnp.float64), 1.0e-300)


def configure_ucisd_k_mode_pair_sampling(
    meas_ctx: UcisdKModeMeasCtx,
    sampling: UcisdKModePairSamplingCfg,
    guide_scores: jax.Array,
) -> UcisdKModeMeasCtx:
    """Attach a prefix or guide-ranked head and normalized tail guide."""
    if meas_ctx.model_sampling is not None:
        raise ValueError("Rebuild the Cholesky layout and contexts before changing the frozen guide.")
    scores = jnp.asarray(guide_scores, dtype=jnp.float64)
    n_chol = int(meas_ctx.base.chol_b.shape[0])
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
    base = replace(specs.base, **{name: P("model") for name in (
        "chol_b", "rot_chol_a", "rot_chol_b", "rot_chol_flat_a", "rot_chol_flat_b",
        "lci1_a", "lci1_b",
    )})
    return replace(specs, base=base)


def _local_cholesky_head_terms(common, indices, h, ctx, trial, *, n_chunks):
    return _ucisd_k_mode_chol_terms_for_walkers(
        common, h, ctx, trial, chol_indices=indices, n_chunks=n_chunks,
    )


def pair_sampled_block_energy(
    walkers: jax.Array,
    weights: jax.Array,
    overlaps: jax.Array,
    rng_key: jax.Array,
    n_chunks: int,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    e_ref: jax.Array,
    energy_clip_threshold: jax.Array,
) -> jax.Array | BlockEnergyEstimate:
    """Evaluate an exact Cholesky head and sample walker--Cholesky tail pairs."""
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
            common_fn=lambda w, h, c, t: _ucisd_k_energy_common(w, h, c, t, _k_mode_apply_realimag),
            moments_fn=_ucisd_k_mode_chol_index_moments_for_walkers, n_chunks=n_chunks,
        )
    else:
        common = wk.vmap_chunked(
            lambda walker: _ucisd_k_energy_common(
                walker,
                ham_data,
                meas_ctx,
                trial_data,
                _k_mode_apply_realimag,
            ),
            n_chunks=n_chunks,
        )(walkers)

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
        head_sum, head_squared_norm = _ucisd_k_mode_chol_index_moments_for_walkers(
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
        head_sum = _ucisd_k_mode_chol_index_sum_for_walkers(
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
            ham_data, meas_ctx, trial_data, pair_fn=_ucisd_k_mode_chol_pair_terms,
            context_specs_fn=_local_cholesky_context_specs, n_chunks=n_chunks,
        )
    elif local_mesh is not None:
        tail_estimate, sampling_noise = local_pair_tail(
            local_mesh, common, accepted_weights, walker_probabilities, rng_key,
            ham_data, meas_ctx, trial_data, pair_fn=_ucisd_k_mode_chol_pair_terms, n_chunks=n_chunks,
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
    pair_n_chunks = math.ceil(sampling.pair_sample_size / walker_batch_size)
    sample_terms = _ucisd_k_mode_chol_pair_terms(
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
    diagnostics[d_energy_sampling_noise] = diagnostic_scale * (first_mean - second_mean)
    return BlockEnergyEstimate(energy=energy, diagnostics=diagnostics)


@jax.jit
def _ucisd_k_mode_population_common_batch(
    walkers,
    ham_data,
    meas_ctx,
    trial_data,
):
    return jax.vmap(
        lambda walker: _ucisd_k_energy_common(
            walker,
            ham_data,
            meas_ctx,
            trial_data,
            _k_mode_apply_realimag,
        )
    )(walkers)


@jax.jit
def _ucisd_k_mode_population_term_batch(
    common,
    chol_indices,
    ham_data,
    meas_ctx,
    trial_data,
):
    return _ucisd_k_mode_chol_terms_for_walkers(
        common,
        ham_data,
        meas_ctx,
        trial_data,
        chol_indices=chol_indices,
        n_chunks=1,
    )


def stream_ucisd_k_mode_population_statistics(
    walkers: jax.Array,
    weights: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    *,
    n_walker_chunks: int = 10,
    chol_batch_size: int = 16,
) -> UcisdKModePopulationStats:
    """Accumulate UCISD pair moments without storing an ``Nw x Nchol`` table."""
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
        common = _ucisd_k_mode_population_common_batch(
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
            terms = _ucisd_k_mode_population_term_batch(
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
    return UcisdKModePopulationStats(
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


def average_ucisd_k_mode_population_statistics(
    population_stats: list[UcisdKModePopulationStats],
) -> UcisdKModePopulationStats:
    """Average temporal UCISD moments using the shared pair-tuning policy."""
    return average_cisd_mode_population_statistics(population_stats)


def _as_ucisd_k_mode_sampling(
    sampling,
) -> UcisdKModePairSamplingCfg:
    return UcisdKModePairSamplingCfg(
        chol_head_size=sampling.chol_head_size,
        pair_sample_size=sampling.pair_sample_size,
        rank_head_by_guide=sampling.rank_head_by_guide,
        guide_chol_batch_size=sampling.guide_chol_batch_size,
        head_chol_batch_size=sampling.head_chol_batch_size,
        tail_probability_uniform_mix=sampling.tail_probability_uniform_mix,
        track_half_sample_diagnostic=sampling.track_half_sample_diagnostic,
        guard_head_deviations=sampling.guard_head_deviations,
        walker_guide_policy=sampling.walker_guide_policy,
        walker_guide_weight_mix=sampling.walker_guide_weight_mix,
        sample_local_walkers=sampling.sample_local_walkers,
    )


def select_ucisd_k_mode_pair_sampling(
    stats: UcisdKModePopulationStats,
    cfg: UcisdKModePairTuningCfg,
    *,
    n_walkers: int,
    reference_guide_scores: np.ndarray | None = None,
    calibration_std_ha: float | None = None,
    calibration_source: str = "independent population standard deviation",
    final_error_target_ha: float | None = None,
    n_blocks: int | None = None,
) -> UcisdKModePairTuningResult:
    """Choose the least-work UCISD estimator meeting the requested noise target."""
    selected = select_cisd_mode_pair_sampling(
        stats,
        cfg,
        n_walkers=n_walkers,
        reference_guide_scores=reference_guide_scores,
        calibration_std_ha=calibration_std_ha,
        calibration_source=calibration_source,
        final_error_target_ha=final_error_target_ha,
        n_blocks=n_blocks,
    )
    return UcisdKModePairTuningResult(
        sampling=_as_ucisd_k_mode_sampling(selected.sampling),
        guide_policy=selected.guide_policy,
        chol_head_fraction=selected.chol_head_fraction,
        in_sample_single_pair_variance_ha2=selected.in_sample_single_pair_variance_ha2,
        estimated_single_pair_variance_ha2=selected.estimated_single_pair_variance_ha2,
        cross_validation_fold_count=selected.cross_validation_fold_count,
        cross_validation_quantile=selected.cross_validation_quantile,
        estimated_tail_std_ha=selected.estimated_tail_std_ha,
        guarded_tail_std_ha=selected.guarded_tail_std_ha,
        target_tail_std_ha=selected.target_tail_std_ha,
        target_tail_std_source=selected.target_tail_std_source,
        calibration_std_ha=selected.calibration_std_ha,
        estimated_pair_evaluations=selected.estimated_pair_evaluations,
    )


def retune_ucisd_k_mode_pair_sampling(
    state,
    equilibration_energies: jax.Array,
    equilibration_weights: jax.Array,
    params,
    ham_data: HamChol,
    meas_ctx: UcisdKModeMeasCtx,
    trial_data: UcisdKModeTrial,
    *,
    advance_blocks: BlockEnergyAdvanceFn,
    tuning_cfg: UcisdKModePairTuningCfg,
    target_error: float | None = None,
) -> BlockEnergyRetuneResult:
    """Tune and install the production UCISD pair estimator after equilibration."""
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
            "[sampling] streaming UCISD-K-mode population statistics: "
            f"population={population_index + 1}/{tuning_cfg.tuning_population_count}, "
            f"walker_chunks={min(tuning_cfg.tuning_n_chunks, int(state.walkers.shape[0]))}, "
            f"chol_batch_size={tuning_cfg.tuning_chol_batch_size}."
        )
        stats_i = stream_ucisd_k_mode_population_statistics(
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

    stats = average_ucisd_k_mode_population_statistics(population_stats)
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

    selected = select_ucisd_k_mode_pair_sampling(
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
        guide_label = "reference"
    production_ctx = configure_ucisd_k_mode_pair_sampling(
        meas_ctx,
        selected.sampling,
        jnp.asarray(production_guide_scores, dtype=jnp.float64),
    )
    print(
        "[sampling] UCISD-K-mode population-moment tuning sweep complete: "
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


def make_ucisd_k_mode_meas_ops(
    sys: System,
    *,
    memory_mode: str = "high",
    mixed_precision: bool = True,
    n_mode_chunks: int = 1,
    energy_sampling: UcisdKModePairSamplingCfg | None = None,
    energy_tuning: UcisdKModePairTuningCfg | None = None,
) -> MeasOps:
    """Build deterministic or pair-sampled combined-K UCISD measurements."""
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "UCISD K-mode MeasOps currently supports only restricted walkers, got: "
            f"{sys.walker_kind}"
        )
    if memory_mode != "high":
        raise ValueError("UCISD K-mode measurements currently require memory_mode='high'.")
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if energy_tuning is not None and energy_sampling is not None and energy_sampling.sample_local_walkers:
        raise ValueError("Local walker sampling currently requires frozen settings (energy_tuning=None).")
    if energy_tuning is not None and energy_sampling is None:
        raise ValueError("energy_tuning requires an equilibration energy_sampling config.")
    cfg = UcisdMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    retune_block_energy = (
        partial(retune_ucisd_k_mode_pair_sampling, tuning_cfg=energy_tuning)
        if energy_tuning is not None
        else None
    )
    meas_ops = MeasOps(
        overlap=ucisd_k_mode_overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(
            ham_data,
            trial_data,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
            energy_sampling=energy_sampling,
        ),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={},
        block_energy=pair_sampled_block_energy if energy_sampling is not None else None,
        retune_block_energy=retune_block_energy,
    )
    object.__setattr__(meas_ops, _UCISD_K_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops
