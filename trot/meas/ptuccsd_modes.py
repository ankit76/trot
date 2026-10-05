from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from jax.sharding import PartitionSpec as P

from .. import walkers as wk
from ..core.ops import (
    BlockComponentEstimate,
    BlockComponentRetuneResult,
    BlockComponentsAdvanceFn,
    EstimatorOps,
    MeasOps,
    d_pt_component_sampling_noise_imag,
    d_pt_component_sampling_noise_real,
    d_pt_estimator_phase_coherence,
    d_pt_walker_proposal_ess,
    k_energy,
    k_force_bias,
)
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ptuccsd_modes import (
    PtuccsdThoulessModeTrial,
    overlap_r,
    overlap_u,
    reference_overlap_r,
)
from ..trial.ptuccsd_thouless import PtuccsdThoulessTrial, greenp_from_green
from .ptuccsd_thouless import (
    PtuccsdThoulessMeasCfg,
    PtuccsdThoulessMeasCtx,
    _chol_contract,
    _energy_gl_batched,
    _energy_gl_scalar,
    build_ptuccsd_thouless_meas_ctx,
    o_pt_components,
)
from .ptccsd_modes import (
    PtccsdModePairTuningCfg as _PtccsdModePairTuningCfg,
    PtccsdModePopulationStats as _PtccsdModePopulationStats,
    select_ptccsd_mode_pair_sampling as _select_ptccsd_mode_pair_sampling,
)
from .pt2ccsd import combine_first_order_energy, project_first_order_energy_terms
from .pair_sampling import ModelPairSamplingData, local_cholesky_component_head, local_cholesky_tail
from .ucisd_modes import _spin_sum_chol_contract as _ucisd_spin_sum_chol_contract

PtuccsdModeMeasCfg = PtuccsdThoulessMeasCfg


@dataclass(frozen=True)
class PtuccsdModePairSamplingCfg:
    """Fixed walker--Cholesky sampling policy for the UCC PT numerator."""

    chol_head_size: int
    pair_sample_size: int
    rank_head_by_guide: bool = False
    guide_chol_batch_size: int = 16
    head_chol_batch_size: int = 0
    tail_probability_uniform_mix: float = 0.0
    track_half_sample_diagnostic: bool = False
    walker_guide_policy: Literal["abs_weight", "head_rms"] = "abs_weight"
    walker_guide_weight_mix: float = 0.1

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
        if self.walker_guide_policy not in ("abs_weight", "head_rms"):
            raise ValueError("walker_guide_policy must be 'abs_weight' or 'head_rms'.")
        if not 0.0 < self.walker_guide_weight_mix <= 1.0:
            raise ValueError("walker_guide_weight_mix must lie in (0, 1].")
        if self.track_half_sample_diagnostic and self.pair_sample_size < 2:
            raise ValueError(
                "track_half_sample_diagnostic requires pair_sample_size to be at least two."
            )


@dataclass(frozen=True)
class PtuccsdModePairTuningCfg(_PtccsdModePairTuningCfg):
    """Automatic UCC PT sampler tuning with the shared PT/CISD policy."""


@dataclass(frozen=True)
class PtuccsdModePopulationStats(_PtccsdModePopulationStats):
    """Real projected UCC residual moments for walker populations."""


@dataclass(frozen=True)
class PtuccsdModePairTuningResult:
    sampling: PtuccsdModePairSamplingCfg
    chol_head_fraction: float
    estimated_tail_std_ha: float
    guarded_tail_std_ha: float
    target_tail_std_ha: float
    target_tail_std_source: str
    estimated_pair_evaluations: int


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdModeMeasCtx:
    """Spin-rotated Hamiltonian intermediates and mode batching policy."""

    base: PtuccsdThoulessMeasCtx
    reference_chol_scores: jax.Array
    chol_head_indices: jax.Array
    chol_tail_indices: jax.Array
    chol_tail_prob: jax.Array
    n_mode_chunks: int
    component_sampling: PtuccsdModePairSamplingCfg | None
    model_sampling: ModelPairSamplingData | None = None

    @property
    def h1_b(self) -> jax.Array:
        return self.base.h1_b

    @property
    def chol_b(self) -> jax.Array:
        return self.base.chol_b

    @property
    def rot_chol_a(self) -> jax.Array:
        return self.base.rot_chol_a

    @property
    def rot_chol_b(self) -> jax.Array:
        return self.base.rot_chol_b

    @property
    def cfg(self) -> PtuccsdModeMeasCfg:
        return self.base.cfg

    def tree_flatten(self):
        children = (
            self.base,
            self.reference_chol_scores,
            self.chol_head_indices,
            self.chol_tail_indices,
            self.chol_tail_prob,
            self.model_sampling,
        )
        return children, (self.n_mode_chunks, self.component_sampling)

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_mode_chunks, component_sampling = aux
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
            n_mode_chunks=n_mode_chunks,
            component_sampling=component_sampling,
            model_sampling=model_sampling,
        )


class PtuccsdModeEnergyCommon(NamedTuple):
    """Per-walker data independent of the Cholesky component sum."""

    half_green_a: jax.Array
    half_green_b: jax.Array
    greenp_a: jax.Array
    greenp_b: jax.Array
    combo2_a: jax.Array
    combo2_b: jax.Array
    theta: jax.Array
    electronic_0_base: jax.Array
    h_t_base: jax.Array


_PTUCCSD_MODE_MEAS_CFG_ATTR = "_ptuccsd_mode_meas_cfg"


def get_ptuccsd_mode_meas_cfg(meas_ops: MeasOps) -> PtuccsdModeMeasCfg | None:
    cfg = getattr(meas_ops, _PTUCCSD_MODE_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, PtuccsdModeMeasCfg) else None


def build_ptuccsd_mode_meas_ctx(
    ham_data: HamChol,
    trial_data: PtuccsdThoulessModeTrial,
    cfg: PtuccsdModeMeasCfg = PtuccsdModeMeasCfg(),
    *,
    n_mode_chunks: int = 1,
    component_sampling: PtuccsdModePairSamplingCfg | None = None,
    guide_data=None,
    guide_ctx=None,
) -> PtuccsdModeMeasCtx:
    """Build the spin-rotated Hamiltonian intermediates used by the mode guide."""

    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    # Context construction accesses only the two Thouless references and the
    # beta orbital rotation, which the dense and mode trials share exactly.
    base = build_ptuccsd_thouless_meas_ctx(
        ham_data, cast(PtuccsdThoulessTrial, trial_data), cfg,
        guide_data=guide_data, guide_ctx=guide_ctx,
    )
    n_chol = int(ham_data.chol.shape[0])
    if component_sampling is not None and component_sampling.chol_head_size > n_chol:
        raise ValueError(
            f"chol_head_size must not exceed the number of Cholesky vectors ({n_chol})."
        )
    chunks = min(int(n_mode_chunks), trial_data.mode_rank) if trial_data.mode_rank else 1
    meas_ctx = PtuccsdModeMeasCtx(
        base=base,
        reference_chol_scores=jnp.empty((0,), dtype=jnp.float64),
        chol_head_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_prob=jnp.empty((0,), dtype=jnp.float64),
        n_mode_chunks=chunks,
        component_sampling=None,
    )
    if component_sampling is None:
        return meas_ctx
    reference_scores = _build_ptuccsd_reference_chol_scores(
        ham_data,
        meas_ctx,
        trial_data,
        chol_batch_size=component_sampling.guide_chol_batch_size,
    )
    return configure_ptuccsd_mode_pair_sampling(
        meas_ctx,
        component_sampling,
        reference_scores,
    )


def _mode_apply_realimag(
    trial_data: PtuccsdThoulessModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    cfg: PtuccsdModeMeasCfg,
) -> tuple[jax.Array, jax.Array]:
    """Apply the retained combined modes using the configured precision."""

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
    projection_r = (
        projection_a_r.astype(jnp.float64) + projection_b_r.astype(jnp.float64)
    ).astype(cfg.mixed_real_dtype)

    vector_a_i = jnp.imag(matrix_a).reshape(-1).astype(cfg.mixed_real_dtype)
    vector_b_i = jnp.imag(matrix_b).reshape(-1).astype(cfg.mixed_real_dtype)
    projection_a_i = jnp.einsum("rp,p->r", modes_a, vector_a_i, optimize="optimal")
    projection_b_i = jnp.einsum("rp,p->r", modes_b, vector_b_i, optimize="optimal")
    projection_i = (
        projection_a_i.astype(jnp.float64) + projection_b_i.astype(jnp.float64)
    ).astype(cfg.mixed_real_dtype)

    applied_r = jnp.einsum("r,rp->p", values * projection_r, modes, optimize="optimal")
    applied_i = jnp.einsum("r,rp->p", values * projection_i, modes, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    applied = applied_r.astype(cfg.mixed_complex_dtype)
    applied += imag_unit * applied_i.astype(cfg.mixed_complex_dtype)

    return applied[:da].reshape(expected_a), applied[da:].reshape(expected_b)


def _mode_quadratic_batched_realimag(
    trial_data: PtuccsdThoulessModeTrial,
    matrices_a: jax.Array,
    matrices_b: jax.Array,
    cfg: PtuccsdModeMeasCfg,
    n_mode_chunks: int = 1,
) -> jax.Array:
    """Evaluate ``0.5 * z.T @ K_modes @ z`` for a batch of matrix pairs."""

    if matrices_a.shape[:-2] != matrices_b.shape[:-2]:
        raise ValueError("alpha and beta matrices must have identical leading shapes.")
    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrices_a.shape[-2:] != expected_a or matrices_b.shape[-2:] != expected_b:
        raise ValueError(
            f"matrix pair must end in shapes {expected_a} and {expected_b}, got "
            f"{matrices_a.shape} and {matrices_b.shape}."
        )

    leading_shape = matrices_a.shape[:-2]
    batch_size = math.prod(leading_shape)
    vectors_a = matrices_a.reshape((batch_size, trial_data.pair_dim[0]))
    vectors_b = matrices_b.reshape((batch_size, trial_data.pair_dim[1]))
    rank = trial_data.mode_rank
    result_dtype = (
        jnp.complex128
        if jnp.issubdtype(vectors_a.dtype, jnp.complexfloating)
        or jnp.issubdtype(vectors_b.dtype, jnp.complexfloating)
        else jnp.float64
    )
    if rank == 0:
        return jnp.zeros(leading_shape, dtype=result_dtype)

    da, _ = trial_data.pair_dim
    vectors_a_r = jnp.real(vectors_a).astype(cfg.mixed_real_dtype_testing)
    vectors_b_r = jnp.real(vectors_b).astype(cfg.mixed_real_dtype_testing)
    vectors_a_i = jnp.imag(vectors_a).astype(cfg.mixed_real_dtype_testing)
    vectors_b_i = jnp.imag(vectors_b).astype(cfg.mixed_real_dtype_testing)

    def evaluate_chunk(values_i: jax.Array, modes_i: jax.Array) -> jax.Array:
        modes_i = modes_i.astype(cfg.mixed_real_dtype_testing)
        modes_a_i = modes_i[:, :da]
        modes_b_i = modes_i[:, da:]
        projection_a_r = jnp.einsum(
            "sp,rp->sr", vectors_a_r, modes_a_i, optimize="optimal"
        )
        projection_b_r = jnp.einsum(
            "sp,rp->sr", vectors_b_r, modes_b_i, optimize="optimal"
        )
        values_t = values_i.astype(jnp.float64)[None, :]
        if result_dtype == jnp.complex128:
            projection_a_i = jnp.einsum(
                "sp,rp->sr", vectors_a_i, modes_a_i, optimize="optimal"
            )
            projection_b_i = jnp.einsum(
                "sp,rp->sr", vectors_b_i, modes_b_i, optimize="optimal"
            )
            projection_a = projection_a_r.astype(jnp.complex128)
            projection_a += 1.0j * projection_a_i.astype(jnp.complex128)
            projection_b = projection_b_r.astype(jnp.complex128)
            projection_b += 1.0j * projection_b_i.astype(jnp.complex128)
            contribution = 0.5 * values_t * (projection_a + projection_b) ** 2
            return jnp.sum(contribution, axis=1, dtype=jnp.complex128)

        projection = projection_a_r.astype(jnp.float64)
        projection += projection_b_r.astype(jnp.float64)
        contribution = 0.5 * values_t * projection**2
        return jnp.sum(contribution, axis=1, dtype=jnp.float64)

    chunks = min(int(n_mode_chunks), rank)
    if chunks == 1:
        result = evaluate_chunk(trial_data.eigenvalues, trial_data.modes)
        return result.reshape(leading_shape)

    base_chunk_size = rank // chunks
    n_larger_chunks = rank % chunks
    chunk_size = base_chunk_size + int(n_larger_chunks > 0)
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    # Local Cholesky tiles vary over the manual model axis; preserve that
    # variation in the mode-scan carry even though its initial value is zero.
    zero = jnp.zeros_like(vectors_a, shape=(vectors_a.shape[0],), dtype=result_dtype)

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


def _green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    _, _, green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = (
        _half_green_blocks(walker, trial_data)
    )
    return green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b


def _half_green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Return spin-resolved half- and full-Green-function blocks."""

    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    overlap_a = trial_data.mo_t_a.conj().T @ walker_a
    overlap_b = trial_data.mo_t_b.conj().T @ walker_b_beta
    half_green_a = jnp.linalg.solve(overlap_a.T, walker_a.T)
    half_green_b = jnp.linalg.solve(overlap_b.T, walker_b_beta.T)
    green_a = trial_data.mo_t_a.conj() @ half_green_a
    green_b = trial_data.mo_t_b.conj() @ half_green_b
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:noa, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_b[:nob, nob : nob + trial_data.nvir[1]]
    greenp_a = greenp_from_green(green_a, noa, trial_data.nvir[0])
    greenp_b = greenp_from_green(green_b, nob, trial_data.nvir[1])
    return (
        half_green_a,
        half_green_b,
        green_a,
        green_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
    )


def _force_bias_half_green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Build only the half-Green blocks needed by the guide force bias."""

    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    overlap_a = trial_data.mo_t_a.conj().T @ walker_a
    overlap_b = trial_data.mo_t_b.conj().T @ walker_b_beta
    half_green_a = jnp.linalg.solve(overlap_a.T, walker_a.T)
    half_green_b = jnp.linalg.solve(overlap_b.T, walker_b_beta.T)
    noa, nob = trial_data.nocc

    green_rows_a = trial_data.mo_t_a.conj()[:noa, :] @ half_green_a
    green_rows_b = trial_data.mo_t_b.conj()[:nob, :] @ half_green_b
    green_occ_a = green_rows_a[:, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_rows_b[:, nob : nob + trial_data.nvir[1]]

    greenp_a = trial_data.mo_t_a.conj() @ half_green_a[:, noa : noa + trial_data.nvir[0]]
    greenp_b = trial_data.mo_t_b.conj() @ half_green_b[:, nob : nob + trial_data.nvir[1]]
    greenp_a = greenp_a.at[noa : noa + trial_data.nvir[0], :].add(
        -jnp.eye(trial_data.nvir[0], dtype=greenp_a.dtype)
    )
    greenp_b = greenp_b.at[nob : nob + trial_data.nvir[1], :].add(
        -jnp.eye(trial_data.nvir[1], dtype=greenp_b.dtype)
    )
    return (
        half_green_a,
        half_green_b,
        green_rows_a,
        green_rows_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
    )


def _force_bias_kernel_uw_rh_full(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Full-Green force bias retained as a correctness oracle."""

    green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = _green_blocks(
        walker,
        trial_data,
    )
    noa, nob = trial_data.nocc

    # As in dense PT-UCCSD and UCISD, keep the determinant-reference force
    # bias in full precision and evaluate only the correlation correction in
    # the configured mixed precision.
    f0_a = jnp.einsum("gij,ij->g", ham_data.chol, green_a, optimize="optimal")
    f0_b = jnp.einsum("gij,ij->g", meas_ctx.chol_b, green_b, optimize="optimal")
    applied_a, applied_b = _mode_apply_realimag(
        trial_data,
        green_occ_a,
        green_occ_b,
        meas_ctx.cfg,
    )
    t2_green_a = (greenp_a @ applied_a.T) @ green_a[:noa, :]
    t2_green_b = (greenp_b @ applied_b.T) @ green_b[:nob, :]
    correction = -_chol_contract(ham_data.chol, t2_green_a, meas_ctx.cfg)
    correction -= _chol_contract(meas_ctx.chol_b, t2_green_b, meas_ctx.cfg)
    return f0_a + f0_b + correction


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Half-Green PT-UCCSD exponential-guide force bias."""

    (
        half_green_a,
        half_green_b,
        green_rows_a,
        green_rows_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
    ) = _force_bias_half_green_blocks(walker, trial_data)

    f0_a = jnp.einsum(
        "giq,iq->g", meas_ctx.rot_chol_a, half_green_a, optimize="optimal"
    )
    f0_b = jnp.einsum(
        "giq,iq->g", meas_ctx.rot_chol_b, half_green_b, optimize="optimal"
    )
    applied_a, applied_b = _mode_apply_realimag(
        trial_data,
        green_occ_a,
        green_occ_b,
        meas_ctx.cfg,
    )
    t2_green_a = (greenp_a @ applied_a.T) @ green_rows_a
    t2_green_b = (greenp_b @ applied_b.T) @ green_rows_b
    correction = _ucisd_spin_sum_chol_contract(
        ham_data.chol,
        t2_green_a,
        t2_green_b,
        trial_data.mo_coeff_b,
        meas_ctx.cfg,
    )
    return f0_a + f0_b - correction


def _force_bias_kernel_rw_rh_full(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Restricted-walker wrapper for the full-Green force-bias oracle."""

    noa, nob = trial_data.nocc
    return _force_bias_kernel_uw_rh_full(
        (walker[:, :noa], walker[:, :nob]),
        ham_data,
        meas_ctx,
        trial_data,
    )


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Mode-native force bias for a restricted open-shell walker."""

    noa, nob = trial_data.nocc
    return force_bias_kernel_uw_rh(
        (walker[:, :noa], walker[:, :nob]),
        ham_data,
        meas_ctx,
        trial_data,
    )


def _energy_components_uw_rh_reference(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return mode-native ``(theta2, electronic_0, h_t)`` components."""

    (
        half_green_a,
        half_green_b,
        green_a,
        green_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
    ) = _half_green_blocks(walker, trial_data)
    noa, nob = trial_data.nocc
    cfg = meas_ctx.cfg
    h1_a = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = meas_ctx.h1_b
    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    e1_0 = jnp.einsum("ij,ij->", h1_a, green_a, optimize="optimal")
    e1_0 += jnp.einsum("ij,ij->", h1_b, green_b, optimize="optimal")

    applied_a, applied_b = _mode_apply_realimag(
        trial_data,
        green_occ_a,
        green_occ_b,
        cfg,
    )
    theta2 = 0.5 * jnp.einsum(
        "pt,pt->", green_occ_a, applied_a, optimize="optimal"
    )
    theta2 += 0.5 * jnp.einsum(
        "pt,pt->", green_occ_b, applied_b, optimize="optimal"
    )
    combo_a = (greenp_a @ applied_a.T) @ green_a[:noa, :]
    combo_b = (greenp_b @ applied_b.T) @ green_b[:nob, :]
    e1_2 = e1_0 * theta2
    e1_2 -= jnp.einsum("ij,ij->", h1_a, combo_a, optimize="optimal")
    e1_2 -= jnp.einsum("ij,ij->", h1_b, combo_b, optimize="optimal")

    lg_a = jnp.einsum("giq,iq->g", rot_chol_a, half_green_a, optimize="optimal")
    lg_b = jnp.einsum("giq,iq->g", rot_chol_b, half_green_b, optimize="optimal")
    lg = lg_a + lg_b
    lg1_a = jnp.einsum(
        "gip,jp->gij", rot_chol_a, half_green_a, optimize="optimal"
    )
    lg1_b = jnp.einsum(
        "gip,jp->gij", rot_chol_b, half_green_b, optimize="optimal"
    )
    e2_0 = 0.5 * (lg @ lg)
    e2_0 -= 0.5 * (
        jnp.sum(lg1_a * jnp.swapaxes(lg1_a, -1, -2))
        + jnp.sum(lg1_b * jnp.swapaxes(lg1_b, -1, -2))
    )
    electronic_0 = e1_0 + e2_0

    combo2_a = 2.0 * combo_a
    combo2_b = 2.0 * combo_b
    lt2g_a = _chol_contract(chol_a, combo2_a, cfg)
    lt2g_b = _chol_contract(chol_b, combo2_b, cfg)
    e2_2_2_1 = -0.5 * ((lt2g_a + lt2g_b) @ lg)

    # Cast before constructing walker--Cholesky batches so the selected mixed
    # precision controls their memory footprint, as in dense PT-UCCSD.
    reference_occ_a = trial_data.mo_t_a.conj()[:noa, :].astype(cfg.mixed_complex_dtype)
    reference_occ_b = trial_data.mo_t_b.conj()[:nob, :].astype(cfg.mixed_complex_dtype)
    greenp_a_mixed = greenp_a.astype(cfg.mixed_complex_dtype)
    greenp_b_mixed = greenp_b.astype(cfg.mixed_complex_dtype)
    combo2_a_mixed = combo2_a.astype(cfg.mixed_complex_dtype)
    combo2_b_mixed = combo2_b.astype(cfg.mixed_complex_dtype)

    if cfg.memory_mode == "low":
        zero_e222 = jnp.zeros_like(e2_2_2_1)
        zero_e23 = jnp.asarray(0.0, dtype=jnp.complex128)

        def scan_doubles(carry, xs):
            e222_acc, e23_acc = carry
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = xs
            gl_half_a_i = _energy_gl_scalar(half_green_a, chol_a_i, cfg)
            gl_half_b_i = _energy_gl_scalar(half_green_b, chol_b_i, cfg)
            lcombo_a_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_a_i.astype(cfg.mixed_complex_dtype),
                combo2_a_mixed,
                optimize="optimal",
            )
            lcombo_b_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_b_i.astype(cfg.mixed_complex_dtype),
                combo2_b_mixed,
                optimize="optimal",
            )
            e222_acc += 0.5 * (
                jnp.einsum("pi,pi->", gl_half_a_i, lcombo_a_i, optimize="optimal")
                + jnp.einsum("pi,pi->", gl_half_b_i, lcombo_b_i, optimize="optimal")
            )

            gl_occ_a_i = reference_occ_a @ gl_half_a_i
            gl_occ_b_i = reference_occ_b @ gl_half_b_i
            glgp_a_i = jnp.einsum(
                "pi,it->pt", gl_occ_a_i, greenp_a_mixed, optimize="optimal"
            ).astype(cfg.mixed_complex_dtype_testing)
            glgp_b_i = jnp.einsum(
                "pi,it->pt", gl_occ_b_i, greenp_b_mixed, optimize="optimal"
            ).astype(cfg.mixed_complex_dtype_testing)
            e23_i = _mode_quadratic_batched_realimag(
                trial_data,
                glgp_a_i[None, ...],
                glgp_b_i[None, ...],
                cfg,
                meas_ctx.n_mode_chunks,
            )[0]
            return (e222_acc, e23_acc + e23_i), None

        (e2_2_2_2, e2_2_3), _ = lax.scan(
            scan_doubles,
            (zero_e222, zero_e23),
            (chol_a, rot_chol_a, chol_b, rot_chol_b),
        )
    else:
        gl_half_a = _energy_gl_batched(half_green_a, chol_a, cfg)
        gl_half_b = _energy_gl_batched(half_green_b, chol_b, cfg)
        lcombo_a = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_a.astype(cfg.mixed_complex_dtype),
            combo2_a_mixed,
            optimize="optimal",
        )
        lcombo_b = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_b.astype(cfg.mixed_complex_dtype),
            combo2_b_mixed,
            optimize="optimal",
        )
        e2_2_2_2 = 0.5 * (
            jnp.einsum("gpi,gpi->", gl_half_a, lcombo_a, optimize="optimal")
            + jnp.einsum("gpi,gpi->", gl_half_b, lcombo_b, optimize="optimal")
        )

        gl_occ_a = jnp.einsum(
            "ij,gjq->giq", reference_occ_a, gl_half_a, optimize="optimal"
        )
        gl_occ_b = jnp.einsum(
            "ij,gjq->giq", reference_occ_b, gl_half_b, optimize="optimal"
        )
        glgp_a = jnp.einsum(
            "gpi,it->gpt", gl_occ_a, greenp_a_mixed, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        glgp_b = jnp.einsum(
            "gpi,it->gpt", gl_occ_b, greenp_b_mixed, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        e2_2_3 = jnp.sum(
            _mode_quadratic_batched_realimag(
                trial_data,
                glgp_a,
                glgp_b,
                cfg,
                meas_ctx.n_mode_chunks,
            ),
            dtype=jnp.complex128,
        )

    e2_2 = e2_0 * theta2 + e2_2_2_1 + e2_2_2_2 + e2_2_3
    h_t = e1_2 + e2_2
    return theta2, electronic_0, h_t


def _ptuccsd_mode_energy_common_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> PtuccsdModeEnergyCommon:
    """Build per-walker data outside the Cholesky component sum."""

    (
        half_green_a,
        half_green_b,
        green_a,
        green_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
    ) = _half_green_blocks(walker, trial_data)
    noa, nob = trial_data.nocc
    cfg = meas_ctx.cfg
    h1_a = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = meas_ctx.h1_b

    e1_0 = jnp.einsum("ij,ij->", h1_a, green_a, optimize="optimal")
    e1_0 += jnp.einsum("ij,ij->", h1_b, green_b, optimize="optimal")

    applied_a, applied_b = _mode_apply_realimag(
        trial_data,
        green_occ_a,
        green_occ_b,
        cfg,
    )
    theta = 0.5 * jnp.einsum(
        "pt,pt->", green_occ_a, applied_a, optimize="optimal"
    )
    theta += 0.5 * jnp.einsum(
        "pt,pt->", green_occ_b, applied_b, optimize="optimal"
    )
    combo_a = (greenp_a @ applied_a.T) @ green_a[:noa, :]
    combo_b = (greenp_b @ applied_b.T) @ green_b[:nob, :]
    e1_2 = e1_0 * theta
    e1_2 -= jnp.einsum("ij,ij->", h1_a, combo_a, optimize="optimal")
    e1_2 -= jnp.einsum("ij,ij->", h1_b, combo_b, optimize="optimal")

    return PtuccsdModeEnergyCommon(
        half_green_a=half_green_a,
        half_green_b=half_green_b,
        greenp_a=greenp_a,
        greenp_b=greenp_b,
        combo2_a=2.0 * combo_a,
        combo2_b=2.0 * combo_b,
        theta=theta,
        electronic_0_base=e1_0,
        h_t_base=e1_2,
    )


def _ptuccsd_mode_energy_common_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> PtuccsdModeEnergyCommon:
    """Restricted-open-shell wrapper for the spin-resolved common data."""

    noa, nob = trial_data.nocc
    return _ptuccsd_mode_energy_common_uw_rh(
        (walker[:, :noa], walker[:, :nob]),
        ham_data,
        meas_ctx,
        trial_data,
    )


def _ptuccsd_mode_chol_terms(
    common: PtuccsdModeEnergyCommon,
    chol_a: jax.Array,
    rot_chol_a: jax.Array,
    chol_b: jax.Array,
    rot_chol_b: jax.Array,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Return ``[e2_0, theta * e2_0 + connected]`` by Cholesky vector."""

    noa, nob = trial_data.nocc
    cfg = meas_ctx.cfg
    reference_occ_a = trial_data.mo_t_a.conj()[:noa, :].astype(
        cfg.mixed_complex_dtype
    )
    reference_occ_b = trial_data.mo_t_b.conj()[:nob, :].astype(
        cfg.mixed_complex_dtype
    )
    greenp_a = common.greenp_a.astype(cfg.mixed_complex_dtype)
    greenp_b = common.greenp_b.astype(cfg.mixed_complex_dtype)
    combo2_a = common.combo2_a.astype(cfg.mixed_complex_dtype)
    combo2_b = common.combo2_b.astype(cfg.mixed_complex_dtype)

    def scalar_term(
        chol_a_i: jax.Array,
        rot_chol_a_i: jax.Array,
        chol_b_i: jax.Array,
        rot_chol_b_i: jax.Array,
    ) -> jax.Array:
        lg_i = jnp.einsum(
            "iq,iq->", rot_chol_a_i, common.half_green_a, optimize="optimal"
        )
        lg_i += jnp.einsum(
            "iq,iq->", rot_chol_b_i, common.half_green_b, optimize="optimal"
        )
        lg1_a_i = jnp.einsum(
            "ip,jp->ij",
            rot_chol_a_i,
            common.half_green_a,
            optimize="optimal",
        )
        lg1_b_i = jnp.einsum(
            "ip,jp->ij",
            rot_chol_b_i,
            common.half_green_b,
            optimize="optimal",
        )
        e20_i = 0.5 * lg_i * lg_i
        e20_i -= 0.5 * jnp.sum(lg1_a_i * jnp.swapaxes(lg1_a_i, -1, -2))
        e20_i -= 0.5 * jnp.sum(lg1_b_i * jnp.swapaxes(lg1_b_i, -1, -2))
        lt2g_a_i = _chol_contract(chol_a_i[None, ...], common.combo2_a, cfg)[0]
        lt2g_b_i = _chol_contract(chol_b_i[None, ...], common.combo2_b, cfg)[0]
        e221_i = -0.5 * (lt2g_a_i + lt2g_b_i) * lg_i

        gl_half_a_i = _energy_gl_scalar(common.half_green_a, chol_a_i, cfg)
        gl_half_b_i = _energy_gl_scalar(common.half_green_b, chol_b_i, cfg)
        lcombo_a_i = jnp.einsum(
            "pi,ji->pj",
            rot_chol_a_i.astype(cfg.mixed_complex_dtype),
            combo2_a,
            optimize="optimal",
        )
        lcombo_b_i = jnp.einsum(
            "pi,ji->pj",
            rot_chol_b_i.astype(cfg.mixed_complex_dtype),
            combo2_b,
            optimize="optimal",
        )
        e222_i = 0.5 * (
            jnp.einsum("pi,pi->", gl_half_a_i, lcombo_a_i, optimize="optimal")
            + jnp.einsum("pi,pi->", gl_half_b_i, lcombo_b_i, optimize="optimal")
        )

        gl_occ_a_i = reference_occ_a @ gl_half_a_i
        gl_occ_b_i = reference_occ_b @ gl_half_b_i
        glgp_a_i = jnp.einsum(
            "pi,it->pt", gl_occ_a_i, greenp_a, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        glgp_b_i = jnp.einsum(
            "pi,it->pt", gl_occ_b_i, greenp_b, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        e223_i = _mode_quadratic_batched_realimag(
            trial_data,
            glgp_a_i[None, ...],
            glgp_b_i[None, ...],
            cfg,
            meas_ctx.n_mode_chunks,
        )[0]
        connected_i = e221_i + e222_i + e223_i
        return jnp.stack((e20_i, common.theta * e20_i + connected_i))

    if cfg.memory_mode == "low":
        zero = jnp.zeros((), dtype=jnp.result_type(common.half_green_a, chol_a))

        def scan_term(carry, xs):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = xs
            return carry, scalar_term(
                chol_a_i,
                rot_chol_a_i,
                chol_b_i,
                rot_chol_b_i,
            )

        _, terms = lax.scan(
            scan_term,
            zero,
            (chol_a, rot_chol_a, chol_b, rot_chol_b),
        )
        return terms

    lg_a = jnp.einsum(
        "giq,iq->g", rot_chol_a, common.half_green_a, optimize="optimal"
    )
    lg_b = jnp.einsum(
        "giq,iq->g", rot_chol_b, common.half_green_b, optimize="optimal"
    )
    lg = lg_a + lg_b
    lg1_a = jnp.einsum(
        "gip,jp->gij", rot_chol_a, common.half_green_a, optimize="optimal"
    )
    lg1_b = jnp.einsum(
        "gip,jp->gij", rot_chol_b, common.half_green_b, optimize="optimal"
    )
    e20 = 0.5 * lg * lg
    e20 -= 0.5 * jnp.sum(
        lg1_a * jnp.swapaxes(lg1_a, -1, -2), axis=(-1, -2)
    )
    e20 -= 0.5 * jnp.sum(
        lg1_b * jnp.swapaxes(lg1_b, -1, -2), axis=(-1, -2)
    )
    lt2g_a = _chol_contract(chol_a, common.combo2_a, cfg)
    lt2g_b = _chol_contract(chol_b, common.combo2_b, cfg)
    e221 = -0.5 * (lt2g_a + lt2g_b) * lg

    gl_half_a = _energy_gl_batched(common.half_green_a, chol_a, cfg)
    gl_half_b = _energy_gl_batched(common.half_green_b, chol_b, cfg)
    lcombo_a = jnp.einsum(
        "gpi,ji->gpj",
        rot_chol_a.astype(cfg.mixed_complex_dtype),
        combo2_a,
        optimize="optimal",
    )
    lcombo_b = jnp.einsum(
        "gpi,ji->gpj",
        rot_chol_b.astype(cfg.mixed_complex_dtype),
        combo2_b,
        optimize="optimal",
    )
    e222 = 0.5 * (
        jnp.einsum("gpi,gpi->g", gl_half_a, lcombo_a, optimize="optimal")
        + jnp.einsum("gpi,gpi->g", gl_half_b, lcombo_b, optimize="optimal")
    )

    gl_occ_a = jnp.einsum(
        "ij,gjq->giq", reference_occ_a, gl_half_a, optimize="optimal"
    )
    gl_occ_b = jnp.einsum(
        "ij,gjq->giq", reference_occ_b, gl_half_b, optimize="optimal"
    )
    glgp_a = jnp.einsum(
        "gpi,it->gpt", gl_occ_a, greenp_a, optimize="optimal"
    ).astype(cfg.mixed_complex_dtype_testing)
    glgp_b = jnp.einsum(
        "gpi,it->gpt", gl_occ_b, greenp_b, optimize="optimal"
    ).astype(cfg.mixed_complex_dtype_testing)
    e223 = _mode_quadratic_batched_realimag(
        trial_data,
        glgp_a,
        glgp_b,
        cfg,
        meas_ctx.n_mode_chunks,
    )
    connected = e221 + e222 + e223
    return jnp.stack((e20, common.theta * e20 + connected), axis=-1)


def _ptuccsd_mode_chol_terms_for_walkers(
    common: PtuccsdModeEnergyCommon,
    chol_a: jax.Array,
    rot_chol_a: jax.Array,
    chol_b: jax.Array,
    rot_chol_b: jax.Array,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
    *,
    n_chunks: int = 1,
) -> jax.Array:
    """Return bounded walker-by-Cholesky PT component terms."""

    return wk.vmap_chunked(
        lambda common_i: _ptuccsd_mode_chol_terms(
            common_i,
            chol_a,
            rot_chol_a,
            chol_b,
            rot_chol_b,
            meas_ctx,
            trial_data,
        ),
        n_chunks=n_chunks,
    )(common)


def _ptuccsd_mode_chol_index_terms(
    common: PtuccsdModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
    *,
    n_chunks: int,
) -> jax.Array:
    """Return one walker's component terms at arbitrary Cholesky indices."""

    return wk.vmap_chunked(
        lambda chol_i: _ptuccsd_mode_chol_terms(
            common,
            ham_data.chol[chol_i][None, ...],
            meas_ctx.rot_chol_a[chol_i][None, ...],
            meas_ctx.chol_b[chol_i][None, ...],
            meas_ctx.rot_chol_b[chol_i][None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        shard_walkers=False,
    )(chol_indices)


def _ptuccsd_mode_chol_index_moments_for_walkers(
    common: PtuccsdModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
    theta_reference: jax.Array | float = 0.0,
    compute_projection_moments: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Stream a UCC component head and final-energy projection moments."""

    head_size = int(chol_indices.shape[0])
    zero_total = jnp.stack(
        (
            jnp.zeros_like(common.electronic_0_base),
            jnp.zeros_like(common.h_t_base),
        ),
        axis=-1,
    )
    zero_real = jnp.zeros_like(jnp.real(common.h_t_base), dtype=jnp.float64)
    if head_size == 0:
        return zero_total, zero_real, zero_real, zero_real

    batch_size = head_size if chol_batch_size <= 0 else min(chol_batch_size, head_size)
    n_batches = math.ceil(head_size / batch_size)
    padded_size = n_batches * batch_size
    padded_indices = jnp.pad(chol_indices, (0, padded_size - head_size)).reshape(
        n_batches, batch_size
    )
    valid = (jnp.arange(padded_size) < head_size).reshape(n_batches, batch_size)

    def scan_batch(carry, xs):
        total, real_sq, imag_sq, real_imag = carry
        indices_i, valid_i = xs
        terms_i = _ptuccsd_mode_chol_terms_for_walkers(
            common,
            ham_data.chol[indices_i],
            meas_ctx.rot_chol_a[indices_i],
            meas_ctx.chol_b[indices_i],
            meas_ctx.rot_chol_b[indices_i],
            meas_ctx,
            trial_data,
            n_chunks=n_walker_chunks,
        )
        terms_i = jnp.where(valid_i[None, :, None], terms_i, 0.0)
        total = total + jnp.sum(terms_i, axis=1)
        if compute_projection_moments:
            effective_i = project_first_order_energy_terms(theta_reference, terms_i)
            terms_real = jnp.real(effective_i).astype(jnp.float64)
            terms_imag = jnp.imag(effective_i).astype(jnp.float64)
            real_sq = real_sq + jnp.sum(terms_real**2, axis=1, dtype=jnp.float64)
            imag_sq = imag_sq + jnp.sum(terms_imag**2, axis=1, dtype=jnp.float64)
            real_imag = real_imag + jnp.sum(
                terms_real * terms_imag,
                axis=1,
                dtype=jnp.float64,
            )
        return (total, real_sq, imag_sq, real_imag), None

    moments, _ = lax.scan(
        scan_batch,
        (zero_total, zero_real, zero_real, zero_real),
        (padded_indices, valid),
    )
    return moments


def _ptuccsd_mode_chol_index_sum_for_walkers(
    common: PtuccsdModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
) -> jax.Array:
    """Stream an exact UCC component head without storing every pair."""

    total, _, _, _ = _ptuccsd_mode_chol_index_moments_for_walkers(
        common,
        chol_indices,
        ham_data,
        meas_ctx,
        trial_data,
        n_walker_chunks=n_walker_chunks,
        chol_batch_size=chol_batch_size,
        compute_projection_moments=False,
    )
    return total


def _ptuccsd_mode_chol_pair_terms(
    common: PtuccsdModeEnergyCommon,
    sample_walker: jax.Array,
    sample_chol: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
    *,
    n_chunks: int = 1,
) -> jax.Array:
    """Evaluate gathered UCC walker--Cholesky component pairs in microbatches."""

    return wk.vmap_chunked(
        lambda walker_i, chol_i: _ptuccsd_mode_chol_terms(
            tree_util.tree_map(lambda value: value[walker_i], common),
            ham_data.chol[chol_i][None, ...],
            meas_ctx.rot_chol_a[chol_i][None, ...],
            meas_ctx.chol_b[chol_i][None, ...],
            meas_ctx.rot_chol_b[chol_i][None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        in_axes=(0, 0),
        shard_walkers=False,
    )(sample_walker, sample_chol)


def _build_ptuccsd_reference_chol_scores(
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
    *,
    chol_batch_size: int,
) -> jax.Array:
    """Build bounded reference scores for the combined PT energy residual."""

    n_chol = int(ham_data.chol.shape[0])
    reference_walker = (
        trial_data.mo_t_a,
        trial_data.mo_coeff_b @ trial_data.mo_t_b,
    )
    common = _ptuccsd_mode_energy_common_uw_rh(
        reference_walker,
        ham_data,
        meas_ctx,
        trial_data,
    )
    indices = jnp.arange(n_chol, dtype=jnp.int32)
    n_chunks = max(1, math.ceil(n_chol / chol_batch_size))
    terms = _ptuccsd_mode_chol_index_terms(
        common,
        indices,
        ham_data,
        meas_ctx,
        trial_data,
        n_chunks=n_chunks,
    )
    effective = project_first_order_energy_terms(common.theta, terms)
    return jnp.maximum(jnp.abs(jnp.real(effective)).astype(jnp.float64), 1.0e-300)


def configure_ptuccsd_mode_pair_sampling(
    meas_ctx: PtuccsdModeMeasCtx,
    sampling: PtuccsdModePairSamplingCfg,
    guide_scores: jax.Array,
) -> PtuccsdModeMeasCtx:
    """Attach an exact UCC head and strictly positive fixed tail proposal."""

    if meas_ctx.model_sampling is not None:
        raise ValueError("Local Cholesky sampling requires a frozen guide; rebuild the context to reconfigure.")
    scores = jnp.asarray(guide_scores, dtype=jnp.float64)
    n_chol = int(meas_ctx.rot_chol_a.shape[0])
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
        reference_chol_scores=scores,
        chol_head_indices=head_indices,
        chol_tail_indices=tail_indices,
        chol_tail_prob=tail_prob,
        component_sampling=sampling,
    )


def _local_cholesky_context_specs(ctx):
    specs = jax.tree.map(lambda _: P(), ctx)
    return replace(specs, base=replace(specs.base, chol_b=P("model"),
                                      rot_chol_a=P("model"), rot_chol_b=P("model")))


def _local_cholesky_head_terms(common, indices, h, ctx, trial, *, n_chunks):
    return _ptuccsd_mode_chol_terms_for_walkers(
        common, h.chol[indices], ctx.rot_chol_a[indices], ctx.chol_b[indices],
        ctx.rot_chol_b[indices], ctx, trial, n_chunks=n_chunks,
    )


def pair_sampled_ptuccsd_block_components(
    walkers: jax.Array,
    candidate_weights: jax.Array,
    rng_key: jax.Array,
    n_chunks: int,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> BlockComponentEstimate:
    """Estimate both Cholesky-resolved UCC PT component numerators."""

    sampling = meas_ctx.component_sampling
    if sampling is None:
        raise ValueError(
            "pair_sampled_ptuccsd_block_components requires a component sampling config."
        )
    n_walkers = int(walkers.shape[0])
    if candidate_weights.shape != (n_walkers,):
        raise ValueError(
            f"candidate_weights must have shape {(n_walkers,)}, got "
            f"{candidate_weights.shape}."
        )

    common = wk.vmap_chunked(
        _ptuccsd_mode_energy_common_rw_rh,
        n_chunks=n_chunks,
        in_axes=(0, None, None, None),
    )(walkers, ham_data, meas_ctx, trial_data)

    finite_common = jnp.ones((n_walkers,), dtype=jnp.bool_)
    for value in tree_util.tree_leaves(common):
        finite_common = finite_common & jnp.all(
            jnp.isfinite(value.reshape(n_walkers, -1)), axis=1
        )
    finite_weights = jnp.isfinite(candidate_weights)
    preliminary_valid = finite_common & finite_weights
    preliminary_weights = jnp.where(preliminary_valid, candidate_weights, 0.0)
    preliminary_weight = jnp.sum(preliminary_weights)
    preliminary_weight_safe = jnp.where(
        preliminary_weight == 0.0, 1.0, preliminary_weight
    )
    theta_reference = jnp.sum(preliminary_weights * common.theta)
    theta_reference /= preliminary_weight_safe
    theta_reference = jnp.where(preliminary_weight == 0.0, 0.0, theta_reference)

    if meas_ctx.model_sampling is not None:
        head_sum, head_real_sq, head_imag_sq, head_real_imag = local_cholesky_component_head(
            common, theta_reference, ham_data, meas_ctx, trial_data,
            terms_fn=_local_cholesky_head_terms, project_fn=project_first_order_energy_terms,
            context_specs_fn=_local_cholesky_context_specs, n_chunks=n_chunks,
        )
    elif sampling.walker_guide_policy == "head_rms":
        head_sum, head_real_sq, head_imag_sq, head_real_imag = (
            _ptuccsd_mode_chol_index_moments_for_walkers(
                common,
                meas_ctx.chol_head_indices,
                ham_data,
                meas_ctx,
                trial_data,
                n_walker_chunks=n_chunks,
                chol_batch_size=sampling.head_chol_batch_size,
                theta_reference=theta_reference,
            )
        )
    else:
        head_sum = _ptuccsd_mode_chol_index_sum_for_walkers(
            common,
            meas_ctx.chol_head_indices,
            ham_data,
            meas_ctx,
            trial_data,
            n_walker_chunks=n_chunks,
            chol_batch_size=sampling.head_chol_batch_size,
        )
        head_real_sq = jnp.zeros_like(
            jnp.real(common.electronic_0_base), dtype=jnp.float64
        )
        head_imag_sq = jnp.zeros_like(head_real_sq)
        head_real_imag = jnp.zeros_like(head_real_sq)
    exact_components = jnp.stack(
        (
            common.theta,
            common.electronic_0_base + head_sum[:, 0],
            common.h_t_base + head_sum[:, 1],
        ),
        axis=1,
    )

    finite_components = jnp.all(jnp.isfinite(exact_components), axis=1)
    valid = preliminary_valid & finite_components
    estimator_weights = jnp.where(valid, candidate_weights, 0.0)
    safe_components = jnp.where(valid[:, None], exact_components, 0.0)
    estimator_weight = jnp.sum(estimator_weights)
    numerator = jnp.sum(estimator_weights[:, None] * safe_components, axis=0)

    abs_weights = jnp.abs(estimator_weights).astype(jnp.float64)
    abs_weight_sum = jnp.sum(abs_weights, dtype=jnp.float64)
    abs_weight_sum_safe = jnp.where(abs_weight_sum == 0.0, 1.0, abs_weight_sum)
    abs_weight_prob = abs_weights / abs_weight_sum_safe
    abs_weight_prob = jnp.where(
        abs_weight_sum == 0.0,
        jnp.full_like(abs_weight_prob, 1.0 / n_walkers),
        abs_weight_prob,
    )
    if sampling.walker_guide_policy == "head_rms":
        estimator_weight_safe = jnp.where(
            estimator_weight == 0.0, 1.0, estimator_weight
        )
        normalized_weights = estimator_weights / estimator_weight_safe
        normalized_real = jnp.real(normalized_weights).astype(jnp.float64)
        normalized_imag = jnp.imag(normalized_weights).astype(jnp.float64)
        projected_head_sq = (
            normalized_real**2 * head_real_sq
            + normalized_imag**2 * head_imag_sq
            - 2.0 * normalized_real * normalized_imag * head_real_imag
        )
        head_scores = jnp.sqrt(jnp.maximum(projected_head_sq, 0.0))
        head_scores = jnp.where(
            valid & (estimator_weight != 0.0) & jnp.isfinite(head_scores),
            head_scores,
            0.0,
        )
        head_score_sum = jnp.sum(head_scores, dtype=jnp.float64)
        head_score_sum_safe = jnp.where(head_score_sum == 0.0, 1.0, head_score_sum)
        guided_prob = head_scores / head_score_sum_safe
        guided_prob = jnp.where(
            head_score_sum == 0.0,
            abs_weight_prob,
            guided_prob,
        )
        weight_mix = sampling.walker_guide_weight_mix
        walker_prob = weight_mix * abs_weight_prob + (1.0 - weight_mix) * guided_prob
    else:
        walker_prob = abs_weight_prob
    diagnostics: dict[str, jax.Array] = {
        d_pt_estimator_phase_coherence: jnp.where(
            abs_weight_sum == 0.0,
            0.0,
            jnp.abs(estimator_weight) / abs_weight_sum_safe,
        ),
        d_pt_walker_proposal_ess: jnp.where(
            abs_weight_sum == 0.0,
            0.0,
            1.0 / jnp.sum(walker_prob**2, dtype=jnp.float64),
        ),
    }

    tail_size = int(meas_ctx.chol_tail_indices.shape[0])
    if tail_size == 0:
        if sampling.track_half_sample_diagnostic:
            diagnostics[d_pt_component_sampling_noise_real] = jnp.asarray(
                0.0, dtype=jnp.float64
            )
            diagnostics[d_pt_component_sampling_noise_imag] = jnp.asarray(
                0.0, dtype=jnp.float64
            )
        return BlockComponentEstimate(
            weight=estimator_weight,
            numerator=numerator,
            diagnostics=diagnostics,
        )

    def sample_tail(key):
        key_walker, key_chol = jax.random.split(key)
        sample_walker = jax.random.choice(
            key_walker,
            n_walkers,
            shape=(sampling.pair_sample_size,),
            replace=True,
            p=walker_prob,
        )
        sample_chol_rel = jax.random.choice(
            key_chol,
            tail_size,
            shape=(sampling.pair_sample_size,),
            replace=True,
            p=meas_ctx.chol_tail_prob,
        )
        sample_chol = meas_ctx.chol_tail_indices[sample_chol_rel]
        walker_batch_size = (n_walkers + n_chunks - 1) // n_chunks
        pair_n_chunks = (
            sampling.pair_sample_size + walker_batch_size - 1
        ) // walker_batch_size
        component_terms = _ptuccsd_mode_chol_pair_terms(
            common,
            sample_walker,
            sample_chol,
            ham_data,
            meas_ctx,
            trial_data,
            n_chunks=pair_n_chunks,
        )
        importance_samples = (
            estimator_weights[sample_walker, None]
            * component_terms
            / (
                walker_prob[sample_walker, None]
                * meas_ctx.chol_tail_prob[sample_chol_rel, None]
            )
        )
        tail_numerator = jnp.mean(importance_samples, axis=0)
        if not sampling.track_half_sample_diagnostic:
            return tail_numerator, jnp.zeros_like(tail_numerator)

        first_size = sampling.pair_sample_size // 2
        second_size = sampling.pair_sample_size - first_size
        first_mean = jnp.mean(importance_samples[:first_size], axis=0)
        second_mean = jnp.mean(importance_samples[first_size:], axis=0)
        scale = math.sqrt(first_size * second_size) / sampling.pair_sample_size
        return tail_numerator, scale * (first_mean - second_mean)

    zero_tail = jnp.zeros((2,), dtype=numerator.dtype)
    if meas_ctx.model_sampling is not None:
        tail_numerator, half_difference = local_cholesky_tail(
            common, estimator_weights, walker_prob, rng_key, ham_data, meas_ctx, trial_data,
            pair_fn=_ptuccsd_mode_chol_pair_terms,
            context_specs_fn=_local_cholesky_context_specs, n_chunks=n_chunks, components=True,
        )
    else:
        tail_numerator, half_difference = lax.cond(
            abs_weight_sum > 0.0, sample_tail, lambda key: (zero_tail, zero_tail), rng_key,
        )
    numerator = numerator.at[1:].add(tail_numerator)
    if sampling.track_half_sample_diagnostic:
        estimator_weight_safe = jnp.where(
            estimator_weight == 0.0, 1.0, estimator_weight
        )
        normalized_difference = half_difference / estimator_weight_safe
        theta_mean = numerator[0] / estimator_weight_safe
        energy_difference = project_first_order_energy_terms(
            theta_mean, normalized_difference
        )
        diagnostics[d_pt_component_sampling_noise_real] = jnp.real(
            energy_difference
        )
        diagnostics[d_pt_component_sampling_noise_imag] = jnp.imag(
            energy_difference
        )
    return BlockComponentEstimate(
        weight=estimator_weight,
        numerator=numerator,
        diagnostics=diagnostics,
    )


@jax.jit
def _ptuccsd_mode_population_common_batch(walkers, ham_data, meas_ctx, trial_data):
    return jax.vmap(
        _ptuccsd_mode_energy_common_rw_rh,
        in_axes=(0, None, None, None),
    )(walkers, ham_data, meas_ctx, trial_data)


@jax.jit
def _ptuccsd_mode_population_term_batch(
    common,
    chol_indices,
    ham_data,
    meas_ctx,
    trial_data,
):
    return _ptuccsd_mode_chol_terms_for_walkers(
        common,
        ham_data.chol[chol_indices],
        meas_ctx.rot_chol_a[chol_indices],
        meas_ctx.chol_b[chol_indices],
        meas_ctx.rot_chol_b[chol_indices],
        meas_ctx,
        trial_data,
        n_chunks=1,
    )


def stream_ptuccsd_mode_population_statistics(
    walkers: jax.Array,
    candidate_weights: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
    *,
    n_walker_chunks: int = 10,
    chol_batch_size: int = 16,
) -> PtuccsdModePopulationStats:
    """Stream real projected UCC residual moments without storing all pairs."""

    if n_walker_chunks <= 0:
        raise ValueError("n_walker_chunks must be positive.")
    if chol_batch_size <= 0:
        raise ValueError("chol_batch_size must be positive.")
    start_time = time.perf_counter()
    n_walkers = int(walkers.shape[0])
    n_chol = int(ham_data.chol.shape[0])
    if n_walkers <= 0:
        raise ValueError("PT-UCCSD calibration requires at least one walker.")
    if n_chol <= 0:
        raise ValueError("PT-UCCSD calibration requires at least one Cholesky vector.")
    if candidate_weights.shape != (n_walkers,):
        raise ValueError(
            f"candidate_weights must have shape {(n_walkers,)}, got "
            f"{candidate_weights.shape}."
        )

    candidate_np = np.asarray(jax.device_get(candidate_weights), dtype=np.complex128)
    walker_batch_size = math.ceil(n_walkers / min(n_walker_chunks, n_walkers))
    valid = np.isfinite(candidate_np)
    theta_values = np.zeros(n_walkers, dtype=np.complex128)

    # Determine the exact denominator before projecting the complex residuals.
    # The common intermediates are recomputed below to keep the calibration
    # bounded in both walker and Cholesky dimensions.
    for walker_start in range(0, n_walkers, walker_batch_size):
        walker_stop = min(walker_start + walker_batch_size, n_walkers)
        valid_walkers = walker_stop - walker_start
        walker_indices = np.minimum(
            walker_start + np.arange(walker_batch_size, dtype=np.int32),
            n_walkers - 1,
        )
        common = _ptuccsd_mode_population_common_batch(
            walkers[jnp.asarray(walker_indices)],
            ham_data,
            meas_ctx,
            trial_data,
        )
        common_np = tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value)),
            common,
        )
        finite_common = np.ones(walker_batch_size, dtype=bool)
        for value in tree_util.tree_leaves(common_np):
            finite_common &= np.all(
                np.isfinite(value.reshape(walker_batch_size, -1)),
                axis=1,
            )
        valid[walker_start:walker_stop] &= finite_common[:valid_walkers]
        theta_values[walker_start:walker_stop] = np.asarray(
            common_np.theta[:valid_walkers],
            dtype=np.complex128,
        )

    estimator_weights = np.where(valid, candidate_np, 0.0)
    estimator_weight = np.sum(estimator_weights, dtype=np.complex128)
    abs_weight_sum = float(np.sum(np.abs(estimator_weights), dtype=np.float64))
    if not np.isfinite(estimator_weight) or abs(estimator_weight) == 0.0:
        raise ValueError(
            "PT-UCCSD calibration population has zero or nonfinite estimator weight."
        )
    if not np.isfinite(abs_weight_sum) or abs_weight_sum <= 0.0:
        raise ValueError(
            "PT-UCCSD calibration population has no finite absolute estimator weight."
        )

    normalized_weights = estimator_weights / estimator_weight
    theta_reference = np.sum(
        normalized_weights * theta_values,
        dtype=np.complex128,
    )
    walker_prob = np.abs(estimator_weights) / abs_weight_sum
    term_means = np.zeros(n_chol, dtype=np.float64)
    term_second_moments = np.zeros(n_chol, dtype=np.float64)
    block_numerator = np.zeros(3, dtype=np.complex128)

    for walker_start in range(0, n_walkers, walker_batch_size):
        walker_stop = min(walker_start + walker_batch_size, n_walkers)
        valid_walkers = walker_stop - walker_start
        walker_indices = np.minimum(
            walker_start + np.arange(walker_batch_size, dtype=np.int32),
            n_walkers - 1,
        )
        common = _ptuccsd_mode_population_common_batch(
            walkers[jnp.asarray(walker_indices)],
            ham_data,
            meas_ctx,
            trial_data,
        )
        common_np = tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value)),
            common,
        )
        batch_weights = np.zeros(walker_batch_size, dtype=np.complex128)
        batch_weights[:valid_walkers] = estimator_weights[walker_start:walker_stop]
        batch_normalized = batch_weights / estimator_weight
        batch_prob = np.zeros(walker_batch_size, dtype=np.float64)
        batch_prob[:valid_walkers] = walker_prob[walker_start:walker_stop]
        component_sum = np.zeros((walker_batch_size, 2), dtype=np.complex128)

        for chol_start in range(0, n_chol, chol_batch_size):
            chol_stop = min(chol_start + chol_batch_size, n_chol)
            valid_chol = chol_stop - chol_start
            chol_indices = np.minimum(
                chol_start + np.arange(chol_batch_size, dtype=np.int32),
                n_chol - 1,
            )
            terms = _ptuccsd_mode_population_term_batch(
                common,
                jnp.asarray(chol_indices),
                ham_data,
                meas_ctx,
                trial_data,
            )
            terms_np = np.asarray(jax.device_get(terms), dtype=np.complex128)[
                :, :valid_chol, :
            ]
            terms_np = np.where(batch_prob[:, None, None] > 0.0, terms_np, 0.0)
            component_sum += np.sum(terms_np, axis=1, dtype=np.complex128)
            effective = project_first_order_energy_terms(theta_reference, terms_np)
            projected = np.real(batch_normalized[:, None] * effective)
            term_means[chol_start:chol_stop] += np.sum(
                projected,
                axis=0,
                dtype=np.float64,
            )
            term_second_moments[chol_start:chol_stop] += np.sum(
                np.where(
                    batch_prob[:, None] > 0.0,
                    projected**2 / np.maximum(batch_prob[:, None], 1.0e-300),
                    0.0,
                ),
                axis=0,
                dtype=np.float64,
            )

        full_components = np.stack(
            (
                np.asarray(common_np.theta, dtype=np.complex128),
                np.asarray(common_np.electronic_0_base, dtype=np.complex128)
                + component_sum[:, 0],
                np.asarray(common_np.h_t_base, dtype=np.complex128)
                + component_sum[:, 1],
            ),
            axis=1,
        )
        full_components = np.where(
            batch_prob[:, None] > 0.0,
            full_components,
            0.0,
        )
        block_numerator += np.sum(
            batch_weights[:, None] * full_components,
            axis=0,
            dtype=np.complex128,
        )

    block_components = block_numerator / estimator_weight
    exact_energy = float(
        np.real(
            np.asarray(
                combine_first_order_energy(ham_data.h0, jnp.asarray(block_components))
            ).reshape(())
        )
    )
    phase_coherence = float(abs(estimator_weight) / abs_weight_sum)
    return PtuccsdModePopulationStats(
        term_means=term_means,
        term_second_moments=term_second_moments,
        rms_scores=np.sqrt(np.maximum(term_second_moments, 0.0)),
        exact_block_energy_ha=exact_energy,
        phase_coherence=phase_coherence,
        wall_seconds=time.perf_counter() - start_time,
        population_term_means=term_means[None, :],
        population_term_second_moments=term_second_moments[None, :],
        population_exact_energies_ha=np.asarray([exact_energy], dtype=np.float64),
    )


def average_ptuccsd_mode_population_statistics(
    population_stats: list[PtuccsdModePopulationStats],
) -> PtuccsdModePopulationStats:
    """Average independent UCC calibration snapshots."""

    if not population_stats:
        raise ValueError("population_stats must be nonempty.")
    population_means = np.stack([stats.term_means for stats in population_stats])
    population_seconds = np.stack(
        [stats.term_second_moments for stats in population_stats]
    )
    exact_energies = np.asarray(
        [stats.exact_block_energy_ha for stats in population_stats], dtype=np.float64
    )
    second_moments = np.mean(population_seconds, axis=0, dtype=np.float64)
    return PtuccsdModePopulationStats(
        term_means=np.mean(population_means, axis=0, dtype=np.float64),
        term_second_moments=second_moments,
        rms_scores=np.sqrt(np.maximum(second_moments, 0.0)),
        exact_block_energy_ha=float(exact_energies[-1]),
        phase_coherence=float(
            np.mean([stats.phase_coherence for stats in population_stats])
        ),
        wall_seconds=float(sum(stats.wall_seconds for stats in population_stats)),
        population_term_means=population_means,
        population_term_second_moments=population_seconds,
        population_exact_energies_ha=exact_energies,
    )


def select_ptuccsd_mode_pair_sampling(
    stats: PtuccsdModePopulationStats,
    cfg: PtuccsdModePairTuningCfg,
    *,
    n_walkers: int,
    reference_guide_scores: np.ndarray,
    calibration_std_ha: float,
    calibration_source: str,
    final_error_target_ha: float | None,
    n_blocks: int,
) -> PtuccsdModePairTuningResult:
    """Apply the shared PT/CISD head and sample-count selector to UCC moments."""

    selected = _select_ptccsd_mode_pair_sampling(
        _PtccsdModePopulationStats(
            term_means=stats.term_means,
            term_second_moments=stats.term_second_moments,
            rms_scores=stats.rms_scores,
            exact_block_energy_ha=stats.exact_block_energy_ha,
            phase_coherence=stats.phase_coherence,
            wall_seconds=stats.wall_seconds,
            population_term_means=stats.population_term_means,
            population_term_second_moments=stats.population_term_second_moments,
            population_exact_energies_ha=stats.population_exact_energies_ha,
        ),
        cfg,
        n_walkers=n_walkers,
        reference_guide_scores=reference_guide_scores,
        calibration_std_ha=calibration_std_ha,
        calibration_source=calibration_source,
        final_error_target_ha=final_error_target_ha,
        n_blocks=n_blocks,
    )
    sampling = PtuccsdModePairSamplingCfg(
        chol_head_size=selected.sampling.chol_head_size,
        pair_sample_size=selected.sampling.pair_sample_size,
        rank_head_by_guide=True,
        guide_chol_batch_size=cfg.tuning_chol_batch_size,
        head_chol_batch_size=cfg.production_head_chol_batch_size,
        tail_probability_uniform_mix=cfg.tail_probability_uniform_mix,
        track_half_sample_diagnostic=cfg.track_half_sample_diagnostic,
        walker_guide_policy=cfg.walker_guide_policy,
        walker_guide_weight_mix=cfg.walker_guide_weight_mix,
    )
    return PtuccsdModePairTuningResult(
        sampling=sampling,
        chol_head_fraction=selected.chol_head_fraction,
        estimated_tail_std_ha=selected.estimated_tail_std_ha,
        guarded_tail_std_ha=selected.guarded_tail_std_ha,
        target_tail_std_ha=selected.target_tail_std_ha,
        target_tail_std_source=selected.target_tail_std_source,
        estimated_pair_evaluations=selected.estimated_pair_evaluations,
    )


def retune_ptuccsd_mode_pair_sampling(
    state,
    equilibration_components: jax.Array,
    equilibration_weights: jax.Array,
    params,
    ham_data: HamChol,
    estimator_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
    *,
    guide_data,
    guide_meas_ops: MeasOps,
    guide_meas_ctx,
    advance_blocks: BlockComponentsAdvanceFn,
    tuning_cfg: PtuccsdModePairTuningCfg,
    target_error: float | None = None,
) -> BlockComponentRetuneResult:
    """Tune real projected UCC residual variance and install the production sampler."""

    if estimator_ctx.model_sampling is not None:
        raise ValueError("Local Cholesky sampling requires frozen settings (no automatic retuning).")
    del guide_meas_ctx
    population_stats = []
    for population_index in range(tuning_cfg.tuning_population_count):
        if population_index > 0:
            spacing = tuning_cfg.tuning_population_spacing_blocks
            print(
                f"[PT-UCCSD sampling] advancing {spacing} calibration blocks before "
                f"population {population_index + 1}/{tuning_cfg.tuning_population_count}."
            )
            state, _, _ = advance_blocks(state, n_blocks=spacing)
            jax.block_until_ready(state)

        guide_overlaps = wk.vmap_chunked(
            guide_meas_ops.overlap,
            n_chunks=min(tuning_cfg.tuning_n_chunks, int(state.walkers.shape[0])),
            in_axes=(0, None),
        )(state.walkers, guide_data)
        reference_overlaps = wk.vmap_chunked(
            reference_overlap_r,
            n_chunks=min(tuning_cfg.tuning_n_chunks, int(state.walkers.shape[0])),
            in_axes=(0, None),
        )(state.walkers, trial_data)
        overlap_ratio = reference_overlaps / guide_overlaps
        candidate_weights = jnp.where(
            jnp.isfinite(overlap_ratio),
            state.weights * overlap_ratio,
            0.0,
        )
        print(
            "[PT-UCCSD sampling] streaming real projected population statistics: "
            f"population={population_index + 1}/{tuning_cfg.tuning_population_count}, "
            f"chol_batch_size={tuning_cfg.tuning_chol_batch_size}."
        )
        stats_i = stream_ptuccsd_mode_population_statistics(
            state.walkers,
            candidate_weights,
            ham_data,
            estimator_ctx,
            trial_data,
            n_walker_chunks=tuning_cfg.tuning_n_chunks,
            chol_batch_size=tuning_cfg.tuning_chol_batch_size,
        )
        population_stats.append(stats_i)
        print(
            f"[PT-UCCSD sampling] population {population_index + 1}: "
            f"exact projected energy={stats_i.exact_block_energy_ha:.10f} Ha, "
            f"phase_coherence={stats_i.phase_coherence:.3e}, "
            f"statistics_seconds={stats_i.wall_seconds:.1f}."
        )

    stats = average_ptuccsd_mode_population_statistics(population_stats)
    snapshot_energies = np.asarray(stats.population_exact_energies_ha, dtype=np.float64)
    if snapshot_energies.size > 1:
        calibration_std = float(np.std(snapshot_energies, ddof=1))
        calibration_source = "exact calibration-population standard deviation"
    else:
        equil_components_np = np.asarray(jax.device_get(equilibration_components))
        equil_weights_np = np.asarray(jax.device_get(equilibration_weights))
        finite = np.isfinite(equil_weights_np) & np.all(
            np.isfinite(equil_components_np), axis=1
        )
        proxy = np.real(
            np.asarray(
                combine_first_order_energy(
                    ham_data.h0,
                    jnp.asarray(equil_components_np[finite]),
                )
            )
        )
        late_proxy = proxy[proxy.size // 2 :]
        calibration_std = (
            float(np.std(late_proxy, ddof=1)) if late_proxy.size > 1 else float("nan")
        )
        calibration_source = "late equilibration projected-energy standard deviation"

    has_absolute_target = tuning_cfg.target_tail_std_ha is not None or (
        tuning_cfg.final_error_target_ha is not None
        or (target_error is not None and target_error > 0.0)
    )
    if not has_absolute_target and (
        not np.isfinite(calibration_std) or calibration_std <= 0.0
    ):
        raise ValueError(
            "PT-UCCSD pair tuning requires multiple calibration populations, a usable "
            "equilibration variance, or an absolute/final-error target."
        )
    selected = select_ptuccsd_mode_pair_sampling(
        stats,
        tuning_cfg,
        n_walkers=int(state.walkers.shape[0]),
        reference_guide_scores=np.asarray(
            jax.device_get(estimator_ctx.reference_chol_scores), dtype=np.float64
        ),
        calibration_std_ha=calibration_std,
        calibration_source=calibration_source,
        final_error_target_ha=target_error,
        n_blocks=int(params.n_blocks),
    )
    production_scores = (
        stats.rms_scores
        if tuning_cfg.guide_policy == "population_rms"
        else np.asarray(jax.device_get(estimator_ctx.reference_chol_scores))
    )
    production_ctx = configure_ptuccsd_mode_pair_sampling(
        estimator_ctx,
        selected.sampling,
        jnp.asarray(production_scores, dtype=jnp.float64),
    )
    print(
        "[PT-UCCSD sampling] selected real-projected estimator: "
        f"chol_head_size={selected.sampling.chol_head_size}/{stats.rms_scores.size} "
        f"({selected.chol_head_fraction:.3%}), "
        f"pair_sample_size={selected.sampling.pair_sample_size}, "
        f"tail_std={selected.estimated_tail_std_ha:.3e} Ha, "
        f"target={selected.target_tail_std_ha:.3e} Ha, "
        f"walker_guide={selected.sampling.walker_guide_policy}, "
        f"phase_coherence={stats.phase_coherence:.3e}, "
        f"work_proxy={selected.estimated_pair_evaluations} pairs."
    )
    return BlockComponentRetuneResult(
        state=state,
        estimator_ctx=production_ctx,
        initial_n_chunks=tuning_cfg.production_initial_n_chunks,
        settling_blocks=tuning_cfg.settling_blocks,
    )


def _energy_components_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return components from exact common data and Cholesky residuals."""

    common = _ptuccsd_mode_energy_common_uw_rh(
        walker,
        ham_data,
        meas_ctx,
        trial_data,
    )
    chol_components = _ptuccsd_mode_chol_terms(
        common,
        ham_data.chol,
        meas_ctx.rot_chol_a,
        meas_ctx.chol_b,
        meas_ctx.rot_chol_b,
        meas_ctx,
        trial_data,
    )
    chol_sum = jnp.sum(chol_components, axis=0)
    return (
        common.theta,
        common.electronic_0_base + chol_sum[0],
        common.h_t_base + chol_sum[1],
    )



def initial_components_ptuccsd_mode_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Exact block-zero components with bounded, vectorized Cholesky batches.

    The initial walker does not use production pair sampling. Sum all residual
    terms in groups of cfg.chol_batch_size (64 by default), retaining only two
    scalars between groups, rather than materializing full-Cholesky products.
    """
    noa, nob = trial_data.nocc
    common = _ptuccsd_mode_energy_common_uw_rh(
        (walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data
    )
    # Vectorize each group even if production selected the scalar low-memory
    # path; initialization has only one representative walker.
    batch_ctx = replace(meas_ctx, base=replace(
        meas_ctx.base, cfg=replace(meas_ctx.cfg, memory_mode="high")
    ))

    def local_sum(common, chol_a, rot_a, chol_b, rot_b, trial):
        tensors = (chol_a, rot_a, chol_b, rot_b)
        batch_size = meas_ctx.cfg.chol_batch_size
        nchol = chol_a.shape[0]

        def terms(tensors):
            values = _ptuccsd_mode_chol_terms(common, *tensors, batch_ctx, trial)
            return jnp.sum(values, axis=0, dtype=jnp.complex128)

        total = jnp.zeros((2,), dtype=jnp.complex128)
        if meas_ctx.cfg.chol_mesh is not None:
            total = lax.pcast(total, ("model",), to="varying")
        nfull, remainder = divmod(nchol, batch_size)

        def accumulate(index, total):
            batches = tuple(lax.dynamic_slice_in_dim(
                tensor, index * batch_size, batch_size, axis=0
            ) for tensor in tensors)
            return total + terms(batches)

        if nfull:
            total = lax.fori_loop(0, nfull, accumulate, total)
        if remainder:
            total = total + terms(tuple(t[nfull * batch_size:] for t in tensors))
        if meas_ctx.cfg.chol_mesh is not None:
            total = lax.psum(total, "model")
        return total

    args = (common, ham_data.chol, meas_ctx.rot_chol_a,
            meas_ctx.chol_b, meas_ctx.rot_chol_b, trial_data)
    if meas_ctx.cfg.chol_mesh is not None:
        chol_sum = jax.shard_map(
            local_sum, mesh=meas_ctx.cfg.chol_mesh,
            in_specs=(P(), P("model"), P("model"), P("model"), P("model"), P()),
            out_specs=P(),
        )(*args)
    else:
        chol_sum = local_sum(*args)
    return jnp.stack((common.theta, common.electronic_0_base + chol_sum[0],
                      common.h_t_base + chol_sum[1]))


def components_ptuccsd_mode_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    """Return guide-independent ``[theta, electronic_0, h_t]`` components."""

    return jnp.stack(_energy_components_uw_rh(walker, ham_data, meas_ctx, trial_data))


def components_ptuccsd_mode_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return components_ptuccsd_mode_uw_rh(
        (walker[:, :noa], walker[:, :nob]),
        ham_data,
        meas_ctx,
        trial_data,
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    components = components_ptuccsd_mode_uw_rh(walker, ham_data, meas_ctx, trial_data)
    return combine_first_order_energy(ham_data.h0, components)


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdModeMeasCtx,
    trial_data: PtuccsdThoulessModeTrial,
) -> jax.Array:
    components = components_ptuccsd_mode_rw_rh(walker, ham_data, meas_ctx, trial_data)
    return combine_first_order_energy(ham_data.h0, components)


def make_ptuccsd_mode_force_bias_ops(
    sys: System,
    *,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    """Build propagation-only mode operations containing the force bias.

    This deliberately omits ``k_energy`` for callers that need only the
    propagation kernel.  Use :func:`make_ptuccsd_mode_meas_ops` for a complete
    guide measurement bundle.
    """

    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    walker_kind = sys.walker_kind.lower()
    if walker_kind == "restricted":
        if sys.nup < sys.ndn:
            raise ValueError("Restricted PT-UCCSD mode force bias requires nup >= ndn.")
        overlap_fn = overlap_r
        force_bias_fn = force_bias_kernel_rw_rh
    elif walker_kind == "unrestricted":
        overlap_fn = overlap_u
        force_bias_fn = force_bias_kernel_uw_rh
    else:
        raise ValueError(
            "PT-UCCSD mode force bias supports restricted/unrestricted walkers, "
            f"got: {sys.walker_kind}"
        )

    meas_ops = MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_ptuccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            cfg,
        ),
        kernels={k_force_bias: force_bias_fn},
    )
    object.__setattr__(meas_ops, _PTUCCSD_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops


def make_ptuccsd_mode_meas_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    """Build complete mode-native PT-UCCSD guide measurements."""

    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    walker_kind = sys.walker_kind.lower()
    if walker_kind == "restricted":
        if sys.nup < sys.ndn:
            raise ValueError("Restricted PT-UCCSD mode measurements require nup >= ndn.")
        overlap_fn = overlap_r
        force_bias_fn = force_bias_kernel_rw_rh
        energy_fn = energy_kernel_rw_rh
        components_fn = components_ptuccsd_mode_rw_rh
    elif walker_kind == "unrestricted":
        overlap_fn = overlap_u
        force_bias_fn = force_bias_kernel_uw_rh
        energy_fn = energy_kernel_uw_rh
        components_fn = components_ptuccsd_mode_uw_rh
    else:
        raise ValueError(
            "PT-UCCSD mode measurements support restricted/unrestricted walkers, "
            f"got: {sys.walker_kind}"
        )

    meas_ops = MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_ptuccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            cfg,
            n_mode_chunks=n_mode_chunks,
        ),
        kernels={k_force_bias: force_bias_fn, k_energy: energy_fn},
        observables={o_pt_components: components_fn},
    )
    object.__setattr__(meas_ops, _PTUCCSD_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops


def make_ptuccsd_mode_estimator_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
    component_sampling: PtuccsdModePairSamplingCfg | None = None,
    component_tuning: PtuccsdModePairTuningCfg | None = None,
) -> EstimatorOps:
    """Build a mode-native PT2-UCCSD estimator for a separate guide.

    A fixed ``component_sampling`` policy keeps the common PT components and
    an exact Cholesky head deterministic while sampling the connected UCC
    residual tail over walker--Cholesky pairs. Supplying ``component_tuning``
    gathers bounded post-equilibration population moments and installs an
    automatically selected production policy. Proposals and tuning use the
    real projected residual, but the block numerator remains fully complex.
    """

    if sys.walker_kind.lower() != "restricted" or sys.nup < sys.ndn:
        raise ValueError(
            "PT-UCCSD mode estimators require a restricted walker with nup >= ndn."
        )
    if component_tuning is not None and component_sampling is None:
        raise ValueError("component_tuning requires an equilibration component_sampling config.")
    cfg = PtuccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=reference_overlap_r,
        components=components_ptuccsd_mode_rw_rh,
        initial_components=initial_components_ptuccsd_mode_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptuccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            cfg,
            n_mode_chunks=n_mode_chunks,
            component_sampling=component_sampling,
        ),
        build_estimator_ctx_from_guide=lambda ham_data, trial_data, guide_data, guide_ctx: build_ptuccsd_mode_meas_ctx(
            ham_data, trial_data, cfg, n_mode_chunks=n_mode_chunks,
            component_sampling=component_sampling, guide_data=guide_data, guide_ctx=guide_ctx,
        ),
        block_components=(
            pair_sampled_ptuccsd_block_components
            if component_sampling is not None
            else None
        ),
        retune_block_components=(
            partial(retune_ptuccsd_mode_pair_sampling, tuning_cfg=component_tuning)
            if component_tuning is not None
            else None
        ),
        use_for_population_control=component_sampling is not None,
    )


__all__ = [
    "PtuccsdModeMeasCfg",
    "PtuccsdModeMeasCtx",
    "PtuccsdModePairSamplingCfg",
    "PtuccsdModePairTuningCfg",
    "PtuccsdModePairTuningResult",
    "PtuccsdModePopulationStats",
    "average_ptuccsd_mode_population_statistics",
    "build_ptuccsd_mode_meas_ctx",
    "components_ptuccsd_mode_rw_rh",
    "components_ptuccsd_mode_uw_rh",
    "energy_kernel_rw_rh",
    "energy_kernel_uw_rh",
    "force_bias_kernel_rw_rh",
    "force_bias_kernel_uw_rh",
    "get_ptuccsd_mode_meas_cfg",
    "make_ptuccsd_mode_estimator_ops",
    "make_ptuccsd_mode_force_bias_ops",
    "make_ptuccsd_mode_meas_ops",
    "pair_sampled_ptuccsd_block_components",
    "retune_ptuccsd_mode_pair_sampling",
    "select_ptuccsd_mode_pair_sampling",
    "stream_ptuccsd_mode_population_statistics",
]
