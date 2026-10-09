from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
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
from ..trial.cisd_modes import mode_apply, mode_quadratic
from ..trial.ptccsd_modes import (
    PtccsdModeTrial,
    PtccsdThoulessModeTrial,
    det_overlap_thouless_r,
    greens_pt_r,
    half_green_thouless_r,
    hf_overlap_r,
    overlap_pt_r,
    overlap_ptccsd_thouless_r,
)
from .cisd_modes import (
    CisdModePairTuningCfg,
    CisdModePopulationStats,
    mode_quadratic_matrices,
    select_cisd_mode_pair_sampling,
)
from .ptccsd import o_pt_components
from .pt2ccsd import combine_first_order_energy, project_first_order_energy_terms
from .pair_sampling import ModelPairSamplingData, local_cholesky_component_head, local_cholesky_tail
from .chol_setup import transform_cholesky

_PTCCSD_SETUP_CHOL_BATCH_SIZE = 256


@dataclass(frozen=True)
class PtccsdModeMeasCfg:
    """Precision and memory policy for mode-native PT-CCSD contractions."""

    memory_mode: Literal["low", "high"] = "high"
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


@dataclass(frozen=True)
class PtccsdModePairSamplingCfg:
    """Fixed walker--Cholesky sampling policy for the RCC PT component numerator."""

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
class PtccsdModePairTuningCfg:
    """Post-equilibration tuning using real projected PT residual moments."""

    guide_policy: Literal["population_rms", "reference"] = "population_rms"
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
    walker_guide_policy: Literal["abs_weight", "head_rms"] = "abs_weight"
    walker_guide_weight_mix: float = 0.1
    settling_blocks: int = 5

    def __post_init__(self) -> None:
        if self.guide_policy not in ("population_rms", "reference"):
            raise ValueError("guide_policy must be 'population_rms' or 'reference'.")
        if self.walker_guide_policy not in ("abs_weight", "head_rms"):
            raise ValueError("walker_guide_policy must be 'abs_weight' or 'head_rms'.")
        if not 0.0 < self.walker_guide_weight_mix <= 1.0:
            raise ValueError("walker_guide_weight_mix must lie in (0, 1].")
        # Reuse the mature scalar-tuner validation for the common controls.
        _ = self.as_cisd_cfg()

    def as_cisd_cfg(self) -> CisdModePairTuningCfg:
        return CisdModePairTuningCfg(
            guide_policy="population_rms" if self.guide_policy == "population_rms" else "hf",
            final_error_target_ha=self.final_error_target_ha,
            final_error_sampling_fraction=self.final_error_sampling_fraction,
            target_tail_std_fraction=self.target_tail_std_fraction,
            target_tail_std_ha=self.target_tail_std_ha,
            safety_factor=self.safety_factor,
            cross_validation_quantile=self.cross_validation_quantile,
            candidate_sample_sizes=self.candidate_sample_sizes,
            minimum_head_fraction=self.minimum_head_fraction,
            maximum_head_fraction=self.maximum_head_fraction,
            head_size_stride=self.head_size_stride,
            tuning_n_chunks=self.tuning_n_chunks,
            tuning_chol_batch_size=self.tuning_chol_batch_size,
            tuning_population_count=self.tuning_population_count,
            tuning_population_spacing_blocks=self.tuning_population_spacing_blocks,
            production_initial_n_chunks=self.production_initial_n_chunks,
            production_head_chol_batch_size=self.production_head_chol_batch_size,
            tail_probability_uniform_mix=self.tail_probability_uniform_mix,
            track_half_sample_diagnostic=self.track_half_sample_diagnostic,
            walker_guide_policy="weight",
            walker_guide_weight_mix=self.walker_guide_weight_mix,
            settling_blocks=self.settling_blocks,
        )


@dataclass(frozen=True)
class PtccsdModePopulationStats:
    """Real projected residual moments for one or more walker populations."""

    term_means: np.ndarray
    term_second_moments: np.ndarray
    rms_scores: np.ndarray
    exact_block_energy_ha: float
    phase_coherence: float
    wall_seconds: float
    population_term_means: np.ndarray | None = None
    population_term_second_moments: np.ndarray | None = None
    population_exact_energies_ha: np.ndarray | None = None


@dataclass(frozen=True)
class PtccsdModePairTuningResult:
    sampling: PtccsdModePairSamplingCfg
    chol_head_fraction: float
    estimated_tail_std_ha: float
    guarded_tail_std_ha: float
    target_tail_std_ha: float
    target_tail_std_source: str
    estimated_pair_evaluations: int


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdModeMeasCtx:
    rot_chol: jax.Array
    l_t1: jax.Array
    n_mode_chunks: int
    cfg: PtccsdModeMeasCfg

    def tree_flatten(self):
        return (self.rot_chol, self.l_t1), (self.n_mode_chunks, self.cfg)

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_mode_chunks, cfg = aux
        rot_chol, l_t1 = children
        return cls(
            rot_chol=rot_chol,
            l_t1=l_t1,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
        )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdThoulessModeMeasCtx:
    rot_chol: jax.Array
    reference_chol_scores: jax.Array
    chol_head_indices: jax.Array
    chol_tail_indices: jax.Array
    chol_tail_prob: jax.Array
    n_mode_chunks: int
    cfg: PtccsdModeMeasCfg
    component_sampling: PtccsdModePairSamplingCfg | None
    model_sampling: ModelPairSamplingData | None = None

    @property
    def memory_mode(self) -> Literal["low", "high"]:
        return self.cfg.memory_mode

    def tree_flatten(self):
        children = (
            self.rot_chol,
            self.reference_chol_scores,
            self.chol_head_indices,
            self.chol_tail_indices,
            self.chol_tail_prob,
            self.model_sampling,
        )
        return children, (self.n_mode_chunks, self.cfg, self.component_sampling)

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_mode_chunks, cfg, component_sampling = aux
        (
            rot_chol,
            reference_chol_scores,
            chol_head_indices,
            chol_tail_indices,
            chol_tail_prob,
            model_sampling,
        ) = children
        return cls(
            rot_chol=rot_chol,
            reference_chol_scores=reference_chol_scores,
            chol_head_indices=chol_head_indices,
            chol_tail_indices=chol_tail_indices,
            chol_tail_prob=chol_tail_prob,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
            component_sampling=component_sampling,
            model_sampling=model_sampling,
        )


class PtccsdThoulessModeEnergyCommon(NamedTuple):
    """Per-walker PT sufficient-statistic data shared by Cholesky terms.

    The two base components contain only one-body contributions. The
    determinant-reference two-body energy and connected T2 correction are both
    represented by Cholesky-resolved component terms.
    """

    half_green: jax.Array
    greenp: jax.Array
    t2_green: jax.Array
    theta: jax.Array
    electronic_0_base: jax.Array
    h_t_base: jax.Array


def build_ptccsd_mode_meas_ctx(
    ham_data: HamChol,
    trial_data: PtccsdModeTrial,
    *,
    n_mode_chunks: int = 1,
    cfg: PtccsdModeMeasCfg = PtccsdModeMeasCfg(),
) -> PtccsdModeMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("PT-CCSD mode kernels require a restricted Hamiltonian.")
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    rot_chol = ham_data.chol[:, : trial_data.nocc, :]
    l_t1 = jnp.einsum(
        "git,pt->gip",
        ham_data.chol[:, :, trial_data.nocc :],
        trial_data.t1,
        optimize="optimal",
    )
    return PtccsdModeMeasCtx(
        rot_chol=rot_chol,
        l_t1=l_t1,
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
        cfg=cfg,
    )


def build_ptccsd_thouless_mode_meas_ctx(
    ham_data: HamChol,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_mode_chunks: int = 1,
    memory_mode: Literal["low", "high"] = "high",
    cfg: PtccsdModeMeasCfg | None = None,
    component_sampling: PtccsdModePairSamplingCfg | None = None,
) -> PtccsdThoulessModeMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("PT-CCSD Thouless mode kernels require a restricted Hamiltonian.")
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if memory_mode not in {"low", "high"}:
        raise ValueError("memory_mode must be 'low' or 'high'.")
    if cfg is None:
        cfg = PtccsdModeMeasCfg(memory_mode=memory_mode)
    elif cfg.memory_mode != memory_mode:
        raise ValueError("cfg.memory_mode and memory_mode must agree.")
    n_chol = int(ham_data.chol.shape[0])
    if component_sampling is not None and component_sampling.chol_head_size > n_chol:
        raise ValueError(
            f"chol_head_size must not exceed the number of Cholesky vectors ({n_chol})."
        )
    meas_ctx = PtccsdThoulessModeMeasCtx(
        rot_chol=transform_cholesky(
            ham_data.chol,
            left=trial_data.mo_t.conj().T,
            batch_size=_PTCCSD_SETUP_CHOL_BATCH_SIZE,
        ),
        reference_chol_scores=jnp.empty((0,), dtype=jnp.float64),
        chol_head_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_indices=jnp.empty((0,), dtype=jnp.int32),
        chol_tail_prob=jnp.empty((0,), dtype=jnp.float64),
        n_mode_chunks=min(int(n_mode_chunks), trial_data.mode_rank),
        cfg=cfg,
        component_sampling=None,
    )
    if component_sampling is None:
        return meas_ctx
    reference_scores = _build_ptccsd_thouless_reference_chol_scores(
        ham_data,
        meas_ctx,
        trial_data,
        chol_batch_size=component_sampling.guide_chol_batch_size,
    )
    return configure_ptccsd_mode_pair_sampling(
        meas_ctx,
        component_sampling,
        reference_scores,
    )


def _chol_contract(
    chol: jax.Array,
    matrix: jax.Array,
    cfg: PtccsdModeMeasCfg,
) -> jax.Array:
    """Contract real Cholesky vectors with a complex matrix in mixed precision."""

    chol_r = chol.astype(cfg.mixed_real_dtype)
    matrix_r = jnp.real(matrix).astype(cfg.mixed_real_dtype)
    matrix_i = jnp.imag(matrix).astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("gij,ij->g", chol_r, matrix_r, optimize="optimal")
    imag_part = jnp.einsum("gij,ij->g", chol_r, matrix_i, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _energy_gl_batched(
    green: jax.Array,
    chol: jax.Array,
    cfg: PtccsdModeMeasCfg,
) -> jax.Array:
    green_r = jnp.real(green).astype(cfg.mixed_real_dtype)
    green_i = jnp.imag(green).astype(cfg.mixed_real_dtype)
    chol_r = chol.astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("pr,gqr->gpq", green_r, chol_r, optimize="optimal")
    imag_part = jnp.einsum("pr,gqr->gpq", green_i, chol_r, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _energy_gl_scalar(
    green: jax.Array,
    chol_i: jax.Array,
    cfg: PtccsdModeMeasCfg,
) -> jax.Array:
    green_r = jnp.real(green).astype(cfg.mixed_real_dtype)
    green_i = jnp.imag(green).astype(cfg.mixed_real_dtype)
    chol_i_r = chol_i.astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("ir,qr->iq", green_r, chol_i_r, optimize="optimal")
    imag_part = jnp.einsum("ir,qr->iq", green_i, chol_i_r, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _pt_green_blocks(
    walker: jax.Array,
    trial_data: PtccsdModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    green = greens_pt_r(walker, trial_data)
    green_occ = green[:, trial_data.nocc :]
    greenp = jnp.vstack(
        [
            green_occ,
            -jnp.eye(trial_data.nvir, dtype=green.dtype),
        ]
    )
    return green, green_occ, greenp


def _thouless_green_blocks(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    _, green, _, green_occ, greenp = _thouless_half_green_blocks(walker, trial_data)
    return green, green_occ, greenp


def _thouless_half_green_force_bias_blocks(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the half-Green blocks needed by the exponential-guide force bias."""

    half_green = half_green_thouless_r(walker, trial_data)
    nocc = trial_data.nocc
    green_rows = trial_data.mo_t.conj()[:nocc, :] @ half_green
    green_occ = green_rows[:, nocc:]
    greenp = trial_data.mo_t.conj() @ half_green[:, nocc:]
    greenp = greenp.at[nocc:, :].add(
        -jnp.eye(trial_data.nvir, dtype=greenp.dtype)
    )
    return half_green, green_rows, green_occ, greenp


def _thouless_half_green_blocks(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return half- and full-Green blocks for a Thouless reference.

    If ``C`` is ``trial_data.mo_t`` and ``R`` is the half Green function, the
    full transition Green function factors as ``G = C.conj() @ R``.  Keeping
    ``R`` makes it possible for the energy kernel to avoid intermediates with
    shape ``(n_chol, norb, norb)``.
    """

    half_green, green_rows, green_occ, greenp = (
        _thouless_half_green_force_bias_blocks(walker, trial_data)
    )
    green = trial_data.mo_t.conj() @ half_green
    return half_green, green, green_rows, green_occ, greenp


def _mode_t2_green(
    trial_data,
    green: jax.Array,
    green_occ: jax.Array,
    greenp: jax.Array,
    *,
    green_rows: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    del green
    _, t2_green, theta2 = _mode_t2_green_factors(
        trial_data,
        green_occ,
        greenp,
        green_rows=green_rows,
    )
    return t2_green, theta2


def _mode_t2_green_factors(
    trial_data,
    green_occ: jax.Array,
    greenp: jax.Array,
    *,
    green_rows: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the left factor, full correction, and scalar T2 contraction."""

    projections, kg = mode_apply(trial_data, green_occ)
    theta2 = mode_quadratic(trial_data, green_occ, projections)
    t2_left = greenp @ kg.T
    t2_green = t2_left @ green_rows
    return t2_left, t2_green, theta2


def force_bias_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    green, green_occ, greenp = _pt_green_blocks(walker, trial_data)
    f0 = 2.0 * jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")

    t1gp = jnp.einsum("pt,it->pi", trial_data.t1, greenp, optimize="optimal")
    gt1gp = jnp.einsum("pj,pi->ij", green, t1gp, optimize="optimal")
    t2_green, _ = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green,
    )
    connected = _chol_contract(
        ham_data.chol,
        -2.0 * t2_green - 2.0 * gt1gp,
        meas_ctx.cfg,
    )
    return f0 + connected


def components_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    """Return unweighted ``[theta, electronic_0, h_t]`` PT sufficient statistics."""

    green, green_occ, greenp = _pt_green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol
    nocc = trial_data.nocc

    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2.0 * hg
    lg = jnp.einsum("gpj,pj->g", meas_ctx.rot_chol, green, optimize="optimal")
    lg1 = jnp.einsum("gpj,qj->gpq", meas_ctx.rot_chol, green, optimize="optimal")
    e2_0 = 2.0 * (lg @ lg) - jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2))
    electronic_0 = e1_0 + e2_0

    t1g = jnp.einsum("pt,pt->", trial_data.t1, green_occ, optimize="optimal")
    theta1 = 2.0 * t1g
    t1_green = (greenp @ trial_data.t1.T) @ green
    e1_1 = 4.0 * t1g * hg - 2.0 * jnp.einsum(
        "ij,ij->", h1, t1_green, optimize="optimal"
    )

    t2_green, theta2 = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green,
    )
    e1_2 = 2.0 * hg * theta2 - 2.0 * jnp.einsum(
        "ij,ij->", h1, t2_green, optimize="optimal"
    )

    lt1g = _chol_contract(chol, t1_green, meas_ctx.cfg)
    t1g1 = trial_data.t1 @ green[:, nocc:].T
    l_t1g = jnp.einsum("gia,qi->gaq", meas_ctx.l_t1, green, optimize="optimal")
    e2_1_3_1 = jnp.einsum("gpq,gqa,ap->", lg1, lg1, t1g1, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gaq,gqa->", l_t1g, lg1, optimize="optimal")
    e2_1 = e2_0 * theta1 + 2.0 * (-2.0 * (lt1g @ lg) + e2_1_3_1 + e2_1_3_2)

    lt2g = _chol_contract(chol, t2_green, meas_ctx.cfg)
    lt2_green = jnp.einsum(
        "gpi,ji->gpj",
        meas_ctx.rot_chol.astype(meas_ctx.cfg.mixed_complex_dtype),
        t2_green.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    gl = _energy_gl_batched(green, chol, meas_ctx.cfg)
    e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lt2_green, optimize="optimal")
    glgp = jnp.einsum(
        "gpi,it->gpt",
        gl,
        greenp.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_3 = jnp.sum(
        mode_quadratic_matrices(
            trial_data,
            glgp,
            n_mode_chunks=meas_ctx.n_mode_chunks,
        )
    )
    e2_2 = e2_0 * theta2 + 4.0 * (-(lt2g @ lg) + e2_2_2_2) + e2_2_3

    theta = theta1 + theta2
    h_t = e1_1 + e1_2 + e2_1 + e2_2
    return jnp.stack([theta, electronic_0, h_t])


def energy_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    theta, electronic_0, h_t = components_pt_rw_rh(walker, ham_data, meas_ctx, trial_data)
    return ham_data.h0 + electronic_0 + h_t - theta * electronic_0


def inverse_guide_components_pt_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdModeMeasCtx,
    trial_data: PtccsdModeTrial,
) -> jax.Array:
    components = components_pt_rw_rh(walker, ham_data, meas_ctx, trial_data)
    reweight = jnp.exp(-components[0])
    return jnp.concatenate([reweight[None], reweight * components])


def _force_bias_pt_thouless_rw_rh_full(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Reference full-Green implementation retained as a test oracle."""

    green, green_occ, greenp = _thouless_green_blocks(walker, trial_data)
    f0 = 2.0 * jnp.einsum("gpq,pq->g", ham_data.chol, green, optimize="optimal")
    t2_green, _ = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green[: trial_data.nocc, :],
    )
    return f0 + _chol_contract(ham_data.chol, -2.0 * t2_green, meas_ctx.cfg)


def force_bias_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Restricted exponential-guide force bias using a half Green function."""

    half_green, green_rows, green_occ, greenp = (
        _thouless_half_green_force_bias_blocks(walker, trial_data)
    )
    f0 = 2.0 * jnp.einsum(
        "giq,iq->g",
        meas_ctx.rot_chol,
        half_green,
        optimize="optimal",
    )
    _, t2_green, _ = _mode_t2_green_factors(
        trial_data,
        green_occ,
        greenp,
        green_rows=green_rows,
    )
    return f0 + _chol_contract(ham_data.chol, -2.0 * t2_green, meas_ctx.cfg)


def _components_pt_thouless_rw_rh_full(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Reference full-Green components retained as a test oracle."""

    green, green_occ, greenp = _thouless_green_blocks(walker, trial_data)
    h1 = ham_data.h1
    chol = ham_data.chol
    nocc = trial_data.nocc

    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2.0 * hg
    t2_green, theta2 = _mode_t2_green(
        trial_data,
        green,
        green_occ,
        greenp,
        green_rows=green[:nocc, :],
    )
    e1_2 = 2.0 * hg * theta2 - 2.0 * jnp.einsum(
        "pq,pq->", h1, t2_green, optimize="optimal"
    )

    lg = jnp.einsum("gpq,pq->g", chol, green, optimize="optimal")
    gl = _energy_gl_batched(green, chol, meas_ctx.cfg)
    e2_0 = 2.0 * (lg @ lg) - jnp.sum(gl * jnp.swapaxes(gl, -1, -2))

    lt2g = _chol_contract(chol, t2_green, meas_ctx.cfg)
    lt2_green = jnp.einsum(
        "gpr,qr->gpq",
        chol.astype(meas_ctx.cfg.mixed_real_dtype),
        t2_green.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_2_2 = 0.5 * jnp.einsum("gpq,gpq->", gl, lt2_green, optimize="optimal")
    glgp = jnp.einsum(
        "gpi,it->gpt",
        gl[:, :nocc, :],
        greenp.astype(meas_ctx.cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_3 = jnp.sum(
        mode_quadratic_matrices(
            trial_data,
            glgp,
            n_mode_chunks=meas_ctx.n_mode_chunks,
        )
    )
    e2_2 = e2_0 * theta2 + 4.0 * (-(lt2g @ lg) + e2_2_2_2) + e2_2_3

    electronic_0 = e1_0 + e2_0
    h_t = e1_2 + e2_2
    return jnp.stack([theta2, electronic_0, h_t])


def _ptccsd_thouless_mode_energy_common(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> PtccsdThoulessModeEnergyCommon:
    """Build per-walker data outside the Cholesky component sum."""

    half_green, green, green_rows, green_occ, greenp = _thouless_half_green_blocks(
        walker,
        trial_data,
    )
    h1 = ham_data.h1
    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2.0 * hg
    _, t2_green, theta2 = _mode_t2_green_factors(
        trial_data,
        green_occ,
        greenp,
        green_rows=green_rows,
    )
    e1_2 = 2.0 * hg * theta2 - 2.0 * jnp.einsum(
        "pq,pq->", h1, t2_green, optimize="optimal"
    )

    return PtccsdThoulessModeEnergyCommon(
        half_green=half_green,
        greenp=greenp,
        t2_green=t2_green,
        theta=theta2,
        electronic_0_base=e1_0,
        h_t_base=e1_2,
    )


def _ptccsd_thouless_mode_chol_terms(
    common: PtccsdThoulessModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Return ``[e2_0, theta * e2_0 + connected]`` by Cholesky vector."""

    reference_occ = trial_data.mo_t.conj()[: trial_data.nocc, :].astype(
        meas_ctx.cfg.mixed_complex_dtype
    )
    greenp_mixed = common.greenp.astype(meas_ctx.cfg.mixed_complex_dtype)
    t2_green_mixed = common.t2_green.astype(meas_ctx.cfg.mixed_complex_dtype)

    def scalar_term(chol_i: jax.Array, rot_chol_i: jax.Array) -> jax.Array:
        lg_i = jnp.einsum(
            "iq,iq->",
            rot_chol_i,
            common.half_green,
            optimize="optimal",
        )
        lg1_i = jnp.einsum(
            "ip,jp->ij",
            rot_chol_i,
            common.half_green,
            optimize="optimal",
        )
        e20_i = 2.0 * lg_i * lg_i
        e20_i -= jnp.sum(lg1_i * jnp.swapaxes(lg1_i, -1, -2))
        lt2g_i = _chol_contract(
            chol_i[None, ...], common.t2_green, meas_ctx.cfg
        )[0]
        gl_half_i = _energy_gl_scalar(common.half_green, chol_i, meas_ctx.cfg)
        lt2_half_i = jnp.einsum(
            "ir,qr->iq",
            rot_chol_i.astype(meas_ctx.cfg.mixed_complex_dtype),
            t2_green_mixed,
            optimize="optimal",
        )
        e222_i = 0.5 * jnp.einsum(
            "iq,iq->", gl_half_i, lt2_half_i, optimize="optimal"
        )
        gl_occ_i = reference_occ @ gl_half_i
        glgp_i = jnp.einsum(
            "pi,it->pt", gl_occ_i, greenp_mixed, optimize="optimal"
        )
        e223_i = mode_quadratic_matrices(
            trial_data,
            glgp_i[None, ...],
            n_mode_chunks=meas_ctx.n_mode_chunks,
        )[0]
        connected_i = 4.0 * (-lt2g_i * lg_i + e222_i) + e223_i
        return jnp.stack((e20_i, common.theta * e20_i + connected_i))

    if meas_ctx.memory_mode == "low":
        zero = jnp.zeros((), dtype=jnp.result_type(common.half_green, chol))

        def scan_term(carry, xs):
            chol_i, rot_chol_i = xs
            return carry, scalar_term(chol_i, rot_chol_i)

        _, terms = jax.lax.scan(scan_term, zero, (chol, rot_chol))
        return terms

    lg = jnp.einsum(
        "giq,iq->g",
        rot_chol,
        common.half_green,
        optimize="optimal",
    )
    lg1 = jnp.einsum(
        "gip,jp->gij",
        rot_chol,
        common.half_green,
        optimize="optimal",
    )
    e20 = 2.0 * lg * lg
    e20 -= jnp.sum(
        lg1 * jnp.swapaxes(lg1, -1, -2),
        axis=(-1, -2),
    )
    lt2g = _chol_contract(chol, common.t2_green, meas_ctx.cfg)
    gl_half = _energy_gl_batched(common.half_green, chol, meas_ctx.cfg)
    lt2_half = jnp.einsum(
        "gir,qr->giq",
        rot_chol.astype(meas_ctx.cfg.mixed_complex_dtype),
        t2_green_mixed,
        optimize="optimal",
    )
    e222 = 0.5 * jnp.einsum(
        "giq,giq->g", gl_half, lt2_half, optimize="optimal"
    )
    gl_occ = jnp.einsum(
        "ji,gip->gjp",
        reference_occ,
        gl_half,
        optimize="optimal",
    )
    glgp = jnp.einsum(
        "gpi,it->gpt", gl_occ, greenp_mixed, optimize="optimal"
    )
    e223 = mode_quadratic_matrices(
        trial_data,
        glgp,
        n_mode_chunks=meas_ctx.n_mode_chunks,
    )
    connected = 4.0 * (-lt2g * lg + e222) + e223
    return jnp.stack((e20, common.theta * e20 + connected), axis=-1)


def _ptccsd_thouless_mode_chol_terms_for_walkers(
    common: PtccsdThoulessModeEnergyCommon,
    chol: jax.Array,
    rot_chol: jax.Array,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_chunks: int = 1,
) -> jax.Array:
    """Return bounded walker-by-Cholesky PT component terms."""

    return wk.vmap_chunked(
        lambda common_i: _ptccsd_thouless_mode_chol_terms(
            common_i,
            chol,
            rot_chol,
            meas_ctx,
            trial_data,
        ),
        n_chunks=n_chunks,
    )(common)


def _ptccsd_thouless_mode_chol_index_terms(
    common: PtccsdThoulessModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_chunks: int,
) -> jax.Array:
    """Return one walker's component terms at arbitrary Cholesky indices."""

    return wk.vmap_chunked(
        lambda chol_i: _ptccsd_thouless_mode_chol_terms(
            common,
            ham_data.chol[chol_i][None, ...],
            meas_ctx.rot_chol[chol_i][None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        shard_walkers=False,
    )(chol_indices)


def _ptccsd_thouless_mode_chol_index_moments_for_walkers(
    common: PtccsdThoulessModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
    theta_reference: jax.Array | float = 0.0,
    compute_projection_moments: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Stream a component head and final-energy projection moments."""

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
        terms_i = _ptccsd_thouless_mode_chol_terms_for_walkers(
            common,
            ham_data.chol[indices_i],
            meas_ctx.rot_chol[indices_i],
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

    moments, _ = jax.lax.scan(
        scan_batch,
        (zero_total, zero_real, zero_real, zero_real),
        (padded_indices, valid),
    )
    return moments


def _ptccsd_thouless_mode_chol_index_sum_for_walkers(
    common: PtccsdThoulessModeEnergyCommon,
    chol_indices: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_walker_chunks: int,
    chol_batch_size: int,
) -> jax.Array:
    """Stream an exact component head without materializing every pair."""

    total, _, _, _ = _ptccsd_thouless_mode_chol_index_moments_for_walkers(
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


def _ptccsd_thouless_mode_chol_pair_terms(
    common: PtccsdThoulessModeEnergyCommon,
    sample_walker: jax.Array,
    sample_chol: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_chunks: int = 1,
) -> jax.Array:
    """Evaluate gathered walker--Cholesky component pairs in microbatches."""

    return wk.vmap_chunked(
        lambda walker_i, chol_i: _ptccsd_thouless_mode_chol_terms(
            tree_util.tree_map(lambda value: value[walker_i], common),
            ham_data.chol[chol_i][None, ...],
            meas_ctx.rot_chol[chol_i][None, ...],
            meas_ctx,
            trial_data,
        )[0],
        n_chunks=n_chunks,
        in_axes=(0, 0),
        shard_walkers=False,
    )(sample_walker, sample_chol)


def _build_ptccsd_thouless_reference_chol_scores(
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    chol_batch_size: int,
) -> jax.Array:
    """Build bounded reference scores for the combined PT energy residual."""

    n_chol = int(ham_data.chol.shape[0])
    common = _ptccsd_thouless_mode_energy_common(
        trial_data.mo_t,
        ham_data,
        meas_ctx,
        trial_data,
    )
    indices = jnp.arange(n_chol, dtype=jnp.int32)
    n_chunks = max(1, math.ceil(n_chol / chol_batch_size))
    terms = _ptccsd_thouless_mode_chol_index_terms(
        common,
        indices,
        ham_data,
        meas_ctx,
        trial_data,
        n_chunks=n_chunks,
    )
    effective = project_first_order_energy_terms(common.theta, terms)
    return jnp.maximum(jnp.abs(jnp.real(effective)).astype(jnp.float64), 1.0e-300)


def configure_ptccsd_mode_pair_sampling(
    meas_ctx: PtccsdThoulessModeMeasCtx,
    sampling: PtccsdModePairSamplingCfg,
    guide_scores: jax.Array,
) -> PtccsdThoulessModeMeasCtx:
    """Attach an exact head and strictly positive fixed tail proposal."""

    if meas_ctx.model_sampling is not None:
        raise ValueError("Local Cholesky sampling requires a frozen guide; rebuild the context to reconfigure.")
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
        reference_chol_scores=scores,
        chol_head_indices=head_indices,
        chol_tail_indices=tail_indices,
        chol_tail_prob=tail_prob,
        component_sampling=sampling,
    )


def _local_cholesky_context_specs(ctx):
    return replace(jax.tree.map(lambda _: P(), ctx), rot_chol=P("model"))


def _local_cholesky_head_terms(common, indices, h, ctx, trial, *, n_chunks):
    return _ptccsd_thouless_mode_chol_terms_for_walkers(
        common, h.chol[indices], ctx.rot_chol[indices], ctx, trial, n_chunks=n_chunks,
    )


def pair_sampled_ptccsd_block_components(
    walkers: jax.Array,
    candidate_weights: jax.Array,
    rng_key: jax.Array,
    n_chunks: int,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> BlockComponentEstimate:
    """Estimate the block PT component numerator with an exact denominator.

    ``theta`` and both one-body bases remain exact. The Cholesky head and tail
    contribute jointly to the unnormalized ``electronic_0`` and ``h_t``
    numerators; no per-walker scalar local energy is formed.
    """

    sampling = meas_ctx.component_sampling
    if sampling is None:
        raise ValueError(
            "pair_sampled_ptccsd_block_components requires a component sampling config."
        )
    n_walkers = int(walkers.shape[0])
    if candidate_weights.shape != (n_walkers,):
        raise ValueError(
            f"candidate_weights must have shape {(n_walkers,)}, got "
            f"{candidate_weights.shape}."
        )

    common = wk.vmap_chunked(
        _ptccsd_thouless_mode_energy_common,
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
            _ptccsd_thouless_mode_chol_index_moments_for_walkers(
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
        head_sum = _ptccsd_thouless_mode_chol_index_sum_for_walkers(
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
        pair_n_chunks = (sampling.pair_sample_size + walker_batch_size - 1) // walker_batch_size
        component_terms = _ptccsd_thouless_mode_chol_pair_terms(
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
            pair_fn=_ptccsd_thouless_mode_chol_pair_terms,
            context_specs_fn=_local_cholesky_context_specs, n_chunks=n_chunks, components=True,
        )
    else:
        tail_numerator, half_difference = jax.lax.cond(
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
def _ptccsd_mode_population_common_batch(walkers, ham_data, meas_ctx, trial_data):
    return jax.vmap(
        _ptccsd_thouless_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham_data, meas_ctx, trial_data)


@jax.jit
def _ptccsd_mode_population_term_batch(
    common,
    chol_indices,
    ham_data,
    meas_ctx,
    trial_data,
):
    return _ptccsd_thouless_mode_chol_terms_for_walkers(
        common,
        ham_data.chol[chol_indices],
        meas_ctx.rot_chol[chol_indices],
        meas_ctx,
        trial_data,
        n_chunks=1,
    )


def stream_ptccsd_mode_population_statistics(
    walkers: jax.Array,
    candidate_weights: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    n_walker_chunks: int = 10,
    chol_batch_size: int = 16,
) -> PtccsdModePopulationStats:
    """Stream real projected residual moments without storing all pairs."""

    if n_walker_chunks <= 0:
        raise ValueError("n_walker_chunks must be positive.")
    if chol_batch_size <= 0:
        raise ValueError("chol_batch_size must be positive.")
    start_time = time.perf_counter()
    n_walkers = int(walkers.shape[0])
    n_chol = int(ham_data.chol.shape[0])
    if n_walkers <= 0:
        raise ValueError("PT calibration requires at least one walker.")
    if n_chol <= 0:
        raise ValueError("PT calibration requires at least one Cholesky vector.")
    if candidate_weights.shape != (n_walkers,):
        raise ValueError(
            f"candidate_weights must have shape {(n_walkers,)}, got "
            f"{candidate_weights.shape}."
        )

    candidate_np = np.asarray(jax.device_get(candidate_weights), dtype=np.complex128)
    walker_batch_size = math.ceil(n_walkers / min(n_walker_chunks, n_walkers))
    valid = np.isfinite(candidate_np)
    theta_values = np.zeros(n_walkers, dtype=np.complex128)

    # Determine the exact denominator before forming the projected moments.
    # Recomputing these relatively small common intermediates below keeps the
    # calibration sweep bounded in both walkers and Cholesky vectors.
    for walker_start in range(0, n_walkers, walker_batch_size):
        walker_stop = min(walker_start + walker_batch_size, n_walkers)
        valid_walkers = walker_stop - walker_start
        walker_indices = np.minimum(
            walker_start + np.arange(walker_batch_size, dtype=np.int32),
            n_walkers - 1,
        )
        common = _ptccsd_mode_population_common_batch(
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
        raise ValueError("PT calibration population has zero or nonfinite estimator weight.")
    if not np.isfinite(abs_weight_sum) or abs_weight_sum <= 0.0:
        raise ValueError("PT calibration population has no finite absolute estimator weight.")

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
        common = _ptccsd_mode_population_common_batch(
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
            terms = _ptccsd_mode_population_term_batch(
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
    return PtccsdModePopulationStats(
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


def average_ptccsd_mode_population_statistics(
    population_stats: list[PtccsdModePopulationStats],
) -> PtccsdModePopulationStats:
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
    return PtccsdModePopulationStats(
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


def select_ptccsd_mode_pair_sampling(
    stats: PtccsdModePopulationStats,
    cfg: PtccsdModePairTuningCfg,
    *,
    n_walkers: int,
    reference_guide_scores: np.ndarray,
    calibration_std_ha: float,
    calibration_source: str,
    final_error_target_ha: float | None,
    n_blocks: int,
) -> PtccsdModePairTuningResult:
    cisd_stats = CisdModePopulationStats(
        term_means=stats.term_means,
        term_second_moments=stats.term_second_moments,
        rms_scores=stats.rms_scores,
        local_energies=np.asarray([stats.exact_block_energy_ha]),
        exact_block_energy_ha=stats.exact_block_energy_ha,
        independent_population_std_ha=calibration_std_ha,
        wall_seconds=stats.wall_seconds,
        population_term_means=stats.population_term_means,
        population_term_second_moments=stats.population_term_second_moments,
    )
    selected = select_cisd_mode_pair_sampling(
        cisd_stats,
        cfg.as_cisd_cfg(),
        n_walkers=n_walkers,
        reference_guide_scores=reference_guide_scores,
        calibration_std_ha=calibration_std_ha,
        calibration_source=calibration_source,
        final_error_target_ha=final_error_target_ha,
        n_blocks=n_blocks,
    )
    sampling = PtccsdModePairSamplingCfg(
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
    return PtccsdModePairTuningResult(
        sampling=sampling,
        chol_head_fraction=selected.chol_head_fraction,
        estimated_tail_std_ha=selected.estimated_tail_std_ha,
        guarded_tail_std_ha=selected.guarded_tail_std_ha,
        target_tail_std_ha=selected.target_tail_std_ha,
        target_tail_std_source=selected.target_tail_std_source,
        estimated_pair_evaluations=selected.estimated_pair_evaluations,
    )


def retune_ptccsd_mode_pair_sampling(
    state,
    equilibration_components: jax.Array,
    equilibration_weights: jax.Array,
    params,
    ham_data: HamChol,
    estimator_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
    *,
    guide_data,
    guide_meas_ops: MeasOps,
    guide_meas_ctx,
    advance_blocks: BlockComponentsAdvanceFn,
    tuning_cfg: PtccsdModePairTuningCfg,
    target_error: float | None = None,
) -> BlockComponentRetuneResult:
    """Tune the real projected residual variance and install a production sampler."""

    if estimator_ctx.model_sampling is not None:
        raise ValueError("Local Cholesky sampling requires frozen settings (no automatic retuning).")
    del guide_meas_ctx
    population_stats = []
    for population_index in range(tuning_cfg.tuning_population_count):
        if population_index > 0:
            spacing = tuning_cfg.tuning_population_spacing_blocks
            print(
                f"[PT sampling] advancing {spacing} calibration blocks before "
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
            det_overlap_thouless_r,
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
            "[PT sampling] streaming real projected population statistics: "
            f"population={population_index + 1}/{tuning_cfg.tuning_population_count}, "
            f"chol_batch_size={tuning_cfg.tuning_chol_batch_size}."
        )
        stats_i = stream_ptccsd_mode_population_statistics(
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
            f"[PT sampling] population {population_index + 1}: "
            f"exact projected energy={stats_i.exact_block_energy_ha:.10f} Ha, "
            f"phase_coherence={stats_i.phase_coherence:.3e}, "
            f"statistics_seconds={stats_i.wall_seconds:.1f}."
        )

    stats = average_ptccsd_mode_population_statistics(population_stats)
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
            "PT pair tuning requires multiple calibration populations, a usable "
            "equilibration variance, or an absolute/final-error target."
        )
    selected = select_ptccsd_mode_pair_sampling(
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
    production_ctx = configure_ptccsd_mode_pair_sampling(
        estimator_ctx,
        selected.sampling,
        jnp.asarray(production_scores, dtype=jnp.float64),
    )
    print(
        "[PT sampling] selected real-projected estimator: "
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


def components_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    """Return PT2/Thouless components using half-Green Cholesky contractions.

    With ``G = C.conj() @ R``, the full-Green implementation forms Cholesky
    intermediates of shape ``(n_chol, norb, norb)``.  This implementation
    factors those contractions through ``R`` and keeps their largest shape at
    ``(n_chol, nocc, norb)`` without changing the estimator algebra.
    """

    common = _ptccsd_thouless_mode_energy_common(
        walker,
        ham_data,
        meas_ctx,
        trial_data,
    )
    chol_components = _ptccsd_thouless_mode_chol_terms(
        common,
        ham_data.chol,
        meas_ctx.rot_chol,
        meas_ctx,
        trial_data,
    )
    chol_sum = jnp.sum(chol_components, axis=0)
    electronic_0 = common.electronic_0_base + chol_sum[0]
    h_t = common.h_t_base + chol_sum[1]
    return jnp.stack([common.theta, electronic_0, h_t])


def energy_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    theta2, electronic_0, h_t = components_pt_thouless_rw_rh(
        walker, ham_data, meas_ctx, trial_data
    )
    return ham_data.h0 + electronic_0 + h_t - theta2 * electronic_0


def inverse_guide_components_pt_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtccsdThoulessModeMeasCtx,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    components = components_pt_thouless_rw_rh(walker, ham_data, meas_ctx, trial_data)
    reweight = jnp.exp(-components[0])
    return jnp.concatenate([reweight[None], reweight * components])


def make_ptccsd_mode_meas_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD mode measurements require a closed-shell restricted walker.")
    cfg = PtccsdModeMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return MeasOps(
        overlap=overlap_pt_r,
        build_meas_ctx=lambda ham_data, trial_data: build_ptccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
        ),
        kernels={k_force_bias: force_bias_pt_rw_rh, k_energy: energy_pt_rw_rh},
        observables={o_pt_components: inverse_guide_components_pt_rw_rh},
    )


def make_ptccsd_mode_estimator_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    mixed_precision: bool = True,
    testing: bool = False,
) -> EstimatorOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD mode estimators require a closed-shell restricted walker.")
    cfg = PtccsdModeMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=hf_overlap_r,
        components=components_pt_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptccsd_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
            cfg=cfg,
        ),
    )


def make_ptccsd_thouless_mode_meas_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    exponentiate_t2: bool = True,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD Thouless mode measurements require a closed-shell restricted walker."
        )
    overlap = overlap_ptccsd_thouless_r if exponentiate_t2 else det_overlap_thouless_r
    observables = (
        {o_pt_components: inverse_guide_components_pt_thouless_rw_rh}
        if exponentiate_t2
        else {}
    )
    cfg = PtccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return MeasOps(
        overlap=overlap,
        build_meas_ctx=lambda ham_data, trial_data: build_ptccsd_thouless_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
            memory_mode=memory_mode,
            cfg=cfg,
        ),
        kernels={
            k_force_bias: force_bias_pt_thouless_rw_rh,
            k_energy: energy_pt_thouless_rw_rh,
        },
        observables=observables,
    )


def make_ptccsd_thouless_mode_estimator_ops(
    sys: System,
    *,
    n_mode_chunks: int = 1,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
    component_sampling: PtccsdModePairSamplingCfg | None = None,
    component_tuning: PtccsdModePairTuningCfg | None = None,
) -> EstimatorOps:
    """Build the common mode-native PT2/Thouless projected estimator.

    A fixed ``component_sampling`` policy samples only the Cholesky-resolved
    contribution to the block-level ``h_t`` numerator. Supplying
    ``component_tuning`` additionally gathers bounded post-equilibration
    population moments and installs an automatically selected production
    policy. Proposal construction and tuning use the real part of the
    normalized residual, while the sampled estimator remains fully complex.
    """

    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "PT-CCSD Thouless mode estimators require a closed-shell restricted walker."
        )
    if component_tuning is not None and component_sampling is None:
        raise ValueError("component_tuning requires an equilibration component_sampling config.")
    cfg = PtccsdModeMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=det_overlap_thouless_r,
        components=components_pt_thouless_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptccsd_thouless_mode_meas_ctx(
            ham_data,
            trial_data,
            n_mode_chunks=n_mode_chunks,
            memory_mode=memory_mode,
            cfg=cfg,
            component_sampling=component_sampling,
        ),
        block_components=(
            pair_sampled_ptccsd_block_components
            if component_sampling is not None
            else None
        ),
        retune_block_components=(
            partial(retune_ptccsd_mode_pair_sampling, tuning_cfg=component_tuning)
            if component_tuning is not None
            else None
        ),
        use_for_population_control=component_sampling is not None,
    )
