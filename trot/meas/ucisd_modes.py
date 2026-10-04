from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ucisd import UcisdTrial
from ..trial.ucisd_modes import (
    UcisdModeTrial,
    doubles_apply,
    doubles_projections,
    doubles_quadratic,
    overlap_r,
)
from .ucisd import UcisdMeasCfg, UcisdMeasCtx, _greenp_from_active
from .ucisd import build_meas_ctx as build_dense_meas_ctx

_UCISD_MODE_MEAS_CFG_ATTR = "_ucisd_mode_meas_cfg"


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdModeMeasCtx:
    """Dense-independent UCISD measurement data plus static mode chunking."""

    base: UcisdMeasCtx
    n_mode_chunks: int

    def tree_flatten(self):
        return (self.base,), (self.n_mode_chunks,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (n_mode_chunks,) = aux
        (base,) = children
        return cls(base=base, n_mode_chunks=n_mode_chunks)


class UcisdModeEnergyCommon(NamedTuple):
    """Per-walker intermediates shared by all Cholesky contributions."""

    green_a: jax.Array
    green_b: jax.Array
    greenp_a: jax.Array
    greenp_b: jax.Array
    overlap: jax.Array
    m1_a: jax.Array
    m1_b: jax.Array
    m2_a: jax.Array
    m2_b: jax.Array
    z1_a: jax.Array
    z1_b: jax.Array
    base: jax.Array


def get_ucisd_mode_meas_cfg(meas_ops: MeasOps) -> UcisdMeasCfg | None:
    cfg = getattr(meas_ops, _UCISD_MODE_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, UcisdMeasCfg) else None


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: UcisdModeTrial,
    *,
    cfg: UcisdMeasCfg = UcisdMeasCfg(memory_mode="high"),
    n_mode_chunks: int = 1,
) -> UcisdModeMeasCtx:
    """Build deterministic retained-mode UCISD measurement intermediates."""
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    if cfg.memory_mode != "high":
        raise ValueError("Mode-native UCISD measurements currently require memory_mode='high'.")
    if ham_data.basis != "restricted":
        raise ValueError("UCISD mode MeasOps requires HamChol.basis == 'restricted'.")

    # Dense UCISD context construction depends only on the reference orbitals,
    # singles, and Hamiltonian. It never reads a doubles tensor.
    base = build_dense_meas_ctx(ham_data, cast(UcisdTrial, trial_data), cfg)
    maximum_rank = max(trial_data.mode_rank, default=0)
    chunks = min(int(n_mode_chunks), maximum_rank) if maximum_rank else 1
    return UcisdModeMeasCtx(base=base, n_mode_chunks=chunks)


def _greens_restricted(
    walker: jax.Array,
    trial_data: UcisdModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build alpha/beta Green functions from one shared restricted walker."""
    noa, nob = trial_data.nocc
    wa = walker[:, :noa]
    wb = trial_data.mo_coeff_b.T @ walker[:, :nob]
    green_a = jnp.linalg.solve(wa[:noa].T, wa.T)
    green_b = jnp.linalg.solve(wb[:nob].T, wb.T)
    green_occ_a = green_a[:, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_b[:, nob : nob + trial_data.nvir[1]]
    greenp_a = _greenp_from_active(green_occ_a, trial_data.norb)
    greenp_b = _greenp_from_active(green_occ_b, trial_data.norb)
    return green_a, green_b, greenp_a, greenp_b


def _chol_contract(chol: jax.Array, matrix: jax.Array, cfg: UcisdMeasCfg) -> jax.Array:
    """Contract through real FP32 kernels before reconstructing the result."""
    chol_r = chol.astype(cfg.mixed_real_dtype)
    matrix_r = jnp.real(matrix).astype(cfg.mixed_real_dtype)
    matrix_i = jnp.imag(matrix).astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("gij,ij->g", chol_r, matrix_r, optimize="optimal")
    imag_part = jnp.einsum("gij,ij->g", chol_r, matrix_i, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _energy_gl_batched_realimag(
    green: jax.Array,
    chol: jax.Array,
    cfg: UcisdMeasCfg,
) -> jax.Array:
    """Build a batched ``green @ chol`` through real mixed-precision GEMMs."""
    green_r = jnp.real(green).astype(cfg.mixed_real_dtype)
    green_i = jnp.imag(green).astype(cfg.mixed_real_dtype)
    chol_r = chol.astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("pj,gji->gpi", green_r, chol_r, optimize="optimal")
    imag_part = jnp.einsum("pj,gji->gpi", green_i, chol_r, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _beta_matrix_to_alpha_basis(
    matrix_b: jax.Array,
    mo_coeff_b: jax.Array,
    cfg: UcisdMeasCfg,
) -> jax.Array:
    """Return ``C_b.conj() @ matrix_b @ C_b.T`` in the mixed dtype."""
    if jnp.issubdtype(mo_coeff_b.dtype, jnp.complexfloating):
        cb = mo_coeff_b.astype(cfg.mixed_complex_dtype)
        return cb.conj() @ matrix_b.astype(cfg.mixed_complex_dtype) @ cb.T

    cb = mo_coeff_b.astype(cfg.mixed_real_dtype)
    matrix_r = jnp.real(matrix_b).astype(cfg.mixed_real_dtype)
    matrix_i = jnp.imag(matrix_b).astype(cfg.mixed_real_dtype)
    rotated_r = (cb @ matrix_r) @ cb.T
    rotated_i = (cb @ matrix_i) @ cb.T
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return rotated_r.astype(cfg.mixed_complex_dtype) + imag_unit * rotated_i.astype(
        cfg.mixed_complex_dtype
    )


def _spin_sum_chol_contract(
    chol_a: jax.Array,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    mo_coeff_b: jax.Array,
    cfg: UcisdMeasCfg,
) -> jax.Array:
    """Contract alpha and beta matrices once in the alpha orbital basis.

    If ``L_b = C_b.conj().T @ L_a @ C_b``, then
    ``L_a:M_a + L_b:M_b = L_a:(M_a + C_b.conj() @ M_b @ C_b.T)``.
    """
    matrix_b_alpha = _beta_matrix_to_alpha_basis(matrix_b, mo_coeff_b, cfg)
    spin_sum = matrix_a.astype(cfg.mixed_complex_dtype) + matrix_b_alpha
    return _chol_contract(chol_a, spin_sum, cfg)


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdModeMeasCtx,
    trial_data: UcisdModeTrial,
) -> jax.Array:
    """Mode-native UCISD force bias for a shared restricted walker."""
    base = meas_ctx.base
    green_a, green_b, greenp_a, greenp_b = _greens_restricted(walker, trial_data)
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_b[:, nob : nob + trial_data.nvir[1]]

    lg_a = jnp.einsum("gpj,pj->g", base.rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", base.rot_chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b
    singles = jnp.einsum("pt,pt->", trial_data.c1a, green_occ_a, optimize="optimal")
    singles += jnp.einsum("pt,pt->", trial_data.c1b, green_occ_b, optimize="optimal")
    m1_a = (greenp_a @ trial_data.c1a.T) @ green_a
    m1_b = (greenp_b @ trial_data.c1b.T) @ green_b

    projections = doubles_projections(trial_data, green_occ_a, green_occ_b)
    ya, yb = doubles_apply(
        trial_data,
        green_occ_a,
        green_occ_b,
        projections=projections,
    )
    doubles = doubles_quadratic(
        trial_data,
        green_occ_a,
        green_occ_b,
        projections=projections,
    )
    m2_a = (greenp_a @ ya.T) @ green_a
    m2_b = (greenp_b @ yb.T) @ green_b
    overlap = 1.0 + singles + doubles

    correction = _spin_sum_chol_contract(
        ham_data.chol,
        m1_a + m2_a,
        m1_b + m2_b,
        trial_data.mo_coeff_b,
        base.cfg,
    )
    return (lg * overlap - correction) / overlap


def _mode_bilinear_matrices(
    values: jax.Array,
    left_modes: jax.Array,
    right_modes: jax.Array,
    left_matrices: jax.Array,
    right_matrices: jax.Array,
    *,
    n_mode_chunks: int,
    shared_projection: bool = False,
) -> jax.Array:
    """Evaluate batched bilinear mode contractions with a bounded mode axis."""
    if left_matrices.shape[:-2] != right_matrices.shape[:-2]:
        raise ValueError("left and right matrices must have identical leading shapes.")
    if left_matrices.shape[-2:] != left_modes.shape[-2:]:
        raise ValueError(
            f"left matrices must end in shape {left_modes.shape[-2:]}, "
            f"got {left_matrices.shape}."
        )
    if right_matrices.shape[-2:] != right_modes.shape[-2:]:
        raise ValueError(
            f"right matrices must end in shape {right_modes.shape[-2:]}, "
            f"got {right_matrices.shape}."
        )

    leading_shape = left_matrices.shape[:-2]
    left_flat = left_matrices.reshape((-1,) + left_matrices.shape[-2:])
    right_flat = right_matrices.reshape((-1,) + right_matrices.shape[-2:])
    result_dtype = (
        jnp.complex128
        if jnp.issubdtype(left_matrices.dtype, jnp.complexfloating)
        or jnp.issubdtype(right_matrices.dtype, jnp.complexfloating)
        else jnp.float64
    )
    rank = int(values.shape[0])
    if rank == 0:
        return jnp.zeros(leading_shape, dtype=result_dtype)

    def evaluate_chunk(values_i, left_modes_i, right_modes_i):
        left_r = jnp.real(left_flat).astype(left_modes_i.dtype)
        projection_left_r = jnp.einsum(
            "spt,rpt->sr", left_r, left_modes_i, optimize="optimal"
        )
        if shared_projection:
            projection_right_r = projection_left_r
        else:
            right_r = jnp.real(right_flat).astype(right_modes_i.dtype)
            projection_right_r = jnp.einsum(
                "spt,rpt->sr", right_r, right_modes_i, optimize="optimal"
            )
        if result_dtype == jnp.complex128:
            left_i = jnp.imag(left_flat).astype(left_modes_i.dtype)
            projection_left_i = jnp.einsum(
                "spt,rpt->sr", left_i, left_modes_i, optimize="optimal"
            )
            if shared_projection:
                projection_right_i = projection_left_i
            else:
                right_i = jnp.imag(right_flat).astype(right_modes_i.dtype)
                projection_right_i = jnp.einsum(
                    "spt,rpt->sr", right_i, right_modes_i, optimize="optimal"
                )
            projection_left = projection_left_r.astype(jnp.complex128)
            projection_left += 1.0j * projection_left_i.astype(jnp.complex128)
            projection_right = projection_right_r.astype(jnp.complex128)
            projection_right += 1.0j * projection_right_i.astype(jnp.complex128)
        else:
            projection_left = projection_left_r.astype(jnp.float64)
            projection_right = projection_right_r.astype(jnp.float64)
        return jnp.sum(
            values_i.astype(jnp.float64)[None, :] * projection_left * projection_right,
            axis=1,
            dtype=result_dtype,
        )

    chunks = min(int(n_mode_chunks), rank)
    if chunks == 1:
        return evaluate_chunk(values, left_modes, right_modes).reshape(leading_shape)

    base_chunk_size = rank // chunks
    n_larger_chunks = rank % chunks
    chunk_size = base_chunk_size + int(n_larger_chunks > 0)
    chunk_offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    zero = jnp.zeros((left_flat.shape[0],), dtype=result_dtype)

    def scan_body(total, chunk_index):
        is_larger = chunk_index < n_larger_chunks
        chunk_length = base_chunk_size + is_larger.astype(jnp.int32)
        start = chunk_index * base_chunk_size + jnp.minimum(chunk_index, n_larger_chunks)
        indices = start + chunk_offsets
        valid = chunk_offsets < chunk_length
        indices = jnp.minimum(indices, rank - 1)
        values_i = jnp.where(valid, values[indices], 0.0)
        contribution = evaluate_chunk(values_i, left_modes[indices], right_modes[indices])
        return total + contribution, None

    result, _ = lax.scan(scan_body, zero, jnp.arange(chunks, dtype=jnp.int32))
    return result.reshape(leading_shape)


def _mode_quadratic_matrices(
    trial_data: UcisdModeTrial,
    meas_ctx: UcisdModeMeasCtx,
    matrices_a: jax.Array,
    matrices_b: jax.Array,
) -> jax.Array:
    aa = _mode_bilinear_matrices(
        trial_data.eigenvalues_aa,
        trial_data.modes_aa,
        trial_data.modes_aa,
        matrices_a,
        matrices_a,
        n_mode_chunks=meas_ctx.n_mode_chunks,
        shared_projection=True,
    )
    ab = _mode_bilinear_matrices(
        trial_data.singular_values_ab,
        trial_data.left_modes_ab,
        trial_data.right_modes_ab,
        matrices_a,
        matrices_b,
        n_mode_chunks=meas_ctx.n_mode_chunks,
    )
    bb = _mode_bilinear_matrices(
        trial_data.eigenvalues_bb,
        trial_data.modes_bb,
        trial_data.modes_bb,
        matrices_b,
        matrices_b,
        n_mode_chunks=meas_ctx.n_mode_chunks,
        shared_projection=True,
    )
    return 0.5 * aa + ab + 0.5 * bb


def _ucisd_mode_energy_common(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdModeMeasCtx,
    trial_data: UcisdModeTrial,
) -> UcisdModeEnergyCommon:
    """Build the exact walker-only energy base and reusable intermediates."""
    base = meas_ctx.base
    green_a, green_b, greenp_a, greenp_b = _greens_restricted(walker, trial_data)
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_b[:, nob : nob + trial_data.nvir[1]]
    h1_a = 0.5 * (ham_data.h1 + ham_data.h1.T)
    h1_b = base.h1_b

    hg = jnp.einsum("pj,pj->", h1_a[:noa], green_a, optimize="optimal")
    hg += jnp.einsum("pj,pj->", h1_b[:nob], green_b, optimize="optimal")
    singles = jnp.einsum("pt,pt->", trial_data.c1a, green_occ_a, optimize="optimal")
    singles += jnp.einsum("pt,pt->", trial_data.c1b, green_occ_b, optimize="optimal")
    m1_a = (greenp_a @ trial_data.c1a.T) @ green_a
    m1_b = (greenp_b @ trial_data.c1b.T) @ green_b

    projections = doubles_projections(trial_data, green_occ_a, green_occ_b)
    ya, yb = doubles_apply(
        trial_data,
        green_occ_a,
        green_occ_b,
        projections=projections,
    )
    doubles = doubles_quadratic(
        trial_data,
        green_occ_a,
        green_occ_b,
        projections=projections,
    )
    m2_a = (greenp_a @ ya.T) @ green_a
    m2_b = (greenp_b @ yb.T) @ green_b
    overlap = 1.0 + singles + doubles

    one_body = hg * overlap
    one_body -= jnp.einsum("ij,ij->", h1_a, m1_a + m2_a, optimize="optimal")
    one_body -= jnp.einsum("ij,ij->", h1_b, m1_b + m2_b, optimize="optimal")
    return UcisdModeEnergyCommon(
        green_a=green_a,
        green_b=green_b,
        greenp_a=greenp_a,
        greenp_b=greenp_b,
        overlap=overlap,
        m1_a=m1_a,
        m1_b=m1_b,
        m2_a=m2_a,
        m2_b=m2_b,
        z1_a=trial_data.c1a @ green_occ_a.T,
        z1_b=trial_data.c1b @ green_occ_b.T,
        base=ham_data.h0 + one_body / overlap,
    )


def _ucisd_mode_chol_terms(
    common: UcisdModeEnergyCommon,
    chol_a: jax.Array,
    rot_chol_a: jax.Array,
    lci1_a: jax.Array,
    chol_b: jax.Array,
    rot_chol_b: jax.Array,
    lci1_b: jax.Array,
    meas_ctx: UcisdModeMeasCtx,
    trial_data: UcisdModeTrial,
) -> jax.Array:
    """Return one walker's normalized contribution for each Cholesky vector."""
    cfg = meas_ctx.base.cfg
    green_a = common.green_a
    green_b = common.green_b
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b
    q_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
    q_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
    e20 = 0.5 * lg * lg
    e20 -= 0.5 * jnp.sum(q_a * jnp.swapaxes(q_a, -1, -2), axis=(-1, -2))
    e20 -= 0.5 * jnp.sum(q_b * jnp.swapaxes(q_b, -1, -2), axis=(-1, -2))

    r1 = jnp.einsum("gpq,gqr,rp->g", q_a, q_a, common.z1_a, optimize="optimal")
    r1 += jnp.einsum("gpq,gqr,rp->g", q_b, q_b, common.z1_b, optimize="optimal")
    lci1g_a = jnp.einsum("gip,qi->gpq", lci1_a, green_a, optimize="optimal")
    lci1g_b = jnp.einsum("gip,qi->gpq", lci1_b, green_b, optimize="optimal")
    r1 -= jnp.einsum("gpq,gqp->g", lci1g_a, q_a, optimize="optimal")
    r1 -= jnp.einsum("gpq,gqp->g", lci1g_b, q_b, optimize="optimal")

    lm12 = _spin_sum_chol_contract(
        chol_a,
        common.m1_a + common.m2_a,
        common.m1_b + common.m2_b,
        trial_data.mo_coeff_b,
        cfg,
    )
    gl_a = _energy_gl_batched_realimag(green_a, chol_a, cfg)
    gl_b = _energy_gl_batched_realimag(green_b, chol_b, cfg)
    rot_m2_a = jnp.einsum("gpi,ji->gpj", rot_chol_a, common.m2_a, optimize="optimal")
    rot_m2_b = jnp.einsum("gpi,ji->gpj", rot_chol_b, common.m2_b, optimize="optimal")
    r2 = jnp.einsum("gpi,gpi->g", gl_a, rot_m2_a, optimize="optimal")
    r2 += jnp.einsum("gpi,gpi->g", gl_b, rot_m2_b, optimize="optimal")

    x_a = jnp.einsum(
        "gpi,it->gpt",
        gl_a,
        common.greenp_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    x_b = jnp.einsum(
        "gpi,it->gpt",
        gl_b,
        common.greenp_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    r3 = _mode_quadratic_matrices(trial_data, meas_ctx, x_a, x_b)
    numerator = common.overlap * e20 - lm12 * lg + r1 + r2 + r3
    return numerator / common.overlap


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdModeMeasCtx,
    trial_data: UcisdModeTrial,
) -> jax.Array:
    """Complete deterministic local energy from every retained UCISD mode."""
    common = _ucisd_mode_energy_common(walker, ham_data, meas_ctx, trial_data)
    base = meas_ctx.base
    chol_terms = _ucisd_mode_chol_terms(
        common,
        ham_data.chol,
        base.rot_chol_a,
        base.lci1_a,
        base.chol_b,
        base.rot_chol_b,
        base.lci1_b,
        meas_ctx,
        trial_data,
    )
    return common.base + jnp.sum(chol_terms, dtype=jnp.complex128)


def make_ucisd_mode_meas_ops(
    sys: System,
    *,
    mixed_precision: bool = True,
    n_mode_chunks: int = 1,
) -> MeasOps:
    """Build deterministic mode-native UCISD measurements for restricted walkers."""
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "UCISD mode MeasOps currently supports only restricted walkers, got: "
            f"{sys.walker_kind}"
        )
    if n_mode_chunks <= 0:
        raise ValueError("n_mode_chunks must be positive.")
    cfg = UcisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    meas_ops = MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(
            ham_data,
            trial_data,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
        ),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={},
    )
    object.__setattr__(meas_ops, _UCISD_MODE_MEAS_CFG_ATTR, cfg)
    return meas_ops
