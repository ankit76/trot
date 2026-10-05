from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.sharding import Mesh

from ..core.ops import EstimatorOps, MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from .chol_setup import shared_beta_cholesky, transform_cholesky
from .chol_contract import contract_cholesky
from ..sharding import cholesky_model_mesh
from ..trial.ptuccsd_thouless import (
    PtuccsdThoulessTrial,
    greenp_from_green,
    overlap_r,
    overlap_u,
    reference_overlap_r,
)
from .pt2ccsd import combine_first_order_energy

o_pt_components = "pt_components"


@dataclass(frozen=True)
class PtuccsdThoulessMeasCfg:
    memory_mode: Literal["low", "high"] = "high"
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64
    chol_batch_size: int = 64  # force bias; also doubles in low-memory mode
    chol_mesh: Mesh | None = None

    def __post_init__(self) -> None:
        if self.memory_mode not in {"low", "high"}:
            raise ValueError("memory_mode must be 'low' or 'high'.")
        if not isinstance(self.chol_batch_size, Integral) or self.chol_batch_size < 1:
            raise ValueError("chol_batch_size must be a positive integer.")


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdThoulessMeasCtx:
    h1_b: jax.Array
    chol_b: jax.Array
    rot_chol_a: jax.Array
    rot_chol_b: jax.Array
    cfg: PtuccsdThoulessMeasCfg

    def tree_flatten(self):
        return (self.h1_b, self.chol_b, self.rot_chol_a, self.rot_chol_b), (self.cfg,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        h1_b, chol_b, rot_chol_a, rot_chol_b = children
        return cls(
            h1_b=h1_b,
            chol_b=chol_b,
            rot_chol_a=rot_chol_a,
            rot_chol_b=rot_chol_b,
            cfg=cfg,
        )


def build_ptuccsd_thouless_meas_ctx(
    ham_data: HamChol,
    trial_data: PtuccsdThoulessTrial,
    cfg: PtuccsdThoulessMeasCfg = PtuccsdThoulessMeasCfg(),
    *,
    guide_data=None,
    guide_ctx=None,
) -> PtuccsdThoulessMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError(
            "PT2-UCCSD Thouless measurements require a restricted-basis Hamiltonian."
        )
    cfg = replace(cfg, chol_mesh=cholesky_model_mesh(ham_data.chol))
    cb = trial_data.mo_coeff_b
    cbh = cb.conj().T
    h1_sym = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = cbh @ h1_sym @ cb
    chol_b = shared_beta_cholesky(ham_data, trial_data, guide_data, guide_ctx)
    if chol_b is None:
        chol_b = transform_cholesky(ham_data.chol, left=cbh, right=cb)
    else:
        print("[setup] reusing guide beta-basis Cholesky tensor for PT2-UCCSD", flush=True)
    rot_chol_a = transform_cholesky(ham_data.chol, left=trial_data.mo_t_a.conj().T)
    rot_chol_b = transform_cholesky(chol_b, left=trial_data.mo_t_b.conj().T)
    return PtuccsdThoulessMeasCtx(
        h1_b=h1_b,
        chol_b=chol_b,
        rot_chol_a=rot_chol_a,
        rot_chol_b=rot_chol_b,
        cfg=cfg,
    )


def _green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessTrial,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    _, _, green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = (
        _half_green_blocks(walker, trial_data)
    )
    return green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b


def _half_green_blocks(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessTrial,
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
    """Return spin-resolved half and full Green-function blocks."""

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


def _chol_contract(
    chol: jax.Array,
    mat: jax.Array,
    cfg: PtuccsdThoulessMeasCfg,
) -> jax.Array:
    """Contract a real Cholesky tensor with a complex matrix in mixed precision."""

    return contract_cholesky(chol, mat, cfg)


def _energy_gl_batched(
    half_green: jax.Array,
    chol: jax.Array,
    cfg: PtuccsdThoulessMeasCfg,
) -> jax.Array:
    """Form all half-Green--Cholesky products without dtype promotion."""

    half_green_r = jnp.real(half_green).astype(cfg.mixed_real_dtype)
    half_green_i = jnp.imag(half_green).astype(cfg.mixed_real_dtype)
    chol_r = chol.astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("pj,gji->gpi", half_green_r, chol_r, optimize="optimal")
    imag_part = jnp.einsum("pj,gji->gpi", half_green_i, chol_r, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _energy_gl_scalar(
    half_green: jax.Array,
    chol_i: jax.Array,
    cfg: PtuccsdThoulessMeasCfg,
) -> jax.Array:
    """Scalar-Cholesky counterpart of :func:`_energy_gl_batched`."""

    half_green_r = jnp.real(half_green).astype(cfg.mixed_real_dtype)
    half_green_i = jnp.imag(half_green).astype(cfg.mixed_real_dtype)
    chol_i_r = chol_i.astype(cfg.mixed_real_dtype)
    real_part = jnp.einsum("pj,ji->pi", half_green_r, chol_i_r, optimize="optimal")
    imag_part = jnp.einsum("pj,ji->pi", half_green_i, chol_i_r, optimize="optimal")
    imag_unit = jnp.asarray(1.0j, dtype=cfg.mixed_complex_dtype)
    return real_part.astype(cfg.mixed_complex_dtype) + imag_unit * imag_part.astype(
        cfg.mixed_complex_dtype
    )


def _theta2_force_terms(
    trial_data: PtuccsdThoulessTrial,
    green_occ_a: jax.Array,
    green_occ_b: jax.Array,
    cfg: PtuccsdThoulessMeasCfg,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    t2aa, t2ab, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2bb
    t2g_a = jnp.einsum(
        "ptqu,pt->qu",
        t2aa.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_b = jnp.einsum(
        "ptqu,pt->qu",
        t2bb.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    theta2a = 0.5 * jnp.einsum("qu,qu->", t2g_a, green_occ_a, optimize="optimal")
    theta2b = 0.5 * jnp.einsum("qu,qu->", t2g_b, green_occ_b, optimize="optimal")
    t2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    theta2ab = jnp.einsum("pt,pt->", t2g_ab_a, green_occ_a, optimize="optimal")
    return theta2a + theta2b + theta2ab, t2g_a, t2g_b, t2g_ab_a


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = _green_blocks(
        walker, trial_data
    )
    noa, nob = trial_data.nocc
    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    cfg = meas_ctx.cfg

    # Keep the determinant-reference force bias at full precision, matching
    # the UCISD policy. Only the correlation correction is evaluated in the
    # configured mixed precision below.
    lg_a = jnp.einsum("gij,ij->g", chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gij,ij->g", chol_b, green_b, optimize="optimal")
    f0 = lg_a + lg_b

    theta2, t2g_a, t2g_b, t2g_ab_a = _theta2_force_terms(
        trial_data, green_occ_a, green_occ_b, cfg
    )
    t2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        trial_data.t2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2_green_a = (greenp_a @ (t2g_a + t2g_ab_a).T) @ green_a[:noa, :]
    t2_green_b = (greenp_b @ (t2g_b + t2g_ab_b).T) @ green_b[:nob, :]

    fb_2_1 = theta2 * f0
    fb_2_2 = -_chol_contract(chol_a, t2_green_a, cfg) - _chol_contract(
        chol_b, t2_green_b, cfg
    )
    return f0 + fb_2_1 + fb_2_2 - f0 * theta2


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return force_bias_kernel_uw_rh(
        (walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data
    )


def _energy_components_uw_rh_full(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reference full-Green implementation retained as a test oracle."""

    green_a, green_b, green_occ_a, green_occ_b, greenp_a, greenp_b = _green_blocks(
        walker, trial_data
    )
    t2aa, t2ab, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2bb
    noa, nob = trial_data.nocc
    cfg = meas_ctx.cfg

    h1_a = 0.5 * (ham_data.h1 + ham_data.h1.T.conj())
    h1_b = meas_ctx.h1_b
    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b

    e1_0 = jnp.einsum("ij,ij->", h1_a, green_a, optimize="optimal")
    e1_0 += jnp.einsum("ij,ij->", h1_b, green_b, optimize="optimal")

    t2g_a = 0.25 * jnp.einsum(
        "ptqu,pt->qu",
        t2aa.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_b = 0.25 * jnp.einsum(
        "ptqu,pt->qu",
        t2bb.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    theta2a = jnp.einsum("qu,qu->", t2g_a, green_occ_a, optimize="optimal")
    theta2b = jnp.einsum("qu,qu->", t2g_b, green_occ_b, optimize="optimal")
    theta2ab = jnp.einsum("pt,pt->", t2g_ab_a, green_occ_a, optimize="optimal")
    theta2 = 2.0 * (theta2a + theta2b) + theta2ab

    t2_green_a = (greenp_a @ t2g_a.T) @ green_a[:noa, :]
    t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_a[:noa, :]
    t2_green_b = (greenp_b @ t2g_b.T) @ green_b[:nob, :]
    t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_b[:nob, :]
    combo_a = 4.0 * t2_green_a + t2_green_ab_a
    combo_b = 4.0 * t2_green_b + t2_green_ab_b
    e1_2_1 = e1_0 * theta2
    e1_2_2 = -jnp.einsum("ij,ij->", h1_a, combo_a, optimize="optimal")
    e1_2_2 -= jnp.einsum("ij,ij->", h1_b, combo_b, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    lg_a = jnp.einsum("gij,ij->g", chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gij,ij->g", chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b
    e2_0_1 = 0.5 * (lg @ lg)
    gl1_a = jnp.einsum("pr,gqr->gpq", green_a, chol_a, optimize="optimal")
    gl1_b = jnp.einsum("pr,gqr->gpq", green_b, chol_b, optimize="optimal")
    e2_0_2 = -0.5 * (
        jnp.sum(gl1_a * jnp.swapaxes(gl1_a, -1, -2))
        + jnp.sum(gl1_b * jnp.swapaxes(gl1_b, -1, -2))
    )
    e2_0 = e2_0_1 + e2_0_2
    electronic_0 = e1_0 + e2_0

    e2_2_1 = e2_0 * theta2
    lt2g_a = _chol_contract(
        chol_a, 8.0 * t2_green_a + 2.0 * t2_green_ab_a, cfg
    )
    lt2g_b = _chol_contract(
        chol_b, 8.0 * t2_green_b + 2.0 * t2_green_ab_b, cfg
    )
    e2_2_2_1 = -0.5 * ((lt2g_a + lt2g_b) @ lg)
    combo2_a = 8.0 * t2_green_a + 2.0 * t2_green_ab_a
    combo2_b = 8.0 * t2_green_b + 2.0 * t2_green_ab_b

    def scan_over_chol(carry, x):
        e222_acc, e23_acc = carry
        chol_a_i, chol_b_i = x
        gl_a_i = jnp.einsum("pr,rq->pq", green_a, chol_a_i, optimize="optimal")
        gl_b_i = jnp.einsum("pr,rq->pq", green_b, chol_b_i, optimize="optimal")
        lt2_green_a_i = jnp.einsum(
            "pi,ji->pj", chol_a_i, combo2_a, optimize="optimal"
        )
        lt2_green_b_i = jnp.einsum(
            "pi,ji->pj", chol_b_i, combo2_b, optimize="optimal"
        )
        e222_acc += 0.5 * (
            jnp.einsum("pi,pi->", gl_a_i, lt2_green_a_i, optimize="optimal")
            + jnp.einsum("pi,pi->", gl_b_i, lt2_green_b_i, optimize="optimal")
        )
        glgp_a_i = jnp.einsum(
            "pi,it->pt", gl_a_i[:noa, :], greenp_a, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        glgp_b_i = jnp.einsum(
            "pi,it->pt", gl_b_i[:nob, :], greenp_b, optimize="optimal"
        ).astype(cfg.mixed_complex_dtype_testing)
        l2t2_a = 0.5 * jnp.einsum(
            "pt,qu,ptqu->",
            glgp_a_i,
            glgp_a_i,
            t2aa.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2t2_b = 0.5 * jnp.einsum(
            "pt,qu,ptqu->",
            glgp_b_i,
            glgp_b_i,
            t2bb.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2t2_ab = jnp.einsum(
            "pt,qu,ptqu->",
            glgp_a_i,
            glgp_b_i,
            t2ab.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        e23_acc += l2t2_a + l2t2_b + l2t2_ab
        return (e222_acc, e23_acc), None

    # The exchange-like term follows the dtype of the uncast Green's-function
    # contractions, while the quartic T2 term deliberately uses the lower
    # ``testing`` precision.  Keeping separate scan seeds is required when the
    # Hamiltonian and walkers are double precision but mixed precision is on.
    zero_e222 = jnp.zeros_like(e2_2_2_1)
    zero_e23 = jnp.array(0.0, dtype=cfg.mixed_complex_dtype_testing)
    (e2_2_2_2, e2_2_3), _ = jax.lax.scan(
        scan_over_chol, (zero_e222, zero_e23), (chol_a, chol_b)
    )
    e2_2 = e2_2_1 + e2_2_2_1 + e2_2_2_2 + e2_2_3

    h_t = e1_2 + e2_2
    return theta2, electronic_0, h_t


def _energy_components_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Spin-resolved PT2-UCCSD components with selectable doubles memory mode."""

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
    t2aa, t2ab, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2bb
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

    t2g_a = 0.25 * jnp.einsum(
        "ptqu,pt->qu",
        t2aa.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_b = 0.25 * jnp.einsum(
        "ptqu,pt->qu",
        t2bb.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    t2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        t2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    theta2a = jnp.einsum("qu,qu->", t2g_a, green_occ_a, optimize="optimal")
    theta2b = jnp.einsum("qu,qu->", t2g_b, green_occ_b, optimize="optimal")
    theta2ab = jnp.einsum("pt,pt->", t2g_ab_a, green_occ_a, optimize="optimal")
    theta2 = 2.0 * (theta2a + theta2b) + theta2ab

    green_rows_a = green_a[:noa, :]
    green_rows_b = green_b[:nob, :]
    t2_green_a = (greenp_a @ t2g_a.T) @ green_rows_a
    t2_green_ab_a = (greenp_a @ t2g_ab_a.T) @ green_rows_a
    t2_green_b = (greenp_b @ t2g_b.T) @ green_rows_b
    t2_green_ab_b = (greenp_b @ t2g_ab_b.T) @ green_rows_b
    combo_a = 4.0 * t2_green_a + t2_green_ab_a
    combo_b = 4.0 * t2_green_b + t2_green_ab_b
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

    combo2_a = 8.0 * t2_green_a + 2.0 * t2_green_ab_a
    combo2_b = 8.0 * t2_green_b + 2.0 * t2_green_ab_b
    lt2g_a = _chol_contract(chol_a, combo2_a, cfg)
    lt2g_b = _chol_contract(chol_b, combo2_b, cfg)
    e2_2_2_1 = -0.5 * ((lt2g_a + lt2g_b) @ lg)

    # These factors must be cast before the large batched contractions. Leaving
    # either at its input float64/complex128 dtype silently promotes the full
    # walker--Cholesky intermediate back to complex128.
    reference_occ_a = trial_data.mo_t_a.conj()[:noa, :].astype(
        cfg.mixed_complex_dtype
    )
    reference_occ_b = trial_data.mo_t_b.conj()[:nob, :].astype(
        cfg.mixed_complex_dtype
    )
    greenp_a_mixed = greenp_a.astype(cfg.mixed_complex_dtype)
    greenp_b_mixed = greenp_b.astype(cfg.mixed_complex_dtype)
    combo2_a_mixed = combo2_a.astype(cfg.mixed_complex_dtype)
    combo2_b_mixed = combo2_b.astype(cfg.mixed_complex_dtype)

    def contract_doubles(chol_a, rot_chol_a, chol_b, rot_chol_b):
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
        l2t2_a = 0.5 * jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_a,
            glgp_a,
            t2aa.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2t2_b = 0.5 * jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_b,
            glgp_b,
            t2bb.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        l2t2_ab = jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_a,
            glgp_b,
            t2ab.astype(cfg.mixed_real_dtype_testing),
            optimize="optimal",
        )
        e2_2_3 = jnp.sum(l2t2_a + l2t2_b + l2t2_ab)
        return e2_2_2_2, e2_2_3

    nchol = chol_a.shape[0]
    chol_tensors = (chol_a, rot_chol_a, chol_b, rot_chol_b)
    if cfg.memory_mode == "high" or nchol <= cfg.chol_batch_size:
        e2_2_2_2, e2_2_3 = contract_doubles(*chol_tensors)
    else:
        batch_size = cfg.chol_batch_size
        nfull, remainder = divmod(nchol, batch_size)

        def accumulate(index, carry):
            batches = tuple(
                jax.lax.dynamic_slice_in_dim(tensor, index * batch_size, batch_size, axis=0)
                for tensor in chol_tensors
            )
            values = contract_doubles(*batches)
            return tuple(old + value for old, value in zip(carry, values))

        zeros = (
            jnp.zeros_like(e2_2_2_1),
            jnp.zeros((), dtype=cfg.mixed_complex_dtype_testing),
        )
        values = jax.lax.fori_loop(0, nfull, accumulate, zeros)
        if remainder:
            # Pad only the final partial batch, never the full Hamiltonian.
            padding = ((0, batch_size - remainder), (0, 0), (0, 0))
            tail = contract_doubles(
                *(jnp.pad(tensor[nfull * batch_size :], padding) for tensor in chol_tensors)
            )
            values = tuple(old + value for old, value in zip(values, tail))
        e2_2_2_2, e2_2_3 = values

    e2_2 = e2_0 * theta2 + e2_2_2_1 + e2_2_2_2 + e2_2_3
    h_t = e1_2 + e2_2
    return theta2, electronic_0, h_t


def components_ptuccsd_thouless_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    """Return guide-independent ``[theta, electronic_0, h_t]`` components."""
    return jnp.stack(_energy_components_uw_rh(walker, ham_data, meas_ctx, trial_data))


def components_ptuccsd_thouless_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    return components_ptuccsd_thouless_uw_rh(
        (walker[:, :noa], walker[:, :nob]), ham_data, meas_ctx, trial_data
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    components = components_ptuccsd_thouless_uw_rh(
        walker, ham_data, meas_ctx, trial_data
    )
    return combine_first_order_energy(ham_data.h0, components)


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: PtuccsdThoulessMeasCtx,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    components = components_ptuccsd_thouless_rw_rh(
        walker, ham_data, meas_ctx, trial_data
    )
    return combine_first_order_energy(ham_data.h0, components)


def make_ptuccsd_thouless_meas_ops(
    sys: System,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
    *,
    chol_batch_size: int = 64,
) -> MeasOps:
    """Dense measurements; low-memory mode batches doubles over Cholesky vectors."""
    cfg = PtuccsdThoulessMeasCfg(
        memory_mode=memory_mode,
        chol_batch_size=chol_batch_size,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    walker_kind = sys.walker_kind.lower()
    if walker_kind == "restricted":
        if sys.nup < sys.ndn:
            raise ValueError("Restricted PT2-UCCSD measurements require nup >= ndn.")
        overlap_fn = overlap_r
        kernels = {k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh}
        components_fn = components_ptuccsd_thouless_rw_rh
    elif walker_kind == "unrestricted":
        overlap_fn = overlap_u
        kernels = {k_force_bias: force_bias_kernel_uw_rh, k_energy: energy_kernel_uw_rh}
        components_fn = components_ptuccsd_thouless_uw_rh
    else:
        raise ValueError(
            "PT2-UCCSD Thouless measurements support restricted/unrestricted walkers, "
            f"got: {sys.walker_kind}"
        )

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_ptuccsd_thouless_meas_ctx(
            ham_data, trial_data, cfg
        ),
        kernels=kernels,
        observables={o_pt_components: components_fn},
    )


def make_ptuccsd_thouless_estimator_ops(
    sys: System,
    memory_mode: Literal["low", "high"] = "high",
    mixed_precision: bool = True,
    testing: bool = False,
    *,
    chol_batch_size: int = 64,
) -> EstimatorOps:
    """Build a PT2-UCCSD estimator for a separately chosen propagation guide.

    ``memory_mode="low"`` bounds doubles work to ``chol_batch_size`` vectors;
    ``"high"`` retains the all-Cholesky contractions.
    """
    if sys.walker_kind.lower() != "restricted" or sys.nup < sys.ndn:
        raise ValueError("PT2-UCCSD estimators require a restricted walker with nup >= ndn.")

    cfg = PtuccsdThoulessMeasCfg(
        memory_mode=memory_mode,
        chol_batch_size=chol_batch_size,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )
    return EstimatorOps(
        reference_overlap=reference_overlap_r,
        components=components_ptuccsd_thouless_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=lambda ham_data, trial_data: build_ptuccsd_thouless_meas_ctx(
            ham_data, trial_data, cfg
        ),
        build_estimator_ctx_from_guide=lambda ham_data, trial_data, guide_data, guide_ctx: build_ptuccsd_thouless_meas_ctx(
            ham_data, trial_data, cfg, guide_data=guide_data, guide_ctx=guide_ctx
        ),
    )


__all__ = [
    "PtuccsdThoulessMeasCfg",
    "PtuccsdThoulessMeasCtx",
    "build_ptuccsd_thouless_meas_ctx",
    "components_ptuccsd_thouless_rw_rh",
    "components_ptuccsd_thouless_uw_rh",
    "energy_kernel_rw_rh",
    "energy_kernel_uw_rh",
    "force_bias_kernel_rw_rh",
    "force_bias_kernel_uw_rh",
    "make_ptuccsd_thouless_estimator_ops",
    "make_ptuccsd_thouless_meas_ops",
    "o_pt_components",
]
