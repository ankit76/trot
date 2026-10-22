"""
The unrestricted pt2CCSD estimator (upt2ccsd) on the unrestricted (uchol) hamiltonian.

Per walker (wa, wb), against the trial exp(T1)|HF> in each spin's own basis,

    t2 = <exp(T1)HF| T2 |phi> / <exp(T1)HF|phi>
    e0 = <exp(T1)HF| H  |phi> / <exp(T1)HF|phi>
    e1 = <exp(T1)HF| T2 H |phi> / <exp(T1)HF|phi>

combined by combine_first_order_energy: E = h0 + <e0> + <e1> - <t2><e0>. The two body
sum is scanned over chunks of nchol_chunk cholesky vectors (meas/pt2ccsd_chunking.py);
alpha and beta share the cholesky index, so the coulomb trace sees both spins while the
exchange term stays within a spin, which is how the opposite spin interaction is
recovered from L_a and L_b.

The greens functions here are full (norb_s, norb_s) matrices. The bar variant in
meas/upt2ccsd_bar_uh.py reaches the same components from half greens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy
from ..ham.chol_u import HamCholU
from ..trial.upt2ccsd_bar_uh import build_bar_intermediates_u
from ..trial.upt2ccsd_uh import Upt2ccsdTrial, overlap_u
from .pt2ccsd import combine_first_order_energy
from .pt2ccsd_chunking import (
    ChunkPlan,
    Pt2ccsdChunkMeasCfg,
    make_chunk_meas_cfg,
    pad_reshape_chol,
    plan_pt2ccsd_chunking,
    resolve_nchol_chunk,
    upt2ccsd_memory_model,
)

_UPT2CCSD_MEAS_CFG_ATTR = "_upt2ccsd_meas_cfg"

COMPONENTS: tuple[str, ...] = ("theta", "electronic_0", "h_t")


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdMeasCtx:
    cfg: Pt2ccsdChunkMeasCfg  # static
    nchol_chunk: int  # static
    # similarity transformed intermediates, only built for the bar kernel
    exp_t1_a: jax.Array | None = None
    exp_t1_b: jax.Array | None = None
    h1_bar_a: jax.Array | None = None
    h1_bar_b: jax.Array | None = None
    chol_bar_a: jax.Array | None = None
    chol_bar_b: jax.Array | None = None

    def tree_flatten(self):
        children = (
            self.exp_t1_a,
            self.exp_t1_b,
            self.h1_bar_a,
            self.h1_bar_b,
            self.chol_bar_a,
            self.chol_bar_b,
        )
        return children, (self.cfg, self.nchol_chunk)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        exp_t1_a, exp_t1_b, h1_bar_a, h1_bar_b, chol_bar_a, chol_bar_b = children
        return cls(
            cfg=cfg,
            nchol_chunk=nchol_chunk,
            exp_t1_a=exp_t1_a,
            exp_t1_b=exp_t1_b,
            h1_bar_a=h1_bar_a,
            h1_bar_b=h1_bar_b,
            chol_bar_a=chol_bar_a,
            chol_bar_b=chol_bar_b,
        )


def nchol_of(ham_data: HamCholU) -> int:
    nchol = ham_data.nchol
    return int(nchol) if nchol is not None else int(ham_data.chol_a.shape[0])


def build_meas_ctx(
    ham_data: HamCholU,
    trial_data: Upt2ccsdTrial,
    cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg(),
    *,
    bar: bool = False,
) -> Upt2ccsdMeasCtx:
    if ham_data.basis != "uchol":
        raise ValueError(
            "unrestricted pt2CCSD MeasOps assume the unrestricted hamiltonian, "
            f"HamCholU.basis == 'uchol'; got {ham_data.basis!r}."
        )
    nchol_chunk = resolve_nchol_chunk(nchol_of(ham_data), cfg.nchol_chunk)
    bar_tensors = build_bar_intermediates_u(ham_data, trial_data) if bar else {}
    return Upt2ccsdMeasCtx(
        cfg=cfg,
        nchol_chunk=nchol_chunk,
        exp_t1_a=bar_tensors.get("exp_t1_a"),
        exp_t1_b=bar_tensors.get("exp_t1_b"),
        h1_bar_a=bar_tensors.get("h1_bar_a"),
        h1_bar_b=bar_tensors.get("h1_bar_b"),
        chol_bar_a=bar_tensors.get("chol_bar_a"),
        chol_bar_b=bar_tensors.get("chol_bar_b"),
    )


# ---------------------------------------------------------------------------------------
# pieces shared with the bar kernel
# ---------------------------------------------------------------------------------------


def t2_one_body(
    trial_data: Upt2ccsdTrial,
    greenov: tuple[jax.Array, jax.Array],
    greenrow: tuple[jax.Array, jax.Array],
    greenp: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    The T2 contractions that need no cholesky vector.

    greenov  (nocc_s, nvir_s)  the occupied-virtual block of the green
    greenrow (nocc_s, norb_s)  its occupied rows
    greenp   (norb_s, nvir_s)  (green - 1) restricted to the virtual columns

    Returns gt2g = <T2>, and t2_green_s = Gp_pb t_iajb G_ia G_jq connected within each
    spin, the (norb_s, norb_s) matrix every T2 term with a one or two body operator goes
    through.
    """
    t2aa, t2ab, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2bb
    gov_a, gov_b = greenov

    t2g_a = jnp.einsum("iajb,ia->jb", t2aa, gov_a, optimize="optimal") / 4
    t2g_b = jnp.einsum("iajb,ia->jb", t2bb, gov_b, optimize="optimal") / 4
    t2g_ab_a = jnp.einsum("iajb,jb->ia", t2ab, gov_b, optimize="optimal")
    t2g_ab_b = jnp.einsum("iajb,ia->jb", t2ab, gov_a, optimize="optimal")

    # t_iajb (G_ia G_jb - G_ib G_ja)
    gt2g_a = jnp.einsum("jb,jb->", t2g_a, gov_a, optimize="optimal")
    gt2g_b = jnp.einsum("jb,jb->", t2g_b, gov_b, optimize="optimal")
    gt2g_ab = jnp.einsum("ia,ia->", t2g_ab_a, gov_a, optimize="optimal")
    gt2g = 2 * (gt2g_a + gt2g_b) + gt2g_ab

    # 4 * (same spin) + (opposite spin, contracted onto this one)
    t2_green_a = greenp[0] @ (4 * t2g_a + t2g_ab_a).T @ greenrow[0]
    t2_green_b = greenp[1] @ (4 * t2g_b + t2g_ab_b).T @ greenrow[1]
    return gt2g, t2_green_a, t2_green_b


def l2t2_g(
    glgp_a: jax.Array, glgp_b: jax.Array, t2_r: tuple[jax.Array, jax.Array, jax.Array]
) -> jax.Array:
    """
    e2_2_3 per cholesky vector: 1/2 L t2aa L + 1/2 L t2bb L + L t2ab L, with glgp_s the
    (k, nocc_s, nvir_s) contraction of gl with greenp. These "iajb" contractions cost
    nocc^2 nvir^2 per vector and dominate the kernel.
    """
    t2aa_r, t2ab_r, t2bb_r = t2_r
    lt2_aa = jnp.einsum("gia,iajb->gjb", glgp_a, t2aa_r, optimize="optimal")
    lt2_bb = jnp.einsum("gia,iajb->gjb", glgp_b, t2bb_r, optimize="optimal")
    lt2_ab = jnp.einsum("gia,iajb->gjb", glgp_a, t2ab_r, optimize="optimal")
    l2t2_aa = 0.5 * jnp.einsum("gjb,gjb->g", lt2_aa, glgp_a, optimize="optimal")
    l2t2_bb = 0.5 * jnp.einsum("gjb,gjb->g", lt2_bb, glgp_b, optimize="optimal")
    l2t2_ab = jnp.einsum("gjb,gjb->g", lt2_ab, glgp_b, optimize="optimal")
    return l2t2_aa + l2t2_bb + l2t2_ab


def e2_0_g(gl_a: jax.Array, gl_b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    <h2> per cholesky vector from the square (k, n_s, n_s) blocks of gl each spin
    contributes, together with the spin summed trace: coulomb sees both spins, exchange
    only its own.
    """
    tr_gl = jnp.einsum("gpp->g", gl_a, optimize="optimal") + jnp.einsum(
        "gpp->g", gl_b, optimize="optimal"
    )
    ex_gl = jnp.einsum("gpq,gqp->g", gl_a, gl_a, optimize="optimal") + jnp.einsum(
        "gpq,gqp->g", gl_b, gl_b, optimize="optimal"
    )
    return (tr_gl * tr_gl - ex_gl) / 2.0, tr_gl


# ---------------------------------------------------------------------------------------
# the chunked kernel
# ---------------------------------------------------------------------------------------


def energy_kernel_uw_uh_chunk(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> jax.Array:
    """
    (t2, e0, e1) of one unrestricted walker against exp(T1)|HF> in each spin's basis,
    the two body sum scanned over chunks of meas_ctx.nchol_chunk cholesky vectors. The
    T2 contractions run in the mixed dtypes of meas_ctx.cfg; e2_0 and every partial sum
    stay in complex128.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    nocc_a, nocc_b = trial_data.nocc
    wu, wd = walker
    mo_a, mo_b = trial_data.mo_t_a, trial_data.mo_t_b
    h1_a, h1_b = ham_data.h1_a, ham_data.h1_b

    # full green's function G_pq per spin
    green_a = (wu @ jnp.linalg.inv(mo_a.T @ wu) @ mo_a.T).T
    green_b = (wd @ jnp.linalg.inv(mo_b.T @ wd) @ mo_b.T).T
    greenp_a = (green_a - jnp.eye(green_a.shape[0]))[:, nocc_a:]
    greenp_b = (green_b - jnp.eye(green_b.shape[0]))[:, nocc_b:]

    # <exp(T1)HF|h1|walker>/<exp(T1)HF|walker>
    e1_0 = jnp.einsum("pq,pq->", h1_a, green_a, optimize="optimal") + jnp.einsum(
        "pq,pq->", h1_b, green_b, optimize="optimal"
    )

    # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>
    gt2g, t2_green_a, t2_green_b = t2_one_body(
        trial_data,
        greenov=(green_a[:nocc_a, nocc_a:], green_b[:nocc_b, nocc_b:]),
        greenrow=(green_a[:nocc_a, :], green_b[:nocc_b, :]),
        greenp=(greenp_a, greenp_b),
    )
    e1_2_1 = e1_0 * gt2g
    e1_2_2 = -jnp.einsum("pq,pq->", h1_a, t2_green_a, optimize="optimal") - jnp.einsum(
        "pq,pq->", h1_b, t2_green_b, optimize="optimal"
    )
    e1_2 = e1_2_1 + e1_2_2

    # <exp(T1)HF|T2 h2|walker>/<exp(T1)HF|walker>, chunked over the shared cholesky index.
    # both spins are padded the same way, so the leftover contributes to neither
    chol_a, _, _, _ = pad_reshape_chol(ham_data.chol_a, meas_ctx.nchol_chunk)
    chol_b, _, _, _ = pad_reshape_chol(ham_data.chol_b, meas_ctx.nchol_chunk)

    t2_r = (
        trial_data.t2aa.astype(rtype),
        trial_data.t2ab.astype(rtype),
        trial_data.t2bb.astype(rtype),
    )

    def scanned_fun(carry, x):
        chol_a_c, chol_b_c = x  # (k, norb_a, norb_a), (k, norb_b, norb_b)

        # e2_0
        gl_a = jnp.einsum("pr,gqr->gpq", green_a, chol_a_c, optimize="optimal")
        gl_b = jnp.einsum("pr,gqr->gpq", green_b, chol_b_c, optimize="optimal")
        e2_0_c, tr_gl = e2_0_g(gl_a, gl_b)
        carry[0] += jnp.sum(e2_0_c).astype(c128)

        # e2_2_2_1
        lt2g_a = jnp.einsum("gpr,qr->gpq", chol_a_c, t2_green_a, optimize="optimal")
        lt2g_b = jnp.einsum("gpr,qr->gpq", chol_b_c, t2_green_b, optimize="optimal")
        tr_lt2g = jnp.einsum("gpp->g", lt2g_a, optimize="optimal") + jnp.einsum(
            "gpp->g", lt2g_b, optimize="optimal"
        )
        carry[1] += -jnp.sum(tr_lt2g.astype(ctype) * tr_gl.astype(ctype)).astype(c128)

        # e2_2_2_2
        carry[2] += (
            jnp.einsum("gpq,gpq->", gl_a.astype(ctype), lt2g_a.astype(ctype), optimize="optimal")
            + jnp.einsum("gpq,gpq->", gl_b.astype(ctype), lt2g_b.astype(ctype), optimize="optimal")
        ).astype(c128)

        # e2_2_3
        glgp_a = jnp.einsum("giq,qa->gia", gl_a[:, :nocc_a, :], greenp_a, optimize="optimal")
        glgp_b = jnp.einsum("giq,qa->gia", gl_b[:, :nocc_b, :], greenp_b, optimize="optimal")
        carry[3] += jnp.sum(l2t2_g(glgp_a.astype(ctype), glgp_b.astype(ctype), t2_r)).astype(c128)

        return carry, None

    zero = jnp.zeros((), dtype=c128)
    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [zero, zero, zero, zero], (chol_a, chol_b)
    )

    e2_2_1 = e2_0 * gt2g
    e2_2 = e2_2_1 + e2_2_2_1 + e2_2_2_2 + e2_2_3

    t2 = gt2g  # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
    e0 = e1_0 + e2_0  # <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
    e1 = e1_2 + e2_2  # <exp(T1)HF|T2 (h1+h2)|walker>/<exp(T1)HF|walker>

    return jnp.stack([t2, e0, e1])


def energy_kernel_uw_uh_chunk_scalar(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> jax.Array:
    """The combined local energy of one walker, for checks against the other kernels."""
    return combine_first_order_energy(
        ham_data.h0, energy_kernel_uw_uh_chunk(walker, ham_data, meas_ctx, trial_data)
    )


def _check_unrestricted(sys: Any) -> None:
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError(
            "unrestricted pt2CCSD MeasOps require walker_kind='unrestricted', "
            f"got: {sys.walker_kind}"
        )


def make_upt2ccsd_meas_ops(
    sys: Any,
    *,
    mixed_precision: bool = True,
    testing: bool = False,
    nchol_chunk: int | None = None,
) -> MeasOps:
    """
    MeasOps of the upt2ccsd trial on the uchol hamiltonian: overlap_u and the "energy"
    kernel returning the three components per walker (a measurement trial for the mixed
    driver, not a guide). nchol_chunk caps the cholesky vectors per scan step; the mixed
    driver's chunk plan sets it from the memory budget when it is None.
    """
    _check_unrestricted(sys)
    cfg = make_chunk_meas_cfg(
        mixed_precision=mixed_precision, testing=testing, nchol_chunk=nchol_chunk
    )
    meas_ops = MeasOps(
        overlap=overlap_u,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel_uw_uh_chunk},
    )
    object.__setattr__(meas_ops, _UPT2CCSD_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_upt2ccsd_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _UPT2CCSD_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, Pt2ccsdChunkMeasCfg) else None


def plan_chunking_for_run_u(
    sys: Any,
    ham_data: HamCholU,
    trial_data: Upt2ccsdTrial,
    *,
    n_walkers: int,
    budget_bytes: int,
    n_chunks: int = 1,
    nchol_chunk: int | None = None,
    mixed_precision: bool = True,
    n_devices: int = 1,
    bar: bool = False,
) -> ChunkPlan:
    """Build the memory model of the chunk (or bar) kernel for a run and plan its chunking."""
    nchol = nchol_of(ham_data)
    model = upt2ccsd_memory_model(
        norb=trial_data.norb,
        nocc=trial_data.nocc,
        nchol=nchol,
        n_walkers=n_walkers,
        real_bytes=4 if mixed_precision else 8,
        complex_bytes=8 if mixed_precision else 16,
        bar=bar,
    )
    return plan_pt2ccsd_chunking(
        model,
        n_walkers=n_walkers,
        nchol=nchol,
        budget_bytes=int(budget_bytes),
        n_chunks=n_chunks,
        nchol_chunk=nchol_chunk,
        n_devices=n_devices,
    )
