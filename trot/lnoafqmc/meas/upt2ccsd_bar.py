"""
The LNO fragment pt2CCSD estimator, unrestricted (uchol) hamiltonian, bar form only.

Ported from afqmc's lno_afqmc/wavefunctions_unrestricted.py (class upt2ccsd:
_calc_e0bar_frag, _t2eorb_tc, _build_measurement_intermediates), on top of the branch's
meas/upt2ccsd_bar_uh.py. As the restricted one (meas/pt2ccsd_bar.py) it measures with
the similarity transformed hamiltonian against the bare reference: each spin's exp(T1_s)
sits on that spin's hamiltonian and walker, alpha and beta each in their own LNO basis
over one shared cholesky index. The kernel returns TRIAL_COMPONENTS = [t2frg, e0frg,
e1frg, e0] per walker and the fragment energy is E_F = <e0frg> + <e1frg> - <t2frg><e0>
(frag_pt2ccsd_energy_fn).

Differences from the branch's energy_kernel_uw_uh_bar, which this otherwise follows line
by line: the doubles are projected on their first occupied index, so the same-spin blocks
are no longer antisymmetric under i <-> j (the exchange half of every T2-contracted
one-body intermediate is formed instead of doubling the direct one) and t2ba is
independent of t2ab; and e0frg needs the transformed fock matrices and the constant
e0t1orb in the ctx.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ...core.ops import MeasOps, k_energy
from ...ham.chol_u import HamCholU
from ...meas.pt2ccsd_chunking import (
    ChunkPlan,
    Pt2ccsdChunkMeasCfg,
    make_chunk_meas_cfg,
    pad_reshape_chol,
    resolve_nchol_chunk,
)
from ...meas.upt2ccsd_uh import e2_0_g, nchol_of
from ...meas.upt2ccsd_uh import plan_chunking_for_run_u as _plan_chunking_for_run_u
from ...trial.upt2ccsd_bar_uh import build_bar_intermediates_u
from ..trial.upt2ccsd import Upt2ccsdTrial, overlap_u
from .pt2ccsd_bar import TRIAL_COMPONENTS, frag_pt2ccsd_energy_fn

__all__ = [
    "TRIAL_COMPONENTS",
    "frag_pt2ccsd_energy_fn",
    "Upt2ccsdFragMeasCtx",
    "ufock_from_chol",
    "e0t1orb_from_chol_u",
    "build_meas_ctx",
    "energy_kernel_uw_uh_bar",
    "make_upt2ccsd_meas_ops",
    "get_upt2ccsd_meas_cfg",
    "plan_chunking_for_run_u",
]

_UPT2CCSD_FRAG_MEAS_CFG_ATTR = "_lno_upt2ccsd_meas_cfg"


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdFragMeasCtx:
    cfg: Pt2ccsdChunkMeasCfg  # static
    nchol_chunk: int  # static; sets the shape the chol tensors are reshaped to
    exp_t1_a: jax.Array  # (norb_a, norb_a)
    exp_t1_b: jax.Array  # (norb_b, norb_b)
    h1_bar_a: jax.Array  # (norb_a, norb_a)
    h1_bar_b: jax.Array  # (norb_b, norb_b)
    chol_bar_a: jax.Array  # (nchol, norb_a, norb_a)
    chol_bar_b: jax.Array  # (nchol, norb_b, norb_b)
    fock_bar_a: jax.Array  # (norb_a, norb_a), fock of H_bar at the reference
    fock_bar_b: jax.Array  # (norb_b, norb_b)
    e0t1orb: jax.Array  # scalar, <exp(T1)HF|H|HF> projected on the fragment

    def tree_flatten(self):
        children = (
            self.exp_t1_a,
            self.exp_t1_b,
            self.h1_bar_a,
            self.h1_bar_b,
            self.chol_bar_a,
            self.chol_bar_b,
            self.fock_bar_a,
            self.fock_bar_b,
            self.e0t1orb,
        )
        return children, (self.cfg, self.nchol_chunk)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        (
            exp_t1_a,
            exp_t1_b,
            h1_bar_a,
            h1_bar_b,
            chol_bar_a,
            chol_bar_b,
            fock_bar_a,
            fock_bar_b,
            e0t1orb,
        ) = children
        return cls(
            cfg=cfg,
            nchol_chunk=nchol_chunk,
            exp_t1_a=exp_t1_a,
            exp_t1_b=exp_t1_b,
            h1_bar_a=h1_bar_a,
            h1_bar_b=h1_bar_b,
            chol_bar_a=chol_bar_a,
            chol_bar_b=chol_bar_b,
            fock_bar_a=fock_bar_a,
            fock_bar_b=fock_bar_b,
            e0t1orb=e0t1orb,
        )


def ufock_from_chol(
    nocc: tuple[int, int], h1: tuple[jax.Array, jax.Array], chol: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """
    Unrestricted fock matrices of (h1_s, chol_s) at the reference occupying each spin's
    first nocc_s orbitals: h1_s + J[a + b] - K[s]. Written for non-symmetric chol (the
    transformed tensors), as afqmc's integral.get_ufock.
    """
    nocc_a, nocc_b = nocc
    chol_a, chol_b = chol
    tr_l = jnp.einsum("gii->g", chol_a[:, :nocc_a, :nocc_a], optimize="optimal") + jnp.einsum(
        "gii->g", chol_b[:, :nocc_b, :nocc_b], optimize="optimal"
    )
    out = []
    for h1_s, chol_s, nocc_s in ((h1[0], chol_a, nocc_a), (h1[1], chol_b, nocc_b)):
        jeff = jnp.einsum("gpq,g->pq", chol_s, tr_l, optimize="optimal")
        keff = jnp.einsum(
            "gpj,gjq->pq", chol_s[:, :, :nocc_s], chol_s[:, :nocc_s, :], optimize="optimal"
        )
        out.append(h1_s + jeff - keff)
    return out[0], out[1]


def _frag_two_body(
    lg_a: jax.Array, lg_b: jax.Array, prjlo_a: jax.Array, prjlo_b: jax.Array
) -> jax.Array:
    """
    1/2 sum_g [ (L_a P_a)(tr L_a + tr L_b) - (L_a L_a P_a) + the same for beta ], with
    lg_s a (k, nocc_s, nocc_s) contraction of the ov cholesky block with an ov matrix.
    The projected two-body form shared by e0t1orb (with t1) and e0frg (with the green).
    """
    tr = jnp.einsum("gjj->g", lg_a, optimize="optimal") + jnp.einsum(
        "gjj->g", lg_b, optimize="optimal"
    )
    coul = jnp.einsum("gik,ik,g->", lg_a, prjlo_a, tr, optimize="optimal") + jnp.einsum(
        "gik,ik,g->", lg_b, prjlo_b, tr, optimize="optimal"
    )
    exch = jnp.einsum("gij,gjk,ik->", lg_a, lg_a, prjlo_a, optimize="optimal") + jnp.einsum(
        "gij,gjk,ik->", lg_b, lg_b, prjlo_b, optimize="optimal"
    )
    return 0.5 * (coul - exch)


def e0t1orb_from_chol_u(ham_data: HamCholU, trial_data: Upt2ccsdTrial) -> jax.Array:
    """
    <exp(T1)HF| H |HF> restricted to the fragment, from the untransformed cholesky
    vectors. afqmc's ham_data['e0t1orb'] (aa + ab + ba + bb).
    """
    nocc_a, nocc_b = trial_data.nocc
    lt1_a = jnp.einsum(
        "ia,gja->gij", trial_data.t1a, ham_data.chol_a[:, :nocc_a, nocc_a:], optimize="optimal"
    )
    lt1_b = jnp.einsum(
        "ia,gja->gij", trial_data.t1b, ham_data.chol_b[:, :nocc_b, nocc_b:], optimize="optimal"
    )
    return _frag_two_body(lt1_a, lt1_b, trial_data.prjlo_a, trial_data.prjlo_b)


def build_meas_ctx(
    ham_data: HamCholU,
    trial_data: Upt2ccsdTrial,
    cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg(),
) -> Upt2ccsdFragMeasCtx:
    if ham_data.basis != "uchol":
        raise ValueError(
            "the unrestricted fragment pt2CCSD MeasOps assume HamCholU.basis == 'uchol'; "
            f"got {ham_data.basis!r}."
        )
    # the branch's builder reads mo_t_a/b and the sizes only, which the fragment trial has too
    bar = build_bar_intermediates_u(ham_data, cast(Any, trial_data))
    fock_bar_a, fock_bar_b = ufock_from_chol(
        trial_data.nocc, (bar["h1_bar_a"], bar["h1_bar_b"]), (bar["chol_bar_a"], bar["chol_bar_b"])
    )
    return Upt2ccsdFragMeasCtx(
        cfg=cfg,
        nchol_chunk=resolve_nchol_chunk(nchol_of(ham_data), cfg.nchol_chunk),
        exp_t1_a=bar["exp_t1_a"],
        exp_t1_b=bar["exp_t1_b"],
        h1_bar_a=bar["h1_bar_a"],
        h1_bar_b=bar["h1_bar_b"],
        chol_bar_a=bar["chol_bar_a"],
        chol_bar_b=bar["chol_bar_b"],
        fock_bar_a=fock_bar_a,
        fock_bar_b=fock_bar_b,
        e0t1orb=e0t1orb_from_chol_u(ham_data, trial_data),
    )


# ---------------------------------------------------------------------------------------
# pieces of the kernel
# ---------------------------------------------------------------------------------------


class _BarWalker(NamedTuple):
    """Chunk independent, per walker intermediates of the fragment bar kernel."""

    green_a: jax.Array  # (nocc_a, norb_a), the half green against the bare reference
    green_b: jax.Array  # (nocc_b, norb_b)
    greenp_a: jax.Array  # (norb_a, nvir_a)
    greenp_b: jax.Array  # (norb_b, nvir_b)
    t2_green_a: jax.Array  # (norb_a, norb_a)
    t2_green_b: jax.Array  # (norb_b, norb_b)
    gt2g: jax.Array  # <P T2>
    e1_0: jax.Array  # <h1>
    e1_2: jax.Array  # <P T2 h1>


def _t2_one_body(
    trial_data: Upt2ccsdTrial,
    greenov: tuple[jax.Array, jax.Array],
    greenrow: tuple[jax.Array, jax.Array],
    greenp: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    The projected-T2 contractions that need no cholesky vector: gt2g = <P T2> and the
    (norb_s, norb_s) matrices t2_green_s = Gp_pb t_iajb G_ia G_jq connected within each
    spin. As the branch's t2_one_body, with the exchange halves of the same-spin blocks
    and the t2ba block kept, since the projection breaks i <-> j. afqmc's _t2eorb_tc.
    """
    t2aa, t2ab, t2ba, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2ba, trial_data.t2bb
    gov_a, gov_b = greenov

    t2g_aa_c = jnp.einsum("iajb,ia->jb", t2aa, gov_a, optimize="optimal") / 4
    t2g_aa_e = jnp.einsum("iajb,ja->ib", t2aa, gov_a, optimize="optimal") / 4
    t2g_bb_c = jnp.einsum("iajb,ia->jb", t2bb, gov_b, optimize="optimal") / 4
    t2g_bb_e = jnp.einsum("iajb,ja->ib", t2bb, gov_b, optimize="optimal") / 4
    t2g_ab_a = jnp.einsum("iajb,ia->jb", t2ab, gov_a, optimize="optimal") / 2  # beta ov
    t2g_ab_b = jnp.einsum("iajb,jb->ia", t2ab, gov_b, optimize="optimal") / 2  # alpha ov
    t2g_ba_a = jnp.einsum("iajb,jb->ia", t2ba, gov_a, optimize="optimal") / 2  # beta ov
    t2g_ba_b = jnp.einsum("iajb,ia->jb", t2ba, gov_b, optimize="optimal") / 2  # alpha ov

    gt2g_aa = jnp.einsum("jb,jb->", t2g_aa_c, gov_a, optimize="optimal")
    gt2g_bb = jnp.einsum("jb,jb->", t2g_bb_c, gov_b, optimize="optimal")
    gt2g_ab = jnp.einsum("jb,jb->", t2g_ab_a, gov_b, optimize="optimal")
    gt2g_ba = jnp.einsum("jb,jb->", t2g_ba_b, gov_a, optimize="optimal")
    gt2g = 2 * (gt2g_aa + gt2g_bb) + (gt2g_ab + gt2g_ba)

    # 2 * (same spin direct - exchange) + (both opposite spin blocks, contracted onto this spin)
    t2_green_a = greenp[0] @ (2 * (t2g_aa_c - t2g_aa_e) + t2g_ab_b + t2g_ba_b).T @ greenrow[0]
    t2_green_b = greenp[1] @ (2 * (t2g_bb_c - t2g_bb_e) + t2g_ba_a + t2g_ab_a).T @ greenrow[1]
    return gt2g, t2_green_a, t2_green_b


def _l2t2_g(
    glgp_a: jax.Array, glgp_b: jax.Array, t2_r: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
) -> jax.Array:
    """
    e2_2_3 per cholesky vector: 1/2 (L t2aa L + L t2ab L + L t2ba L + L t2bb L), glgp_s the
    (k, nocc_s, nvir_s) contraction of gl with greenp.
    """
    t2aa_r, t2ab_r, t2ba_r, t2bb_r = t2_r
    lt2_aa = jnp.einsum("gia,iajb->gjb", glgp_a, t2aa_r, optimize="optimal")
    lt2_ab = jnp.einsum("gia,iajb->gjb", glgp_a, t2ab_r, optimize="optimal")
    lt2_ba = jnp.einsum("gia,iajb->gjb", glgp_b, t2ba_r, optimize="optimal")
    lt2_bb = jnp.einsum("gia,iajb->gjb", glgp_b, t2bb_r, optimize="optimal")
    l2t2_aa = jnp.einsum("gjb,gjb->g", lt2_aa, glgp_a, optimize="optimal")
    l2t2_ab = jnp.einsum("gjb,gjb->g", lt2_ab, glgp_b, optimize="optimal")
    l2t2_ba = jnp.einsum("gjb,gjb->g", lt2_ba, glgp_a, optimize="optimal")
    l2t2_bb = jnp.einsum("gjb,gjb->g", lt2_bb, glgp_b, optimize="optimal")
    return 0.5 * (l2t2_aa + l2t2_ab + l2t2_ba + l2t2_bb)


def _bar_walker(
    walker: tuple[jax.Array, jax.Array],
    meas_ctx: Upt2ccsdFragMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> _BarWalker:
    """
    Transform the walker with exp(T1) and build everything that does not touch a cholesky
    vector. As the branch's _bar_walker, with the fragment _t2_one_body.
    """
    h1_a, h1_b = meas_ctx.h1_bar_a, meas_ctx.h1_bar_b
    nocc_a, nocc_b = trial_data.nocc
    nvir_a, nvir_b = trial_data.nvir
    wu, wd = walker
    walker_bar_a = meas_ctx.exp_t1_a @ wu
    walker_bar_b = meas_ctx.exp_t1_b @ wd

    green_a = (walker_bar_a @ jnp.linalg.inv(walker_bar_a[:nocc_a, :])).T
    green_b = (walker_bar_b @ jnp.linalg.inv(walker_bar_b[:nocc_b, :])).T
    greenp_a = jnp.vstack((green_a[:, nocc_a:], -jnp.eye(nvir_a, dtype=green_a.dtype)))
    greenp_b = jnp.vstack((green_b[:, nocc_b:], -jnp.eye(nvir_b, dtype=green_b.dtype)))

    # one body energy. only the occupied rows of h1 meet a nonzero row of the green
    e1_0 = jnp.einsum("pq,pq->", h1_a[:nocc_a, :], green_a, optimize="optimal") + jnp.einsum(
        "pq,pq->", h1_b[:nocc_b, :], green_b, optimize="optimal"
    )

    gt2g, t2_green_a, t2_green_b = _t2_one_body(
        trial_data,
        greenov=(green_a[:, nocc_a:], green_b[:, nocc_b:]),
        greenrow=(green_a, green_b),
        greenp=(greenp_a, greenp_b),
    )
    e1_2_1 = e1_0 * gt2g
    e1_2_2 = -jnp.einsum("pq,pq->", h1_a, t2_green_a, optimize="optimal") - jnp.einsum(
        "pq,pq->", h1_b, t2_green_b, optimize="optimal"
    )

    return _BarWalker(
        green_a=green_a,
        green_b=green_b,
        greenp_a=greenp_a,
        greenp_b=greenp_b,
        t2_green_a=t2_green_a,
        t2_green_b=t2_green_b,
        gt2g=gt2g,
        e1_0=e1_0,
        e1_2=e1_2_1 + e1_2_2,
    )


def _bar_chunk_terms(
    chol_a_c: jax.Array,
    chol_b_c: jax.Array,
    bw: _BarWalker,
    trial_data: Upt2ccsdTrial,
    t2_r: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    rtype: Any,
    ctype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    The two-body terms of one chunk of k cholesky vectors, per vector, as the branch's
    _bar_chunk_terms (e2_0 exact in complex128, the three T2-contracted ones in ctype),
    with the fragment _l2t2_g.
    """
    nocc_a, nocc_b = trial_data.nocc

    gl_a = jnp.einsum(
        "ir,gqr->giq", bw.green_a, chol_a_c, optimize="optimal"
    )  # (k, nocc_a, norb_a)
    gl_b = jnp.einsum(
        "ir,gqr->giq", bw.green_b, chol_b_c, optimize="optimal"
    )  # (k, nocc_b, norb_b)
    e2_0_c, tr_gl = e2_0_g(gl_a[:, :, :nocc_a], gl_b[:, :, :nocc_b])

    # e2_2_2_1: only the trace of chol . t2_green is needed
    lt2g = jnp.einsum(
        "gpr,pr->g", chol_a_c.astype(rtype), bw.t2_green_a.astype(ctype), optimize="optimal"
    ) + jnp.einsum(
        "gpr,pr->g", chol_b_c.astype(rtype), bw.t2_green_b.astype(ctype), optimize="optimal"
    )
    e2_2_2_1_g = -lt2g.astype(ctype) * tr_gl.astype(ctype)

    # e2_2_2_2: only the occupied rows of chol . t2_green meet a nonzero row of gl
    lt2_green_a = jnp.einsum(
        "gir,qr->giq",
        chol_a_c[:, :nocc_a, :].astype(rtype),
        bw.t2_green_a.astype(ctype),
        optimize="optimal",
    )
    lt2_green_b = jnp.einsum(
        "gir,qr->giq",
        chol_b_c[:, :nocc_b, :].astype(rtype),
        bw.t2_green_b.astype(ctype),
        optimize="optimal",
    )
    e2_2_2_2_g = jnp.einsum(
        "giq,giq->g", gl_a.astype(ctype), lt2_green_a.astype(ctype), optimize="optimal"
    ) + jnp.einsum("giq,giq->g", gl_b.astype(ctype), lt2_green_b.astype(ctype), optimize="optimal")

    # e2_2_3
    glgp_a = jnp.einsum(
        "giq,qa->gia", gl_a.astype(ctype), bw.greenp_a.astype(ctype), optimize="optimal"
    )
    glgp_b = jnp.einsum(
        "giq,qa->gia", gl_b.astype(ctype), bw.greenp_b.astype(ctype), optimize="optimal"
    )
    e2_2_3_g = _l2t2_g(glgp_a, glgp_b, t2_r)

    return e2_0_c.astype(jnp.complex128), e2_2_2_1_g, e2_2_2_2_g, e2_2_3_g


def _e0bar_frag(
    bw: _BarWalker, meas_ctx: Upt2ccsdFragMeasCtx, trial_data: Upt2ccsdTrial
) -> jax.Array:
    """
    The projected correlation part of <HF| H_bar |walker_bar> / <HF|walker_bar>:
    e0t1orb + the projected fock ov term + the projected ov-ov two-body term. afqmc's
    _calc_e0bar_frag.
    """
    nocc_a, nocc_b = trial_data.nocc
    prjlo_a, prjlo_b = trial_data.prjlo_a, trial_data.prjlo_b
    gov_a = bw.green_a[:, nocc_a:]
    gov_b = bw.green_b[:, nocc_b:]

    e1 = jnp.einsum(
        "ia,ik,ka->", gov_a, prjlo_a, meas_ctx.fock_bar_a[:nocc_a, nocc_a:], optimize="optimal"
    ) + jnp.einsum(
        "ia,ik,ka->", gov_b, prjlo_b, meas_ctx.fock_bar_b[:nocc_b, nocc_b:], optimize="optimal"
    )

    chol_ov_a, _, _, _ = pad_reshape_chol(
        meas_ctx.chol_bar_a[:, :nocc_a, nocc_a:], meas_ctx.nchol_chunk
    )
    chol_ov_b, _, _, _ = pad_reshape_chol(
        meas_ctx.chol_bar_b[:, :nocc_b, nocc_b:], meas_ctx.nchol_chunk
    )

    def scanned_fun(carry, x):
        lg_a = jnp.einsum("gia,ka->gik", x[0], gov_a, optimize="optimal")
        lg_b = jnp.einsum("gia,ka->gik", x[1], gov_b, optimize="optimal")
        return carry + _frag_two_body(lg_a, lg_b, prjlo_a, prjlo_b), None

    e2, _ = lax.scan(scanned_fun, jnp.zeros((), dtype=gov_a.dtype), (chol_ov_a, chol_ov_b))
    return meas_ctx.e0t1orb + e1 + e2


# ---------------------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------------------


def energy_kernel_uw_uh_bar(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdFragMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> jax.Array:
    """
    The fragment pt2CCSD estimator for one unrestricted walker, [t2frg, e0frg, e1frg, e0].

    Follows the branch's energy_kernel_uw_uh_bar: each spin's similarity transformed
    hamiltonian in meas_ctx, the walker exp_t1_s @ walker_s, and the (nocc_s, norb_s)
    half greens against the bare reference. ham_data is unused; the tensors are in
    meas_ctx.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    bw = _bar_walker(walker, meas_ctx, trial_data)
    e0frg = _e0bar_frag(bw, meas_ctx, trial_data)

    chol_a, _, _, _ = pad_reshape_chol(meas_ctx.chol_bar_a, meas_ctx.nchol_chunk)
    chol_b, _, _, _ = pad_reshape_chol(meas_ctx.chol_bar_b, meas_ctx.nchol_chunk)

    t2_r = (
        trial_data.t2aa.astype(rtype),
        trial_data.t2ab.astype(rtype),
        trial_data.t2ba.astype(rtype),
        trial_data.t2bb.astype(rtype),
    )

    def scanned_fun(carry, x):
        e2_0_c, e2_2_2_1_c, e2_2_2_2_c, e2_2_3_c = _bar_chunk_terms(
            x[0], x[1], bw, trial_data, t2_r, rtype, ctype
        )
        carry[0] += jnp.sum(e2_0_c)
        carry[1] += jnp.sum(e2_2_2_1_c).astype(c128)
        carry[2] += jnp.sum(e2_2_2_2_c).astype(c128)
        carry[3] += jnp.sum(e2_2_3_c).astype(c128)
        return carry, None

    zero = jnp.zeros((), dtype=c128)
    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [zero, zero, zero, zero], (chol_a, chol_b)
    )

    e2_2 = e2_0 * bw.gt2g + e2_2_2_1 + e2_2_2_2 + e2_2_3

    t2frg = bw.gt2g  # <HF| P T2 |walker_bar> / <HF|walker_bar>
    e0 = bw.e1_0 + e2_0  # <HF| H_bar |walker_bar> / <HF|walker_bar>
    e1frg = bw.e1_2 + e2_2  # <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>

    return jnp.stack([t2frg, e0frg, e1frg, e0])


def make_upt2ccsd_meas_ops(
    sys: Any,
    *,
    mixed_precision: bool = True,
    testing: bool = False,
    nchol_chunk: int | None = None,
) -> MeasOps:
    """
    MeasOps of the unrestricted fragment pt2CCSD trial: overlap_u and the "energy" kernel
    that returns TRIAL_COMPONENTS per walker. The same signature as the branch's
    make_upt2ccsd_bar_meas_ops, so setup_mixed drives it unchanged.
    """
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError(
            "the unrestricted fragment pt2CCSD MeasOps require walker_kind='unrestricted', "
            f"got: {sys.walker_kind}"
        )
    cfg = make_chunk_meas_cfg(
        mixed_precision=mixed_precision, testing=testing, nchol_chunk=nchol_chunk
    )
    meas_ops = MeasOps(
        overlap=overlap_u,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel_uw_uh_bar},
    )
    object.__setattr__(meas_ops, _UPT2CCSD_FRAG_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_upt2ccsd_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _UPT2CCSD_FRAG_MEAS_CFG_ATTR, None)
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
) -> ChunkPlan:
    """
    The recipe's memory hook, with the branch's unrestricted bar memory model. The
    fragment ctx adds the fock matrices and a constant, and the trial the t2ba block (the
    size of t2ab); both are small against the cholesky copies the model already counts.
    """
    return _plan_chunking_for_run_u(
        sys,
        ham_data,
        cast(Any, trial_data),
        n_walkers=n_walkers,
        budget_bytes=budget_bytes,
        n_chunks=n_chunks,
        nchol_chunk=nchol_chunk,
        mixed_precision=mixed_precision,
        n_devices=n_devices,
        bar=True,
    )
