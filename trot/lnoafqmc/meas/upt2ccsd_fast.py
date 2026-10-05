"""
The unrestricted LNO fragment pt2CCSD estimator with the fragment projectors in factored
form ("upt2ccsd_fast"): the same four numbers per walker as meas/upt2ccsd_bar.py
(TRIAL_COMPONENTS, combined by frag_pt2ccsd_energy_fn), to roundoff, with the
contraction path of the restricted pt2ccsd_fast kernel (meas/pt2ccsd_fast.py) applied
per spin.

Sizes: nlo = L local orbitals, nocc_s = o_s, nvir_s = v_s, norb_s = n_s, k cholesky
vectors per scan step; costs are per walker and per step.

Projectors. prjlo_s = U_s U_s^H with U_s = <act_occ_s|lo_s> of shape (o_s, L). Every
projected quantity is built as (something)_{I...} = sum_i U_{iI} (something)_{i...} on
both sides and I is contracted last, so the doubles (trial/upt2ccsd_fast.py) carry the
local index on their first slot and the T2 contractions cost L o v^2 per block instead of
o^2 v^2. The same-spin blocks are antisymmetric in (a, b), which the bar kernel already
uses to have one contraction per index pair (the exchange term is the direct term with
a and b swapped); that carries over unchanged, and the four blocks aa, ab, ba, bb each
need one contraction over their first (local) pair.

Green's functions. green_s = [1 | gf_s], so gl_oo,s = L_oo,s^T + gf_s L_ov,s^T and
gl_ov,s = L_vo,s^T + gf_s L_vv,s^T are formed from the blocks of L_g,s (o_s n_s v_s), and
glgp_s = gl_oo,s gf_s - gl_ov,s (o_s^2 v_s). The e2_2_2 terms go through glgp_s and the
(o_s, v_s) one-body intermediate t2g_s, so no (n_s, n_s) t2_green enters the scan.

Terms, with gc_s = U_s^H gf_s, gu_s = U_s^T gf_s, fu_s = U_s^H f_ov,s and cu_g,s = U_s^T L_ov,g,s
(the last two built once in the ctx):

    e0frg fock     sum_s <gu_s, fu_s>
    e0frg coulomb  1/2 sum_g (sum_s <cu_g,s, gc_s>) (tr lg_a + tr lg_b), tr lg_s = <L_ov,s, gf_s>
    e0frg exchange 1/2 sum_g sum_s <A_s, B_s^T>, A_s = cu_g,s gf_s^T, B_s = L_ov,s gc_s^T  (L o v per spin)
    t2g_s          the (o_s, v_s) one-body T2 intermediate, two contractions per block
    e2_2_2_1       -tr(gl) sum_s <t2g_s, glgp_s>                                (o v)
    e2_2_2_2       sum_s sum_{ij} (glgp_s t2g_s^T)_{ij} gl_oo,s_{ji}             (o^2 v)
    e2_2_3         1/2 sum_g sum_blocks <glgpu_(first spin) t2*_u, glgp_(second spin)>
                   with glgpu_s = U_s^H glgp_s                                    (L o v^2 per block)

Precision: everything that carries T2 (t2g_s and the one-body term built from it, the
e2_2_2 terms, the T2 contraction) runs in the mixed dtypes of the cfg; the greens
function pieces, e0 and the projected e0frg stay in double.
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
from ...trial.upt2ccsd_bar_uh import build_bar_intermediates_u
from ..trial.upt2ccsd_fast import Upt2ccsdFastTrial, overlap_u
from .pt2ccsd_bar import TRIAL_COMPONENTS, frag_pt2ccsd_energy_fn
from .upt2ccsd_bar import e0t1orb_from_chol_u, plan_chunking_for_run_u, ufock_from_chol

__all__ = [
    "TRIAL_COMPONENTS",
    "frag_pt2ccsd_energy_fn",
    "Upt2ccsdFastMeasCtx",
    "build_meas_ctx",
    "energy_kernel_uw_uh_fast",
    "make_upt2ccsd_fast_meas_ops",
    "get_upt2ccsd_fast_meas_cfg",
    "plan_chunking_for_run_u_fast",
]

_UPT2CCSD_FAST_MEAS_CFG_ATTR = "_lno_upt2ccsd_fast_meas_cfg"


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdFastMeasCtx:
    cfg: Pt2ccsdChunkMeasCfg  # static
    nchol_chunk: int  # static; sets the shape the chol tensors are reshaped to
    exp_t1_a: jax.Array  # (norb_a, norb_a)
    exp_t1_b: jax.Array  # (norb_b, norb_b)
    h1_bar_a: jax.Array  # (norb_a, norb_a)
    h1_bar_b: jax.Array  # (norb_b, norb_b)
    chol_bar_a: jax.Array  # (nchol, norb_a, norb_a)
    chol_bar_b: jax.Array  # (nchol, norb_b, norb_b)
    chol_ov_u_a: jax.Array  # (nchol, nlo, nvir_a): U_a^T applied to the ov block of chol_bar_a
    chol_ov_u_b: jax.Array  # (nchol, nlo, nvir_b)
    fock_ov_u_a: jax.Array  # (nlo, nvir_a): U_a^H applied to the ov block of fock_bar_a
    fock_ov_u_b: jax.Array  # (nlo, nvir_b)
    e0t1orb: jax.Array  # scalar, <exp(T1)HF|H|HF> projected on the fragment

    def tree_flatten(self):
        children = (
            self.exp_t1_a,
            self.exp_t1_b,
            self.h1_bar_a,
            self.h1_bar_b,
            self.chol_bar_a,
            self.chol_bar_b,
            self.chol_ov_u_a,
            self.chol_ov_u_b,
            self.fock_ov_u_a,
            self.fock_ov_u_b,
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
            chol_ov_u_a,
            chol_ov_u_b,
            fock_ov_u_a,
            fock_ov_u_b,
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
            chol_ov_u_a=chol_ov_u_a,
            chol_ov_u_b=chol_ov_u_b,
            fock_ov_u_a=fock_ov_u_a,
            fock_ov_u_b=fock_ov_u_b,
            e0t1orb=e0t1orb,
        )


def build_meas_ctx(
    ham_data: HamCholU,
    trial_data: Upt2ccsdFastTrial,
    cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg(),
) -> Upt2ccsdFastMeasCtx:
    if ham_data.basis != "uchol":
        raise ValueError(
            "the unrestricted fragment pt2CCSD MeasOps assume HamCholU.basis == 'uchol'; "
            f"got {ham_data.basis!r}."
        )
    nocc_a, nocc_b = trial_data.nocc
    # the branch's builder reads mo_t_a/b and the sizes only, which the fragment trial has too
    bar = build_bar_intermediates_u(ham_data, cast(Any, trial_data))
    fock_bar_a, fock_bar_b = ufock_from_chol(
        trial_data.nocc, (bar["h1_bar_a"], bar["h1_bar_b"]), (bar["chol_bar_a"], bar["chol_bar_b"])
    )
    u_a, u_b = trial_data.u_a, trial_data.u_b
    return Upt2ccsdFastMeasCtx(
        cfg=cfg,
        nchol_chunk=resolve_nchol_chunk(nchol_of(ham_data), cfg.nchol_chunk),
        exp_t1_a=bar["exp_t1_a"],
        exp_t1_b=bar["exp_t1_b"],
        h1_bar_a=bar["h1_bar_a"],
        h1_bar_b=bar["h1_bar_b"],
        chol_bar_a=bar["chol_bar_a"],
        chol_bar_b=bar["chol_bar_b"],
        chol_ov_u_a=jnp.einsum(
            "iI,gia->gIa", u_a, bar["chol_bar_a"][:, :nocc_a, nocc_a:], optimize="optimal"
        ),
        chol_ov_u_b=jnp.einsum(
            "iI,gia->gIa", u_b, bar["chol_bar_b"][:, :nocc_b, nocc_b:], optimize="optimal"
        ),
        fock_ov_u_a=jnp.einsum(
            "kI,ka->Ia", u_a.conj(), fock_bar_a[:nocc_a, nocc_a:], optimize="optimal"
        ),
        fock_ov_u_b=jnp.einsum(
            "kI,ka->Ia", u_b.conj(), fock_bar_b[:nocc_b, nocc_b:], optimize="optimal"
        ),
        # the constant needs prjlo_s and t1_s, which the fast trial exposes as well
        e0t1orb=e0t1orb_from_chol_u(ham_data, cast(Any, trial_data)),
    )


class _Walker(NamedTuple):
    """Chunk independent, per walker intermediates, one entry per spin."""

    green: tuple[jax.Array, jax.Array]  # (o_s, n_s) half greens against the bare reference
    gf: tuple[jax.Array, jax.Array]  # (o_s, v_s)
    gc: tuple[jax.Array, jax.Array]  # (L, v_s) = U_s^H gf_s
    t2g: tuple[jax.Array, jax.Array]  # (o_s, v_s) one-body T2 intermediates
    gt2g: jax.Array  # <P T2>
    e1_0: jax.Array  # <h1>
    e1_2: jax.Array  # <P T2 h1>
    e0frg_1: jax.Array  # the projected fock term of e0frg


def _t2g(
    trial_data: Upt2ccsdFastTrial,
    gf: tuple[jax.Array, jax.Array],
    gc: tuple[jax.Array, jax.Array],
    rtype: Any,
    ctype: Any,
) -> tuple[jax.Array, jax.Array]:
    """
    The (o_s, v_s) one-body T2 intermediates t2g_s of the bar kernel (the matrices its
    t2_green_s = greenp_s t2g_s^T green_s are built from), with the doubles carrying the
    local index: a block's first pair closes with gc_s, its second pair with gf of the
    other slot and the projector's second factor after. The (a, b) antisymmetry of the
    same-spin blocks turns their exchange terms into the direct ones (see the module
    docstring), so every block takes two contractions. Runs in the mixed dtypes.
    """
    t2aa, t2ab, t2ba, t2bb = (
        trial_data.t2aa_u.astype(rtype),
        trial_data.t2ab_u.astype(rtype),
        trial_data.t2ba_u.astype(rtype),
        trial_data.t2bb_u.astype(rtype),
    )
    uc_a, uc_b = trial_data.u_a.conj().astype(ctype), trial_data.u_b.conj().astype(ctype)
    gf_a, gf_b = (g.astype(ctype) for g in gf)
    gc_a, gc_b = (g.astype(ctype) for g in gc)

    # alpha ov: 2 (aa direct - aa exchange) + ab (contracted over beta) + ba (over beta)
    t2g_a = (
        0.5 * jnp.einsum("Ia,Iajb->jb", gc_a, t2aa, optimize="optimal")
        + 0.5
        * jnp.einsum("kI,Ia->ka", uc_a, jnp.einsum("Iajb,jb->Ia", t2aa, gf_a, optimize="optimal"))
        + 0.5
        * jnp.einsum("kI,Ia->ka", uc_a, jnp.einsum("Iajb,jb->Ia", t2ab, gf_b, optimize="optimal"))
        + 0.5 * jnp.einsum("Ia,Iajb->jb", gc_b, t2ba, optimize="optimal")
    )
    # beta ov: 2 (bb direct - bb exchange) + ba (contracted over alpha) + ab (over alpha)
    t2g_b = (
        0.5 * jnp.einsum("Ia,Iajb->jb", gc_b, t2bb, optimize="optimal")
        + 0.5
        * jnp.einsum("kI,Ia->ka", uc_b, jnp.einsum("Iajb,jb->Ia", t2bb, gf_b, optimize="optimal"))
        + 0.5
        * jnp.einsum("kI,Ia->ka", uc_b, jnp.einsum("Iajb,jb->Ia", t2ba, gf_a, optimize="optimal"))
        + 0.5 * jnp.einsum("Ia,Iajb->jb", gc_a, t2ab, optimize="optimal")
    )
    return t2g_a, t2g_b


def _walker(
    walker: tuple[jax.Array, jax.Array],
    meas_ctx: Upt2ccsdFastMeasCtx,
    trial_data: Upt2ccsdFastTrial,
    rtype: Any,
    ctype: Any,
) -> _Walker:
    c128 = jnp.complex128
    nocc = trial_data.nocc
    nvir = trial_data.nvir
    u = (trial_data.u_a, trial_data.u_b)
    exp_t1 = (meas_ctx.exp_t1_a, meas_ctx.exp_t1_b)
    h1 = (meas_ctx.h1_bar_a, meas_ctx.h1_bar_b)
    fu = (meas_ctx.fock_ov_u_a, meas_ctx.fock_ov_u_b)

    green, gf, gc, gu, greenp = [], [], [], [], []
    for s in range(2):
        wb = exp_t1[s] @ walker[s]
        g = (wb @ jnp.linalg.inv(wb[: nocc[s], :])).T  # (o_s, n_s) = [1 | gf_s]
        green.append(g)
        gf.append(g[:, nocc[s] :])
        gc.append(jnp.einsum("kI,ka->Ia", u[s].conj(), gf[s], optimize="optimal"))
        gu.append(jnp.einsum("iI,ia->Ia", u[s], gf[s], optimize="optimal"))
        greenp.append(jnp.vstack((gf[s], -jnp.eye(nvir[s], dtype=g.dtype))))

    e0frg_1 = sum(jnp.einsum("Ia,Ia->", gu[s], fu[s], optimize="optimal") for s in range(2))
    e1_0 = sum(
        jnp.einsum("pq,pq->", h1[s][: nocc[s], :], green[s], optimize="optimal") for s in range(2)
    )
    # everything that carries T2 runs in the mixed dtypes (the precision policy of
    # meas/pt2ccsd_fast.py); the greens functions, e0 and e0frg stay in double
    t2g = _t2g(trial_data, (gf[0], gf[1]), (gc[0], gc[1]), rtype, ctype)  # ctype
    # <P T2>: the spin sum of <t2g_s, gf_s> counts every block twice
    gt2g_s = [
        jnp.einsum("ia,ia->", t2g[s], gf[s].astype(ctype), optimize="optimal") for s in range(2)
    ]
    gt2g = (0.5 * (gt2g_s[0] + gt2g_s[1])).astype(c128)
    # the one-body term needs the (n_s, n_s) t2_green_s once per walker
    h1t2_s = [
        jnp.einsum(
            "pq,pq->",
            h1[s].astype(rtype),
            greenp[s].astype(ctype) @ t2g[s].T @ green[s].astype(ctype),
            optimize="optimal",
        )
        for s in range(2)
    ]
    e1_2 = e1_0 * gt2g - (h1t2_s[0] + h1t2_s[1]).astype(c128)
    return _Walker(
        green=(green[0], green[1]),
        gf=(gf[0], gf[1]),
        gc=(gc[0], gc[1]),
        t2g=t2g,
        gt2g=gt2g,
        e1_0=e1_0,
        e1_2=e1_2,
        e0frg_1=e0frg_1,
    )


def energy_kernel_uw_uh_fast(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdFastMeasCtx,
    trial_data: Upt2ccsdFastTrial,
) -> jax.Array:
    """
    The fragment pt2CCSD estimator for one unrestricted walker, [t2frg, e0frg, e1frg, e0],
    with the projectors applied in factored form (see the module docstring). ham_data is
    unused: the tensors are in meas_ctx. The T2 contractions of the two-body term run in
    the mixed dtypes of meas_ctx.cfg; everything else is double.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    w = _walker(walker, meas_ctx, trial_data, rtype, ctype)
    nocc = trial_data.nocc
    uc = (trial_data.u_a.conj().astype(ctype), trial_data.u_b.conj().astype(ctype))
    t2_r = (
        trial_data.t2aa_u.astype(rtype),
        trial_data.t2ab_u.astype(rtype),
        trial_data.t2ba_u.astype(rtype),
        trial_data.t2bb_u.astype(rtype),
    )

    chol_a, _, _, _ = pad_reshape_chol(meas_ctx.chol_bar_a, meas_ctx.nchol_chunk)
    chol_b, _, _, _ = pad_reshape_chol(meas_ctx.chol_bar_b, meas_ctx.nchol_chunk)
    cu_a, _, _, _ = pad_reshape_chol(meas_ctx.chol_ov_u_a, meas_ctx.nchol_chunk)
    cu_b, _, _, _ = pad_reshape_chol(meas_ctx.chol_ov_u_b, meas_ctx.nchol_chunk)

    def scanned_fun(carry, x):
        chol_c = (x[0], x[1])
        cu_c = (x[2], x[3])
        gl_oo, glgp, l_ov, trl = [], [], [], []
        for s in range(2):
            o = nocc[s]
            lc = chol_c[s]
            gf = w.gf[s]
            # gl_s = green_s L_s^T with green_s = [1 | gf_s], from the blocks of L_s
            goo = jnp.transpose(lc[:, :o, :o], (0, 2, 1)) + jnp.einsum(
                "ia,gqa->giq", gf, lc[:, :o, o:], optimize="optimal"
            )  # (k, o, o)
            gov = jnp.transpose(lc[:, o:, :o], (0, 2, 1)) + jnp.einsum(
                "ia,gqa->giq", gf, lc[:, o:, o:], optimize="optimal"
            )  # (k, o, v)
            gl_oo.append(goo)
            glgp.append(jnp.einsum("gij,jb->gib", goo, gf, optimize="optimal") - gov)
            l_ov.append(lc[:, :o, o:])
            trl.append(jnp.einsum("gia,ia->g", lc[:, :o, o:], gf, optimize="optimal"))

        # e2_0 (both spins, exact)
        e2_0_c, tr_gl = e2_0_g(gl_oo[0], gl_oo[1])
        carry[0] += jnp.sum(e2_0_c).astype(c128)

        # the projected two-body term of e0frg: coulomb sees the spin summed trace,
        # exchange stays within a spin; lg_s is never formed
        tr_both = trl[0] + trl[1]
        e2frg = jnp.zeros((), dtype=c128)
        for s in range(2):
            pl = jnp.einsum("gIa,Ia->g", cu_c[s], w.gc[s], optimize="optimal")
            a_ij = jnp.einsum("gIa,ja->gIj", cu_c[s], w.gf[s], optimize="optimal")
            b_ji = jnp.einsum("gja,Ia->gjI", l_ov[s], w.gc[s], optimize="optimal")
            e2frg += 0.5 * (
                jnp.einsum("g,g->", pl, tr_both, optimize="optimal")
                - jnp.einsum("gIj,gjI->", a_ij, b_ji, optimize="optimal")
            )
        carry[1] += e2frg.astype(c128)

        glgp_c = [g.astype(ctype) for g in glgp]

        # e2_2_2_1 = -sum_g tr(gl_g) sum_s <L_s, t2_green_s>, with <L_s, t2_green_s> = <t2g_s, glgp_s>
        lt2g = sum(
            jnp.einsum("jb,gjb->g", w.t2g[s], glgp_c[s], optimize="optimal") for s in range(2)
        )
        carry[2] += -jnp.einsum("g,g->", lt2g, tr_gl.astype(ctype), optimize="optimal").astype(c128)

        # e2_2_2_2 = sum_s <gl_s, L_s t2_green_s^T> = sum_s sum_{ij} (glgp_s t2g_s^T)_{ij} gl_oo,s_{ji}
        e2222 = jnp.zeros((), dtype=c128)
        for s in range(2):
            z = jnp.einsum("gib,jb->gij", glgp_c[s], w.t2g[s], optimize="optimal")
            e2222 += jnp.einsum("gij,gji->", z, gl_oo[s].astype(ctype), optimize="optimal").astype(
                c128
            )
        carry[3] += e2222

        # e2_2_3 = 1/2 sum_blocks: the projector's second factor closes glgp on the local
        # index of the block's first spin, then one contraction per block
        glgpu = [jnp.einsum("kI,gka->gIa", uc[s], glgp_c[s], optimize="optimal") for s in range(2)]
        lt2_a = jnp.einsum("gIa,Iajb->gjb", glgpu[0], t2_r[0], optimize="optimal") + jnp.einsum(
            "gIa,Iajb->gjb", glgpu[1], t2_r[2], optimize="optimal"
        )  # aa + ba, closes with glgp_a
        lt2_b = jnp.einsum("gIa,Iajb->gjb", glgpu[1], t2_r[3], optimize="optimal") + jnp.einsum(
            "gIa,Iajb->gjb", glgpu[0], t2_r[1], optimize="optimal"
        )  # bb + ab, closes with glgp_b
        e223 = jnp.einsum("gjb,gjb->", lt2_a, glgp_c[0], optimize="optimal") + jnp.einsum(
            "gjb,gjb->", lt2_b, glgp_c[1], optimize="optimal"
        )
        carry[4] += (0.5 * e223).astype(c128)

        return carry, None

    zero = jnp.zeros((), dtype=c128)
    [e2_0, e2frg, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [zero, zero, zero, zero, zero], (chol_a, chol_b, cu_a, cu_b)
    )

    e2_2 = e2_0 * w.gt2g + e2_2_2_1 + e2_2_2_2 + e2_2_3

    t2frg = w.gt2g  # <HF| P T2 |walker_bar> / <HF|walker_bar>
    e0frg = meas_ctx.e0t1orb + w.e0frg_1 + e2frg  # projected correlation energy
    e0 = w.e1_0 + e2_0  # <HF| H_bar |walker_bar> / <HF|walker_bar>
    e1frg = w.e1_2 + e2_2  # <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>

    return jnp.stack([t2frg, e0frg, e1frg, e0])


def make_upt2ccsd_fast_meas_ops(
    sys: Any,
    *,
    mixed_precision: bool = True,
    testing: bool = False,
    nchol_chunk: int | None = None,
) -> MeasOps:
    """MeasOps of the unrestricted fragment upt2ccsd_fast trial; see make_upt2ccsd_meas_ops."""
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
        kernels={k_energy: energy_kernel_uw_uh_fast},
    )
    object.__setattr__(meas_ops, _UPT2CCSD_FAST_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_upt2ccsd_fast_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _UPT2CCSD_FAST_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, Pt2ccsdChunkMeasCfg) else None


def plan_chunking_for_run_u_fast(
    sys: Any,
    ham_data: HamCholU,
    trial_data: Upt2ccsdFastTrial,
    *,
    n_walkers: int,
    budget_bytes: int,
    n_chunks: int = 1,
    nchol_chunk: int | None = None,
    mixed_precision: bool = True,
    n_devices: int = 1,
) -> ChunkPlan:
    """The recipe's memory hook: the bar kernel's model, which over-counts this kernel."""
    return plan_chunking_for_run_u(
        sys,
        cast(Any, ham_data),
        cast(Any, trial_data),
        n_walkers=n_walkers,
        budget_bytes=budget_bytes,
        n_chunks=n_chunks,
        nchol_chunk=nchol_chunk,
        mixed_precision=mixed_precision,
        n_devices=n_devices,
    )
