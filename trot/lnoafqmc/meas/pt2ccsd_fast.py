"""
The LNO fragment pt2CCSD estimator with the fragment projector in factored form
("pt2ccsd_fast"): the same four numbers per walker as meas/pt2ccsd_bar.py,

    t2frg = <HF| P T2 |walker_bar> / <HF|walker_bar>
    e0frg = correlation part of <HF| H_bar |walker_bar> / <HF|walker_bar>, projected
    e1frg = <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>
    e0    = <HF| H_bar |walker_bar> / <HF|walker_bar>, electronic, unprojected

(TRIAL_COMPONENTS, combined by frag_pt2ccsd_energy_fn), to roundoff, at a fraction of
the cost. Sizes below: nlo < nocc < nvir, norb = nocc + nvir, k cholesky vectors per
scan step; every cost is per walker and per step.

Projector. prjlo_{ik} = sum_I U_{iI} U^*_{kI} with U = <act_occ|lo> of shape (nocc, nlo).
The bar kernel contracts prjlo as a dense (nocc, nocc) matrix and keeps T2 projected on
its first index, so every contraction over that index costs nocc. Here every projected
quantity is built as (something)_{I...} = sum_i U_{iI} (something)_{i...} on both sides
of the projector and the local index I is contracted last, so those cost nlo instead.
The doubles are stored as t2x_{Iajb} = 2 t2u_{Iajb} - t2u_{Ibja} with
t2u_{Iajb} = sum_i t2_{iajb} U_{iI} (trial/pt2ccsd_fast.py): nlo/nocc of the memory,
and every direct/exchange pair of the bar kernel is one contraction with t2x.

Green's function. The half green against the bare reference is green = [1 | gf] with gf
the (nocc, nvir) occupied-virtual block, so nothing is ever contracted over its unit
block: gl_{g,iq} = sum_r green_{ir} L_{g,qr} = L_{g,qi} + sum_a gf_{ia} L_{g,q,nocc+a}
(k nocc norb nvir), and the (norb, nvir) greenp = [gf; -1] gives
glgp = gl_oo gf - gl_ov (k nocc^2 nvir) instead of gl greenp (k nocc norb nvir).

Terms, with gu_{Ia} = sum_i U_{iI} gf_{ia}, gc_{Ia} = sum_k U^*_{kI} gf_{ka} (equal for a
real U), fu = U^H f_ov and cu_g = U^T L_{g,ov} built once in the ctx:

    e0frg fock     2 sum_{Ia} gu_{Ia} fu_{Ia}
    e0frg coulomb  2 sum_g (sum_{Ia} cu_{g,Ia} gc_{Ia}) tr(lg_g), tr(lg_g) = sum_{ia} L_{g,ia} gf_{ia}
    e0frg exchange sum_g sum_{Ij} A_{g,Ij} B_{g,jI}, A = cu_g gf^T, B = L_{g,ov} gc^T   (k nlo nocc nvir)
    t2g            the (nocc, nvir) one-body T2 intermediate, two contractions with t2x
                   (nlo nocc nvir^2 each), gt2g = t2frg = <t2g, gf>
    e2_0           from gl_oo, as in the bar kernel
    e2_2_2_1       -sum_g tr(gl_g) sum_{jb} t2g_{jb} glgp_{g,jb}            (k nocc nvir)
    e2_2_2_2       1/2 sum_g sum_{ij} (glgp_g t2g^T)_{ij} gl_{g,ji}          (k nocc^2 nvir)
    e2_2_3         sum_g sum_{Iajb} glgpu_{g,Ia} t2x_{Iajb} glgp_{g,jb}, glgpu = U^H glgp
                                                                          (k nlo nocc nvir^2)

The two e2_2_2 terms use t2_green = greenp t2g^T green only through glgp and t2g, so the
(norb, norb) t2_green and the half rotated cholesky copy of the bar kernel are not
needed in the scan; t2_green is formed once per walker for the one-body e1_2 only. The
scan passes the cholesky vectors once and carries all accumulators. The T2 contraction
(e2_2_3) runs in the mixed dtypes of the cfg; everything else is double.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ...core.ops import MeasOps, k_energy
from ...ham.chol import HamChol
from ...meas.pt2ccsd_chunking import (
    ChunkPlan,
    Pt2ccsdChunkMeasCfg,
    make_chunk_meas_cfg,
    pad_reshape_chol,
    resolve_nchol_chunk,
)
from ...trial.pt2ccsd_bar import build_bar_intermediates
from ..trial.pt2ccsd_fast import Pt2ccsdFastTrial, overlap_r
from .pt2ccsd_bar import (
    TRIAL_COMPONENTS,
    e0t1orb_from_chol,
    fock_from_chol,
    frag_pt2ccsd_energy_fn,
    plan_chunking_for_run,
)

__all__ = [
    "TRIAL_COMPONENTS",
    "frag_pt2ccsd_energy_fn",
    "Pt2ccsdFastMeasCtx",
    "build_meas_ctx",
    "energy_kernel_rw_rh_fast",
    "make_pt2ccsd_fast_meas_ops",
    "get_pt2ccsd_fast_meas_cfg",
    "plan_chunking_for_run_fast",
]

_PT2CCSD_FAST_MEAS_CFG_ATTR = "_lno_pt2ccsd_fast_meas_cfg"


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdFastMeasCtx:
    cfg: Pt2ccsdChunkMeasCfg  # static
    nchol_chunk: int  # static; sets the shape the chol tensor is reshaped to
    exp_t1: jax.Array  # (norb, norb)
    h1_bar: jax.Array  # (norb, norb)
    chol_bar: jax.Array  # (nchol, norb, norb)
    chol_ov_u: jax.Array  # (nchol, nlo, nvir): U^T applied to the ov block of chol_bar
    fock_ov_u: jax.Array  # (nlo, nvir): U^H applied to the ov block of fock_bar
    e0t1orb: jax.Array  # scalar, <exp(T1)HF|H|HF> projected on the fragment

    def tree_flatten(self):
        children = (
            self.exp_t1,
            self.h1_bar,
            self.chol_bar,
            self.chol_ov_u,
            self.fock_ov_u,
            self.e0t1orb,
        )
        return children, (self.cfg, self.nchol_chunk)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        exp_t1, h1_bar, chol_bar, chol_ov_u, fock_ov_u, e0t1orb = children
        return cls(
            cfg=cfg,
            nchol_chunk=nchol_chunk,
            exp_t1=exp_t1,
            h1_bar=h1_bar,
            chol_bar=chol_bar,
            chol_ov_u=chol_ov_u,
            fock_ov_u=fock_ov_u,
            e0t1orb=e0t1orb,
        )


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: Pt2ccsdFastTrial,
    cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg(),
) -> Pt2ccsdFastMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("the fragment pt2CCSD MeasOps assume HamChol.basis == 'restricted'.")
    nchol = int(ham_data.nchol) if ham_data.nchol is not None else int(ham_data.chol.shape[0])
    nocc = trial_data.nocc
    u = trial_data.u
    # the branch's builder reads mo_t / nocc / norb only, which the fragment trial has too
    bar = build_bar_intermediates(ham_data, trial_data)  # type: ignore[arg-type]
    fock_bar = fock_from_chol(nocc, bar["h1_bar"], bar["chol_bar"])
    # the projector's left factor folded into the constants once: U^T L_ov and U^H f_ov
    chol_ov_u = jnp.einsum("iI,gia->gIa", u, bar["chol_bar"][:, :nocc, nocc:], optimize="optimal")
    fock_ov_u = jnp.einsum("kI,ka->Ia", u.conj(), fock_bar[:nocc, nocc:], optimize="optimal")
    e0t1orb = e0t1orb_from_chol(ham_data.chol, trial_data.t1, trial_data.prjlo)
    return Pt2ccsdFastMeasCtx(
        cfg=cfg,
        nchol_chunk=resolve_nchol_chunk(nchol, cfg.nchol_chunk),
        exp_t1=bar["exp_t1"],
        h1_bar=bar["h1_bar"],
        chol_bar=bar["chol_bar"],
        chol_ov_u=chol_ov_u,
        fock_ov_u=fock_ov_u,
        e0t1orb=e0t1orb,
    )


def energy_kernel_rw_rh_fast(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: Pt2ccsdFastMeasCtx,
    trial_data: Pt2ccsdFastTrial,
) -> jax.Array:
    """
    The fragment pt2CCSD estimator for one restricted walker, [t2frg, e0frg, e1frg, e0],
    with the projector applied in factored form (see the module docstring). ham_data is
    unused: the tensors are in meas_ctx.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    u = trial_data.u  # (nocc, nlo)
    uc = u.conj()
    t2x = trial_data.t2x
    nocc, nvir = trial_data.nocc, trial_data.nvir

    h1 = meas_ctx.h1_bar
    walker_bar = meas_ctx.exp_t1 @ walker  # (norb, nocc)

    # half green, (nocc, norb) = [1 | gf]: the trial is the bare reference, so the full
    # green is this padded with zero rows and its occupied block is the unit matrix
    green = (walker_bar @ jnp.linalg.inv(walker_bar[:nocc, :])).T
    gf = green[:, nocc:]  # (nocc, nvir)
    # (full green - 1) restricted to the virtual columns, (norb, nvir)
    greenp = jnp.vstack((gf, -jnp.eye(nvir, dtype=green.dtype)))

    # the green contracted with either factor of the projector, (nlo, nvir)
    gu = jnp.einsum("iI,ia->Ia", u, gf, optimize="optimal")
    gc = jnp.einsum("kI,ka->Ia", uc, gf, optimize="optimal")

    # ---- e0frg: the constant, the fock term, and the two-body term from the scan
    e0frg_1 = 2 * jnp.einsum("Ia,Ia->", gu, meas_ctx.fock_ov_u, optimize="optimal")

    # ---- one body energy; only the occupied rows of h1 meet a nonzero row of the green
    hg = jnp.einsum("pi,pi->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2 * hg

    # ---- the one-body T2 intermediate t2g (nocc, nvir): the direct and exchange halves
    # of the bar kernel in one contraction each with t2x, the first index closed with gc,
    # the second factor of the projector applied after the contraction over (j, b)
    # Precision policy: everything that carries T2 (t2g and what is built from it, the
    # T2 contractions in the scan) runs in the mixed dtypes of the cfg; the greens
    # function pieces, e0 and the projected e0frg stay in double. The casts of the
    # constant tensors are done once, outside the scan.
    t2x_r = t2x.astype(rtype)
    uc_c = uc.astype(ctype)
    gf_c = gf.astype(ctype)
    t2g = 0.5 * jnp.einsum("Ia,Iajb->jb", gc.astype(ctype), t2x_r, optimize="optimal")
    t2g = t2g + 0.5 * jnp.einsum(
        "kI,Ia->ka",
        uc_c,
        jnp.einsum("Iajb,jb->Ia", t2x_r, gf_c, optimize="optimal"),
        optimize="optimal",
    )  # ctype
    gt2g = jnp.einsum("ia,ia->", t2g, gf_c, optimize="optimal").astype(c128)
    # (norb, norb), for the one-body term only
    t2_green = greenp.astype(ctype) @ t2g.T @ green.astype(ctype)
    e1_2 = 2 * hg * gt2g - 2 * jnp.einsum(
        "pq,pq->", h1.astype(rtype), t2_green, optimize="optimal"
    ).astype(c128)

    # ---- two body terms, one scan over the cholesky chunks; both tensors are padded the
    # same way, so the leftover contributes to nothing
    chol, _, _, _ = pad_reshape_chol(meas_ctx.chol_bar, meas_ctx.nchol_chunk)
    chol_ov_u, _, _, _ = pad_reshape_chol(meas_ctx.chol_ov_u, meas_ctx.nchol_chunk)

    def scanned_fun(carry, x):
        chol_c, cu_c = x  # (k, norb, norb), (k, nlo, nvir)

        # gl_{g,iq} = sum_r green_{ir} L_{g,qr} with green = [1 | gf]
        gl = jnp.transpose(chol_c[:, :, :nocc], (0, 2, 1)) + jnp.einsum(
            "ia,gqa->giq", gf, chol_c[:, :, nocc:], optimize="optimal"
        )  # (k, nocc, norb)
        gl_oo = gl[:, :, :nocc]
        gl_ov = gl[:, :, nocc:]

        # e2_0
        tr_gl = jnp.einsum("gii->g", gl_oo, optimize="optimal")
        e2_0_c = 2 * jnp.einsum("g,g->", tr_gl, tr_gl, optimize="optimal")
        e2_0_e = -jnp.einsum("gij,gji->", gl_oo, gl_oo, optimize="optimal")
        carry[0] += (e2_0_c + e2_0_e).astype(c128)

        # glgp = gl greenp with greenp = [gf; -1]
        glgp = jnp.einsum("gij,jb->gib", gl_oo, gf, optimize="optimal") - gl_ov  # (k, nocc, nvir)

        # the projected two-body term of e0frg. lg_{g,ik} = sum_a L_{g,ia} gf_{ka} is never
        # formed: coulomb needs sum_{ik} lg prjlo = sum_{Ia} cu gc, exchange the two half
        # projected (k, nlo, nocc) products
        chol_ov_c = chol_c[:, :nocc, nocc:]
        trl = jnp.einsum("gia,ia->g", chol_ov_c, gf, optimize="optimal")  # tr(lg_g)
        pl = jnp.einsum("gIa,Ia->g", cu_c, gc, optimize="optimal")
        a_ij = jnp.einsum("gIa,ja->gIj", cu_c, gf, optimize="optimal")  # (k, nlo, nocc)
        b_ji = jnp.einsum("gja,Ia->gjI", chol_ov_c, gc, optimize="optimal")  # (k, nocc, nlo)
        e2frg_c = 2 * jnp.einsum("g,g->", pl, trl, optimize="optimal")
        e2frg_e = jnp.einsum("gIj,gjI->", a_ij, b_ji, optimize="optimal")
        carry[1] += (e2frg_c - e2frg_e).astype(c128)

        glgp_c = glgp.astype(ctype)

        # e2_2_2_1 = -sum_g tr(L_g t2_green) tr(gl_g), with tr(L_g t2_green) = <t2g, glgp_g>
        lt2g = jnp.einsum("jb,gjb->g", t2g, glgp_c, optimize="optimal")
        carry[2] += -jnp.einsum("g,g->", lt2g, tr_gl.astype(ctype), optimize="optimal").astype(c128)

        # e2_2_2_2 = 1/2 sum_g <gl_g, L_g t2_green^T> = 1/2 sum_g sum_{ij} (glgp_g t2g^T)_{ij} gl_{g,ji}
        z = jnp.einsum("gib,jb->gij", glgp_c, t2g, optimize="optimal")  # (k, nocc, nocc)
        carry[3] += 0.5 * jnp.einsum(
            "gij,gji->", z, gl_oo.astype(ctype), optimize="optimal"
        ).astype(c128)

        # e2_2_3: the projector's second factor closes glgp on the local index, then one
        # contraction with the exchange folded doubles
        glgpu = jnp.einsum("kI,gka->gIa", uc_c, glgp_c, optimize="optimal")
        lt2 = jnp.einsum("gIa,Iajb->gjb", glgpu, t2x_r, optimize="optimal")
        carry[4] += jnp.einsum("gjb,gjb->", lt2, glgp_c, optimize="optimal").astype(c128)

        return carry, None

    zero = jnp.zeros((), dtype=c128)
    [e2_0, e2frg, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [zero, zero, zero, zero, zero], (chol, chol_ov_u)
    )

    e2_2 = e2_0 * gt2g + 4 * (e2_2_2_1 + e2_2_2_2) + e2_2_3

    t2frg = gt2g  # <HF| P T2 |walker_bar> / <HF|walker_bar>
    e0frg = meas_ctx.e0t1orb + e0frg_1 + e2frg  # projected correlation energy
    e0 = e1_0 + e2_0  # <HF| H_bar |walker_bar> / <HF|walker_bar>
    e1frg = e1_2 + e2_2  # <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>

    return jnp.stack([t2frg, e0frg, e1frg, e0])


def make_pt2ccsd_fast_meas_ops(
    sys: Any,
    *,
    mixed_precision: bool = True,
    testing: bool = False,
    nchol_chunk: int | None = None,
) -> MeasOps:
    """
    MeasOps of the fragment pt2ccsd_fast trial: overlap_r and the "energy" kernel that
    returns TRIAL_COMPONENTS per walker. The same signature as make_pt2ccsd_meas_ops.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"the fragment pt2CCSD MeasOps support restricted walkers only, got: {sys.walker_kind}"
        )
    cfg = make_chunk_meas_cfg(
        mixed_precision=mixed_precision, testing=testing, nchol_chunk=nchol_chunk
    )
    meas_ops = MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel_rw_rh_fast},
    )
    object.__setattr__(meas_ops, _PT2CCSD_FAST_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_pt2ccsd_fast_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _PT2CCSD_FAST_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, Pt2ccsdChunkMeasCfg) else None


def plan_chunking_for_run_fast(
    sys: Any,
    ham_data: HamChol,
    trial_data: Pt2ccsdFastTrial,
    *,
    n_walkers: int,
    budget_bytes: int,
    n_chunks: int = 1,
    nchol_chunk: int | None = None,
    mixed_precision: bool = True,
    n_devices: int = 1,
) -> ChunkPlan:
    """
    The recipe's memory hook: the bar kernel's model, which over-counts this kernel (its
    T2 block and chunk intermediates carry nlo where the model counts nocc), so the plan
    errs on the safe side.
    """
    return plan_chunking_for_run(
        sys,
        ham_data,
        trial_data,  # type: ignore[arg-type]
        n_walkers=n_walkers,
        budget_bytes=budget_bytes,
        n_chunks=n_chunks,
        nchol_chunk=nchol_chunk,
        mixed_precision=mixed_precision,
        n_devices=n_devices,
    )
