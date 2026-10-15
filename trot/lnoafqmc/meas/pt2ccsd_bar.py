"""
The LNO fragment pt2CCSD estimator, restricted hamiltonian, bar form only.

Ported from afqmc's lno_afqmc/wavefunctions_restricted.py (class pt2ccsd:
_calc_e0bar_frag, _t2eorb_tc, _build_measurement_intermediates), on top of the branch's
meas/pt2ccsd_bar.py: the fragment energy is measured with the similarity transformed
hamiltonian H_bar = exp(T1) H exp(-T1) against the bare reference, walker_bar = exp(T1)
walker, and the two body sums are scanned over chunks of nchol_chunk cholesky vectors
sized by meas/pt2ccsd_chunking.py. The kernel returns four numbers per walker,

    t2frg = <HF| P T2 |walker_bar> / <HF|walker_bar>
    e0frg = correlation part of <HF| H_bar |walker_bar> / <HF|walker_bar>, projected
    e1frg = <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>
    e0    = <HF| H_bar |walker_bar> / <HF|walker_bar>, electronic, unprojected

(TRIAL_COMPONENTS) and the fragment correlation energy is

    E_F = <e0frg> + <e1frg> - <t2frg><e0>          (frag_pt2ccsd_energy_fn)

over the wp-weighted block averages; there is no h0. P projects the first occupied
index of T2 on the fragment (trial/pt2ccsd.py), which is the LNO partition of the
correlation energy: the fragment energies add up to the full-space AFQMC/pt2CCSD energy
when the local active spaces are complete.

Differences from the branch's energy_kernel_rw_rh_bar, which this otherwise follows line
by line: the projected t2 is not symmetric under (ia)<->(jb), so both halves of every
T2-contracted one-body intermediate are formed instead of doubling one; and e0frg needs
two more intermediates in the ctx, the transformed fock matrix (projected ov block) and
the constant e0t1orb = <exp(T1)HF| H |HF>_F.
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
    plan_pt2ccsd_chunking,
    pt2ccsd_memory_model,
    resolve_nchol_chunk,
)
from ...trial.pt2ccsd_bar import build_bar_intermediates
from ..trial.pt2ccsd import Pt2ccsdTrial, overlap_r

__all__ = [
    "TRIAL_COMPONENTS",
    "frag_pt2ccsd_energy_fn",
    "Pt2ccsdFragMeasCtx",
    "fock_from_chol",
    "e0t1orb_from_chol",
    "build_meas_ctx",
    "energy_kernel_rw_rh_bar",
    "make_pt2ccsd_meas_ops",
    "get_pt2ccsd_meas_cfg",
    "plan_chunking_for_run",
]

TRIAL_COMPONENTS: tuple[str, ...] = ("t2frg", "e0frg", "e1frg", "e0")

_PT2CCSD_FRAG_MEAS_CFG_ATTR = "_lno_pt2ccsd_meas_cfg"


def frag_pt2ccsd_energy_fn(h0: Any, components: Any) -> Any:
    """
    The fragment correlation energy from the averaged components, in the recipe's
    energy_fn signature: components (..., 4) in TRIAL_COMPONENTS order,

        E_F = <e0frg> + <e1frg> - <t2frg><e0>.

    h0 is ignored: the fragment estimator is a correlation energy. Applied to a
    (n_blocks, 4) array it gives the per block proxy energies the outlier filter uses.
    """
    c = jnp.asarray(components)
    return c[..., 1] + c[..., 2] - c[..., 0] * c[..., 3]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdFragMeasCtx:
    cfg: Pt2ccsdChunkMeasCfg  # static
    nchol_chunk: int  # static; sets the shape the chol tensor is reshaped to
    exp_t1: jax.Array  # (norb, norb)
    h1_bar: jax.Array  # (norb, norb)
    chol_bar: jax.Array  # (nchol, norb, norb)
    fock_bar: jax.Array  # (norb, norb), fock of H_bar at the reference
    e0t1orb: jax.Array  # scalar, <exp(T1)HF|H|HF> projected on the fragment

    def tree_flatten(self):
        children = (self.exp_t1, self.h1_bar, self.chol_bar, self.fock_bar, self.e0t1orb)
        return children, (self.cfg, self.nchol_chunk)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        exp_t1, h1_bar, chol_bar, fock_bar, e0t1orb = children
        return cls(
            cfg=cfg,
            nchol_chunk=nchol_chunk,
            exp_t1=exp_t1,
            h1_bar=h1_bar,
            chol_bar=chol_bar,
            fock_bar=fock_bar,
            e0t1orb=e0t1orb,
        )


def fock_from_chol(nocc: int, h1: jax.Array, chol: jax.Array) -> jax.Array:
    """
    Closed-shell fock matrix of (h1, chol) at the reference occupying the first nocc
    orbitals: h1 + 2 J - K. Written for a non-symmetric chol (the transformed tensor),
    as afqmc's integral.get_rfock.
    """
    jeff = jnp.einsum("gpq,gjj->pq", chol, chol[:, :nocc, :nocc], optimize="optimal")
    keff = jnp.einsum("gpj,gjq->pq", chol[:, :, :nocc], chol[:, :nocc, :], optimize="optimal")
    return h1 + 2 * jeff - keff


def e0t1orb_from_chol(chol: jax.Array, t1: jax.Array, prjlo: jax.Array) -> jax.Array:
    """
    <exp(T1)HF| H |HF> restricted to the fragment: the T1-contracted two-body term,
    2 sum_g (L t1)_ik P_ik tr(L t1)_g - sum_g (L t1)_ij (L t1)_jk P_ik, from the
    untransformed cholesky vectors. afqmc's ham_data['e0t1orb'].
    """
    nocc = t1.shape[0]
    lt1 = jnp.einsum("ia,gja->gij", t1, chol[:, :nocc, nocc:], optimize="optimal")
    coul = 2 * jnp.einsum("gik,ik,gjj->", lt1, prjlo, lt1, optimize="optimal")
    exch = jnp.einsum("gij,gjk,ik->", lt1, lt1, prjlo, optimize="optimal")
    return coul - exch


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: Pt2ccsdTrial,
    cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg(),
) -> Pt2ccsdFragMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("the fragment pt2CCSD MeasOps assume HamChol.basis == 'restricted'.")
    nchol = int(ham_data.nchol) if ham_data.nchol is not None else int(ham_data.chol.shape[0])
    # the branch's builder reads mo_t / nocc / norb only, which the fragment trial has too
    bar = build_bar_intermediates(ham_data, trial_data)  # type: ignore[arg-type]
    fock_bar = fock_from_chol(trial_data.nocc, bar["h1_bar"], bar["chol_bar"])
    e0t1orb = e0t1orb_from_chol(ham_data.chol, trial_data.t1, trial_data.prjlo)
    return Pt2ccsdFragMeasCtx(
        cfg=cfg,
        nchol_chunk=resolve_nchol_chunk(nchol, cfg.nchol_chunk),
        exp_t1=bar["exp_t1"],
        h1_bar=bar["h1_bar"],
        chol_bar=bar["chol_bar"],
        fock_bar=fock_bar,
        e0t1orb=e0t1orb,
    )


def _e0bar_frag(
    green: jax.Array,
    prjlo: jax.Array,
    fock_bar: jax.Array,
    chol_bar: jax.Array,
    e0t1orb: jax.Array,
    nchol_chunk: int,
) -> jax.Array:
    """
    The projected correlation part of <HF| H_bar |walker_bar> / <HF|walker_bar>:
    e0t1orb + the projected fock ov term + the projected ov-ov two-body term. afqmc's
    _calc_e0bar_frag. green is the (nocc, norb) half green of walker_bar.
    """
    nocc = green.shape[0]
    gf_ov = green[:, nocc:]  # (nocc, nvir)
    fock_ov = fock_bar[:nocc, nocc:]
    e1 = 2 * jnp.einsum("ia,ik,ka->", gf_ov, prjlo, fock_ov, optimize="optimal")

    chol_ov, _, _, _ = pad_reshape_chol(chol_bar[:, :nocc, nocc:], nchol_chunk)

    def scanned_fun(carry, chol_c):
        lg = jnp.einsum("gia,ka->gik", chol_c, gf_ov, optimize="optimal")
        e2_c = 2 * jnp.einsum("gik,ik,gjj->", lg, prjlo, lg, optimize="optimal")
        e2_e = jnp.einsum("gij,gjk,ik->", lg, lg, prjlo, optimize="optimal")
        return carry + e2_c - e2_e, None

    e2, _ = lax.scan(scanned_fun, jnp.zeros((), dtype=gf_ov.dtype), chol_ov)
    return e0t1orb + e1 + e2


def energy_kernel_rw_rh_bar(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: Pt2ccsdFragMeasCtx,
    trial_data: Pt2ccsdTrial,
) -> jax.Array:
    """
    The fragment pt2CCSD estimator for one restricted walker, [t2frg, e0frg, e1frg, e0].

    Follows the branch's energy_kernel_rw_rh_bar: the similarity transformed hamiltonian
    in meas_ctx, the walker exp_t1 @ walker, and the (nocc, norb) half green against the
    bare reference. ham_data is unused: the tensors are in meas_ctx. The T2 contractions
    run in the mixed dtypes of meas_ctx.cfg; e2_0 and every partial sum stay in
    complex128.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    t2 = trial_data.t2
    prjlo = trial_data.prjlo
    nocc, nvir = trial_data.nocc, trial_data.nvir

    h1 = meas_ctx.h1_bar
    walker_bar = meas_ctx.exp_t1 @ walker  # (norb, nocc)

    # half green, (nocc, norb): the trial is the bare reference, so the full green is
    # this padded with zero rows
    green = (walker_bar @ jnp.linalg.inv(walker_bar[:nocc, :])).T
    green_occ = green[:, nocc:]  # (nocc, nvir)
    # (full green - 1) restricted to the virtual columns, (norb, nvir)
    greenp = jnp.vstack((green_occ, -jnp.eye(nvir, dtype=green.dtype)))

    # the projected correlation energy at the walker
    e0frg = _e0bar_frag(
        green, prjlo, meas_ctx.fock_bar, meas_ctx.chol_bar, meas_ctx.e0t1orb, meas_ctx.nchol_chunk
    )

    # one body energy; only the occupied rows of h1 meet a nonzero row of the green
    hg = jnp.einsum("pi,pi->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2 * hg

    # one-body double excitations. t_iajb != t_jbia since the first index is projected,
    # so both halves of each term are formed
    t2g_c_1 = jnp.einsum("iajb,ia->jb", t2, green_occ, optimize="optimal")
    t2g_c_2 = jnp.einsum("iajb,jb->ia", t2, green_occ, optimize="optimal")
    t2g_e_1 = jnp.einsum("iajb,ib->ja", t2, green_occ, optimize="optimal")
    t2g_e_2 = jnp.einsum("iajb,ja->ib", t2, green_occ, optimize="optimal")
    t2_green_c_1 = jnp.einsum("pb,jb,jq->pq", greenp, t2g_c_1, green, optimize="optimal")
    t2_green_c_2 = jnp.einsum("pa,ia,iq->pq", greenp, t2g_c_2, green, optimize="optimal")
    t2_green_e_1 = jnp.einsum("pa,ja,jq->pq", greenp, t2g_e_1, green, optimize="optimal")
    t2_green_e_2 = jnp.einsum("pb,ib,iq->pq", greenp, t2g_e_2, green, optimize="optimal")
    t2_green = (t2_green_c_1 + t2_green_c_2) - 0.5 * (t2_green_e_1 + t2_green_e_2)
    t2g = (t2g_c_1 + t2g_c_2) - 0.5 * (t2g_e_1 + t2g_e_2)
    gt2g = jnp.einsum("ia,ia->", t2g, green_occ, optimize="optimal")
    e1_2_1 = 2 * hg * gt2g
    e1_2_2 = -2 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    # two body energy, chunked over the cholesky index; both the full and the half
    # rotated tensors are padded the same way, so the leftover contributes to neither
    chol, _, _, _ = pad_reshape_chol(meas_ctx.chol_bar, meas_ctx.nchol_chunk)
    rot_chol, _, _, _ = pad_reshape_chol(meas_ctx.chol_bar[:, :nocc, :], meas_ctx.nchol_chunk)

    t2_r = t2.astype(rtype)

    def scanned_fun(carry, x):
        chol_c, rot_chol_c = x  # (k, norb, norb), (k, nocc, norb)

        # e2_0
        gl = jnp.einsum("ir,gqr->giq", green, chol_c, optimize="optimal")  # (k, nocc, norb)
        tr_gl = jnp.einsum("gii->g", gl[:, :, :nocc], optimize="optimal")
        e2_0_c = 2 * jnp.einsum("g,g->", tr_gl, tr_gl, optimize="optimal")
        e2_0_e = -jnp.einsum("gij,gji->", gl[:, :, :nocc], gl[:, :, :nocc], optimize="optimal")
        carry[0] += (e2_0_c + e2_0_e).astype(c128)

        # e2_2_2_1
        lt2g = jnp.einsum(
            "gpr,pr->g", chol_c.astype(rtype), t2_green.astype(ctype), optimize="optimal"
        )
        carry[1] += -jnp.einsum(
            "g,g->", lt2g.astype(ctype), tr_gl.astype(ctype), optimize="optimal"
        ).astype(c128)

        # e2_2_2_2
        lt2_green = jnp.einsum(
            "gir,qr->giq", rot_chol_c.astype(rtype), t2_green.astype(ctype), optimize="optimal"
        )
        carry[2] += 0.5 * jnp.einsum(
            "giq,giq->", gl.astype(ctype), lt2_green.astype(ctype), optimize="optimal"
        ).astype(c128)

        # e2_2_3
        glgp = jnp.einsum("gir,rb->gib", gl.astype(ctype), greenp.astype(ctype), optimize="optimal")
        lt2_c = jnp.einsum("gia,iajb->gjb", glgp, t2_r, optimize="optimal")
        lt2_e = jnp.einsum("gib,iajb->gja", glgp, t2_r, optimize="optimal")
        l2t2_c = jnp.einsum("gjb,gjb->", lt2_c.astype(ctype), glgp, optimize="optimal").astype(c128)
        l2t2_e = jnp.einsum("gja,gja->", lt2_e.astype(ctype), glgp, optimize="optimal").astype(c128)
        carry[3] += (2 * l2t2_c - l2t2_e).astype(c128)

        return carry, None

    zero = jnp.zeros((), dtype=c128)
    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [zero, zero, zero, zero], (chol, rot_chol)
    )

    e2_2_1 = e2_0 * gt2g
    e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    t2frg = gt2g  # <HF| P T2 |walker_bar> / <HF|walker_bar>
    e0 = e1_0 + e2_0  # <HF| H_bar |walker_bar> / <HF|walker_bar>
    e1frg = e1_2 + e2_2  # <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>

    return jnp.stack([t2frg, e0frg, e1frg, e0])


def make_pt2ccsd_meas_ops(
    sys: Any,
    *,
    mixed_precision: bool = True,
    testing: bool = False,
    nchol_chunk: int | None = None,
) -> MeasOps:
    """
    MeasOps of the fragment pt2CCSD trial: overlap_r and the "energy" kernel that returns
    TRIAL_COMPONENTS per walker. The same signature as the branch's
    make_pt2ccsd_bar_meas_ops, so setup_mixed drives it unchanged; nchol_chunk caps the
    cholesky vectors per scan step, and the chunk plan sets it from the memory budget
    when it is None.
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
        kernels={k_energy: energy_kernel_rw_rh_bar},
    )
    object.__setattr__(meas_ops, _PT2CCSD_FRAG_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_pt2ccsd_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _PT2CCSD_FRAG_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, Pt2ccsdChunkMeasCfg) else None


def plan_chunking_for_run(
    sys: Any,
    ham_data: HamChol,
    trial_data: Pt2ccsdTrial,
    *,
    n_walkers: int,
    budget_bytes: int,
    n_chunks: int = 1,
    nchol_chunk: int | None = None,
    mixed_precision: bool = True,
    n_devices: int = 1,
) -> ChunkPlan:
    """
    The recipe's memory hook, with the branch's bar memory model. The fragment ctx adds
    the fock matrix and a constant, both negligible against the cholesky copies the
    model already counts, and the e0frg scan reuses the (k, nocc, ...) shapes it sizes.
    """
    nchol = int(ham_data.nchol) if ham_data.nchol is not None else int(ham_data.chol.shape[0])
    model = pt2ccsd_memory_model(
        norb=trial_data.norb,
        nocc=trial_data.nocc,
        nchol=nchol,
        n_walkers=n_walkers,
        real_bytes=4 if mixed_precision else 8,
        complex_bytes=8 if mixed_precision else 16,
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
