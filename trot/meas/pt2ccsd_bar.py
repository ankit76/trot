"""
The restricted pt2CCSD estimator with exp(T1) applied to the right (pt2ccsd_bar).

Same (t2, e0, e1) per walker as meas/pt2ccsd.py's energy_kernel_rw_rh,

    t2 = <exp(T1)HF| T2 |phi> / <exp(T1)HF|phi>
    e0 = <exp(T1)HF| H  |phi> / <exp(T1)HF|phi>
    e1 = <exp(T1)HF| T2 H |phi> / <exp(T1)HF|phi>

combined by combine_first_order_energy: E = h0 + <e0> + <e1> - <t2><e0>. They are
reached from the similarity transformed hamiltonian in the measurement context (h1_bar,
chol_bar) and the walker exp_t1 @ walker, measured against the bare reference
determinant (trial/pt2ccsd_bar.py). Because the reference occupies the first nocc
orbitals, the greens function against it has nonzero entries only in its first nocc rows;
the kernel carries just those, an (nocc, norb) half green, so every chunk intermediate is
(k, nocc, norb) rather than (k, norb, norb).

The two body sum is scanned over chunks of nchol_chunk cholesky vectors, sized by
meas/pt2ccsd_chunking.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy
from ..ham.chol import HamChol
from ..trial.pt2ccsd_bar import Pt2ccsdTrial, build_bar_intermediates, overlap_r
from .pt2ccsd import combine_first_order_energy
from .pt2ccsd_chunking import (
    ChunkPlan,
    Pt2ccsdChunkMeasCfg,
    make_chunk_meas_cfg,
    pad_reshape_chol,
    plan_pt2ccsd_chunking,
    pt2ccsd_memory_model,
    resolve_nchol_chunk,
)

_PT2CCSD_BAR_MEAS_CFG_ATTR = "_pt2ccsd_bar_meas_cfg"

COMPONENTS: tuple[str, ...] = ("theta", "electronic_0", "h_t")


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdBarMeasCtx:
    cfg: Pt2ccsdChunkMeasCfg  # static
    nchol_chunk: int  # static: it sets the shape the chol tensor is reshaped to
    exp_t1: jax.Array  # (norb, norb)
    h1_bar: jax.Array  # (norb, norb)
    chol_bar: jax.Array  # (nchol, norb, norb)

    def tree_flatten(self):
        return (self.exp_t1, self.h1_bar, self.chol_bar), (self.cfg, self.nchol_chunk)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        exp_t1, h1_bar, chol_bar = children
        return cls(
            cfg=cfg, nchol_chunk=nchol_chunk, exp_t1=exp_t1, h1_bar=h1_bar, chol_bar=chol_bar
        )


def build_meas_ctx(
    ham_data: HamChol, trial_data: Pt2ccsdTrial, cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg()
) -> Pt2ccsdBarMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("pt2ccsd_bar MeasOps assumes HamChol.basis == 'restricted'.")
    nchol = int(ham_data.nchol) if ham_data.nchol is not None else int(ham_data.chol.shape[0])
    bar = build_bar_intermediates(ham_data, trial_data)
    return Pt2ccsdBarMeasCtx(
        cfg=cfg,
        nchol_chunk=resolve_nchol_chunk(nchol, cfg.nchol_chunk),
        exp_t1=bar["exp_t1"],
        h1_bar=bar["h1_bar"],
        chol_bar=bar["chol_bar"],
    )


def energy_kernel_rw_rh_bar(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: Pt2ccsdBarMeasCtx,
    trial_data: Pt2ccsdTrial,
) -> jax.Array:
    """
    (t2, e0, e1) of one restricted walker. ham_data is unused: the transformed tensors
    are in meas_ctx. The T2 contractions run in the mixed dtypes of meas_ctx.cfg; e2_0
    and every partial sum stay in complex128.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    t2 = trial_data.t2
    nocc, nvir = trial_data.nocc, trial_data.nvir

    h1 = meas_ctx.h1_bar
    walker_bar = meas_ctx.exp_t1 @ walker  # (norb, nocc)

    # half green, (nocc, norb): the full green is this padded with zero rows, since the
    # trial here is the bare reference determinant
    green = (walker_bar @ jnp.linalg.inv(walker_bar[:nocc, :])).T
    # (full green - 1) restricted to the virtual columns, (norb, nvir)
    greenp = jnp.vstack((green[:, nocc:], -jnp.eye(nvir, dtype=green.dtype)))

    # one body energy. only the occupied rows of h1 meet a nonzero row of the green
    hg = jnp.einsum("pi,pi->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2 * hg

    # one body double excitations
    t2g_c = jnp.einsum("iajb,ia->jb", t2, green[:, nocc:], optimize="optimal")
    t2g_e = jnp.einsum("iajb,ib->ja", t2, green[:, nocc:], optimize="optimal")
    t2_green_c = jnp.einsum("pb,jb,jq->pq", greenp, t2g_c, green, optimize="optimal")
    t2_green_e = jnp.einsum("pa,ja,jq->pq", greenp, t2g_e, green, optimize="optimal")
    t2_green = 2 * t2_green_c - t2_green_e
    t2g = 2 * t2g_c - t2g_e
    gt2g = jnp.einsum("ia,ia->", t2g, green[:, nocc:], optimize="optimal")
    e1_2_1 = 2 * hg * gt2g
    e1_2_2 = -2 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

    # two body energy, chunked over the cholesky index. both the full and the half
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

    t2 = gt2g  # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
    e0 = e1_0 + e2_0  # <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
    e1 = e1_2 + e2_2  # <exp(T1)HF|T2 (h1+h2)|walker>/<exp(T1)HF|walker>

    return jnp.stack([t2, e0, e1])


def energy_kernel_rw_rh_bar_scalar(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: Pt2ccsdBarMeasCtx,
    trial_data: Pt2ccsdTrial,
) -> jax.Array:
    """The combined local energy of one walker, for checks against the other kernels."""
    return combine_first_order_energy(
        ham_data.h0, energy_kernel_rw_rh_bar(walker, ham_data, meas_ctx, trial_data)
    )


def make_pt2ccsd_bar_meas_ops(
    sys: Any,
    *,
    mixed_precision: bool = True,
    testing: bool = False,
    nchol_chunk: int | None = None,
) -> MeasOps:
    """
    MeasOps of the restricted pt2ccsd_bar trial: overlap_r and the "energy" kernel that
    returns the three components per walker (so it is a measurement trial for the mixed
    driver, not a guide). nchol_chunk caps the cholesky vectors per scan step; the mixed
    driver's chunk plan sets it from the memory budget when it is None.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"pt2ccsd_bar MeasOps supports restricted walkers only, got: {sys.walker_kind}"
        )
    cfg = make_chunk_meas_cfg(
        mixed_precision=mixed_precision, testing=testing, nchol_chunk=nchol_chunk
    )
    meas_ops = MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel_rw_rh_bar},
    )
    object.__setattr__(meas_ops, _PT2CCSD_BAR_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_pt2ccsd_bar_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _PT2CCSD_BAR_MEAS_CFG_ATTR, None)
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
    """Build the memory model of the bar kernel for a run and plan its chunking."""
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
