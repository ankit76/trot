"""
The unrestricted pt2CCSD estimator with exp(T1) applied to the right (upt2ccsd_bar), on
the unrestricted (uchol) hamiltonian.

Same (t2, e0, e1) as meas/upt2ccsd_uh.py's energy_kernel_uw_uh_chunk, reached from each
spin's similarity transformed hamiltonian in the measurement context (h1_bar_s,
chol_bar_s) and the walkers exp_t1_s @ walker_s, measured against the bare reference
determinants (trial/upt2ccsd_bar_uh.py). The greens functions are (nocc_s, norb_s)
halves, so every chunk intermediate is (k, nocc_s, norb_s) rather than (k, norb_s, norb_s).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ..core.ops import MeasOps, k_energy
from ..ham.chol_u import HamCholU
from ..trial.upt2ccsd_bar_uh import Upt2ccsdTrial, overlap_u
from .pt2ccsd import combine_first_order_energy
from .pt2ccsd_chunking import (
    ChunkPlan,
    Pt2ccsdChunkMeasCfg,
    make_chunk_meas_cfg,
    pad_reshape_chol,
)
from .upt2ccsd_uh import (
    Upt2ccsdMeasCtx,
    build_meas_ctx as _build_meas_ctx,
    e2_0_g,
    l2t2_g,
    plan_chunking_for_run_u,
    t2_one_body,
)

_UPT2CCSD_BAR_MEAS_CFG_ATTR = "_upt2ccsd_bar_meas_cfg"

COMPONENTS: tuple[str, ...] = ("theta", "electronic_0", "h_t")


def build_meas_ctx(
    ham_data: HamCholU,
    trial_data: Upt2ccsdTrial,
    cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg(),
) -> Upt2ccsdMeasCtx:
    """The upt2ccsd context with the similarity transformed tensors built."""
    return _build_meas_ctx(ham_data, trial_data, cfg, bar=True)


class _BarWalker(NamedTuple):
    """Chunk independent, per walker intermediates of the bar kernel."""

    green_a: jax.Array  # (nocc_a, norb_a), the half green against the bare reference
    green_b: jax.Array  # (nocc_b, norb_b)
    greenp_a: jax.Array  # (norb_a, nvir_a)
    greenp_b: jax.Array  # (norb_b, nvir_b)
    t2_green_a: jax.Array  # (norb_a, norb_a)
    t2_green_b: jax.Array  # (norb_b, norb_b)
    gt2g: jax.Array  # <T2>
    e1_0: jax.Array  # <h1>
    e1_2: jax.Array  # <T2 h1>


def _bar_walker(
    walker: tuple[jax.Array, jax.Array],
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> _BarWalker:
    """
    Transform the walker with exp(T1) and build everything that does not touch a cholesky
    vector. The reference is the bare determinant, so the green has nonzero entries only
    in its first nocc rows and is carried as that (nocc, norb) half.
    """
    h1_a, h1_b = meas_ctx.h1_bar_a, meas_ctx.h1_bar_b
    if (
        h1_a is None
        or h1_b is None
        or meas_ctx.exp_t1_a is None
        or meas_ctx.exp_t1_b is None
        or meas_ctx.chol_bar_a is None
        or meas_ctx.chol_bar_b is None
    ):
        raise ValueError(
            "energy_kernel_uw_uh_bar needs the similarity transformed hamiltonian; build "
            "the measurement context with meas.upt2ccsd_bar_uh.build_meas_ctx."
        )

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

    gt2g, t2_green_a, t2_green_b = t2_one_body(
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
        e1_2=e1_2_1 + e1_2_2,  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>
    )


def _bar_chunk_terms(
    chol_a_c: jax.Array,
    chol_b_c: jax.Array,
    bw: _BarWalker,
    trial_data: Upt2ccsdTrial,
    t2_r: tuple[jax.Array, jax.Array, jax.Array],
    rtype: Any,
    ctype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    The two body terms of one chunk of k cholesky vectors, per vector:

        e2_0_g      <h2>                  complex128, exact
        e2_2_2_1_g  -tr(L t2_green) tr(G L)
        e2_2_2_2_g  G L . L t2_green
        e2_2_3_g    L t2 L

    The last three are in ctype.
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
    e2_2_3_g = l2t2_g(glgp_a, glgp_b, t2_r)

    return e2_0_c.astype(jnp.complex128), e2_2_2_1_g, e2_2_2_2_g, e2_2_3_g


def energy_kernel_uw_uh_bar(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> jax.Array:
    """
    (t2, e0, e1) of one unrestricted walker. ham_data is unused: the transformed tensors
    are in meas_ctx, built once by build_meas_ctx.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    bw = _bar_walker(walker, meas_ctx, trial_data)
    chol_bar_a, chol_bar_b = meas_ctx.chol_bar_a, meas_ctx.chol_bar_b
    assert chol_bar_a is not None and chol_bar_b is not None

    chol_a, _, _, _ = pad_reshape_chol(chol_bar_a, meas_ctx.nchol_chunk)
    chol_b, _, _, _ = pad_reshape_chol(chol_bar_b, meas_ctx.nchol_chunk)

    t2_r = (
        trial_data.t2aa.astype(rtype),
        trial_data.t2ab.astype(rtype),
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

    t2 = bw.gt2g  # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
    e0 = bw.e1_0 + e2_0  # <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
    e1 = bw.e1_2 + e2_2  # <exp(T1)HF|T2 (h1+h2)|walker>/<exp(T1)HF|walker>

    return jnp.stack([t2, e0, e1])


def energy_kernel_uw_uh_bar_scalar(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> jax.Array:
    """The combined local energy of one walker, for checks against the other kernels."""
    return combine_first_order_energy(
        ham_data.h0, energy_kernel_uw_uh_bar(walker, ham_data, meas_ctx, trial_data)
    )


def make_upt2ccsd_bar_meas_ops(
    sys: Any,
    *,
    mixed_precision: bool = True,
    testing: bool = False,
    nchol_chunk: int | None = None,
) -> MeasOps:
    """MeasOps of the upt2ccsd_bar trial on the uchol hamiltonian; see make_upt2ccsd_meas_ops."""
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError(
            "unrestricted pt2CCSD MeasOps require walker_kind='unrestricted', "
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
    object.__setattr__(meas_ops, _UPT2CCSD_BAR_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_upt2ccsd_bar_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _UPT2CCSD_BAR_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, Pt2ccsdChunkMeasCfg) else None


def plan_chunking_for_run_u_bar(
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
    """plan_chunking_for_run_u with the bar kernel's memory model."""
    return plan_chunking_for_run_u(
        sys,
        ham_data,
        trial_data,
        n_walkers=n_walkers,
        budget_bytes=budget_bytes,
        n_chunks=n_chunks,
        nchol_chunk=nchol_chunk,
        mixed_precision=mixed_precision,
        n_devices=n_devices,
        bar=True,
    )
