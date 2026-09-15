from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy
from ..ham.chol_u import HamCholU
from ..trial.upt2ccsd import Upt2ccsdTrial, overlap_u
from .pt2ccsd import (
    _MEMORY_MODES,
    _PT2CCSD_MEAS_CFG_ATTR,
    DEFAULT_NCHOL_CHUNK,
    ChunkPlan,
    Pt2ccsdMeasCfg,
    Pt2ccsdMemoryModel,
    _equal_chunks,
    chol_sampling_proposal,
    max_equal_chunk_pad,
    plan_pt2ccsd_chunking,
    resolve_chol_budget,
    t1_from_mo_t,
)

# Unrestricted pt2CCSD energy estimators, ported from afqmc's upt2ccsd, upt2ccsd_bar and
# upt2ccsd_sto_chol (afqmc/wavefunctions/wavefunctions_unrestricted.py).
#
# They measure against the unrestricted (uchol) hamiltonian, where alpha and beta each
# keep their own MO basis and share only the cholesky field index. The walker is the
# (walker_up, walker_dn) pair, and every kernel returns (t2, e0, e1) per walker, so the
# block function and pt2ccsd_blocking are shared with the restricted estimators:
#
#     E = h0 + <e0> + <e1> - <t2><e0>
#
# The config, the chunking and the semistochastic head/tail machinery are the restricted
# ones from meas/pt2ccsd.py, reused rather than copied, so Pt2ccsdMeasCfg means the same
# thing for either spin treatment.
#
# afqmc also returns <exp(T1)HF|walker>; here that is MeasOps.overlap, as in the
# restricted port. afqmc sizes both spins with a single norb; here each spin takes its
# own, so norb_a != norb_b works as it does for HamCholU.

_MEASURE_TYPES_U = ("chunk", "bar", "sto_chol")


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdMeasCtx:
    cfg: Pt2ccsdMeasCfg  # static
    # cholesky vectors per scan step, resolved against the hamiltonian in build_meas_ctx;
    # static, since it sets the shape the chol tensors are reshaped to
    nchol_chunk: int = 1

    # similarity transformed intermediates per spin, only built for "bar" and "sto_chol"
    exp_t1_a: jax.Array | None = None  # (norb_a, norb_a)
    exp_t1_b: jax.Array | None = None  # (norb_b, norb_b)
    h1_bar_a: jax.Array | None = None  # (norb_a, norb_a)
    h1_bar_b: jax.Array | None = None  # (norb_b, norb_b)
    chol_bar_a: jax.Array | None = None  # (nchol, norb_a, norb_a)
    chol_bar_b: jax.Array | None = None  # (nchol, norb_b, norb_b)

    def tree_flatten(self):
        children = (
            self.exp_t1_a,
            self.exp_t1_b,
            self.h1_bar_a,
            self.h1_bar_b,
            self.chol_bar_a,
            self.chol_bar_b,
        )
        aux = (self.cfg, self.nchol_chunk)
        return children, aux

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


def _nchol(ham_data: HamCholU) -> int:
    nchol = ham_data.nchol
    return int(nchol) if nchol is not None else int(ham_data.chol_a.shape[0])


def upt2ccsd_memory_model(
    *,
    norb: tuple[int, int],
    nocc: tuple[int, int],
    nchol: int,
    n_walkers: int,
    real_bytes: int = 8,
    complex_bytes: int = 16,
    bar: bool = False,
) -> Pt2ccsdMemoryModel:
    """
    pt2ccsd_memory_model for the unrestricted kernels: every term is counted per spin and
    summed, since the two spins carry separate hamiltonians, greens functions and chunk
    intermediates over one shared cholesky index. The amplitudes add the opposite spin
    block t2ab, and the t2ab contraction's output rides on the alpha chunk intermediates.

    bar=False models energy_kernel_uw_uh_chunk, bar=True energy_kernel_uw_uh_bar and
    energy_kernel_uw_uh_sto. As for the restricted model, it counts the arrays the kernel
    names, not XLA's transient buffers, so treat it as a floor.
    """
    f8, c16 = 8, 16
    r, c = int(real_bytes), int(complex_bytes)
    norb_a, norb_b = (int(n) for n in norb)
    nocc_a, nocc_b = (int(n) for n in nocc)
    ov_a = nocc_a * (norb_a - nocc_a)
    ov_b = nocc_b * (norb_b - nocc_b)
    pad = max_equal_chunk_pad(nchol)

    t2 = ov_a * ov_a + ov_a * ov_b + ov_b * ov_b
    resident = (
        t2 * f8  # trial_data.t2aa, t2ab, t2bb
        + (t2 * r if r != f8 else 0)  # their casts, walker independent so hoisted
        + n_walkers * (norb_a * nocc_a + norb_b * nocc_b) * c16  # the population
    )
    per_walker = 0
    per_walker_chol = 0

    # (norb, nocc, ov of this spin, ov of the block t2ab pairs it with)
    for norb_s, nocc_s, ov_s, ov_cross in ((norb_a, nocc_a, ov_a, ov_b), (norb_b, nocc_b, ov_b, 0)):
        nvir_s = norb_s - nocc_s
        n2 = norb_s * norb_s
        chol = nchol * n2 * f8
        chol_padded = (nchol + pad) * n2 * f8

        if bar:
            resident += (
                chol  # ham_data.chol_s, still resident for the guide propagator
                + chol  # chol_bar_s
                + chol_padded  # its padded, reshaped copy
                + 3 * n2 * f8  # exp_t1_s, exp_mt1_s, h1_bar_s
            )
            per_walker += c16 * (
                2 * norb_s * nocc_s  # walker and walker_bar
                + nocc_s * norb_s  # the half green
                + norb_s * nvir_s  # greenp
                + n2  # t2_green
                + 2 * ov_s  # t2g, t2g_ab
            )
            # gl and its cast, lt2_green; glgp and the same spin lt2, plus t2ab's lt2
            per_walker_chol += nocc_s * norb_s * (c16 + 2 * c) + ov_s * (2 * c) + ov_cross * c
        else:
            resident += chol + chol_padded  # ham_data.chol_s and the kernel's padded copy
            per_walker += c16 * (
                norb_s * nocc_s  # walker
                + n2  # green
                + norb_s * nvir_s  # greenp
                + n2  # t2_green
                + 2 * ov_s  # t2g, t2g_ab
            )
            # gl, lt2g and their casts; glgp, its cast and the same spin lt2, plus t2ab's lt2
            per_walker_chol += n2 * (2 * c16 + 2 * c) + ov_s * (c16 + 2 * c) + ov_cross * c

    return Pt2ccsdMemoryModel(
        resident=resident,
        per_walker=per_walker,
        per_walker_chol=per_walker_chol,
    )


def plan_chunking_for_run_u(
    sys: Any,
    ham_data: HamCholU,
    trial_data: Upt2ccsdTrial,
    *,
    n_walkers: int,
    max_memory_mb: float,
    n_chunks: int = 1,
    nchol_chunk: int | None = None,
    mixed_precision: bool = False,
    n_devices: int = 1,
    measure_type: str = "chunk",
) -> ChunkPlan:
    """Build the memory model for a run and plan its chunking. The MixedRecipe hook."""
    nchol = _nchol(ham_data)
    model = upt2ccsd_memory_model(
        norb=trial_data.norb,
        nocc=trial_data.nocc,
        nchol=nchol,
        n_walkers=n_walkers,
        real_bytes=4 if mixed_precision else 8,
        complex_bytes=8 if mixed_precision else 16,
        bar=measure_type in ("bar", "sto_chol"),
    )
    return plan_pt2ccsd_chunking(
        model,
        n_walkers=n_walkers,
        nchol=nchol,
        budget_bytes=int(max_memory_mb * (1024**2)),
        n_chunks=n_chunks,
        nchol_chunk=nchol_chunk,
        n_devices=n_devices,
    )


def build_bar_intermediates_u(ham_data: HamCholU, trial_data: Upt2ccsdTrial) -> dict:
    """
    build_bar_intermediates, once per spin: each spin's exp(T1) moves onto that spin's
    hamiltonian and walker,

        exp_t1_s   = 1 + X_s,  X_s[:nocc_s, nocc_s:] = t1_s    (X_s**2 = 0, so this is exact)
        exp_mt1_s  = 1 - X_s
        h1_bar_s   = exp_t1_s @ h1_s   @ exp_mt1_s
        chol_bar_s = exp_t1_s @ chol_s @ exp_mt1_s              (per cholesky vector)

    The two transforms never mix, since alpha and beta have separate orbital spaces.
    """
    nocc_a, nocc_b = trial_data.nocc
    out = {}
    for s, mo_t, nocc, h1, chol in (
        ("a", trial_data.mo_t_a, nocc_a, ham_data.h1_a, ham_data.chol_a),
        ("b", trial_data.mo_t_b, nocc_b, ham_data.h1_b, ham_data.chol_b),
    ):
        norb = mo_t.shape[0]
        t1 = t1_from_mo_t(mo_t, nocc)
        x = jnp.zeros((norb, norb), dtype=t1.dtype).at[:nocc, nocc:].set(t1)
        eye = jnp.eye(norb, dtype=t1.dtype)
        exp_t1 = eye + x
        exp_mt1 = eye - x
        out[f"exp_t1_{s}"] = exp_t1
        out[f"exp_mt1_{s}"] = exp_mt1
        out[f"h1_bar_{s}"] = exp_t1 @ h1 @ exp_mt1
        out[f"chol_bar_{s}"] = jnp.einsum(
            "pr,grs,sq->gpq", exp_t1, chol, exp_mt1, optimize="optimal"
        )
    return out


def build_meas_ctx(
    ham_data: HamCholU,
    trial_data: Upt2ccsdTrial,
    cfg: Pt2ccsdMeasCfg = Pt2ccsdMeasCfg(measure_type="chunk"),
) -> Upt2ccsdMeasCtx:
    if ham_data.basis != "uchol":
        raise ValueError(
            "unrestricted pt2CCSD MeasOps assume the unrestricted hamiltonian, "
            f"HamCholU.basis == 'uchol'; got {ham_data.basis!r}."
        )
    if cfg.measure_type not in _MEASURE_TYPES_U:
        raise ValueError(
            f"unknown measure_type {cfg.measure_type!r}; expected one of {_MEASURE_TYPES_U}"
        )
    if cfg.memory_mode not in _MEMORY_MODES:
        raise ValueError(
            f"unknown memory_mode {cfg.memory_mode!r}; expected one of {_MEMORY_MODES}"
        )

    nchol = _nchol(ham_data)
    requested = DEFAULT_NCHOL_CHUNK if cfg.nchol_chunk is None else int(cfg.nchol_chunk)
    if requested < 1:
        raise ValueError(f"nchol_chunk must be >= 1, got {cfg.nchol_chunk}")

    # cfg.nchol_chunk is a cap; resolve the even division the kernels will really scan
    cap = min(requested, nchol) if nchol > 0 else requested
    _, nchol_chunk, _ = _equal_chunks(nchol, cap)

    needs_bar = cfg.measure_type in ("bar", "sto_chol")
    bar = build_bar_intermediates_u(ham_data, trial_data) if needs_bar else {}

    return Upt2ccsdMeasCtx(
        cfg=cfg,
        nchol_chunk=nchol_chunk,
        exp_t1_a=bar.get("exp_t1_a"),
        exp_t1_b=bar.get("exp_t1_b"),
        h1_bar_a=bar.get("h1_bar_a"),
        h1_bar_b=bar.get("h1_bar_b"),
        chol_bar_a=bar.get("chol_bar_a"),
        chol_bar_b=bar.get("chol_bar_b"),
    )


# ---------------------------------------------------------------------------------------
# pieces shared by the kernels
# ---------------------------------------------------------------------------------------


def _t2_one_body(
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
    spin, the (norb_s, norb_s) matrix every T2 term with a one- or two-body operator goes
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


def _l2t2_g(
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


def _e2_0_g(gl_a: jax.Array, gl_b: jax.Array) -> tuple[jax.Array, jax.Array]:
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


class _BarWalker(NamedTuple):
    """Chunk independent, per walker intermediates of the bar and sto_chol kernels."""

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
    name: str,
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
            f"{name} needs the similarity transformed hamiltonian; build the measurement "
            "context with measure_type='bar' or 'sto_chol'."
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
    The two-body terms of one chunk of k cholesky vectors, per vector:

        e2_0_g      <h2>                  complex128, exact
        e2_2_2_1_g  -tr(L t2_green) tr(G L)
        e2_2_2_2_g  G L . L t2_green
        e2_2_3_g    L t2 L

    The last three are in ctype. The bar kernel sums them; the sto_chol kernel weights
    them, so both see the same numbers by construction.
    """
    nocc_a, nocc_b = trial_data.nocc

    gl_a = jnp.einsum("ir,gqr->giq", bw.green_a, chol_a_c, optimize="optimal")  # (k, nocc_a, norb_a)
    gl_b = jnp.einsum("ir,gqr->giq", bw.green_b, chol_b_c, optimize="optimal")  # (k, nocc_b, norb_b)
    e2_0_g, tr_gl = _e2_0_g(gl_a[:, :, :nocc_a], gl_b[:, :, :nocc_b])

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

    return e2_0_g.astype(jnp.complex128), e2_2_2_1_g, e2_2_2_2_g, e2_2_3_g


def _pad_reshape(chol: jax.Array, n_chunks: int, chunk: int, npad: int) -> jax.Array:
    if npad:
        chol = jnp.pad(chol, ((0, npad), (0, 0), (0, 0)))
    return chol.reshape(n_chunks, chunk, *chol.shape[-2:])


# ---------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------


def energy_kernel_uw_uh_chunk(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> jax.Array:
    """
    The unrestricted pt2CCSD estimator (afqmc's upt2ccsd), measured against the trial
    exp(T1)|HF> in each spin's MO basis, with the two-body sum scanned over chunks of
    meas_ctx.nchol_chunk cholesky vectors.

    The greens functions are full (norb_s, norb_s) matrices here. The T2 contractions run
    in the mixed dtypes from meas_ctx.cfg; e2_0 and every partial sum stay in complex128.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    nocc_a, nocc_b = trial_data.nocc
    wu, wd = walker
    mo_a, mo_b = trial_data.mo_t_a, trial_data.mo_t_b
    h1_a, h1_b = ham_data.h1_a, ham_data.h1_b
    chol_a, chol_b = ham_data.chol_a, ham_data.chol_b

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
    gt2g, t2_green_a, t2_green_b = _t2_one_body(
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
    nchunks, nchol_chunk, pad = _equal_chunks(chol_a.shape[0], meas_ctx.nchol_chunk)
    chol_a = _pad_reshape(chol_a, nchunks, nchol_chunk, pad)
    chol_b = _pad_reshape(chol_b, nchunks, nchol_chunk, pad)

    t2_r = (trial_data.t2aa.astype(rtype), trial_data.t2ab.astype(rtype), trial_data.t2bb.astype(rtype))

    def scanned_fun(carry, x):
        chol_a_c, chol_b_c = x  # (k, norb_a, norb_a), (k, norb_b, norb_b)

        # e2_0
        gl_a = jnp.einsum("pr,gqr->gpq", green_a, chol_a_c, optimize="optimal")
        gl_b = jnp.einsum("pr,gqr->gpq", green_b, chol_b_c, optimize="optimal")
        e2_0_g, tr_gl = _e2_0_g(gl_a, gl_b)
        carry[0] += jnp.sum(e2_0_g).astype(c128)

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
        carry[3] += jnp.sum(_l2t2_g(glgp_a.astype(ctype), glgp_b.astype(ctype), t2_r)).astype(c128)

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


def energy_kernel_uw_uh_bar(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> jax.Array:
    """
    The unrestricted pt2CCSD estimator with exp(T1) applied to the right (afqmc's
    upt2ccsd_bar).

    Same (t2, e0, e1) as energy_kernel_uw_uh_chunk, reached from each spin's similarity
    transformed hamiltonian in meas_ctx and the walker exp_t1_s @ walker_s, measured
    against the bare reference. The greens functions are (nocc_s, norb_s) halves, so
    every chunk intermediate is (k, nocc_s, norb_s) rather than (k, norb_s, norb_s).

    ham_data is unused: the transformed tensors are in meas_ctx, built once by
    build_bar_intermediates_u.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    bw = _bar_walker(walker, meas_ctx, trial_data, "energy_kernel_uw_uh_bar")
    chol_a, chol_b = meas_ctx.chol_bar_a, meas_ctx.chol_bar_b
    assert chol_a is not None and chol_b is not None

    nchunks, nchol_chunk, pad = _equal_chunks(chol_a.shape[0], meas_ctx.nchol_chunk)
    chol_a = _pad_reshape(chol_a, nchunks, nchol_chunk, pad)
    chol_b = _pad_reshape(chol_b, nchunks, nchol_chunk, pad)

    t2_r = (trial_data.t2aa.astype(rtype), trial_data.t2ab.astype(rtype), trial_data.t2bb.astype(rtype))

    def scanned_fun(carry, x):
        e2_0_g, e2_2_2_1_g, e2_2_2_2_g, e2_2_3_g = _bar_chunk_terms(
            x[0], x[1], bw, trial_data, t2_r, rtype, ctype
        )
        carry[0] += jnp.sum(e2_0_g)
        carry[1] += jnp.sum(e2_2_2_1_g).astype(c128)
        carry[2] += jnp.sum(e2_2_2_2_g).astype(c128)
        carry[3] += jnp.sum(e2_2_3_g).astype(c128)
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


def energy_kernel_uw_uh_sto(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
    key: jax.Array | None = None,
) -> jax.Array:
    """
    The unrestricted bar estimator with a semistochastic cholesky sum in the T2-contracted
    energy (afqmc's upt2ccsd_sto_chol).

    As in energy_kernel_rw_rh_sto: e2_0, and with it e2_2_1 = e2_0 * gt2g, stays exact,
    and the three T2-contracted accumulators are split into an exactly summed head and an
    importance sampled tail. Both spins share one head/tail split and one set of draws,
    since they share the cholesky index, and the proposal is scored by the spin summed
    e2_0 per vector.

    n_chol_head="full" removes the sampling and reproduces energy_kernel_uw_uh_bar
    exactly; no key is drawn or needed in that limit. The tail is scanned over indices,
    with the gather inside the scan body, for the memory reason given on the restricted
    kernel.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    nocc_a, nocc_b = trial_data.nocc
    bw = _bar_walker(walker, meas_ctx, trial_data, "energy_kernel_uw_uh_sto")
    chol_a, chol_b = meas_ctx.chol_bar_a, meas_ctx.chol_bar_b
    assert chol_a is not None and chol_b is not None
    nchol = chol_a.shape[0]
    nchol_chunk = meas_ctx.nchol_chunk
    t2_r = (trial_data.t2aa.astype(rtype), trial_data.t2ab.astype(rtype), trial_data.t2bb.astype(rtype))

    # ---- pass 1: e2_0 per cholesky vector, exact, no T2 anywhere ----
    # fed the half rotated chol_s[:, :nocc_s, :]: e2_0 only touches the occupied blocks
    def scan_e2_0(carry, x):
        rot_a_c, rot_b_c = x
        gl_occ_a = jnp.einsum("ir,gqr->giq", bw.green_a, rot_a_c, optimize="optimal")
        gl_occ_b = jnp.einsum("ir,gqr->giq", bw.green_b, rot_b_c, optimize="optimal")
        e2_0_g, _ = _e2_0_g(gl_occ_a, gl_occ_b)
        e2_0_g = e2_0_g.astype(c128)
        return carry + jnp.sum(e2_0_g), e2_0_g

    n_chunk1, chunk1, npad1 = _equal_chunks(nchol, nchol_chunk)
    e2_0, e2_0_chunks = lax.scan(
        scan_e2_0,
        jnp.zeros((), dtype=c128),
        (
            _pad_reshape(chol_a[:, :nocc_a, :], n_chunk1, chunk1, npad1),
            _pad_reshape(chol_b[:, :nocc_b, :], n_chunk1, chunk1, npad1),
        ),
    )
    e2_0_g = e2_0_chunks.reshape(-1)[:nchol]

    # ---- head / tail split ----
    n_head, n_samples = resolve_chol_budget(
        nchol,
        cfg.n_chol_head,
        cfg.head_chol_ratio,
        cfg.n_chol_samples,
        cfg.chol_cost_ratio,
        cfg.head_sample_ratio,
    )

    head_prefix: int | None = None
    head_idx = None
    if n_head >= nchol:
        # deterministic limit: every vector summed exactly, empty tail, no draw
        head_prefix = nchol
        tail = jnp.zeros((0,), dtype=jnp.int32)
        tail_prob = jnp.zeros((0,))
    else:
        pi_g = chol_sampling_proposal(
            e2_0_g, score_floor=cfg.chol_score_floor, uniform_mix=cfg.chol_uniform_mix
        )
        if cfg.head_from_guide:
            # per-walker ranking, at the cost of a batched gather for the head
            order = jnp.argsort(-pi_g)
            head_idx = jnp.sort(order[:n_head])
            tail = jnp.sort(order[n_head:])
        else:
            # a contiguous prefix is a plain slice, shared across the vmap batch
            head_prefix = n_head
            tail = jnp.arange(n_head, nchol, dtype=jnp.int32)
        tail_prob = pi_g[tail]
        tail_prob = tail_prob / jnp.sum(tail_prob)

    # ---- pass 2: only the accumulators that contract with T2 ----
    def accum(carry, chol_a_c, chol_b_c, w_c):
        _, e2_2_2_1_g, e2_2_2_2_g, e2_2_3_g = _bar_chunk_terms(
            chol_a_c, chol_b_c, bw, trial_data, t2_r, rtype, ctype
        )
        w_c = w_c.astype(ctype)
        carry[0] += jnp.sum(w_c * e2_2_2_1_g).astype(c128)
        carry[1] += jnp.sum(w_c * e2_2_2_2_g).astype(c128)
        carry[2] += jnp.sum(w_c * e2_2_3_g).astype(c128)
        return carry

    zero = jnp.zeros((), dtype=c128)

    def run_slice(chol_a_s, chol_b_s, weights):
        """Scan over the cholesky vectors themselves; only for the contiguous head."""
        n = weights.shape[0]
        if n == 0:
            return zero, zero, zero
        n_ch, chunk, npad = _equal_chunks(n, nchol_chunk)
        if npad:
            weights = jnp.pad(weights, (0, npad))
        out, _ = lax.scan(
            lambda carry, x: (accum(carry, x[0], x[1], x[2]), None),
            [zero, zero, zero],
            (
                _pad_reshape(chol_a_s, n_ch, chunk, npad),
                _pad_reshape(chol_b_s, n_ch, chunk, npad),
                weights.reshape(n_ch, chunk),
            ),
        )
        return out[0], out[1], out[2]

    def run_indices(idx, weights):
        """Same sum over cholesky *indices*, gathering both spins inside the scan body."""
        n = weights.shape[0]
        if n == 0:
            return zero, zero, zero
        n_ch, chunk, npad = _equal_chunks(n, nchol_chunk)
        if npad:
            # pad with index 0 at zero weight, which contributes nothing
            idx = jnp.pad(idx, (0, npad))
            weights = jnp.pad(weights, (0, npad))
        out, _ = lax.scan(
            lambda carry, x: (accum(carry, chol_a[x[0]], chol_b[x[0]], x[1]), None),
            [zero, zero, zero],
            (idx.reshape(n_ch, chunk), weights.reshape(n_ch, chunk)),
        )
        return out[0], out[1], out[2]

    # head: exact, unit weights
    if head_prefix is not None:
        b_h, c_h, d_h = run_slice(
            chol_a[:head_prefix], chol_b[:head_prefix], jnp.ones(head_prefix, dtype=c128)
        )
    else:
        assert head_idx is not None
        b_h, c_h, d_h = run_indices(head_idx, jnp.ones(head_idx.shape[0], dtype=c128))

    # tail: sampled, so walker dependent and therefore index scanned
    if tail.shape[0] == 0:
        b_t = c_t = d_t = zero
    else:
        if key is None:
            raise ValueError(
                "energy_kernel_uw_uh_sto draws a sampled tail and so needs a PRNG key; "
                "only n_chol_head='full' can run without one."
            )
        sel = jax.random.choice(key, tail.shape[0], shape=(n_samples,), replace=True, p=tail_prob)
        samp_w = (1.0 / (n_samples * tail_prob[sel])).astype(c128)
        b_t, c_t, d_t = run_indices(tail[sel], samp_w)

    # e2_2_1 = e2_0 * gt2g is exact, since e2_0 is
    e2_2 = e2_0 * bw.gt2g + (b_h + b_t) + (c_h + c_t) + (d_h + d_t)

    e0 = bw.e1_0 + e2_0  # fully exact
    e1 = bw.e1_2 + e2_2

    return jnp.stack([bw.gt2g, e0, e1])


def make_upt2ccsd_meas_ops(
    sys: Any,
    measure_type: str = "chunk",
    memory_mode: str = "low",
    mixed_precision: bool = False,
    testing: bool = False,
    nchol_chunk: int | None = None,
    **cfg_fields: Any,
) -> MeasOps:
    """
    measure_type selects the energy kernel: "chunk" afqmc's upt2ccsd, "bar" upt2ccsd_bar
    with exp(T1) moved onto the hamiltonian, "sto_chol" upt2ccsd_sto_chol with the
    T2-contracted two-body sum sampled. All three are chunked over the cholesky index;
    there is no unchunked unrestricted kernel.

    Any remaining Pt2ccsdMeasCfg field may be given by name, as for make_pt2ccsd_meas_ops.
    """
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError(
            "unrestricted pt2CCSD MeasOps require walker_kind='unrestricted', "
            f"got: {sys.walker_kind}"
        )
    if measure_type not in _MEASURE_TYPES_U:
        raise ValueError(
            f"unknown measure_type {measure_type!r}; expected one of {_MEASURE_TYPES_U}"
        )
    if memory_mode not in _MEMORY_MODES:
        raise ValueError(f"unknown memory_mode {memory_mode!r}; expected one of {_MEMORY_MODES}")

    owned = {"measure_type", "memory_mode", "nchol_chunk"} | {
        f.name for f in fields(Pt2ccsdMeasCfg) if f.name.startswith("mixed_")
    }
    valid = {f.name for f in fields(Pt2ccsdMeasCfg)} - owned
    unknown = sorted(set(cfg_fields) - valid)
    if unknown:
        raise ValueError(
            f"unknown or non-overridable Pt2ccsdMeasCfg field(s) {unknown}; "
            f"settable here: {sorted(valid)}"
        )

    cfg = Pt2ccsdMeasCfg(
        measure_type=measure_type,
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
        nchol_chunk=nchol_chunk,
        **cfg_fields,
    )

    # a full head never draws, so the block function must not advance the key stream for it
    samples_tail = measure_type == "sto_chol" and not (
        isinstance(cfg.n_chol_head, str) and cfg.n_chol_head.lower() == "full"
    )

    energy_kernel = {
        "chunk": energy_kernel_uw_uh_chunk,
        "bar": energy_kernel_uw_uh_bar,
        "sto_chol": energy_kernel_uw_uh_sto,
    }[measure_type]

    meas_ops = MeasOps(
        overlap=overlap_u,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel},
        stochastic_kernels=frozenset({k_energy} if samples_tail else ()),
    )
    # same attribute as the restricted ops, so get_pt2ccsd_meas_cfg and the flag dump
    # need no unrestricted variant
    object.__setattr__(meas_ops, _PT2CCSD_MEAS_CFG_ATTR, cfg)
    return meas_ops
