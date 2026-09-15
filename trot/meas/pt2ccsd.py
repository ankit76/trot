from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.pt2ccsd import Pt2ccsdTrial
from ..trial.pt2ccsd import overlap_r
from .. import walkers as wk
from ..prop.types import PropState, QmcParams


_PT2CCSD_MEAS_CFG_ATTR = "_pt2ccsd_meas_cfg"

# default number of cholesky vectors per chunk when a chunking kernel is selected and no
# memory budget is given
DEFAULT_NCHOL_CHUNK = 100

_MEMORY_MODES = ("low", "high")
_MEASURE_TYPES = (None, "chunk", "bar", "sto_chol")


def _equal_chunks(n: int, max_chunk: int) -> tuple[int, int, int]:
    """
    Split n cholesky vectors into equal chunks of at most max_chunk, returning
    (n_chunks, chunk, n_pad).

    Take the fewest chunks the cap allows, then divide evenly. That gives the same number
    of scan steps as slicing at exactly max_chunk and padding the remainder, but spreads
    the vectors out, so the zero padding is the minimum a fixed scan shape admits: with
    nchol=1600 and a cap of 300 it pads 2 vectors rather than 200.

    Padding is always < n_chunks, and is bounded over all caps by about sqrt(n) -- worst
    when the chunk count and the chunk size meet, e.g. 19 vectors at n=381, cap=20.

    Every chunking kernel here uses this, on the whole cholesky set and on the subsets the
    semistochastic kernel forms (its head and its sampled tail). Subsets are why an
    exactly-max_chunk rule will not do: with a cap sized for the full set, a short head
    would pad out to one whole chunk and run the T2 contractions on mostly zeros.
    """
    if max_chunk < 1:
        raise ValueError(f"max_chunk must be >= 1, got {max_chunk}")
    n_chunks = max(1, -(-n // max_chunk))
    chunk = max(1, -(-n // n_chunks))  # max(1, ...) keeps n == 0 from giving a zero axis
    return n_chunks, chunk, n_chunks * chunk - n


def max_equal_chunk_pad(n: int) -> int:
    """
    Worst-case padding from _equal_chunks over every cap, i.e. how many zero cholesky
    vectors the padded copy can carry whatever chunk size is chosen. Small (about sqrt(n)),
    and k independent, which is what lets the memory model charge it once up front instead
    of scaling it with the chunk.
    """
    if n <= 0:
        return 1
    return max(_equal_chunks(n, cap)[2] for cap in range(1, n + 1))




# per-walker cholesky budget for "sto_chol", as a fraction of nchol, split head : samples
# by head_sample_ratio (3:1 by default)
DEFAULT_CHOL_COST_RATIO = 0.2


@dataclass(frozen=True)
class Pt2ccsdMeasCfg:
    """
    measure_type picks the energy kernel, and with it what build_meas_ctx has to prepare:

      None     energy_kernel_rw_rh, one cholesky vector per scan step
      "chunk"  energy_kernel_rw_rh_chunk, nchol_chunk vectors per scan step, with the
               two-body contractions in the mixed dtypes below
      "bar"    energy_kernel_rw_rh_bar, the same estimator with exp(T1) moved onto the
               hamiltonian and the walker instead of the trial. Also chunked, and the
               context carries the transformed tensors; see build_bar_intermediates.
      "sto_chol"
               energy_kernel_rw_rh_sto, the bar estimator with the T2-contracted part of
               the two-body sum evaluated semistochastically: an exactly summed head plus
               an importance sampled tail. Needs a PRNG key per walker, so it is declared
               in MeasOps.stochastic_kernels. The n_chol_* / head_* / chol_* fields below
               size the head and the tail; they mean nothing to the other kernels.

    nchol_chunk is the chunk size for "chunk" and "bar", defaulting to
    DEFAULT_NCHOL_CHUNK. To size it against a memory budget instead, use
    plan_pt2ccsd_chunking, which also decides whether the walkers themselves need
    chunking.

    memory_mode is the estimator's memory layout, kept for consistency with the other
    meas configs. None of the pt2CCSD kernels branch on it today.
    """

    measure_type: str | None = None  # or Literal["chunk","bar"]
    memory_mode: str = "low"  # or Literal["low","high"]
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64
    nchol_chunk: int | None = None

    # semistochastic cholesky sum, measure_type == "sto_chol" only
    n_chol_head: int | str = 0  # head size; a positive int, or "full" to disable sampling
    head_chol_ratio: float | None = None  # head as a fraction of nchol, if n_chol_head == 0
    n_chol_samples: int | None = None  # tail draws per walker per block
    chol_cost_ratio: float = DEFAULT_CHOL_COST_RATIO  # per-walker budget as a fraction of nchol
    head_sample_ratio: float = 3.0  # how that budget splits head : samples
    chol_score_floor: float = 1.0e-6  # drop proposal weight negligible against the max
    chol_uniform_mix: float = 0.01  # uniform floor on the proposal, bounds 1/pi
    head_from_guide: bool = False  # rank the head per walker instead of taking a prefix


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdMeasCtx:
    cfg: Pt2ccsdMeasCfg  # static
    # cholesky vectors per scan step, resolved against the hamiltonian in build_meas_ctx;
    # static, since it sets the shape the chol tensor is reshaped to
    nchol_chunk: int = 1

    # similarity transformed intermediates, only built for measure_type == "bar"
    exp_t1: jax.Array | None = None  # (norb, norb)
    h1_bar: jax.Array | None = None  # (norb, norb)
    chol_bar: jax.Array | None = None  # (nchol, norb, norb)

    def tree_flatten(self):
        children = (self.exp_t1, self.h1_bar, self.chol_bar)
        aux = (self.cfg, self.nchol_chunk)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        exp_t1, h1_bar, chol_bar = children
        return cls(
            cfg=cfg,
            nchol_chunk=nchol_chunk,
            exp_t1=exp_t1,
            h1_bar=h1_bar,
            chol_bar=chol_bar,
        )


@dataclass(frozen=True)
class Pt2ccsdMemoryModel:
    """
    Byte counts for one measurement pass of energy_kernel_rw_rh_chunk, split by how each
    term scales. With w walkers in flight and a chunk of k cholesky vectors:

        bytes(w, k) = resident + w*per_walker + w*k*per_walker_chol

    The two chunking knobs move w and k, so the model is what makes them comparable:
    n_chunks divides w, nchol_chunk sets k, and only the product w*k multiplies the term
    that dominates for any system worth chunking.

    Nothing shared scales with k. The scan slice is a view into the padded copy, not a new
    buffer, and _equal_chunks keeps that copy's zero padding bounded by about sqrt(nchol)
    whatever k is -- so resident carries it once, at its worst case over k.
    """

    resident: int  # alive for the whole pass, whatever the chunking
    per_walker: int  # per walker in flight, independent of the chunk size
    per_walker_chol: int  # per walker per cholesky vector in the chunk

    def bytes(self, *, walkers_in_flight: int, nchol_chunk: int) -> int:
        w, k = int(walkers_in_flight), int(nchol_chunk)
        return self.resident + w * self.per_walker + w * k * self.per_walker_chol


def pt2ccsd_memory_model(
    *,
    norb: int,
    nocc: int,
    nchol: int,
    n_walkers: int,
    real_bytes: int = 8,
    complex_bytes: int = 16,
    bar: bool = False,
) -> Pt2ccsdMemoryModel:
    """
    Estimate what energy_kernel_rw_rh_chunk holds, by counting the arrays it names.

    real_bytes / complex_bytes are the mixed dtypes; the hamiltonian, the walkers and the
    greens-function intermediates are f8/c16 regardless, since only the T2 contractions
    are cast down.

    resident
      chol            nchol*norb^2 * 8    the hamiltonian tensor
      chol (padded)  (nchol + p)*norb^2 * 8   the kernel's own zero padded copy, where p
                                          is the worst-case padding _equal_chunks can
                                          leave over any chunk size (about sqrt(nchol))
      t2              no^2*nv^2    * 8    the trial amplitudes
      t2 (cast)       no^2*nv^2    * r    hoisted out of the scan, only if r != 8
      walkers      n_w*norb*nocc   * 16   the whole population stays resident

    per_walker (all c16)
      walker, green, greenp, t2_green and its two halves, t2g and its two halves

    per_walker_chol
      norb^2 * (2*16 + 2*c)   gl_c and lt2g_c, plus their casts
      no*nv  * (16 + 3*c)     glgp_c and its cast, lt2_1, lt2_2

    bar=True models energy_kernel_rw_rh_bar instead, which trades resident memory for
    chunk memory: chol_bar and its half rotation are two further copies of the cholesky
    tensor, but the half green makes every chunk intermediate (k, nocc, norb) rather than
    (k, norb, norb). It is the better trade whenever nocc << norb, which is the regime
    that needs chunking in the first place.

    It counts the arrays the kernel names, not XLA's transient buffers or scan double
    buffering, so treat it as a floor and leave headroom in the budget.
    """
    nvir = norb - nocc
    f8, c16 = 8, 16
    r, c = int(real_bytes), int(complex_bytes)
    n2 = norb * norb
    ov = nocc * nvir

    pad = max_equal_chunk_pad(nchol)
    chol = nchol * n2 * f8
    chol_padded = (nchol + pad) * n2 * f8
    rot = nchol * nocc * norb * f8
    rot_padded = (nchol + pad) * nocc * norb * f8
    t2 = ov * ov * f8

    common = (
        t2  # trial_data.t2
        + (ov * ov * r if r != f8 else 0)  # t2_r, walker independent so hoisted
        + n_walkers * norb * nocc * c16  # the population, not just those in flight
    )

    if bar:
        resident = (
            chol  # ham_data.chol, still resident for the guide propagator
            + chol  # chol_bar, built once by build_bar_intermediates
            + chol_padded  # its padded, reshaped copy
            + rot  # rot_chol = chol_bar[:, :nocc, :]
            + rot_padded  # its padded copy
            + 3 * n2 * f8  # exp_t1, exp_mt1, h1_bar
            + common
        )

        per_walker = c16 * (
            2 * norb * nocc  # walker and walker_bar
            + nocc * norb  # green, the half green
            + norb * nvir  # greenp
            + 3 * n2  # t2_green, t2_green_c, t2_green_e
            + 3 * ov  # t2g, t2g_c, t2g_e
        )
        # gl and its cast, lt2_green; glgp, lt2_c, lt2_e
        per_walker_chol = nocc * norb * (c16 + 2 * c) + ov * (3 * c)
    else:
        resident = (
            chol  # ham_data.chol
            + chol_padded  # the padded, reshaped copy the kernel builds
            + common
        )

        per_walker = c16 * (
            norb * nocc  # walker
            + n2  # green
            + norb * nvir  # greenp
            + 3 * n2  # t2_green, t2_green_c, t2_green_e
            + 3 * ov  # t2g, t2g_c, t2g_e
        )
        per_walker_chol = n2 * (2 * c16 + 2 * c) + ov * (c16 + 3 * c)

    return Pt2ccsdMemoryModel(
        resident=resident,
        per_walker=per_walker,
        per_walker_chol=per_walker_chol,
    )


@dataclass(frozen=True)
class ChunkPlan:
    """How a memory budget was split between the two chunking knobs."""

    nchol_chunk: int
    n_chunks: int
    walkers_in_flight: int  # per device, after both n_chunks and sharding
    bytes_used: int
    budget_bytes: int
    model: Pt2ccsdMemoryModel
    note: str

    def describe(self) -> str:
        mb = 1024**2
        return (
            f"nchol_chunk={self.nchol_chunk}, n_chunks={self.n_chunks} "
            f"({self.walkers_in_flight} walkers in flight), "
            f"{self.bytes_used / mb:.1f} / {self.budget_bytes / mb:.1f} MB [{self.note}]"
        )


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _budget_error(model: Pt2ccsdMemoryModel, budget_bytes: int, detail: str) -> ValueError:
    mb = 1024**2
    return ValueError(
        f"max_memory of {budget_bytes / mb:.4g} MB is too small: {detail}. "
        f"Resident arrays alone (cholesky, amplitudes, walkers) need "
        f"{model.resident / mb:.4g} MB, which no amount of chunking reduces."
    )


def plan_pt2ccsd_chunking(
    model: Pt2ccsdMemoryModel,
    *,
    n_walkers: int,
    nchol: int,
    budget_bytes: int,
    n_chunks: int = 1,
    nchol_chunk: int | None = None,
    n_devices: int = 1,
) -> ChunkPlan:
    """
    Split a memory budget between the cholesky chunk and the walker chunk.

    The cholesky chunk gives way first. Both knobs multiply the same dominant term, so
    spending the budget on one or the other buys the same arithmetic; the difference is
    that walkers in flight also batch the per-walker work that no cholesky chunk touches
    -- the greens function, the determinant, the one-body amplitude contractions. So all
    the walkers stay in flight and nchol_chunk shrinks to fit.

    Only when a single cholesky vector per step still does not fit does n_chunks rise,
    and then nchol_chunk is chosen again against the smaller walker count: halving the
    walkers in flight usually buys back a chunk well above one.

    n_chunks is a floor, never lowered -- a caller who set it had a reason not visible
    here. nchol_chunk, if given, is taken as fixed and only n_chunks is derived.
    """
    n_walkers = int(n_walkers)
    n_devices = max(1, int(n_devices))
    n_chunks_floor = max(1, int(n_chunks))

    def in_flight(nc: int) -> int:
        return _ceil_div(_ceil_div(n_walkers, nc), n_devices)

    avail = budget_bytes - model.resident
    if avail <= 0:
        raise _budget_error(model, budget_bytes, "it does not even cover the resident arrays")

    def chunk_for(w: int) -> int:
        return (avail - w * model.per_walker) // (w * model.per_walker_chol)

    def walkers_for(k: int) -> int:
        return avail // (model.per_walker + k * model.per_walker_chol)

    if nchol_chunk is not None:
        k = max(1, min(int(nchol_chunk), nchol))
        w_max = walkers_for(k)
        if w_max < 1:
            raise _budget_error(
                model, budget_bytes, f"one walker at nchol_chunk={k} does not fit"
            )
        nc = max(n_chunks_floor, _ceil_div(n_walkers, w_max * n_devices))
        note = "nchol_chunk fixed by the caller"
    else:
        nc = n_chunks_floor
        k = chunk_for(in_flight(nc))

        if k >= 1:
            note = "cholesky chunk set by the budget"
        else:
            # even one cholesky vector per step overflows, so the walkers give way, and
            # the cholesky chunk is then chosen again against the smaller walker count
            w_max = walkers_for(1)
            if w_max < 1:
                raise _budget_error(
                    model, budget_bytes, "one walker at nchol_chunk=1 does not fit"
                )
            nc = max(nc, _ceil_div(n_walkers, w_max * n_devices))
            k = chunk_for(in_flight(nc))
            note = "walkers chunked; cholesky chunk re-derived"

        if k >= nchol:
            note = "whole cholesky tensor fits in one step"
        k = max(1, min(int(k), nchol))

    # the kernels divide the cap evenly, so report the size that will really be scanned
    _, k, _ = _equal_chunks(nchol, k) if nchol > 0 else (1, k, 0)

    w = in_flight(nc)
    return ChunkPlan(
        nchol_chunk=k,
        n_chunks=nc,
        walkers_in_flight=w,
        bytes_used=model.bytes(walkers_in_flight=w, nchol_chunk=k),
        budget_bytes=int(budget_bytes),
        model=model,
        note=note,
    )


def plan_chunking_for_run(
    sys: System,
    ham_data: HamChol,
    trial_data: Pt2ccsdTrial,
    *,
    n_walkers: int,
    max_memory_mb: float,
    n_chunks: int = 1,
    nchol_chunk: int | None = None,
    mixed_precision: bool = False,
    n_devices: int = 1,
    measure_type: str | None = "chunk",
) -> ChunkPlan:
    """Build the memory model for a run and plan its chunking. The MixedRecipe hook."""
    nchol = ham_data.nchol
    model = pt2ccsd_memory_model(
        norb=trial_data.norb,
        nocc=trial_data.nocc,
        nchol=int(nchol) if nchol is not None else int(ham_data.chol.shape[0]),
        n_walkers=n_walkers,
        real_bytes=4 if mixed_precision else 8,
        complex_bytes=8 if mixed_precision else 16,
        bar=measure_type in ("bar", "sto_chol"),
    )
    return plan_pt2ccsd_chunking(
        model,
        n_walkers=n_walkers,
        nchol=int(nchol) if nchol is not None else int(ham_data.chol.shape[0]),
        budget_bytes=int(max_memory_mb * (1024**2)),
        n_chunks=n_chunks,
        nchol_chunk=nchol_chunk,
        n_devices=n_devices,
    )


def build_meas_ctx(
    ham_data: HamChol, trial_data: Pt2ccsdTrial, cfg: Pt2ccsdMeasCfg = Pt2ccsdMeasCfg()
) -> Pt2ccsdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("pt2CCSD MeasOps currently assumes HamChol.basis == 'restricted'.")
    if cfg.measure_type not in _MEASURE_TYPES:
        raise ValueError(
            f"unknown measure_type {cfg.measure_type!r}; expected one of {_MEASURE_TYPES}"
        )
    if cfg.memory_mode not in _MEMORY_MODES:
        raise ValueError(
            f"unknown memory_mode {cfg.memory_mode!r}; expected one of {_MEMORY_MODES}"
        )

    nchol = ham_data.nchol
    nchol = int(nchol) if nchol is not None else int(ham_data.chol.shape[0])
    requested = DEFAULT_NCHOL_CHUNK if cfg.nchol_chunk is None else int(cfg.nchol_chunk)
    if requested < 1:
        raise ValueError(f"nchol_chunk must be >= 1, got {cfg.nchol_chunk}")

    # cfg.nchol_chunk is a cap; what the kernels actually scan is the even division of it,
    # so resolve that here and let everything downstream -- kernels, the flags dump, the
    # memory accounting -- see the size that really runs
    cap = min(requested, nchol) if nchol > 0 else requested
    _, nchol_chunk, _ = _equal_chunks(nchol, cap)

    # sto_chol is the bar estimator with a sampled two-body sum, so it needs the same
    # transformed tensors
    needs_bar = cfg.measure_type in ("bar", "sto_chol")
    bar = build_bar_intermediates(ham_data, trial_data) if needs_bar else {}

    return Pt2ccsdMeasCtx(
        cfg=cfg,
        nchol_chunk=nchol_chunk,
        exp_t1=bar.get("exp_t1"),
        h1_bar=bar.get("h1_bar"),
        chol_bar=bar.get("chol_bar"),
    )


def t1_from_mo_t(mo_t: jax.Array, nocc: int) -> jax.Array:
    """
    Recover the singles amplitudes from the Thouless reference.

    trot stages mo_t as exp_t1[:nocc].T, so its occupied block is the identity and its
    virtual block is t1.T. Dividing the gauge out anyway keeps this correct for any
    equivalent mo_t, since the orbitals of a determinant are fixed only up to right
    multiplication by an nocc x nocc matrix.
    """
    return (mo_t[nocc:, :] @ jnp.linalg.inv(mo_t[:nocc, :])).T  # (nocc, nvir)


def build_bar_intermediates(ham_data: HamChol, trial_data: Pt2ccsdTrial) -> dict:
    """
    Move exp(T1) off the trial and onto the hamiltonian and the walker.

    The T1 generator has only an occupied-virtual block, so its matrix X squares to zero
    and exp(X) = 1 + X exactly -- no expm needed, and the inverse is 1 - X.

        exp_t1  = 1 + X,  X[:nocc, nocc:] = t1
        exp_mt1 = 1 - X
        h1_bar   = exp_t1 @ h1   @ exp_mt1
        chol_bar = exp_t1 @ chol @ exp_mt1     (per cholesky vector)

    With the hamiltonian carrying the transformation, the trial in the energy kernel is
    the bare reference determinant and the walker becomes exp_t1 @ walker. The estimator
    is unchanged: the same e0, e1 and t2 come out, and the overlap det(mo_t^T walker) is
    literally det((exp_t1 @ walker)[:nocc]), so trial.overlap_r needs no change either.

    What it buys is the greens function: against the bare reference only its occupied
    rows are nonzero, so the kernel carries an (nocc, norb) half green rather than a full
    (norb, norb) one, and every chunk intermediate shrinks with it. What it costs is
    chol_bar, a second copy of the cholesky tensor.
    """
    norb, nocc = trial_data.norb, trial_data.nocc
    t1 = t1_from_mo_t(trial_data.mo_t, nocc)

    x = jnp.zeros((norb, norb), dtype=t1.dtype).at[:nocc, nocc:].set(t1)
    eye = jnp.eye(norb, dtype=t1.dtype)
    exp_t1 = eye + x
    exp_mt1 = eye - x

    h1_bar = exp_t1 @ ham_data.h1 @ exp_mt1
    chol_bar = jnp.einsum(
        "pr,grs,sq->gpq", exp_t1, ham_data.chol, exp_mt1, optimize="optimal"
    )

    return {"exp_t1": exp_t1, "exp_mt1": exp_mt1, "h1_bar": h1_bar, "chol_bar": chol_bar}


def _greens_restricted(walker: jax.Array, mo_t: jax.Array) -> jax.Array:
    return (walker @ (jnp.linalg.inv(mo_t.T @ walker)) @ mo_t.T).T


def _greenp_from_green(green: jax.Array, nocc: int) -> jax.Array:
    norb = green.shape[0]
    return (green - jnp.eye(norb))[:, nocc:]


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: Pt2ccsdMeasCtx, trial_data: Pt2ccsdTrial
) -> jax.Array:
    mo_t, t2 = trial_data.mo_t, trial_data.t2
    nocc = trial_data.nocc

    green = _greens_restricted(walker, mo_t)  # (norb, norb)
    greenp = _greenp_from_green(green, nocc)  # (norb, nvir)

    h1 = ham_data.h1
    chol = ham_data.chol

    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2 * hg

    # one-body double excitations
    t2g_c = jnp.einsum("iajb,ia->jb", t2, green[:nocc, nocc:], optimize="optimal")
    t2g_e = jnp.einsum("iajb,ib->ja", t2, green[:nocc, nocc:], optimize="optimal")
    t2_green_c = (greenp @ t2g_c.T) @ green[:nocc, :]
    t2_green_e = (greenp @ t2g_e.T) @ green[:nocc, :]
    t2_green = 2 * t2_green_c - t2_green_e
    t2g = 2 * t2g_c - t2g_e
    gt2g = jnp.einsum("ia,ia->", t2g, green[:nocc, nocc:], optimize="optimal")
    e1_2_1 = 2 * hg * gt2g
    e1_2_2 = -2 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

    # two body energy
    lg = jnp.einsum("gpq,pq->g", chol, green, optimize="optimal")

    # two body double excitations
    lt2g = jnp.einsum("gpq,pq->g", chol, t2_green, optimize="optimal")
    e2_2_2_1 = -lt2g @ lg

    def scanned_fun(carry, x):
        chol_i = x
        # e2_0
        gl_i = jnp.einsum("pr,qr->pq", green, chol_i, optimize="optimal")
        e2_0_1_i = (2 * jnp.trace(gl_i)) ** 2 / 2.0
        e2_0_2_i = -jnp.einsum("pq,qp->", gl_i, gl_i, optimize="optimal")
        carry[0] += e2_0_1_i + e2_0_2_i
        # e2_2_2_2
        lt2_green_i = jnp.einsum("pr,qr->pq", chol_i, t2_green, optimize="optimal")
        carry[1] += 0.5 * jnp.einsum("pq,pq->", gl_i, lt2_green_i, optimize="optimal")
        # e2_2_3
        glgp_i = jnp.einsum("iq,qa->ia", gl_i[:nocc, :], greenp, optimize="optimal")
        l2t2_1 = jnp.einsum("ia,jb,iajb->", glgp_i, glgp_i, t2, optimize="optimal")
        l2t2_2 = jnp.einsum("ib,ja,iajb->", glgp_i, glgp_i, t2, optimize="optimal")
        carry[2] += 2 * l2t2_1 - l2t2_2
        return carry, 0.0

    [e2_0, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0, 0.0], chol)
    e2_2_1 = e2_0 * gt2g
    e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    t2 = gt2g  # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
    e0 = e1_0 + e2_0  # * t1 # <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
    e1 = e1_2 + e2_2  # * t1 # <exp(T1)HF|T2 (h1+h2)|walker>/<exp(T1)HF|walker>

    return jnp.stack([t2, e0, e1])


def energy_kernel_rw_rh_chunk(
    walker: jax.Array, ham_data: HamChol, meas_ctx: Pt2ccsdMeasCtx, trial_data: Pt2ccsdTrial
) -> jax.Array:
    """
    Same estimator as energy_kernel_rw_rh, with the two-body scan run over chunks of
    meas_ctx.nchol_chunk cholesky vectors and its heaviest contractions carried out in
    the mixed dtypes from meas_ctx.cfg.

    Chunking trades memory for speed: a step holds nchol_chunk copies of the (norb, norb)
    and (nocc, nvir) intermediates but contracts them in one call. The one-body terms and
    the e2_0 accumulation stay in full precision; only the T2 pieces are cast down, and
    every partial sum is accumulated back in complex128.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype

    mo_t, t2 = trial_data.mo_t, trial_data.t2
    nocc = trial_data.nocc

    green = _greens_restricted(walker, mo_t)  # (norb, norb)
    greenp = _greenp_from_green(green, nocc)  # (norb, nvir)

    h1 = ham_data.h1
    chol = ham_data.chol
    norb = chol.shape[-1]

    hg = jnp.einsum("pq,pq->", h1, green, optimize="optimal")
    e1_0 = 2 * hg

    # one-body double excitations
    t2g_c = jnp.einsum("iajb,ia->jb", t2, green[:nocc, nocc:], optimize="optimal")
    t2g_e = jnp.einsum("iajb,ib->ja", t2, green[:nocc, nocc:], optimize="optimal")
    t2_green_c = (greenp @ t2g_c.T) @ green[:nocc, :]
    t2_green_e = (greenp @ t2g_e.T) @ green[:nocc, :]
    t2_green = 2 * t2_green_c - t2_green_e
    t2g = 2 * t2g_c - t2g_e
    gt2g = jnp.einsum("ia,ia->", t2g, green[:nocc, nocc:], optimize="optimal")
    e1_2_1 = 2 * hg * gt2g
    e1_2_2 = -2 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2  # <exp(T1)HF|T2 h1|walker>/<exp(T1)HF|walker>

    # two body energy, chunked over the cholesky index. _equal_chunks divides the set
    # evenly under the requested cap; the leftover is zero padded and contributes nothing.
    nchunks, nchol_chunk, pad = _equal_chunks(chol.shape[0], meas_ctx.nchol_chunk)
    chol = jnp.pad(chol, ((0, pad), (0, 0), (0, 0)))
    chol = chol.reshape(nchunks, nchol_chunk, norb, norb)

    t2_r = t2.astype(rtype)

    def scanned_fun(carry, chol_c):
        # chol_c: (nchol_chunk, norb, norb)
        # e2_0
        gl_c = jnp.einsum("pr,gqr->gpq", green, chol_c, optimize="optimal")
        tr_gl_c = jnp.einsum("gpp->g", gl_c, optimize="optimal")
        e2_0_1_c = jnp.sum((2 * tr_gl_c) ** 2) / 2.0
        e2_0_2_c = -jnp.einsum("gpq,gqp->", gl_c, gl_c, optimize="optimal")
        carry[0] += e2_0_1_c + e2_0_2_c

        # e2_2_2_1 and e2_2_2_2
        lt2g_c = jnp.einsum("gpr,qr->gpq", chol_c, t2_green, optimize="optimal")
        tr_lt2g_c = jnp.einsum("gpp->g", lt2g_c, optimize="optimal")
        carry[1] += -jnp.einsum(
            "g,g->", tr_lt2g_c.astype(ctype), tr_gl_c.astype(ctype), optimize="optimal"
        ).astype(jnp.complex128)
        carry[2] += 0.5 * jnp.einsum(
            "gpq,gpq->", gl_c.astype(ctype), lt2g_c.astype(ctype), optimize="optimal"
        ).astype(jnp.complex128)

        # e2_2_3
        glgp_c = jnp.einsum("giq,qa->gia", gl_c[:, :nocc, :], greenp, optimize="optimal")
        glgp_c = glgp_c.astype(ctype)
        lt2_1 = jnp.einsum("gia,iajb->gjb", glgp_c, t2_r, optimize="optimal")
        lt2_2 = jnp.einsum("gib,iajb->gja", glgp_c, t2_r, optimize="optimal")
        l2t2_1 = jnp.einsum(
            "gjb,gjb->", lt2_1.astype(ctype), glgp_c, optimize="optimal"
        ).astype(jnp.complex128)
        l2t2_2 = jnp.einsum(
            "gja,gja->", lt2_2.astype(ctype), glgp_c, optimize="optimal"
        ).astype(jnp.complex128)
        carry[3] += (2 * l2t2_1 - l2t2_2).astype(jnp.complex128)

        return carry, 0.0

    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [0.0, 0.0, 0.0, 0.0], chol
    )

    e2_2_1 = e2_0 * gt2g
    e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    t2 = gt2g  # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
    e0 = e1_0 + e2_0  # <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
    e1 = e1_2 + e2_2  # <exp(T1)HF|T2 (h1+h2)|walker>/<exp(T1)HF|walker>

    return jnp.stack([t2, e0, e1])


def energy_kernel_rw_rh_bar(
    walker: jax.Array, ham_data: HamChol, meas_ctx: Pt2ccsdMeasCtx, trial_data: Pt2ccsdTrial
) -> jax.Array:
    """
    The pt2CCSD estimator with exp(T1) applied to the right.

    Same (t2, e0, e1) as energy_kernel_rw_rh and energy_kernel_rw_rh_chunk, reached from
    the similarity transformed hamiltonian in meas_ctx: h1_bar, chol_bar and the walker
    exp_t1 @ walker, measured against the bare reference determinant.

    Because the reference occupies the first nocc orbitals, the greens function against
    it has nonzero entries only in its first nocc rows. The kernel carries just those --
    an (nocc, norb) half green -- and the full green is that padded with zeros. Every
    contraction below is the corresponding one in energy_kernel_rw_rh_chunk with the zero
    rows dropped, which is what makes the chunk intermediates (k, nocc, norb) instead of
    (k, norb, norb).

    ham_data is unused: the transformed tensors are in meas_ctx, built once by
    build_bar_intermediates.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype

    t2 = trial_data.t2
    nocc, nvir, norb = trial_data.nocc, trial_data.nvir, trial_data.norb

    h1 = meas_ctx.h1_bar
    chol = meas_ctx.chol_bar
    if h1 is None or chol is None or meas_ctx.exp_t1 is None:
        raise ValueError(
            "energy_kernel_rw_rh_bar needs the similarity transformed hamiltonian; "
            "build the measurement context with measure_type='bar'."
        )

    walker_bar = meas_ctx.exp_t1 @ walker  # (norb, nocc)

    # half green, (nocc, norb): the full green is this padded with zero rows, since the
    # trial here is the bare reference determinant
    green = (walker_bar @ jnp.linalg.inv(walker_bar[:nocc, :])).T
    # (full green - 1) restricted to the virtual columns, (norb, nvir)
    greenp = jnp.vstack((green[:, nocc:], -jnp.eye(nvir, dtype=green.dtype)))
    rot_chol = chol[:, :nocc, :]  # (nchol, nocc, norb)

    # one body energy. only the occupied rows of h1 meet a nonzero row of the green
    hg = jnp.einsum("pi,pi->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2 * hg

    # one-body double excitations
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
    # rotated tensors are padded the same way, so the leftover contributes to neither.
    nchunks, nchol_chunk, pad = _equal_chunks(chol.shape[0], meas_ctx.nchol_chunk)
    chol = jnp.pad(chol, ((0, pad), (0, 0), (0, 0)))
    rot_chol = jnp.pad(rot_chol, ((0, pad), (0, 0), (0, 0)))
    chol = chol.reshape(nchunks, nchol_chunk, norb, norb)
    rot_chol = rot_chol.reshape(nchunks, nchol_chunk, nocc, norb)

    t2_r = t2.astype(rtype)

    def scanned_fun(carry, x):
        chol_c, rot_chol_c = x  # (k, norb, norb), (k, nocc, norb)

        # e2_0
        gl = jnp.einsum("ir,gqr->giq", green, chol_c, optimize="optimal")  # (k, nocc, norb)
        tr_gl = jnp.einsum("gii->g", gl[:, :, :nocc], optimize="optimal")
        e2_0_c = 2 * jnp.einsum("g,g->", tr_gl, tr_gl, optimize="optimal")
        e2_0_e = -jnp.einsum(
            "gij,gji->", gl[:, :, :nocc], gl[:, :, :nocc], optimize="optimal"
        )
        carry[0] += e2_0_c + e2_0_e

        # e2_2_2_1
        lt2g = jnp.einsum(
            "gpr,pr->g", chol_c.astype(rtype), t2_green.astype(ctype), optimize="optimal"
        )
        carry[1] += -jnp.einsum(
            "g,g->", lt2g.astype(ctype), tr_gl.astype(ctype), optimize="optimal"
        ).astype(jnp.complex128)

        # e2_2_2_2
        lt2_green = jnp.einsum(
            "gir,qr->giq",
            rot_chol_c.astype(rtype),
            t2_green.astype(ctype),
            optimize="optimal",
        )
        carry[2] += 0.5 * jnp.einsum(
            "giq,giq->", gl.astype(ctype), lt2_green.astype(ctype), optimize="optimal"
        ).astype(jnp.complex128)

        # e2_2_3
        glgp = jnp.einsum(
            "gir,rb->gib", gl.astype(ctype), greenp.astype(ctype), optimize="optimal"
        )
        lt2_c = jnp.einsum("gia,iajb->gjb", glgp, t2_r, optimize="optimal")
        lt2_e = jnp.einsum("gib,iajb->gja", glgp, t2_r, optimize="optimal")
        l2t2_c = jnp.einsum(
            "gjb,gjb->", lt2_c.astype(ctype), glgp, optimize="optimal"
        ).astype(jnp.complex128)
        l2t2_e = jnp.einsum(
            "gja,gja->", lt2_e.astype(ctype), glgp, optimize="optimal"
        ).astype(jnp.complex128)
        carry[3] += (2 * l2t2_c - l2t2_e).astype(jnp.complex128)

        return carry, 0.0

    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [0.0, 0.0, 0.0, 0.0], (chol, rot_chol)
    )

    e2_2_1 = e2_0 * gt2g
    e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    t2 = gt2g  # <exp(T1)HF|T2|walker>/<exp(T1)HF|walker>
    e0 = e1_0 + e2_0  # <exp(T1)HF|h1+h2|walker>/<exp(T1)HF|walker>
    e1 = e1_2 + e2_2  # <exp(T1)HF|T2 (h1+h2)|walker>/<exp(T1)HF|walker>

    return jnp.stack([t2, e0, e1])


def resolve_chol_budget(
    nchol: int,
    n_chol_head: int | str,
    head_chol_ratio: float | None,
    n_chol_samples: int | None,
    chol_cost_ratio: float | None,
    head_sample_ratio: float = 3.0,
) -> tuple[int, int]:
    """
    Resolve (n_head, n_samples) for the semistochastic cholesky sum.

    chol_cost_ratio fixes the per-walker budget C = chol_cost_ratio * nchol and splits it
    head : samples = head_sample_ratio : 1, i.e. n_head = 3 * n_samples by default. That
    split is near the measured optimum: at fixed cost the variance is minimised where the
    head reaches the point at which a tail vector would be drawn about once
    (n_samples * pi ~ 1), and the minimum is shallow between roughly 50% and 87% of the
    budget in the head.

    chol_cost_ratio=None falls back to DEFAULT_CHOL_COST_RATIO (0.2), so by default every
    walker touches 20% of the cholesky vectors, 3/4 of them in the head.

    Precedence, each half independently:
      n_chol_head      int > 0 or "full"  -- sets the head outright
      head_chol_ratio  not None           -- head = round(ratio * nchol)
      otherwise                           -- head = C * head_sample_ratio / (head_sample_ratio + 1)
    and for the tail:
      n_chol_samples   not None           -- used as given
      otherwise                           -- n_samples = C / (head_sample_ratio + 1)

    So head_chol_ratio and/or n_chol_samples override the chol_cost_ratio split for that
    half only.
    """
    if chol_cost_ratio is None:
        chol_cost_ratio = DEFAULT_CHOL_COST_RATIO
    if not 0.0 < chol_cost_ratio <= 1.0:
        raise ValueError(f"chol_cost_ratio must lie in (0, 1], got {chol_cost_ratio}")
    if head_sample_ratio < 0.0:
        raise ValueError("head_sample_ratio must be nonnegative")
    budget = chol_cost_ratio * nchol
    default_head = budget * head_sample_ratio / (head_sample_ratio + 1.0)
    default_samples = budget / (head_sample_ratio + 1.0)

    if isinstance(n_chol_head, str):
        if n_chol_head.lower() != "full":
            raise ValueError(f"n_chol_head must be an int or 'full', got {n_chol_head!r}")
        n_head = nchol
    elif n_chol_head > 0:
        n_head = int(n_chol_head)
    elif head_chol_ratio is not None:
        if not 0.0 <= head_chol_ratio <= 1.0:
            raise ValueError(f"head_chol_ratio must lie in [0, 1], got {head_chol_ratio}")
        n_head = int(round(head_chol_ratio * nchol))
    else:
        n_head = int(round(default_head))
    n_head = min(max(n_head, 0), nchol)

    n_samples = int(n_chol_samples) if n_chol_samples is not None else int(round(default_samples))
    return n_head, max(1, n_samples)


def chol_sampling_proposal(
    e2_g: jax.Array, *, score_floor: float, uniform_mix: float
) -> jax.Array:
    """
    Sampling probability for each cholesky vector from its two-body energy:

        pi_g = (1 - u) * |e2_g| / sum_g |e2_g|  +  u / nchol

    The floor drops guided weight negligible against the largest score; the uniform
    component then keeps pi_g >= u / nchol > 0 for every vector, which bounds the
    importance weights 1/pi_g and keeps the estimator unbiased. A vector with pi_g = 0
    would never be sampled yet would still contribute to the energy -- that is a bias, not
    extra variance.
    """
    scores = jnp.abs(e2_g)
    scores = jnp.where(scores >= score_floor * jnp.max(scores), scores, 0.0)
    nchol = scores.shape[0]
    uniform = jnp.full((nchol,), 1.0 / nchol)
    total = jnp.sum(scores)
    guided = jnp.where(total > 0.0, scores / jnp.where(total > 0.0, total, 1.0), uniform)
    return (1.0 - uniform_mix) * guided + uniform_mix * uniform


def energy_kernel_rw_rh_sto(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: Pt2ccsdMeasCtx,
    trial_data: Pt2ccsdTrial,
    key: jax.Array | None = None,
) -> jax.Array:
    """
    The bar estimator with a semistochastic cholesky sum in the T2-contracted energy.

    e2_0, and with it e2_2_1 = e2_0 * gt2g, stays exact: it is needed to build the
    sampling proposal anyway and costs only the gl = green.chol contraction. The three
    accumulators that contract with T2 -- e2_2_2_1, e2_2_2_2 and e2_2_3, whose "iajb"
    contractions cost nocc^2 nvir^2 per cholesky vector -- are split into an exactly
    summed head and an importance sampled tail.

    The estimator is unbiased. n_chol_head="full" puts every vector in the head, which
    removes the sampling entirely and reproduces energy_kernel_rw_rh_bar exactly; that is
    the reference to check this against. In that limit no key is drawn and none is needed,
    which is why key is optional and why make_pt2ccsd_meas_ops then declares no stochastic
    kernel: a run with a full head must not even perturb the RNG stream, or it would take
    a different trajectory than the bar run it is meant to reproduce.

    Cholesky vectors are never gathered into a new array here. The head is a contiguous
    prefix, so it is a plain slice shared across the vmap batch, and the sampled tail is
    scanned over *indices* with the gather inside the scan body -- only one chunk of
    walker-dependent vectors is live at a time. Passing gathered vectors as the scan's xs
    instead would materialise the whole n_walkers * n * norb^2 block for the length of the
    scan, and n is the sample count, not bounded by nchol_chunk, so it could not be
    recovered by shrinking the chunk.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    t2 = trial_data.t2
    nocc, nvir, norb = trial_data.nocc, trial_data.nvir, trial_data.norb

    h1 = meas_ctx.h1_bar
    chol = meas_ctx.chol_bar
    if h1 is None or chol is None or meas_ctx.exp_t1 is None:
        raise ValueError(
            "energy_kernel_rw_rh_sto needs the similarity transformed hamiltonian; "
            "build the measurement context with measure_type='sto_chol'."
        )

    nchol = chol.shape[0]
    nchol_chunk = meas_ctx.nchol_chunk
    walker_bar = meas_ctx.exp_t1 @ walker

    green = (walker_bar @ jnp.linalg.inv(walker_bar[:nocc, :])).T  # (nocc, norb)
    greenp = jnp.vstack((green[:, nocc:], -jnp.eye(nvir, dtype=green.dtype)))

    # one body, exactly as in the bar kernel
    hg = jnp.einsum("pi,pi->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2 * hg
    t2g_c = jnp.einsum("iajb,ia->jb", t2, green[:, nocc:], optimize="optimal")
    t2g_e = jnp.einsum("iajb,ib->ja", t2, green[:, nocc:], optimize="optimal")
    t2_green_c = jnp.einsum("pb,jb,jq->pq", greenp, t2g_c, green, optimize="optimal")
    t2_green_e = jnp.einsum("pa,ja,jq->pq", greenp, t2g_e, green, optimize="optimal")
    t2_green = 2 * t2_green_c - t2_green_e
    t2g = 2 * t2g_c - t2g_e
    gt2g = jnp.einsum("ia,ia->", t2g, green[:, nocc:], optimize="optimal")
    e1_2 = 2 * hg * gt2g - 2 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")

    # ---- pass 1: e2_0 per cholesky vector, exact, no T2 anywhere ----
    # fed the half rotated chol[:, :nocc, :]: e2_0 only ever touches the occupied block,
    # so gl comes out (chunk, nocc, nocc) rather than (chunk, nocc, norb), and under vmap
    # that factor is paid per walker
    def scan_e2_0(carry, rot_c):
        gl_occ = jnp.einsum("ir,gqr->giq", green, rot_c, optimize="optimal")
        tr_gl = jnp.einsum("gii->g", gl_occ, optimize="optimal")
        e2_0_g = 2 * tr_gl * tr_gl - jnp.einsum(
            "gij,gji->g", gl_occ, gl_occ, optimize="optimal"
        )
        e2_0_g = e2_0_g.astype(c128)
        return carry + jnp.sum(e2_0_g), e2_0_g

    n_chunk1, chunk1, npad1 = _equal_chunks(nchol, nchol_chunk)
    rot_all = chol[:, :nocc, :]
    if npad1:
        rot_all = jnp.pad(rot_all, ((0, npad1), (0, 0), (0, 0)))
    e2_0, e2_0_chunks = lax.scan(
        scan_e2_0, jnp.zeros((), dtype=c128), rot_all.reshape(n_chunk1, chunk1, nocc, norb)
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
        # deterministic limit: every vector summed exactly, empty tail, no draw, and the
        # proposal and sort are skipped entirely
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
            # a contiguous prefix is a plain slice, shared across the vmap batch, and
            # since cholesky vectors come out of the decomposition in decreasing
            # importance it is also a good head
            head_prefix = n_head
            tail = jnp.arange(n_head, nchol, dtype=jnp.int32)
        tail_prob = pi_g[tail]
        tail_prob = tail_prob / jnp.sum(tail_prob)

    # ---- pass 2: only the accumulators that contract with T2 ----
    def accum(carry, chol_c, w_c):
        """The three T2-contracted accumulators for one chunk, weighted per vector."""
        rot_chol_c = chol_c[:, :nocc, :]
        w_c = w_c.astype(ctype)

        gl = jnp.einsum("ir,gqr->giq", green, chol_c, optimize="optimal")
        tr_gl = jnp.einsum("gii->g", gl[:, :, :nocc], optimize="optimal")

        # e2_2_2_1
        lt2g = jnp.einsum(
            "gpr,pr->g", chol_c.astype(rtype), t2_green.astype(ctype), optimize="optimal"
        )
        carry[0] += jnp.sum(w_c * (-lt2g.astype(ctype) * tr_gl.astype(ctype))).astype(c128)

        # e2_2_2_2
        lt2_green = jnp.einsum(
            "gir,qr->giq",
            rot_chol_c.astype(rtype),
            t2_green.astype(ctype),
            optimize="optimal",
        )
        carry[1] += jnp.sum(
            w_c
            * 0.5
            * jnp.einsum(
                "giq,giq->g", gl.astype(ctype), lt2_green.astype(ctype), optimize="optimal"
            )
        ).astype(c128)

        # e2_2_3
        glgp = jnp.einsum(
            "gir,rb->gib", gl.astype(ctype), greenp.astype(ctype), optimize="optimal"
        )
        t2_r = t2.astype(rtype)
        lt2_c = jnp.einsum("gia,iajb->gjb", glgp, t2_r, optimize="optimal")
        lt2_e = jnp.einsum("gib,iajb->gja", glgp, t2_r, optimize="optimal")
        l2t2_c = jnp.einsum("gjb,gjb->g", lt2_c.astype(ctype), glgp, optimize="optimal")
        l2t2_e = jnp.einsum("gja,gja->g", lt2_e.astype(ctype), glgp, optimize="optimal")
        carry[2] += jnp.sum(w_c * (2 * l2t2_c - l2t2_e)).astype(c128)
        return carry

    zero = jnp.zeros((), dtype=c128)

    def run_slice(chol_s, weights):
        """Scan over cholesky vectors themselves. Only for the contiguous head, where the
        slice is shared across the vmap batch and costs nothing."""
        n = weights.shape[0]
        if n == 0:
            return zero, zero, zero
        n_ch, chunk, npad = _equal_chunks(n, nchol_chunk)
        if npad:
            chol_s = jnp.pad(chol_s, ((0, npad), (0, 0), (0, 0)))
            weights = jnp.pad(weights, (0, npad))
        out, _ = lax.scan(
            lambda carry, x: (accum(carry, x[0], x[1]), 0.0),
            [zero, zero, zero],
            (chol_s.reshape(n_ch, chunk, norb, norb), weights.reshape(n_ch, chunk)),
        )
        return out[0], out[1], out[2]

    def run_indices(idx, weights):
        """Same sum, scanning over cholesky *indices*: the gather happens inside the scan
        body, so only one chunk of walker-dependent vectors is ever live."""
        n = weights.shape[0]
        if n == 0:
            return zero, zero, zero
        n_ch, chunk, npad = _equal_chunks(n, nchol_chunk)
        if npad:
            # pad with index 0 at zero weight, which contributes nothing
            idx = jnp.pad(idx, (0, npad))
            weights = jnp.pad(weights, (0, npad))
        out, _ = lax.scan(
            lambda carry, x: (accum(carry, chol[x[0]], x[1]), 0.0),
            [zero, zero, zero],
            (idx.reshape(n_ch, chunk), weights.reshape(n_ch, chunk)),
        )
        return out[0], out[1], out[2]

    # head: exact, unit weights
    if head_prefix is not None:
        b_h, c_h, d_h = run_slice(chol[:head_prefix], jnp.ones(head_prefix, dtype=c128))
    else:
        assert head_idx is not None
        b_h, c_h, d_h = run_indices(head_idx, jnp.ones(head_idx.shape[0], dtype=c128))

    # tail: sampled, so walker dependent and therefore index scanned
    if tail.shape[0] == 0:
        b_t = c_t = d_t = zero
    else:
        if key is None:
            raise ValueError(
                "energy_kernel_rw_rh_sto draws a sampled tail and so needs a PRNG key; "
                "only n_chol_head='full' can run without one."
            )
        sel = jax.random.choice(
            key, tail.shape[0], shape=(n_samples,), replace=True, p=tail_prob
        )
        samp_w = (1.0 / (n_samples * tail_prob[sel])).astype(c128)
        b_t, c_t, d_t = run_indices(tail[sel], samp_w)

    # e2_2_1 = e2_0 * gt2g is exact, since e2_0 is
    e2_2 = e2_0 * gt2g + 4 * (b_h + c_h + b_t + c_t) + d_h + d_t

    e0 = e1_0 + e2_0  # fully exact
    e1 = e1_2 + e2_2

    return jnp.stack([gt2g, e0, e1])


def make_pt2ccsd_meas_ops(
    sys: System,
    measure_type: str | None = None,
    memory_mode: str = "low",
    mixed_precision: bool = False,
    testing: bool = False,
    nchol_chunk: int | None = None,
    **cfg_fields: Any,
) -> MeasOps:
    """
    measure_type selects the energy kernel: None the original one-vector-at-a-time one,
    which ignores mixed_precision and nchol_chunk; "chunk" the chunked one; "bar" the
    chunked one with exp(T1) moved onto the hamiltonian; "sto_chol" that one with the
    T2-contracted two-body sum sampled. See Pt2ccsdMeasCfg.

    Any remaining Pt2ccsdMeasCfg field may be given by name -- the semistochastic knobs
    are passed this way, so they need no signature of their own here.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"pt2CCSD MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    if measure_type not in _MEASURE_TYPES:
        raise ValueError(
            f"unknown measure_type {measure_type!r}; expected one of {_MEASURE_TYPES}"
        )
    if memory_mode not in _MEMORY_MODES:
        raise ValueError(f"unknown memory_mode {memory_mode!r}; expected one of {_MEMORY_MODES}")
    if nchol_chunk is not None and measure_type is None:
        raise ValueError(
            "nchol_chunk means nothing to the unchunked energy kernel; "
            "pass measure_type='chunk', 'bar' or 'sto_chol' to use it."
        )

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

    # "full" means every cholesky vector is summed exactly, so no draw is ever made and
    # the block function must not advance the key stream on this kernel's behalf
    samples_tail = measure_type == "sto_chol" and not (
        isinstance(cfg.n_chol_head, str) and cfg.n_chol_head.lower() == "full"
    )

    energy_kernel = {
        None: energy_kernel_rw_rh,
        "chunk": energy_kernel_rw_rh_chunk,
        "bar": energy_kernel_rw_rh_bar,
        "sto_chol": energy_kernel_rw_rh_sto,
    }[measure_type]

    meas_ops = MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel},
        # a kernel that draws its own samples needs the block function to hand it a key
        # per walker
        stochastic_kernels=frozenset({k_energy} if samples_tail else ()),
    )
    object.__setattr__(meas_ops, _PT2CCSD_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_pt2ccsd_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdMeasCfg | None:
    cfg = getattr(meas_ops, _PT2CCSD_MEAS_CFG_ATTR, None)
    if isinstance(cfg, Pt2ccsdMeasCfg):
        return cfg
    return None


def get_init_pt2trial_energy(
    init_state: PropState,
    ham_data: HamChol,
    trial_data: Pt2ccsdTrial,
    trial_meas_ops: MeasOps,
    trial_meas_ctx: Pt2ccsdMeasCtx,
    params: QmcParams,
):

    walker_0 = wk.take_walkers(init_state.walkers, jnp.array([0]))
    trial_e_kernel = trial_meas_ops.require_kernel(k_energy)
    if trial_meas_ops.needs_rng(k_energy):
        # the tau = 0 row is one sample of a stochastic estimator like any other, so it
        # gets a key too, split off the initial state's
        keys = jax.random.split(init_state.rng_key, wk.n_walkers(walker_0))
        pt2results = wk.vmap_chunked(
            trial_e_kernel, n_chunks=1, in_axes=(0, None, None, None, 0)
        )(walker_0, ham_data, trial_meas_ctx, trial_data, keys)
    else:
        pt2results = wk.vmap_chunked(trial_e_kernel, n_chunks=1, in_axes=(0, None, None, None))(
            walker_0, ham_data, trial_meas_ctx, trial_data
        )
    t2, e0, e1 = pt2results[:, 0], pt2results[:, 1], pt2results[:, 2]
    trial_overlap = wk.vmap_chunked(
        trial_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walker_0, trial_data)
    guide_overlap = init_state.overlaps[0]
    trial_weights = init_state.weights * trial_overlap / guide_overlap
    trial_energy = (ham_data.h0 + e0 + e1 - t2 * e0).mean()

    return trial_energy + 0j, jnp.sum(trial_weights)
