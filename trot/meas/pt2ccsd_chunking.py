"""
Cholesky chunking for the chunked pt2CCSD energy kernels (pt2ccsd_bar, upt2ccsd,
upt2ccsd_bar).

The kernels scan the two body sum over chunks of nchol_chunk cholesky vectors, so their
peak memory is set by the chunk size rather than by the number of vectors. This module
holds what decides that size:

    Pt2ccsdChunkMeasCfg       the mixed dtypes and the requested nchol_chunk
    resolve_nchol_chunk       the even division of the cholesky index the kernels scan
    Pt2ccsdMemoryModel        bytes(walkers_in_flight, nchol_chunk) of one kernel pass
    pt2ccsd_memory_model      the model of the restricted bar kernel
    upt2ccsd_memory_model     the model of the unrestricted chunk and bar kernels
    plan_pt2ccsd_chunking     split a byte budget between the cholesky and walker chunks
    device_memory_budget_bytes
                              a budget from the device allocator limit, when it reports one

The chunk is chosen against a budget with minimal zero padding (cholesky.equal_chunks).
Only when a single cholesky vector per step still does not fit are the walkers chunked,
and the cholesky chunk is then chosen again against the smaller walker count. The walker
chunk count the plan settles on is a floor: the driver's automatic walker chunking may
raise it further after compilation, never lower it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..cholesky import equal_chunks, max_equal_chunk_pad

# cholesky vectors per scan step when neither a budget nor a chunk size is given
DEFAULT_NCHOL_CHUNK = 100

# the share of the device allocator limit the trial measurement may plan against; the
# rest is left to the propagator, the guide kernels and XLA's transient buffers
DEVICE_MEMORY_FRACTION = 0.5


@dataclass(frozen=True)
class Pt2ccsdChunkMeasCfg:
    """
    The T2 contractions of the chunked kernels run in mixed_real_dtype /
    mixed_complex_dtype; the hamiltonian, the walkers, the greens functions and every
    partial sum stay in double precision. The *_testing dtypes are the pair the
    restricted unchunked kernel uses, kept so the two configs read alike.

    nchol_chunk is a cap on the cholesky vectors per scan step; None takes
    DEFAULT_NCHOL_CHUNK or what the memory plan chose. build_meas_ctx resolves the even
    division actually scanned.
    """

    mixed_real_dtype: Any = jnp.float64
    mixed_complex_dtype: Any = jnp.complex128
    mixed_real_dtype_testing: Any = jnp.float32
    mixed_complex_dtype_testing: Any = jnp.complex64
    nchol_chunk: int | None = None


def make_chunk_meas_cfg(
    *, mixed_precision: bool, testing: bool, nchol_chunk: int | None
) -> Pt2ccsdChunkMeasCfg:
    if nchol_chunk is not None and int(nchol_chunk) < 1:
        raise ValueError(f"nchol_chunk must be >= 1, got {nchol_chunk}")
    return Pt2ccsdChunkMeasCfg(
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
        nchol_chunk=None if nchol_chunk is None else int(nchol_chunk),
    )


def resolve_nchol_chunk(nchol: int, requested: int | None) -> int:
    """
    The chunk size the kernels really scan: the requested cap (or the default), capped
    by nchol, divided evenly with minimal padding.
    """
    nchol = int(nchol)
    cap = DEFAULT_NCHOL_CHUNK if requested is None else int(requested)
    if cap < 1:
        raise ValueError(f"nchol_chunk must be >= 1, got {requested}")
    cap = min(cap, nchol) if nchol > 0 else cap
    _, chunk, _ = equal_chunks(nchol, cap)
    return int(chunk)


def pad_reshape_chol(chol: jax.Array, nchol_chunk: int) -> tuple[jax.Array, int, int, int]:
    """(nchol, ...) -> (n_chunks, chunk, ...), zero padded; also returns the division."""
    n_chunks, chunk, n_pad = equal_chunks(int(chol.shape[0]), nchol_chunk)
    if n_pad:
        chol = jnp.pad(chol, ((0, n_pad),) + ((0, 0),) * (chol.ndim - 1))
    return chol.reshape(n_chunks, chunk, *chol.shape[1:]), n_chunks, chunk, n_pad


# ======================================================================================
# memory models
# ======================================================================================


@dataclass(frozen=True)
class Pt2ccsdMemoryModel:
    """
    Byte counts for one measurement pass of a chunked kernel, split by how each term
    scales. With w walkers in flight and a chunk of k cholesky vectors:

        bytes(w, k) = resident + w*per_walker + w*k*per_walker_chol

    The two chunking knobs move w and k, so the model is what makes them comparable:
    n_chunks divides w, nchol_chunk sets k, and only the product w*k multiplies the term
    that dominates for any system worth chunking. Nothing shared scales with k: the scan
    slice is a view into the padded copy, and equal_chunks keeps that copy's padding
    bounded by about sqrt(nchol) whatever k is, so resident carries it once at its worst
    case. It counts the arrays the kernels name, not XLA's transient buffers, so treat it
    as a floor and leave headroom in the budget.
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
) -> Pt2ccsdMemoryModel:
    """
    What energy_kernel_rw_rh_bar (meas/pt2ccsd_bar.py) holds, by counting the arrays it
    names. real_bytes / complex_bytes are the mixed dtypes; the hamiltonian, the walkers
    and the greens function intermediates are f8/c16 regardless.

    resident: the hamiltonian's cholesky tensor (still needed by the guide propagator),
    chol_bar and its padded reshaped copy, the half rotated chol_bar[:, :nocc, :] and its
    padded copy, the transforms, the amplitudes and their cast, and the whole walker
    population. per_walker: the walker and its transform, the half green, greenp, the
    t2_green matrices and the t2g vectors. per_walker_chol: gl and its cast, lt2_green,
    glgp and the two T2 contractions.
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

    resident = (
        chol
        + chol
        + chol_padded
        + rot
        + rot_padded
        + 3 * n2 * f8  # exp_t1, exp_mt1, h1_bar
        + t2
        + (ov * ov * r if r != f8 else 0)
        + n_walkers * norb * nocc * c16
    )
    per_walker = c16 * (2 * norb * nocc + nocc * norb + norb * nvir + 3 * n2 + 3 * ov)
    per_walker_chol = nocc * norb * (c16 + 2 * c) + ov * (3 * c)
    return Pt2ccsdMemoryModel(
        resident=resident, per_walker=per_walker, per_walker_chol=per_walker_chol
    )


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
    bar=False models energy_kernel_uw_uh_chunk, bar=True energy_kernel_uw_uh_bar.
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
        t2 * f8 + (t2 * r if r != f8 else 0) + n_walkers * (norb_a * nocc_a + norb_b * nocc_b) * c16
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
            resident += chol + chol + chol_padded + 3 * n2 * f8
            per_walker += c16 * (
                2 * norb_s * nocc_s + nocc_s * norb_s + norb_s * nvir_s + n2 + 2 * ov_s
            )
            per_walker_chol += nocc_s * norb_s * (c16 + 2 * c) + ov_s * (2 * c) + ov_cross * c
        else:
            resident += chol + chol_padded
            per_walker += c16 * (norb_s * nocc_s + n2 + norb_s * nvir_s + n2 + 2 * ov_s)
            per_walker_chol += n2 * (2 * c16 + 2 * c) + ov_s * (c16 + 2 * c) + ov_cross * c

    return Pt2ccsdMemoryModel(
        resident=resident, per_walker=per_walker, per_walker_chol=per_walker_chol
    )


# ======================================================================================
# planning
# ======================================================================================


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
        f"a memory budget of {budget_bytes / mb:.4g} MB is too small: {detail}. "
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
    that walkers in flight also batch the per walker work that no cholesky chunk touches
    (the greens function, the determinant, the one body amplitude contractions). So all
    the walkers stay in flight and nchol_chunk shrinks to fit.

    Only when a single cholesky vector per step still does not fit does n_chunks rise,
    and then nchol_chunk is chosen again against the smaller walker count: halving the
    walkers in flight usually buys back a chunk well above one.

    n_chunks is a floor, never lowered. nchol_chunk, if given, is taken as fixed and only
    n_chunks is derived.
    """
    n_walkers = int(n_walkers)
    n_devices = max(1, int(n_devices))
    n_chunks_floor = max(1, int(n_chunks))
    nchol = int(nchol)

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
            raise _budget_error(model, budget_bytes, f"one walker at nchol_chunk={k} does not fit")
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
                raise _budget_error(model, budget_bytes, "one walker at nchol_chunk=1 does not fit")
            nc = max(nc, _ceil_div(n_walkers, w_max * n_devices))
            k = chunk_for(in_flight(nc))
            note = "walkers chunked; cholesky chunk re-derived"

        if k >= nchol:
            note = "whole cholesky tensor fits in one step"
        k = max(1, min(int(k), nchol))

    # the kernels divide the cap evenly, so report the size that will really be scanned
    _, k, _ = equal_chunks(nchol, k) if nchol > 0 else (1, k, 0)

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


def device_memory_budget_bytes(fraction: float = DEVICE_MEMORY_FRACTION) -> int | None:
    """
    A byte budget for the trial measurement: fraction of the smallest allocator limit
    the devices report (the same statistic the driver's automatic walker chunking reads),
    or None when the backend reports none (CPU), in which case the callers fall back to
    DEFAULT_NCHOL_CHUNK.
    """
    limits: list[int] = []
    for device in jax.devices():
        try:
            stats = device.memory_stats()
        except (RuntimeError, NotImplementedError):
            stats = None
        if not stats:
            return None
        value = next(
            (stats[key] for key in ("bytes_limit", "memory_limit", "total_memory") if key in stats),
            None,
        )
        if value is None or int(value) <= 0:
            return None
        limits.append(int(value))
    if not limits:
        return None
    return int(float(fraction) * min(limits))
