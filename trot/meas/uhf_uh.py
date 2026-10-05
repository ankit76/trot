"""
UHF measurement kernels on the unrestricted (uchol) hamiltonian.

Unrestricted walker + unrestricted hamiltonian (kernel suffix _uw_uh, versus meas/uhf.py's
_uw_rh: unrestricted walker on the restricted hamiltonian). The UhfMeasCtx of meas/uhf.py
is reused: it already keeps rot_h1 and rot_chol per spin, and here each spin is rotated
with its own h1 and cholesky vectors, so the two rot_chol share the field axis but not
the orbital axis and norb_a may differ from norb_b.

u_rot_force_bias / u_rot_energy follow afqmc's slater_tools.u_rot_force_bias /
u_rot_energy; the half green's function is trot's _half_green_from_overlap_matrix.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ..cholesky import chunk_rot_chol
from ..core.ops import MeasOps, k_energy, k_force_bias
from ..ham.chol_u import HamCholU
from ..trial.uhf import UhfTrial, overlap_u
from .uhf import UhfMeasCtx, _half_green_from_overlap_matrix

# cholesky vectors per lax.scan step in the energy kernel; None sums them in one chunk
DEFAULT_NCHOL_CHUNK: int | None = None


def _u_half_green(bra: tuple, ket: tuple) -> tuple[jax.Array, jax.Array]:
    """Half green's function per spin, (nocc_s, norb_s)."""
    ga = _half_green_from_overlap_matrix(ket[0], bra[0].conj().T @ ket[0])
    gb = _half_green_from_overlap_matrix(ket[1], bra[1].conj().T @ ket[1])
    return (ga, gb)


def u_rot_force_bias(bra: tuple, ket: tuple, rot_chol: tuple) -> jax.Array:
    """
    Force bias against an unrestricted half rotated hamiltonian.

    rot_chol is (rot_chol_a, rot_chol_b), each (n_chol, nocc_s, norb_s). The two spins
    share only the field axis. Returns one length n_chol vector, summed over spin.
    """
    green = _u_half_green(bra, ket)
    fb_a = jnp.einsum("gij,ij->g", rot_chol[0], green[0], optimize="optimal")
    fb_b = jnp.einsum("gij,ij->g", rot_chol[1], green[1], optimize="optimal")
    return fb_a + fb_b


def u_rot_energy(
    bra: tuple,
    ket: tuple,
    h0: jax.Array,
    rot_h1: tuple,
    rot_chol: tuple,
) -> jax.Array:
    """
    Local energy <T|H|phi>/<T|phi> against a spin unrestricted half rotated hamiltonian.

    rot_chol_a and rot_chol_b are (n_chunks, chunk, nocc_s, norb_s); a plain
    (n_chol, nocc_s, norb_s) is accepted and treated as a single chunk. The two body
    term is reduced with lax.scan over the chunks, so peak memory is set by the chunk
    size rather than by n_chol. Zero padded chunks contribute nothing.
    """
    chol_a, chol_b = rot_chol
    if chol_a.ndim == 3:
        chol_a = chol_a.reshape(1, *chol_a.shape)
    if chol_b.ndim == 3:
        chol_b = chol_b.reshape(1, *chol_b.shape)

    green = _u_half_green(bra, ket)
    e1 = jnp.einsum("pq,pq->", rot_h1[0], green[0], optimize="optimal") + jnp.einsum(
        "pq,pq->", rot_h1[1], green[1], optimize="optimal"
    )

    zero = jnp.array(0.0, dtype=jnp.result_type(chol_a, green[0], green[1]))

    def scanned_fun(carry: jax.Array, x) -> tuple[jax.Array, None]:
        chol_a_c, chol_b_c = x  # (chunk, nocc_s, norb_s) each
        lg_a_c = jnp.einsum("gpr,qr->gpq", chol_a_c, green[0], optimize="optimal")
        lg_b_c = jnp.einsum("gpr,qr->gpq", chol_b_c, green[1], optimize="optimal")
        trlg_a_c = jnp.einsum("gpp->g", lg_a_c, optimize="optimal")
        trlg_b_c = jnp.einsum("gpp->g", lg_b_c, optimize="optimal")

        e2aa_c = jnp.sum(trlg_a_c**2) - jnp.einsum("gpq,gqp->", lg_a_c, lg_a_c, optimize="optimal")
        e2ab_c = jnp.sum(trlg_a_c * trlg_b_c) * 2
        e2bb_c = jnp.sum(trlg_b_c**2) - jnp.einsum("gpq,gqp->", lg_b_c, lg_b_c, optimize="optimal")

        carry += (e2aa_c + e2ab_c + e2bb_c) / 2
        return carry, None

    e2, _ = lax.scan(scanned_fun, zero, (chol_a, chol_b))

    return h0 + e1 + e2


def force_bias_kernel_uw_uh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    bra = (trial_data.mo_coeff_a, trial_data.mo_coeff_b)
    return u_rot_force_bias(bra, walker, (meas_ctx.rot_chol_a, meas_ctx.rot_chol_b))


def energy_kernel_uw_uh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
    *,
    nchol_chunk: int | None = DEFAULT_NCHOL_CHUNK,
) -> jax.Array:
    bra = (trial_data.mo_coeff_a, trial_data.mo_coeff_b)
    rot_chol = (
        chunk_rot_chol(meas_ctx.rot_chol_a, nchol_chunk),
        chunk_rot_chol(meas_ctx.rot_chol_b, nchol_chunk),
    )
    return u_rot_energy(
        bra,
        walker,
        ham_data.h0,
        (meas_ctx.rot_h1_a, meas_ctx.rot_h1_b),
        rot_chol,
    )


def build_meas_ctx_uh(ham_data: HamCholU, trial_data: UhfTrial) -> UhfMeasCtx:
    """
    Half rotated h1 and chol for the unrestricted hamiltonian.

    Same UhfMeasCtx as meas.uhf.build_meas_ctx; the only change is the source: each spin
    is rotated with its own h1 and cholesky vectors rather than with one shared set.
    """
    if ham_data.basis != "uchol":
        raise ValueError("UHF unrestricted MeasOps assumes HamCholU.basis == 'uchol'.")
    caH = trial_data.mo_coeff_a.conj().T  # (nocc[0], norb_a)
    cbH = trial_data.mo_coeff_b.conj().T  # (nocc[1], norb_b)
    rot_h1_a = caH @ ham_data.h1_a  # (nocc[0], norb_a)
    rot_h1_b = cbH @ ham_data.h1_b  # (nocc[1], norb_b)
    rot_chol_a = jnp.einsum("pi,gij->gpj", caH, ham_data.chol_a, optimize="optimal")
    rot_chol_b = jnp.einsum("pi,gij->gpj", cbH, ham_data.chol_b, optimize="optimal")
    rot_chol_flat_a = rot_chol_a.reshape(rot_chol_a.shape[0], -1)
    rot_chol_flat_b = rot_chol_b.reshape(rot_chol_b.shape[0], -1)
    return UhfMeasCtx(
        rot_h1_a=rot_h1_a,
        rot_h1_b=rot_h1_b,
        rot_chol_a=rot_chol_a,
        rot_chol_b=rot_chol_b,
        rot_chol_flat_a=rot_chol_flat_a,
        rot_chol_flat_b=rot_chol_flat_b,
    )


def make_uhf_meas_ops_uh(sys: Any, *, nchol_chunk: int | None = DEFAULT_NCHOL_CHUNK) -> MeasOps:
    """
    MeasOps of the UHF trial on the unrestricted hamiltonian.

    Only unrestricted walkers are meaningful: alpha and beta live in different orbital
    spaces, so a shared or spin blocked walker cannot represent them. No observables are
    wired: meas.uhf's rdm1 and density correlation kernels stack the two spin blocks and
    so do not survive norb_a != norb_b.
    """
    wk = sys.walker_kind.lower()
    if wk != "unrestricted":
        raise ValueError(
            f"the unrestricted hamiltonian path requires walker_kind='unrestricted', got {wk!r}"
        )

    energy_kernel = partial(energy_kernel_uw_uh, nchol_chunk=nchol_chunk)

    return MeasOps(
        overlap=overlap_u,
        build_meas_ctx=build_meas_ctx_uh,
        kernels={k_force_bias: force_bias_kernel_uw_uh, k_energy: energy_kernel},
        observables={},
    )
