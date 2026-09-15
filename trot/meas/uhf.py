from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias, o_density_corr, o_rdm1
from ..core.system import System
from ..ham.chol import HamChol
from ..ham.chol_u import HamCholU
from ..trial.uhf import UhfTrial, overlap_g, overlap_r, overlap_u


def _half_green_from_overlap_matrix(w: jax.Array, ovlp_mat: jax.Array) -> jax.Array:
    """
    green_half = (w @ inv(ovlp_mat)).T
    """
    return jnp.linalg.solve(ovlp_mat.T, w.T)


def _build_bra_generalized(trial_data: UhfTrial) -> jax.Array:
    Atrial = trial_data.mo_coeff_a
    Btrial = trial_data.mo_coeff_b
    bra = jnp.block([[Atrial, 0 * Btrial], [0 * Atrial, Btrial]])
    return bra


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return force_bias_kernel_uw_rh(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff_a.conj().T @ wu
    md = trial_data.mo_coeff_b.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)  # (nocc[0], norb)
    gd = _half_green_from_overlap_matrix(wd, md)  # (nocc[1], norb)

    fb_u = jnp.einsum("gij,ij->g", meas_ctx.rot_chol_a, gu, optimize="optimal")
    fb_d = jnp.einsum("gij,ij->g", meas_ctx.rot_chol_b, gd, optimize="optimal")
    return fb_u + fb_d


def force_bias_kernel_gw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    w = walker
    norb = trial_data.norb
    na, _ = trial_data.nocc

    bra = _build_bra_generalized(trial_data)
    g = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)

    g_aa, g_bb = g[:na, :norb], g[na:, norb:]

    rot_chol_aa = meas_ctx.rot_chol_a
    rot_chol_bb = meas_ctx.rot_chol_b

    fb = jnp.einsum("gij,ij->g", rot_chol_aa, g_aa, optimize="optimal")
    fb += jnp.einsum("gij,ij->g", rot_chol_bb, g_bb, optimize="optimal")

    return fb


def rdm1_kernel_rw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return rdm1_kernel_uw(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def rdm1_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff_a.conj().T @ wu
    md = trial_data.mo_coeff_b.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)
    dm_a = gu.T @ trial_data.mo_coeff_a.conj().T
    dm_b = gd.T @ trial_data.mo_coeff_b.conj().T
    return jnp.stack([dm_a, dm_b], axis=0)


def rdm1_kernel_gw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    w = walker
    norb = trial_data.norb
    na, _ = trial_data.nocc
    bra = _build_bra_generalized(trial_data)
    g = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)
    dm_a = g[:na, :norb].T @ trial_data.mo_coeff_a.conj().T
    dm_b = g[na:, norb:].T @ trial_data.mo_coeff_b.conj().T
    return jnp.stack([dm_a, dm_b], axis=0)


def _density_corr_from_greens(ga: jax.Array, gb: jax.Array) -> jax.Array:
    """
    Density correlation from spin-resolved Green's functions ga (norb, norb)
    and gb (norb, norb). Returns (3, norb, norb) array of uu, ud, dd correlations.
    """
    na = jnp.diagonal(ga)
    nb = jnp.diagonal(gb)

    # same-spin: n_i n_j = G_ii G_jj - G_ij G_ji + delta_ij G_ii
    uu = na[:, None] * na[None, :] - ga * ga.T + jnp.diag(na)
    dd = nb[:, None] * nb[None, :] - gb * gb.T + jnp.diag(nb)

    # opposite-spin: n_ia n_jb = G^a_ii G^b_jj  (no exchange)
    ud = na[:, None] * nb[None, :]

    return jnp.stack([uu, ud, dd], axis=0)


def density_corr_kernel_uw(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff_a.conj().T @ wu
    md = trial_data.mo_coeff_b.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)
    ga = gu.T @ trial_data.mo_coeff_a.conj().T
    gb = gd.T @ trial_data.mo_coeff_b.conj().T
    return _density_corr_from_greens(ga, gb)


def density_corr_kernel_rw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return density_corr_kernel_uw(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def density_corr_kernel_gw(
    walker: jax.Array,
    ham_data: Any,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    w = walker
    norb = trial_data.norb
    na, _ = trial_data.nocc
    bra = _build_bra_generalized(trial_data)
    g = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)
    ga = g[:na, :norb].T @ trial_data.mo_coeff_a.conj().T
    gb = g[na:, norb:].T @ trial_data.mo_coeff_b.conj().T
    return _density_corr_from_greens(ga, gb)


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return energy_kernel_uw_rh(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    wu, wd = walker
    mu = trial_data.mo_coeff_a.conj().T @ wu
    md = trial_data.mo_coeff_b.conj().T @ wd
    gu = _half_green_from_overlap_matrix(wu, mu)
    gd = _half_green_from_overlap_matrix(wd, md)

    e0 = ham_data.h0
    e1 = jnp.sum(gu * meas_ctx.rot_h1_a) + jnp.sum(gd * meas_ctx.rot_h1_b)

    f_up = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol_a, gu.T, optimize="optimal")
    f_dn = jnp.einsum("gij,jk->gik", meas_ctx.rot_chol_b, gd.T, optimize="optimal")
    c_up = jax.vmap(jnp.trace)(f_up)
    c_dn = jax.vmap(jnp.trace)(f_dn)
    exc_up = jnp.sum(jax.vmap(lambda x: x * x.T)(f_up))
    exc_dn = jnp.sum(jax.vmap(lambda x: x * x.T)(f_dn))

    e2 = (
        jnp.sum(c_up * c_up) + jnp.sum(c_dn * c_dn) + 2.0 * jnp.sum(c_up * c_dn) - exc_up - exc_dn
    ) / 2.0

    return e0 + e1 + e2


def energy_kernel_gw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UhfMeasCtx,
    trial_data: UhfTrial,
) -> jax.Array:
    w = walker
    norb = trial_data.norb
    na, nb = trial_data.nocc

    bra = _build_bra_generalized(trial_data)
    g = _half_green_from_overlap_matrix(w, bra.T.conj() @ w)

    g_aa = g[:na, :norb]
    g_bb = g[na:, norb:]
    g_ab = g[:na, norb:]
    g_ba = g[na:, :norb]

    rot_h1_a = meas_ctx.rot_h1_a
    rot_h1_b = meas_ctx.rot_h1_b

    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    e0 = ham_data.h0

    e1 = jnp.sum(g_aa * rot_h1_a) + jnp.sum(g_bb * rot_h1_b)

    f_up = jnp.einsum("gij,jk->gik", rot_chol_a, g_aa.T, optimize="optimal")
    f_dn = jnp.einsum("gij,jk->gik", rot_chol_b, g_bb.T, optimize="optimal")
    c_up = jax.vmap(jnp.trace)(f_up)
    c_dn = jax.vmap(jnp.trace)(f_dn)
    J = jnp.sum(c_up * c_up) + jnp.sum(c_dn * c_dn) + 2.0 * jnp.sum(c_up * c_dn)

    K = jnp.sum(jax.vmap(lambda x: x * x.T)(f_up)) + jnp.sum(jax.vmap(lambda x: x * x.T)(f_dn))

    f_ab = jnp.einsum("gip,pj->gij", rot_chol_a, g_ba.T, optimize="optimal")
    f_ba = jnp.einsum("gip,pj->gij", rot_chol_b, g_ab.T, optimize="optimal")
    K += 2.0 * jnp.sum(jax.vmap(lambda x, y: x * y.T)(f_ab, f_ba))

    return e0 + e1 + (J - K) / 2.0


# unrestricted walker + unrestricted (uchol) hamiltonian.
#
# u_rot_force_bias and u_rot_energy are ported from afqmc/slater_tools.py; the kernels
# below are the trot facing wrappers, matching what afqmc/wavefunctions/uhf_wfn.py does
# in rot_force_bias / rot_energy (the wrapper chunks rot_chol, slater_tools consumes it).
#
# afqmc's u_half_green is NOT copied: trot's _half_green_from_overlap_matrix already
# computes the same thing, since solve(m.T, w.T) == (w @ inv(m)).T with m = C^H w.
#
# Only the half rotated forms are carried over, since trot always half rotates.

DEFAULT_NCHOL_CHUNK: int | None = None


def _u_half_green(bra: tuple, ket: tuple) -> tuple[jax.Array, jax.Array]:
    """Half green's function per spin, (nocc_sigma, norb_sigma)."""
    ga = _half_green_from_overlap_matrix(ket[0], bra[0].conj().T @ ket[0])
    gb = _half_green_from_overlap_matrix(ket[1], bra[1].conj().T @ ket[1])
    return (ga, gb)


def u_rot_force_bias(bra: tuple, ket: tuple, rot_chol: tuple) -> jax.Array:
    """
    Force bias against an unrestricted half rotated hamiltonian.

    rot_chol is (rot_chol_a, rot_chol_b), each (n_chol, nocc_sigma, norb_sigma). The two
    spins share only the field axis, so norb_a and norb_b may differ. Returns one length
    n_chol vector, summed over spin.
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
    Energy against a spin unrestricted half rotated hamiltonian.

    rot_chol_a and rot_chol_b are expected as (n_chunks, nchol_chunk, nocc, norb); a
    plain (n_chol, nocc, norb) is accepted and treated as a single chunk. The two body
    term is reduced with lax.scan over chunks, so peak memory is set by the chunk size
    rather than by n_chol.
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
        chol_a_c, chol_b_c = x  # (nchol_chunk, nocc_sigma, norb_sigma) each
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


def _chunk_rot_chol(rot_chol: jax.Array, nchol_chunk: int | None) -> jax.Array:
    """(n_chol, nocc, norb) -> (n_chunks, nchol_chunk, nocc, norb), zero padded."""
    n_chol = int(rot_chol.shape[0])
    chunk = n_chol if nchol_chunk is None else int(nchol_chunk)
    chunk = max(1, min(chunk, n_chol))
    n_chunks = -(-n_chol // chunk)
    pad = n_chunks * chunk - n_chol
    if pad:
        rot_chol = jnp.pad(rot_chol, ((0, pad), (0, 0), (0, 0)))
    return rot_chol.reshape(n_chunks, chunk, *rot_chol.shape[1:])


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
        _chunk_rot_chol(meas_ctx.rot_chol_a, nchol_chunk),
        _chunk_rot_chol(meas_ctx.rot_chol_b, nchol_chunk),
    )
    return u_rot_energy(
        bra,
        walker,
        ham_data.h0,
        (meas_ctx.rot_h1_a, meas_ctx.rot_h1_b),
        rot_chol,
    )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UhfMeasCtx:
    # half-rotated:
    rot_h1_a: jax.Array  # (nocc[0], norb)
    rot_h1_b: jax.Array  # (nocc[1], norb)
    rot_chol_a: jax.Array  # (n_chol, nocc[0], norb)
    rot_chol_b: jax.Array  # (n_chol, nocc[1], norb)
    rot_chol_flat_a: jax.Array  # (n_chol, nocc[0]*norb)
    rot_chol_flat_b: jax.Array  # (n_chol, nocc[1]*norb)

    def tree_flatten(self):
        return (
            self.rot_h1_a,
            self.rot_h1_b,
            self.rot_chol_a,
            self.rot_chol_b,
            self.rot_chol_flat_a,
            self.rot_chol_flat_b,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            rot_h1_a,
            rot_h1_b,
            rot_chol_a,
            rot_chol_b,
            rot_chol_flat_a,
            rot_chol_flat_b,
        ) = children
        return cls(
            rot_h1_a=rot_h1_a,
            rot_h1_b=rot_h1_b,
            rot_chol_a=rot_chol_a,
            rot_chol_b=rot_chol_b,
            rot_chol_flat_a=rot_chol_flat_a,
            rot_chol_flat_b=rot_chol_flat_b,
        )


def build_meas_ctx(ham_data: HamChol, trial_data: UhfTrial) -> UhfMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("UHF MeasOps currently assumes HamChol.basis == 'restricted'.")
    caH = trial_data.mo_coeff_a.conj().T  # (nocc[0], norb)
    cbH = trial_data.mo_coeff_b.conj().T  # (nocc[1], norb)
    rot_h1_a = caH @ ham_data.h1  # (nocc[0], norb)
    rot_h1_b = cbH @ ham_data.h1  # (nocc[1], norb)
    rot_chol_a = jnp.einsum("pi,gij->gpj", caH, ham_data.chol, optimize="optimal")
    rot_chol_b = jnp.einsum("pi,gij->gpj", cbH, ham_data.chol, optimize="optimal")
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


def build_meas_ctx_uh(ham_data: HamCholU, trial_data: UhfTrial) -> UhfMeasCtx:
    """
    Build half rotated h1 and chol for unrestricted hamiltonian,
    where alpha and beta may live in different orbital spaces.

    Same UhfMeasCtx as build_meas_ctx, which already keeps the two spins separate. The
    only change is the source: each spin is rotated with its own h1 and cholesky vectors
    rather than with one shared set. The resulting rot_chol_a and rot_chol_b share the
    field axis but not the orbital axis.
    """
    if ham_data.basis != "uchol":
        raise ValueError("UHF unrestricted MeasOps assumes HamCholU.basis == 'uchol'.")
    caH = trial_data.mo_coeff_a.conj().T  # (nocc[0], norb_a)
    cbH = trial_data.mo_coeff_b.conj().T  # (nocc[1], norb_b)
    rot_h1_a = caH @ ham_data.h1_a  # (nocc[0], norb_a)
    rot_h1_b = cbH @ ham_data.h1_b  # (nocc[1], norb_b)
    rot_chol_a = jnp.einsum("ip,gpq->giq", caH, ham_data.chol_a, optimize="optimal")
    rot_chol_b = jnp.einsum("ip,gpq->giq", cbH, ham_data.chol_b, optimize="optimal")
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


def make_uhf_meas_ops(sys: System) -> MeasOps:
    wk = sys.walker_kind.lower()
    if wk == "restricted":
        overlap_fn = overlap_r
        build_meas_ctx_fn = build_meas_ctx
        kernels = {
            k_force_bias: force_bias_kernel_rw_rh,
            k_energy: energy_kernel_rw_rh,
        }
        observables = {
            o_rdm1: rdm1_kernel_rw,
            o_density_corr: density_corr_kernel_rw,
        }
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        build_meas_ctx_fn = build_meas_ctx
        kernels = {
            k_force_bias: force_bias_kernel_uw_rh,
            k_energy: energy_kernel_uw_rh,
        }
        observables = {
            o_rdm1: rdm1_kernel_uw,
            o_density_corr: density_corr_kernel_uw,
        }
    elif wk == "generalized":
        overlap_fn = overlap_g
        build_meas_ctx_fn = build_meas_ctx
        kernels = {k_force_bias: force_bias_kernel_gw_rh, k_energy: energy_kernel_gw_rh}
        observables = {
            o_rdm1: rdm1_kernel_gw,
            o_density_corr: density_corr_kernel_gw,
        }
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=build_meas_ctx_fn,
        kernels=kernels,
        observables=observables,
    )


def make_uhf_meas_ops_uh(sys: Any, *, nchol_chunk: int | None = DEFAULT_NCHOL_CHUNK) -> MeasOps:
    """
    MeasOps for an unrestricted (uchol) hamiltonian.

    Only the unrestricted walker kind is meaningful here: alpha and beta live in
    different orbital spaces, so a shared or spin blocked walker cannot represent them.

    No observables are wired: rdm1_kernel_uw and density_corr_kernel_uw stack the two
    spin blocks and so do not survive norb_a != norb_b.
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
