"""
UCISD measurement kernels on the unrestricted (uchol) hamiltonian.

Overlap, force bias and local energy of an unrestricted walker (wa, wb), each spin in
its own orbital basis, against

    <T| = <HF| (1 + C1 + C2),   C1 = sum c1_ia i+ a,   C2 = 1/2 sum c2_iajb ... (same spin)
                                                       + sum c2_iajb ...     (alpha-beta)

with the reference of each spin in the leading nocc orbitals of that spin's basis
(kernel suffix _uw_uh: unrestricted walker, unrestricted hamiltonian; meas/ucisd.py's
_uw_rh kernels are the same trial on the restricted hamiltonian, where beta is rotated
into the alpha basis). Ported from afqmc's wavefunctions_unrestricted.ucisd
(_calc_overlap, _calc_force_bias, _calc_energy, _build_measurement_intermediates), whose
ham_data {"h1": [h1a, h1b], "chol": [La, Lb]} is HamCholU and whose wave_data
{ci1A, ci2AA, ...} is UcisdTrial. The lci1 intermediates live in the meas ctx.

Conventions, per spin (dropping the spin label):
    green      (nocc, norb)   [phi (phi_occ)^-1]^T,   the half green's function
    green_occ  (nocc, nvir)   its virtual columns
    greenp     (norb, nvir)   [green_occ; -1]
    rot_chol   (g, nocc, norb) the occupied rows of chol

With every coefficient zero the kernels reduce to meas.uhf_uh's, and with identical
alpha and beta bases to meas.ucisd's _uw_rh kernels (tests/test_uchol_ucisd.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..ham.chol_u import HamCholU
from ..trial.ucisd import UcisdTrial


def _greens(wa: jax.Array, wb: jax.Array, nocc_a: int, nocc_b: int):
    green_a = (wa @ jnp.linalg.inv(wa[:nocc_a, :])).T
    green_b = (wb @ jnp.linalg.inv(wb[:nocc_b, :])).T
    return green_a, green_b


def overlap_uw_uh(walker: tuple[jax.Array, jax.Array], trial_data: UcisdTrial) -> jax.Array:
    wa, wb = walker
    nocc_a, nocc_b = trial_data.nocc
    c1a, c1b = trial_data.c1a, trial_data.c1b
    c2aa, c2ab, c2bb = trial_data.c2aa, trial_data.c2ab, trial_data.c2bb
    green_a, green_b = _greens(wa, wb, nocc_a, nocc_b)
    green_a, green_b = green_a[:, nocc_a:], green_b[:, nocc_b:]
    o0 = jnp.linalg.det(wa[:nocc_a, :]) * jnp.linalg.det(wb[:nocc_b, :])
    o1 = jnp.einsum("ia,ia", c1a, green_a) + jnp.einsum("ia,ia", c1b, green_b)
    o2 = (
        0.5 * jnp.einsum("iajb,ia,jb", c2aa, green_a, green_a, optimize="optimal")
        + 0.5 * jnp.einsum("iajb,ia,jb", c2bb, green_b, green_b, optimize="optimal")
        + jnp.einsum("iajb,ia,jb", c2ab, green_a, green_b, optimize="optimal")
    )
    return (1.0 + o1 + o2) * o0


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdMeasCtxUh:
    # afqmc's lci1 intermediates: chol's virtual columns contracted with c1
    lci1_a: jax.Array  # (n_chol, norb_a, nocc_a)
    lci1_b: jax.Array  # (n_chol, norb_b, nocc_b)

    def tree_flatten(self):
        return (self.lci1_a, self.lci1_b), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def build_meas_ctx_uh(ham_data: HamCholU, trial_data: UcisdTrial) -> UcisdMeasCtxUh:
    if ham_data.basis != "uchol":
        raise ValueError("UCISD unrestricted MeasOps assumes HamCholU.basis == 'uchol'.")
    nocc_a, nocc_b = trial_data.nocc
    lci1_a = jnp.einsum("git,pt->gip", ham_data.chol_a[:, :, nocc_a:], trial_data.c1a)
    lci1_b = jnp.einsum("git,pt->gip", ham_data.chol_b[:, :, nocc_b:], trial_data.c1b)
    return UcisdMeasCtxUh(lci1_a=lci1_a, lci1_b=lci1_b)


def force_bias_kernel_uw_uh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: UcisdMeasCtxUh,
    trial_data: UcisdTrial,
) -> jax.Array:
    """<T| L_g |phi> / <T|phi> for every cholesky vector g."""
    wa, wb = walker
    nocc_a, nocc_b = trial_data.nocc
    norb_a, norb_b = wa.shape[0], wb.shape[0]
    c1a, c1b = trial_data.c1a, trial_data.c1b
    c2aa, c2ab, c2bb = trial_data.c2aa, trial_data.c2ab, trial_data.c2bb
    green_a, green_b = _greens(wa, wb, nocc_a, nocc_b)
    green_occ_a = green_a[:, nocc_a:]
    green_occ_b = green_b[:, nocc_b:]
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

    chol_a, chol_b = ham_data.chol_a, ham_data.chol_b
    rot_chol_a = chol_a[:, :nocc_a, :]
    rot_chol_b = chol_b[:, :nocc_b, :]
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a)
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b)
    lg = lg_a + lg_b

    # reference
    fb_0 = lg

    # single excitations
    ci1g = jnp.einsum("pt,pt->", c1a, green_occ_a) + jnp.einsum("pt,pt->", c1b, green_occ_b)
    fb_1_1 = ci1g * lg
    ci1gp_a = jnp.einsum("pt,it->pi", c1a, greenp_a)
    ci1gp_b = jnp.einsum("pt,it->pi", c1b, greenp_b)
    gci1gp_a = jnp.einsum("pj,pi->ij", green_a, ci1gp_a)
    gci1gp_b = jnp.einsum("pj,pi->ij", green_b, ci1gp_b)
    fb_1_2 = -jnp.einsum("gij,ij->g", chol_a, gci1gp_a) - jnp.einsum("gij,ij->g", chol_b, gci1gp_b)
    fb_1 = fb_1_1 + fb_1_2

    # double excitations
    ci2g_a = jnp.einsum("ptqu,pt->qu", c2aa, green_occ_a)
    ci2g_b = jnp.einsum("ptqu,pt->qu", c2bb, green_occ_b)
    ci2g_ab_a = jnp.einsum("ptqu,qu->pt", c2ab, green_occ_b)
    ci2g_ab_b = jnp.einsum("ptqu,pt->qu", c2ab, green_occ_a)
    gci2g = (
        0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a)
        + 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b)
        + jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a)
    )
    fb_2_1 = lg * gci2g
    ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
    ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
    fb_2_2 = -jnp.einsum("gij,ij->g", chol_a, ci2_green_a) - jnp.einsum(
        "gij,ij->g", chol_b, ci2_green_b
    )
    fb_2 = fb_2_1 + fb_2_2

    overlap = 1.0 + ci1g + gci2g
    return (fb_0 + fb_1 + fb_2) / overlap


def energy_kernel_uw_uh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: UcisdMeasCtxUh,
    trial_data: UcisdTrial,
) -> jax.Array:
    """<T| H |phi> / <T|phi>, h0 included."""
    wa, wb = walker
    nocc_a, nocc_b = trial_data.nocc
    norb_a, norb_b = wa.shape[0], wb.shape[0]
    c1a, c1b = trial_data.c1a, trial_data.c1b
    c2aa, c2ab, c2bb = trial_data.c2aa, trial_data.c2ab, trial_data.c2bb
    green_a, green_b = _greens(wa, wb, nocc_a, nocc_b)
    green_occ_a = green_a[:, nocc_a:]
    green_occ_b = green_b[:, nocc_b:]
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

    chol_a, chol_b = ham_data.chol_a, ham_data.chol_b
    rot_chol_a = chol_a[:, :nocc_a, :]
    rot_chol_b = chol_b[:, :nocc_b, :]
    h1_a, h1_b = ham_data.h1_a, ham_data.h1_b
    hg = jnp.einsum("pj,pj->", h1_a[:nocc_a, :], green_a) + jnp.einsum(
        "pj,pj->", h1_b[:nocc_b, :], green_b
    )

    e0 = ham_data.h0

    # ---- one body
    e1_0 = hg

    ci1g = jnp.einsum("pt,pt->", c1a, green_occ_a) + jnp.einsum("pt,pt->", c1b, green_occ_b)
    e1_1_1 = ci1g * hg
    ci1_green_a = (greenp_a @ c1a.T) @ green_a
    ci1_green_b = (greenp_b @ c1b.T) @ green_b
    e1_1_2 = -(jnp.einsum("ij,ij->", h1_a, ci1_green_a) + jnp.einsum("ij,ij->", h1_b, ci1_green_b))
    e1_1 = e1_1_1 + e1_1_2

    ci2g_a = jnp.einsum("ptqu,pt->qu", c2aa, green_occ_a) / 4
    ci2g_b = jnp.einsum("ptqu,pt->qu", c2bb, green_occ_b) / 4
    ci2g_ab_a = jnp.einsum("ptqu,qu->pt", c2ab, green_occ_b)
    ci2g_ab_b = jnp.einsum("ptqu,pt->qu", c2ab, green_occ_a)
    gci2g_a = jnp.einsum("qu,qu->", ci2g_a, green_occ_a)
    gci2g_b = jnp.einsum("qu,qu->", ci2g_b, green_occ_b)
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a)
    gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
    e1_2_1 = hg * gci2g
    ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
    ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
    ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
    ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
    e1_2_2 = -jnp.einsum("ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a) - jnp.einsum(
        "ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b
    )
    e1_2 = e1_2_1 + e1_2_2

    e1 = e1_0 + e1_1 + e1_2

    # ---- two body
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a)
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b)
    lg1_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a)
    lg1_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b)

    # reference: 1/2 sum_g [ (tr L_g G)^2 - tr(L_g G L_g G) ] over both spins
    def _ref_scan(carry, x):
        chol_a_c, chol_b_c = x  # (nocc, norb) each, one vector
        lg_a_c = jnp.einsum("pr,qr->pq", chol_a_c, green_a)
        lg_b_c = jnp.einsum("pr,qr->pq", chol_b_c, green_b)
        trlg_a_c = jnp.trace(lg_a_c)
        trlg_b_c = jnp.trace(lg_b_c)
        e2aa_c = trlg_a_c**2 - jnp.einsum("pq,qp->", lg_a_c, lg_a_c)
        e2ab_c = trlg_a_c * trlg_b_c * 2
        e2bb_c = trlg_b_c**2 - jnp.einsum("pq,qp->", lg_b_c, lg_b_c)
        return carry + (e2aa_c + e2ab_c + e2bb_c) / 2, 0.0

    e2_0, _ = lax.scan(_ref_scan, 0.0 + 0j * hg, (rot_chol_a, rot_chol_b))

    # single excitations
    e2_1_1 = e2_0 * ci1g
    lci1g_a = jnp.einsum("gij,ij->g", chol_a, ci1_green_a)
    lci1g_b = jnp.einsum("gij,ij->g", chol_b, ci1_green_b)
    e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
    ci1g1_a = c1a @ green_occ_a.T
    ci1g1_b = c1b @ green_occ_b.T
    e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, optimize="optimal") + jnp.einsum(
        "gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, optimize="optimal"
    )
    lci1g_a = jnp.einsum("gip,qi->gpq", meas_ctx.lci1_a, green_a)
    lci1g_b = jnp.einsum("gip,qi->gpq", meas_ctx.lci1_b, green_b)
    e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g_a, lg1_a) - jnp.einsum("gpq,gqp->", lci1g_b, lg1_b)
    e2_1 = e2_1_1 + e2_1_2 + e2_1_3_1 + e2_1_3_2

    # double excitations
    e2_2_1 = e2_0 * gci2g
    lci2g_a = jnp.einsum("gij,ij->g", chol_a, 8 * ci2_green_a + 2 * ci2_green_ab_a)
    lci2g_b = jnp.einsum("gij,ij->g", chol_b, 8 * ci2_green_b + 2 * ci2_green_ab_b)
    e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

    def _dbl_scan(carry, x):
        chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
        gl_a_i = jnp.einsum("pj,ji->pi", green_a, chol_a_i)
        gl_b_i = jnp.einsum("pj,ji->pi", green_b, chol_b_i)
        lci2_green_a_i = jnp.einsum("pi,ji->pj", rot_chol_a_i, 8 * ci2_green_a + 2 * ci2_green_ab_a)
        lci2_green_b_i = jnp.einsum("pi,ji->pj", rot_chol_b_i, 8 * ci2_green_b + 2 * ci2_green_ab_b)
        c0 = carry[0] + 0.5 * (
            jnp.einsum("pi,pi->", gl_a_i, lci2_green_a_i)
            + jnp.einsum("pi,pi->", gl_b_i, lci2_green_b_i)
        )
        glgp_a_i = jnp.einsum("pi,it->pt", gl_a_i, greenp_a)
        glgp_b_i = jnp.einsum("pi,it->pt", gl_b_i, greenp_b)
        l2ci2_a = 0.5 * jnp.einsum("pt,qu,ptqu->", glgp_a_i, glgp_a_i, c2aa, optimize="optimal")
        l2ci2_b = 0.5 * jnp.einsum("pt,qu,ptqu->", glgp_b_i, glgp_b_i, c2bb, optimize="optimal")
        l2ci2_ab = jnp.einsum("pt,qu,ptqu->", glgp_a_i, glgp_b_i, c2ab, optimize="optimal")
        c1 = carry[1] + l2ci2_a + l2ci2_b + l2ci2_ab
        return (c0, c1), 0.0

    zero = 0.0 + 0j * hg
    (e2_2_2_2, e2_2_3), _ = lax.scan(
        _dbl_scan, (zero, zero), (chol_a, rot_chol_a, chol_b, rot_chol_b)
    )
    e2_2 = e2_2_1 + e2_2_2_1 + e2_2_2_2 + e2_2_3

    e2 = e2_0 + e2_1 + e2_2

    overlap = 1.0 + ci1g + gci2g
    return (e1 + e2) / overlap + e0


def make_ucisd_meas_ops_uh(sys: Any, mixed_precision: bool = False) -> MeasOps:
    """
    MeasOps of the UCISD trial on the unrestricted hamiltonian. Only unrestricted walkers
    are meaningful, as for make_uhf_meas_ops_uh. mixed_precision is accepted for the
    pipeline's sake and not honoured: every contraction runs in the working precision.
    """
    wk = sys.walker_kind.lower()
    if wk != "unrestricted":
        raise ValueError(
            f"the unrestricted hamiltonian path requires walker_kind='unrestricted', got {wk!r}"
        )
    return MeasOps(
        overlap=overlap_uw_uh,
        build_meas_ctx=build_meas_ctx_uh,
        kernels={k_force_bias: force_bias_kernel_uw_uh, k_energy: energy_kernel_uw_uh},
        observables={},
    )
