from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, tree_util, vmap

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from .chol_setup import transform_cholesky
from ..trial.ucisdt import UcisdtTrial, overlap_r, overlap_u


def _half_green_from_overlap_matrix(w: jax.Array, ovlp_mat: jax.Array) -> jax.Array:
    return jnp.linalg.solve(ovlp_mat.T, w.T)


def _greenp_from_occ(green_occ: jax.Array) -> jax.Array:
    nvir = green_occ.shape[1]
    return jnp.concatenate((green_occ, -jnp.eye(nvir, dtype=green_occ.dtype)), axis=0)


@dataclass(frozen=True)
class UcisdtMeasCfg:
    memory_mode: str = "high"
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdtMeasCtx:
    h1_b: jax.Array  # (norb, norb)
    chol_b: jax.Array  # (n_chol, norb, norb)

    rot_h1_a: jax.Array  # (nocc[0], norb)
    rot_h1_b: jax.Array  # (nocc[1], norb)
    rot_chol_a: jax.Array  # (n_chol, nocc[0], norb)
    rot_chol_b: jax.Array  # (n_chol, nocc[1], norb)
    rot_chol_flat_a: jax.Array  # (n_chol, nocc[0]*norb)
    rot_chol_flat_b: jax.Array  # (n_chol, nocc[1]*norb)

    lci1_a: jax.Array  # (n_chol, norb, nocc[0])
    lci1_b: jax.Array  # (n_chol, norb, nocc[1])

    cfg: UcisdtMeasCfg

    def tree_flatten(self):
        children = (
            self.h1_b,
            self.chol_b,
            self.rot_h1_a,
            self.rot_h1_b,
            self.rot_chol_a,
            self.rot_chol_b,
            self.rot_chol_flat_a,
            self.rot_chol_flat_b,
            self.lci1_a,
            self.lci1_b,
        )
        return children, (self.cfg,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        (
            h1_b,
            chol_b,
            rot_h1_a,
            rot_h1_b,
            rot_chol_a,
            rot_chol_b,
            rot_chol_flat_a,
            rot_chol_flat_b,
            lci1_a,
            lci1_b,
        ) = children
        return cls(
            h1_b=h1_b,
            chol_b=chol_b,
            rot_h1_a=rot_h1_a,
            rot_h1_b=rot_h1_b,
            rot_chol_a=rot_chol_a,
            rot_chol_b=rot_chol_b,
            rot_chol_flat_a=rot_chol_flat_a,
            rot_chol_flat_b=rot_chol_flat_b,
            lci1_a=lci1_a,
            lci1_b=lci1_b,
            cfg=cfg,
        )


def _real_c3_contract_qurs(c3: jax.Array, a: jax.Array, *, low_memory: bool) -> jax.Array:
    if not low_memory:
        return jnp.einsum("ptqurs,pt->qurs", c3, a, optimize="optimal")

    out_dtype = jnp.result_type(c3.dtype, a.dtype)
    out = jnp.zeros(c3.shape[2:6], dtype=out_dtype)
    n_t = a.shape[1]

    def body(i, acc):
        p = i // n_t
        t = i - p * n_t
        return acc + a[p, t] * c3[p, t, :, :, :, :]

    return lax.fori_loop(0, a.size, body, out)


def _real_c3_contract_ptqu(c3: jax.Array, c: jax.Array, *, low_memory: bool) -> jax.Array:
    if not low_memory:
        return jnp.einsum("ptqurs,rs->ptqu", c3, c, optimize="optimal")

    out_dtype = jnp.result_type(c3.dtype, c.dtype)
    out = jnp.zeros(c3.shape[0:4], dtype=out_dtype)
    n_s = c.shape[1]

    def body(i, acc):
        r = i // n_s
        s = i - r * n_s
        return acc + c[r, s] * c3[:, :, :, :, r, s]

    return lax.fori_loop(0, c.size, body, out)


def _real_c3_contract_rs(
    c3: jax.Array, a: jax.Array, b: jax.Array, *, low_memory: bool
) -> jax.Array:
    if not low_memory:
        if a.size <= b.size:
            tmp = jnp.einsum("ptqurs,pt->qurs", c3, a, optimize="optimal")
            return jnp.einsum("qurs,qu->rs", tmp, b, optimize="optimal")
        tmp = jnp.einsum("ptqurs,qu->ptrs", c3, b, optimize="optimal")
        return jnp.einsum("ptrs,pt->rs", tmp, a, optimize="optimal")

    out_dtype = jnp.result_type(c3.dtype, a.dtype, b.dtype)
    out = jnp.zeros(c3.shape[4:6], dtype=out_dtype)

    if a.size <= b.size:
        n_t = a.shape[1]

        def body_pt(i, acc):
            p = i // n_t
            t = i - p * n_t
            c_pt = c3[p, t, :, :, :, :]
            contrib = jnp.einsum("qu,qurs->rs", b, c_pt, optimize="optimal")
            return acc + a[p, t] * contrib

        return lax.fori_loop(0, a.size, body_pt, out)

    n_u = b.shape[1]

    def body_qu(i, acc):
        q = i // n_u
        u = i - q * n_u
        c_qu = c3[:, :, q, u, :, :]
        contrib = jnp.einsum("pt,ptrs->rs", a, c_qu, optimize="optimal")
        return acc + b[q, u] * contrib

    return lax.fori_loop(0, b.size, body_qu, out)


def _real_c3_contract_qu(
    c3: jax.Array, a: jax.Array, c: jax.Array, *, low_memory: bool
) -> jax.Array:
    if not low_memory:
        if a.size <= c.size:
            tmp = jnp.einsum("ptqurs,pt->qurs", c3, a, optimize="optimal")
            return jnp.einsum("qurs,rs->qu", tmp, c, optimize="optimal")
        tmp = jnp.einsum("ptqurs,rs->ptqu", c3, c, optimize="optimal")
        return jnp.einsum("ptqu,pt->qu", tmp, a, optimize="optimal")

    out_dtype = jnp.result_type(c3.dtype, a.dtype, c.dtype)
    out = jnp.zeros(c3.shape[2:4], dtype=out_dtype)

    if a.size <= c.size:
        n_t = a.shape[1]

        def body_pt(i, acc):
            p = i // n_t
            t = i - p * n_t
            c_pt = c3[p, t, :, :, :, :]
            contrib = jnp.einsum("rs,qurs->qu", c, c_pt, optimize="optimal")
            return acc + a[p, t] * contrib

        return lax.fori_loop(0, a.size, body_pt, out)

    n_s = c.shape[1]

    def body_rs(i, acc):
        r = i // n_s
        s = i - r * n_s
        c_rs = c3[:, :, :, :, r, s]
        contrib = jnp.einsum("pt,ptqu->qu", a, c_rs, optimize="optimal")
        return acc + c[r, s] * contrib

    return lax.fori_loop(0, c.size, body_rs, out)


def _real_c3_contract_pt(
    c3: jax.Array, b: jax.Array, c: jax.Array, *, low_memory: bool
) -> jax.Array:
    if not low_memory:
        if b.size <= c.size:
            tmp = jnp.einsum("ptqurs,qu->ptrs", c3, b, optimize="optimal")
            return jnp.einsum("ptrs,rs->pt", tmp, c, optimize="optimal")
        tmp = jnp.einsum("ptqurs,rs->ptqu", c3, c, optimize="optimal")
        return jnp.einsum("ptqu,qu->pt", tmp, b, optimize="optimal")

    out_dtype = jnp.result_type(c3.dtype, b.dtype, c.dtype)
    out = jnp.zeros(c3.shape[0:2], dtype=out_dtype)

    if b.size <= c.size:
        n_u = b.shape[1]

        def body_qu(i, acc):
            q = i // n_u
            u = i - q * n_u
            c_qu = c3[:, :, q, u, :, :]
            contrib = jnp.einsum("rs,ptrs->pt", c, c_qu, optimize="optimal")
            return acc + b[q, u] * contrib

        return lax.fori_loop(0, b.size, body_qu, out)

    n_s = c.shape[1]

    def body_rs(i, acc):
        r = i // n_s
        s = i - r * n_s
        c_rs = c3[:, :, :, :, r, s]
        contrib = jnp.einsum("qu,ptqu->pt", b, c_rs, optimize="optimal")
        return acc + c[r, s] * contrib

    return lax.fori_loop(0, c.size, body_rs, out)


def _complex_from_real_parts(real_part: jax.Array, imag_part: jax.Array) -> jax.Array:
    return lax.complex(real_part, imag_part)


def _c3_contract_qurs(c3: jax.Array, a: jax.Array, *, low_memory: bool) -> jax.Array:
    a_real = jnp.real(a)
    a_imag = jnp.imag(a)
    real_part = _real_c3_contract_qurs(c3, a_real, low_memory=low_memory)
    imag_part = _real_c3_contract_qurs(c3, a_imag, low_memory=low_memory)
    return _complex_from_real_parts(real_part, imag_part)


def _c3_contract_ptqu(c3: jax.Array, c: jax.Array, *, low_memory: bool) -> jax.Array:
    c_real = jnp.real(c)
    c_imag = jnp.imag(c)
    real_part = _real_c3_contract_ptqu(c3, c_real, low_memory=low_memory)
    imag_part = _real_c3_contract_ptqu(c3, c_imag, low_memory=low_memory)
    return _complex_from_real_parts(real_part, imag_part)


def _c3_contract_rs(c3: jax.Array, a: jax.Array, b: jax.Array, *, low_memory: bool) -> jax.Array:
    a_real = jnp.real(a)
    a_imag = jnp.imag(a)
    b_real = jnp.real(b)
    b_imag = jnp.imag(b)

    rr = _real_c3_contract_rs(c3, a_real, b_real, low_memory=low_memory)
    ri = _real_c3_contract_rs(c3, a_real, b_imag, low_memory=low_memory)
    ir = _real_c3_contract_rs(c3, a_imag, b_real, low_memory=low_memory)
    ii = _real_c3_contract_rs(c3, a_imag, b_imag, low_memory=low_memory)

    return _complex_from_real_parts(rr - ii, ri + ir)


def _c3_contract_qu(c3: jax.Array, a: jax.Array, c: jax.Array, *, low_memory: bool) -> jax.Array:
    a_real = jnp.real(a)
    a_imag = jnp.imag(a)
    c_real = jnp.real(c)
    c_imag = jnp.imag(c)

    rr = _real_c3_contract_qu(c3, a_real, c_real, low_memory=low_memory)
    ri = _real_c3_contract_qu(c3, a_real, c_imag, low_memory=low_memory)
    ir = _real_c3_contract_qu(c3, a_imag, c_real, low_memory=low_memory)
    ii = _real_c3_contract_qu(c3, a_imag, c_imag, low_memory=low_memory)

    return _complex_from_real_parts(rr - ii, ri + ir)


def _c3_contract_pt(c3: jax.Array, b: jax.Array, c: jax.Array, *, low_memory: bool) -> jax.Array:
    b_real = jnp.real(b)
    b_imag = jnp.imag(b)
    c_real = jnp.real(c)
    c_imag = jnp.imag(c)

    rr = _real_c3_contract_pt(c3, b_real, c_real, low_memory=low_memory)
    ri = _real_c3_contract_pt(c3, b_real, c_imag, low_memory=low_memory)
    ir = _real_c3_contract_pt(c3, b_imag, c_real, low_memory=low_memory)
    ii = _real_c3_contract_pt(c3, b_imag, c_imag, low_memory=low_memory)

    return _complex_from_real_parts(rr - ii, ri + ir)


def _c3_contract_scalar(
    c3: jax.Array, a: jax.Array, b: jax.Array, c: jax.Array, *, low_memory: bool
) -> jax.Array:
    pt = _c3_contract_pt(c3, b, c, low_memory=low_memory)
    return jnp.einsum("pt,pt->", a, pt, optimize="optimal")


def _triples_overlap(
    trial_data: UcisdtTrial, go_a: jax.Array, go_b: jax.Array, *, low_memory: bool
) -> jax.Array:
    return (
        (1 / 6) * _c3_contract_scalar(trial_data.c3aaa, go_a, go_a, go_a, low_memory=low_memory)
        + (1 / 6) * _c3_contract_scalar(trial_data.c3bbb, go_b, go_b, go_b, low_memory=low_memory)
        + (1 / 2) * _c3_contract_scalar(trial_data.c3aab, go_a, go_a, go_b, low_memory=low_memory)
        + (1 / 2) * _c3_contract_scalar(trial_data.c3abb, go_a, go_b, go_b, low_memory=low_memory)
    )


# ---------------------------------------------------------------------------
# Triples helper: force-bias contribution
# ---------------------------------------------------------------------------


def _force_bias_triples(
    trial_data: UcisdtTrial,
    green_a: jax.Array,  # (n_oa, norb)
    green_b: jax.Array,  # (n_ob, norb)
    go_a: jax.Array,  # (n_oa, n_va)
    go_b: jax.Array,  # (n_ob, n_vb)
    gp_a: jax.Array,  # (norb, n_va)
    gp_b: jax.Array,  # (norb, n_vb)
    chol_a: jax.Array,  # (n_chol, norb, norb)
    chol_b: jax.Array,  # (n_chol, norb, norb)
    low_memory: bool = False,
) -> jax.Array:
    """<psi_T(triples)| chol_g |w> / <psi_T|w>  (numerator contribution only)."""
    n_oa, n_ob = trial_data.nocc

    lo_a = chol_a[:, :n_oa, :]  # (n_chol, n_oa, norb)
    lo_b = chol_b[:, :n_ob, :]  # (n_chol, n_ob, norb)

    c3aaa = trial_data.c3aaa
    c3aab = trial_data.c3aab
    c3abb = trial_data.c3abb
    c3bbb = trial_data.c3bbb

    # X_g = sum_{ij} L^a_{g,ij} G^a_{ij} + L^b_{g,ij} G^b_{ij}
    xa = jnp.einsum("gij,ij->g", lo_a, green_a)
    xb = jnp.einsum("gij,ij->g", lo_b, green_b)
    x = xa + xb

    # Y^s_{g,p,t} = sum_{i,j} G^s_{p,j} L^s_{g,i,j} Gp^s_{i,t}
    ya = jnp.einsum("pj,gij,it->gpt", green_a, chol_a, gp_a)  # (n_chol, n_oa, n_va)
    yb = jnp.einsum("pj,gij,it->gpt", green_b, chol_b, gp_b)  # (n_chol, n_ob, n_vb)

    # --- AAA ---
    cgg_a = _c3_contract_pt(c3aaa, go_a, go_a, low_memory=low_memory)  # (n_oa, n_va)
    cggg_a = jnp.einsum("pt,pt->", cgg_a, go_a)
    fb_aaa = (1 / 6) * cggg_a * x - (1 / 2) * jnp.einsum("pt,gpt->g", cgg_a, ya)

    # --- BBB ---
    cgg_b = _c3_contract_pt(c3bbb, go_b, go_b, low_memory=low_memory)  # (n_ob, n_vb)
    cggg_b = jnp.einsum("pt,pt->", cgg_b, go_b)
    fb_bbb = (1 / 6) * cggg_b * x - (1 / 2) * jnp.einsum("pt,gpt->g", cgg_b, yb)

    # --- AAB ---
    caab_ga_gb = _c3_contract_pt(c3aab, go_a, go_b, low_memory=low_memory)  # (n_oa, n_va)
    caab_ga_ga_gb = jnp.einsum("pt,pt->", caab_ga_gb, go_a)
    ga_ga_caab = _c3_contract_rs(c3aab, go_a, go_a, low_memory=low_memory)  # (n_ob, n_vb)

    fb_aab = (
        (1 / 2) * caab_ga_ga_gb * x
        - jnp.einsum("gpt,pt->g", ya, caab_ga_gb)
        - (1 / 2) * jnp.einsum("grs,rs->g", yb, ga_ga_caab)
    )

    # --- ABB ---
    cabb_ga_gb_gb = _c3_contract_scalar(c3abb, go_a, go_b, go_b, low_memory=low_memory)
    ga_cabb_gb = _c3_contract_qu(c3abb, go_a, go_b, low_memory=low_memory)  # (n_ob, n_vb)
    cabb_gb_gb = _c3_contract_pt(c3abb, go_b, go_b, low_memory=low_memory)  # (n_oa, n_va)

    fb_abb = (
        (1 / 2) * cabb_ga_gb_gb * x
        - jnp.einsum("gqu,qu->g", yb, ga_cabb_gb)
        - (1 / 2) * jnp.einsum("gpt,pt->g", ya, cabb_gb_gb)
    )

    return fb_aaa + fb_bbb + fb_aab + fb_abb


# ---------------------------------------------------------------------------
# Triples helper: one-body energy contribution from triples
# ---------------------------------------------------------------------------


def _one_body_energy_triples(
    trial_data: UcisdtTrial,
    green_a: jax.Array,  # (n_oa, norb)
    green_b: jax.Array,  # (n_ob, norb)
    go_a: jax.Array,  # (n_oa, n_va)
    go_b: jax.Array,  # (n_ob, n_vb)
    gp_a: jax.Array,  # (norb, n_va)
    gp_b: jax.Array,  # (norb, n_vb)
    h1_a: jax.Array,  # (norb, norb)
    h1_b: jax.Array,  # (norb, norb)
    low_memory: bool = False,
) -> jax.Array:
    """
    One-body local energy contribution from the triples sector.
    Equivalent to calling _force_bias_triples with h1 as a single Cholesky
    vector but avoids the extra g-dimension overhead.
    """
    n_oa, n_ob = trial_data.nocc

    c3aaa = trial_data.c3aaa
    c3aab = trial_data.c3aab
    c3abb = trial_data.c3abb
    c3bbb = trial_data.c3bbb

    # x = Tr(h1_oa @ G^a) + Tr(h1_ob @ G^b)
    xa = jnp.einsum("ij,ij->", h1_a[:n_oa, :], green_a)
    xb = jnp.einsum("ij,ij->", h1_b[:n_ob, :], green_b)
    x = xa + xb

    # ya[p,t] = sum_{i,j} G^a_{p,j} h1_a_{i,j} Gp_a_{i,t}
    ya = jnp.einsum("pj,ij,it->pt", green_a, h1_a, gp_a)  # (n_oa, n_va)
    yb = jnp.einsum("pj,ij,it->pt", green_b, h1_b, gp_b)  # (n_ob, n_vb)

    # --- AAA ---
    cgg_a = _c3_contract_pt(c3aaa, go_a, go_a, low_memory=low_memory)
    cggg_a = jnp.einsum("pt,pt->", cgg_a, go_a)
    e_aaa = (1 / 6) * cggg_a * x - (1 / 2) * jnp.einsum("pt,pt->", cgg_a, ya)

    # --- BBB ---
    cgg_b = _c3_contract_pt(c3bbb, go_b, go_b, low_memory=low_memory)
    cggg_b = jnp.einsum("pt,pt->", cgg_b, go_b)
    e_bbb = (1 / 6) * cggg_b * x - (1 / 2) * jnp.einsum("pt,pt->", cgg_b, yb)

    # --- AAB ---
    caab_ga_gb = _c3_contract_pt(c3aab, go_a, go_b, low_memory=low_memory)
    caab_ga_ga_gb = jnp.einsum("pt,pt->", caab_ga_gb, go_a)
    ga_ga_caab = _c3_contract_rs(c3aab, go_a, go_a, low_memory=low_memory)

    e_aab = (
        (1 / 2) * caab_ga_ga_gb * x
        - jnp.einsum("pt,pt->", ya, caab_ga_gb)
        - (1 / 2) * jnp.einsum("rs,rs->", yb, ga_ga_caab)
    )

    # --- ABB ---
    cabb_ga_gb_gb = _c3_contract_scalar(c3abb, go_a, go_b, go_b, low_memory=low_memory)
    ga_cabb_gb = _c3_contract_qu(c3abb, go_a, go_b, low_memory=low_memory)
    cabb_gb_gb = _c3_contract_pt(c3abb, go_b, go_b, low_memory=low_memory)

    e_abb = (
        (1 / 2) * cabb_ga_gb_gb * x
        - jnp.einsum("qu,qu->", yb, ga_cabb_gb)
        - (1 / 2) * jnp.einsum("pt,pt->", ya, cabb_gb_gb)
    )

    return e_aaa + e_bbb + e_aab + e_abb


# ---------------------------------------------------------------------------
# Triples helper: two-body energy contribution from triples
# ---------------------------------------------------------------------------


def _two_body_energy_triples(
    trial_data: UcisdtTrial,
    green_a: jax.Array,  # (n_oa, norb)
    green_b: jax.Array,  # (n_ob, norb)
    go_a: jax.Array,  # (n_oa, n_va)
    go_b: jax.Array,  # (n_ob, n_vb)
    gp_a: jax.Array,  # (norb, n_va)
    gp_b: jax.Array,  # (norb, n_vb)
    chol_a: jax.Array,  # (n_chol, norb, norb)
    chol_b: jax.Array,  # (n_chol, norb, norb)
    low_memory: bool = False,
) -> jax.Array:
    n_oa, n_ob = trial_data.nocc

    lo_a = chol_a[:, :n_oa, :]  # (n_chol, n_oa, norb)
    lo_b = chol_b[:, :n_ob, :]  # (n_chol, n_ob, norb)

    c3aaa = trial_data.c3aaa
    c3aab = trial_data.c3aab
    c3abb = trial_data.c3abb
    c3bbb = trial_data.c3bbb

    xa = jnp.einsum("gij,ij->g", lo_a, green_a)
    xb = jnp.einsum("gij,ij->g", lo_b, green_b)
    x = xa + xb

    ya = jnp.einsum("pj,gij,it->gpt", green_a, chol_a, gp_a)  # (n_chol, n_oa, n_va)
    yb = jnp.einsum("pj,gij,it->gpt", green_b, chol_b, gp_b)  # (n_chol, n_ob, n_vb)

    # --- AAA & BBB ---
    x2 = jnp.einsum("g,g->", x, x)

    caaaga = _c3_contract_ptqu(c3aaa, go_a, low_memory=low_memory)  # (n_oa,n_va,n_oa,n_va)
    caaagaga = _c3_contract_pt(c3aaa, go_a, go_a, low_memory=low_memory)  # (n_oa, n_va)
    caaagagaga = jnp.einsum("pt,pt->", caaagaga, go_a)

    cbbbgb = _c3_contract_ptqu(c3bbb, go_b, low_memory=low_memory)
    cbbbgbgb = _c3_contract_pt(c3bbb, go_b, go_b, low_memory=low_memory)
    cbbbgbgbgb = jnp.einsum("pt,pt->", cbbbgbgb, go_b)

    eaaa1 = (1 / 12) * x2 * caaagagaga - (1 / 4) * jnp.einsum("g,gpt,pt->", x, ya, caaagaga)
    ebbb1 = (1 / 12) * x2 * cbbbgbgbgb - (1 / 4) * jnp.einsum("g,gpt,pt->", x, yb, cbbbgbgb)

    galaga = jnp.einsum("il,gij,kj->gkl", green_a, lo_a, green_a)  # (n_chol, n_oa, norb)
    gblbgb = jnp.einsum("il,gij,kj->gkl", green_b, lo_b, green_b)

    lglg = jnp.einsum("gkl,gkl->", galaga, lo_a) + jnp.einsum("gkl,gkl->", gblbgb, lo_b)

    eaaa2 = -(1 / 12) * caaagagaga * lglg + (1 / 4) * jnp.einsum(
        "git,gij,pj,pt->", ya, lo_a, green_a, caaagaga
    )
    ebbb2 = -(1 / 12) * cbbbgbgbgb * lglg + (1 / 4) * jnp.einsum(
        "git,gij,pj,pt->", yb, lo_b, green_b, cbbbgbgb
    )

    eaaa3 = (
        (1 / 4) * jnp.einsum("gkt,pt,gkl,pl->", ya, caaagaga, lo_a, green_a)
        + (1 / 2) * jnp.einsum("gpt,gqu,ptqu->", ya, ya, caaaga)
        - (1 / 4) * jnp.einsum("gpt,pt,g->", ya, caaagaga, x)
    )

    ebbb3 = (
        (1 / 4) * jnp.einsum("gkt,pt,gkl,pl->", yb, cbbbgbgb, lo_b, green_b)
        + (1 / 2) * jnp.einsum("gpt,gqu,ptqu->", yb, yb, cbbbgb)
        - (1 / 4) * jnp.einsum("gpt,pt,g->", yb, cbbbgbgb, x)
    )

    eaaa = eaaa1 + eaaa2 + eaaa3
    ebbb = ebbb1 + ebbb2 + ebbb3

    # --- AAB ---
    caabgagb = _c3_contract_pt(c3aab, go_a, go_b, low_memory=low_memory)  # (n_oa, n_va)
    caabgagagb = jnp.einsum("pt,pt->", caabgagb, go_a)
    gagacaab = _c3_contract_rs(c3aab, go_a, go_a, low_memory=low_memory)  # (n_ob, n_vb)

    eaab1 = (
        (1 / 4) * x2 * caabgagagb
        - (1 / 2) * jnp.einsum("pt,g,gpt->", caabgagb, x, ya)
        - (1 / 4) * jnp.einsum("g,grs,rs->", x, yb, gagacaab)
    )
    eaab2 = (
        -(1 / 4) * lglg * caabgagagb
        + (1 / 2) * jnp.einsum("gij,pj,git,pt->", lo_a, green_a, ya, caabgagb)
        + (1 / 4) * jnp.einsum("gij,rj,gis,rs->", lo_b, green_b, yb, gagacaab)
    )

    caabgb = _c3_contract_ptqu(c3aab, go_b, low_memory=low_memory)  # (n_oa,n_va,n_oa,n_va)
    gacaab = _c3_contract_qurs(c3aab, go_a, low_memory=low_memory)  # (n_oa,n_va,n_ob,n_vb)

    # terms 3 and 4 (4=3 so multiplied by 2)
    eaab3 = (
        (1 / 2) * jnp.einsum("gkt,gkl,pl,pt->", ya, lo_a, green_a, caabgagb)
        - (1 / 2) * jnp.einsum("g,gpt,pt->", x, ya, caabgagb)
        + (1 / 2) * jnp.einsum("gpt,gqu,ptqu->", ya, ya, caabgb)
        + (1 / 2) * jnp.einsum("gpt,grs,ptrs->", ya, yb, gacaab)
    )
    eaab5 = (
        (1 / 4) * jnp.einsum("gks,gkl,rl,rs->", yb, lo_b, green_b, gagacaab)
        - (1 / 4) * jnp.einsum("g,grs,rs->", x, yb, gagacaab)
        + (1 / 2) * jnp.einsum("grs,gpt,ptrs->", yb, ya, gacaab)
    )

    eaab = eaab1 + eaab2 + eaab3 + eaab5

    # --- ABB ---
    cabbgb = _c3_contract_ptqu(c3abb, go_b, low_memory=low_memory)  # (n_oa,n_va,n_ob,n_vb)
    cabbgbgb = jnp.einsum("ptqu,qu->pt", cabbgb, go_b)  # (n_oa, n_va)
    cabbgagbgb = jnp.einsum("pt,pt->", cabbgbgb, go_a)
    gacabbgb = _c3_contract_qu(c3abb, go_a, go_b, low_memory=low_memory)  # (n_ob, n_vb)

    eabb1 = (
        (1 / 4) * x2 * cabbgagbgb
        - (1 / 4) * jnp.einsum("g,pt,gpt->", x, cabbgbgb, ya)
        - (1 / 2) * jnp.einsum("g,qu,gqu->", x, gacabbgb, yb)
    )
    eabb2 = (
        -(1 / 4) * cabbgagbgb * lglg
        + (1 / 4) * jnp.einsum("pt,git,pj,gij->", cabbgbgb, ya, green_a, lo_a)
        + (1 / 2) * jnp.einsum("qu,giu,gij,qj->", gacabbgb, yb, lo_b, green_b)
    )
    eabb3 = (
        (1 / 4) * jnp.einsum("gkt,pt,gkl,pl->", ya, cabbgbgb, lo_a, green_a)
        - (1 / 4) * jnp.einsum("g,pt,gpt->", x, cabbgbgb, ya)
        + (1 / 2) * jnp.einsum("ptqu,gpt,gqu->", cabbgb, ya, yb)
    )

    gacabb = _c3_contract_qurs(c3abb, go_a, low_memory=low_memory)  # (n_ob,n_vb,n_ob,n_vb)

    # terms 4 and 5 (4=5 so multiplied by 2)
    eabb4 = (
        (1 / 2) * jnp.einsum("qu,gku,gkl,ql->", gacabbgb, yb, lo_b, green_b)
        - (1 / 2) * jnp.einsum("g,gqu,qu->", x, yb, gacabbgb)
        + (1 / 2) * jnp.einsum("ptqu,gqu,gpt->", cabbgb, yb, ya)
        + (1 / 2) * jnp.einsum("qurs,gqu,grs->", gacabb, yb, yb)
    )

    eabb = eabb1 + eabb2 + eabb3 + eabb4

    return eaaa + ebbb + eaab + eabb


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def force_bias_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdtMeasCtx,
    trial_data: UcisdtTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return force_bias_kernel_uw_rh(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def force_bias_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UcisdtMeasCtx,
    trial_data: UcisdtTrial,
) -> jax.Array:
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb
    c_b = trial_data.mo_coeff_b

    cfg = meas_ctx.cfg

    wb = c_b.T @ wb[:, :n_ob]
    woa = wa[:n_oa, :]
    wob = wb[:n_ob, :]

    green_a = _half_green_from_overlap_matrix(wa, woa)  # (n_oa, norb)
    green_b = _half_green_from_overlap_matrix(wb, wob)  # (n_ob, norb)

    green_occ_a = green_a[:, n_oa:]  # (n_oa, n_va)
    green_occ_b = green_b[:, n_ob:]  # (n_ob, n_vb)
    greenp_a = _greenp_from_occ(green_occ_a)
    greenp_b = _greenp_from_occ(green_occ_b)

    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    lg = lg_a + lg_b

    # ref
    fb_0 = lg_a + lg_b

    # single excitations
    ci1g_a = jnp.einsum("pt,pt->", c1a, green_occ_a, optimize="optimal")
    ci1g_b = jnp.einsum("pt,pt->", c1b, green_occ_b, optimize="optimal")
    ci1g = ci1g_a + ci1g_b
    fb_1_1 = ci1g * lg
    ci1gp_a = jnp.einsum("pt,it->pi", c1a, greenp_a, optimize="optimal")
    ci1gp_b = jnp.einsum("pt,it->pi", c1b, greenp_b, optimize="optimal")
    gci1gp_a = jnp.einsum("pj,pi->ij", green_a, ci1gp_a, optimize="optimal")
    gci1gp_b = jnp.einsum("pj,pi->ij", green_b, ci1gp_b, optimize="optimal")
    fb_1_2 = -jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        gci1gp_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    ) - jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        gci1gp_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_1 = fb_1_1 + fb_1_2

    # double excitations
    ci2g_a = jnp.einsum(
        "ptqu,pt->qu",
        c2aa.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
    )
    ci2g_b = jnp.einsum(
        "ptqu,pt->qu",
        c2bb.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
    )
    ci2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        c2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
    )
    ci2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        c2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
    )
    gci2g_a = 0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
    gci2g_b = 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
    gci2g = gci2g_a + gci2g_b + gci2g_ab
    fb_2_1 = lg * gci2g
    ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
    ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
    fb_2_2_a = -jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        ci2_green_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_2_2_b = -jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        ci2_green_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    fb_2_2 = fb_2_2_a + fb_2_2_b
    fb_2 = fb_2_1 + fb_2_2

    # overlap (singles + doubles + triples)
    o3 = _triples_overlap(
        trial_data, green_occ_a, green_occ_b, low_memory=(cfg.memory_mode == "low")
    )
    overlap = 1.0 + ci1g + gci2g + o3

    fb_3 = _force_bias_triples(
        trial_data,
        green_a,
        green_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
        chol_a,
        chol_b,
        low_memory=(cfg.memory_mode == "low"),
    )

    return (fb_0 + fb_1 + fb_2 + fb_3) / overlap


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: UcisdtMeasCtx,
    trial_data: UcisdtTrial,
) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return energy_kernel_uw_rh(
        (walker[:, :n_elec_0], walker[:, :n_elec_1]), ham_data, meas_ctx, trial_data
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: UcisdtMeasCtx,
    trial_data: UcisdtTrial,
) -> jax.Array:
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb
    c_b = trial_data.mo_coeff_b

    cfg = meas_ctx.cfg

    wb = c_b.T @ wb[:, :n_ob]
    woa = wa[:n_oa, :]
    wob = wb[:n_ob, :]

    green_a = _half_green_from_overlap_matrix(wa, woa)
    green_b = _half_green_from_overlap_matrix(wb, wob)

    green_occ_a = green_a[:, n_oa:]
    green_occ_b = green_b[:, n_ob:]
    greenp_a = _greenp_from_occ(green_occ_a)
    greenp_b = _greenp_from_occ(green_occ_b)

    lci1_a = meas_ctx.lci1_a
    lci1_b = meas_ctx.lci1_b

    chol_a = ham_data.chol
    chol_b = meas_ctx.chol_b
    rot_chol_a = meas_ctx.rot_chol_a
    rot_chol_b = meas_ctx.rot_chol_b

    h1_a = (ham_data.h1 + ham_data.h1.T) / 2.0
    h1_b = meas_ctx.h1_b
    hg_a = jnp.einsum("pj,pj->", h1_a[:n_oa, :], green_a)
    hg_b = jnp.einsum("pj,pj->", h1_b[:n_ob, :], green_b)
    hg = hg_a + hg_b

    e0 = ham_data.h0

    # 1-body energy: ref
    e1_0 = hg

    # 1-body energy: singles
    ci1g_a = jnp.einsum("pt,pt->", c1a, green_occ_a, optimize="optimal")
    ci1g_b = jnp.einsum("pt,pt->", c1b, green_occ_b, optimize="optimal")
    ci1g = ci1g_a + ci1g_b
    e1_1_1 = ci1g * hg
    gpc1a = greenp_a @ c1a.T
    gpc1b = greenp_b @ c1b.T
    ci1_green_a = gpc1a @ green_a
    ci1_green_b = gpc1b @ green_b
    e1_1_2 = -(
        jnp.einsum("ij,ij->", h1_a, ci1_green_a, optimize="optimal")
        + jnp.einsum("ij,ij->", h1_b, ci1_green_b, optimize="optimal")
    )
    e1_1 = e1_1_1 + e1_1_2

    # 1-body energy: doubles
    ci2g_a = (
        jnp.einsum(
            "ptqu,pt->qu",
            c2aa.astype(cfg.mixed_real_dtype),
            green_occ_a.astype(cfg.mixed_complex_dtype),
        )
        / 4
    )
    ci2g_b = (
        jnp.einsum(
            "ptqu,pt->qu",
            c2bb.astype(cfg.mixed_real_dtype),
            green_occ_b.astype(cfg.mixed_complex_dtype),
        )
        / 4
    )
    ci2g_ab_a = jnp.einsum(
        "ptqu,qu->pt",
        c2ab.astype(cfg.mixed_real_dtype),
        green_occ_b.astype(cfg.mixed_complex_dtype),
    )
    ci2g_ab_b = jnp.einsum(
        "ptqu,pt->qu",
        c2ab.astype(cfg.mixed_real_dtype),
        green_occ_a.astype(cfg.mixed_complex_dtype),
    )
    gci2g_a = jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
    gci2g_b = jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
    gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
    e1_2_1 = hg * gci2g
    ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
    ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
    ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
    ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
    e1_2_2_a = -jnp.einsum("ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a, optimize="optimal")
    e1_2_2_b = -jnp.einsum("ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2_a + e1_2_2_b

    e1 = e1_0 + e1_1 + e1_2

    # 2-body energy: ref
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
    e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
    lg1_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
    lg1_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
    e2_0_2 = (
        -(jnp.sum(vmap(lambda x: x * x.T)(lg1_a)) + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))) / 2.0
    )
    e2_0 = e2_0_1 + e2_0_2

    # 2-body energy: singles
    e2_1_1 = e2_0 * ci1g
    lci1g_a = jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        ci1_green_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    lci1g_b = jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        ci1_green_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
    ci1g1_a = c1a @ green_occ_a.T
    ci1g1_b = c1b @ green_occ_b.T
    e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, optimize="optimal") + jnp.einsum(
        "gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, optimize="optimal"
    )
    lci1g_a = jnp.einsum("gip,qi->gpq", lci1_a, green_a, optimize="optimal")
    lci1g_b = jnp.einsum("gip,qi->gpq", lci1_b, green_b, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g_a, lg1_a, optimize="optimal") - jnp.einsum(
        "gpq,gqp->", lci1g_b, lg1_b, optimize="optimal"
    )
    e2_1 = e2_1_1 + e2_1_2 + e2_1_3_1 + e2_1_3_2

    # 2-body energy: doubles
    e2_2_1 = e2_0 * gci2g
    ci2_mix_a = 8 * ci2_green_a + 2 * ci2_green_ab_a
    ci2_mix_b = 8 * ci2_green_b + 2 * ci2_green_ab_b
    c2aa_test = c2aa.astype(cfg.mixed_real_dtype_testing)
    c2bb_test = c2bb.astype(cfg.mixed_real_dtype_testing)
    c2ab_test = c2ab.astype(cfg.mixed_real_dtype_testing)
    lci2g_a = jnp.einsum(
        "gij,ij->g",
        chol_a.astype(cfg.mixed_real_dtype),
        ci2_mix_a.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    lci2g_b = jnp.einsum(
        "gij,ij->g",
        chol_b.astype(cfg.mixed_real_dtype),
        ci2_mix_b.astype(cfg.mixed_complex_dtype),
        optimize="optimal",
    )
    e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

    if cfg.memory_mode == "low":

        def scan_over_chol(carry, x):
            chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
            gl_a_i = jnp.einsum("pj,ji->pi", green_a, chol_a_i, optimize="optimal")
            gl_b_i = jnp.einsum("pj,ji->pi", green_b, chol_b_i, optimize="optimal")
            lci2_green_a_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_a_i,
                ci2_mix_a,
                optimize="optimal",
            )
            lci2_green_b_i = jnp.einsum(
                "pi,ji->pj",
                rot_chol_b_i,
                ci2_mix_b,
                optimize="optimal",
            )
            carry[0] += 0.5 * (
                jnp.einsum("pi,pi->", gl_a_i, lci2_green_a_i, optimize="optimal")
                + jnp.einsum("pi,pi->", gl_b_i, lci2_green_b_i, optimize="optimal")
            )
            glgp_a_i = jnp.einsum("pi,it->pt", gl_a_i, greenp_a, optimize="optimal").astype(
                cfg.mixed_complex_dtype_testing
            )
            glgp_b_i = jnp.einsum("pi,it->pt", gl_b_i, greenp_b, optimize="optimal").astype(
                cfg.mixed_complex_dtype_testing
            )
            l2ci2_a = 0.5 * jnp.einsum(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_a_i,
                c2aa_test,
                optimize="optimal",
            )
            l2ci2_b = 0.5 * jnp.einsum(
                "pt,qu,ptqu->",
                glgp_b_i,
                glgp_b_i,
                c2bb_test,
                optimize="optimal",
            )
            l2c2ab = jnp.einsum(
                "pt,qu,ptqu->",
                glgp_a_i,
                glgp_b_i,
                c2ab_test,
                optimize="optimal",
            )
            carry[1] += l2ci2_a + l2ci2_b + l2c2ab
            return carry, 0.0

        [e2_2_2_2, e2_2_3], _ = lax.scan(
            scan_over_chol, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
        )
    else:
        gl_a = jnp.einsum(
            "pj,gji->gpi",
            green_a.astype(cfg.mixed_complex_dtype),
            chol_a.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        gl_b = jnp.einsum(
            "pj,gji->gpi",
            green_b.astype(cfg.mixed_complex_dtype),
            chol_b.astype(cfg.mixed_real_dtype),
            optimize="optimal",
        )
        lci2_green_a = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_a,
            ci2_mix_a,
            optimize="optimal",
        )
        lci2_green_b = jnp.einsum(
            "gpi,ji->gpj",
            rot_chol_b,
            ci2_mix_b,
            optimize="optimal",
        )
        e2_2_2_2 = 0.5 * (
            jnp.einsum("gpi,gpi->", gl_a, lci2_green_a, optimize="optimal")
            + jnp.einsum("gpi,gpi->", gl_b, lci2_green_b, optimize="optimal")
        )
        glgp_a = jnp.einsum("gpi,it->gpt", gl_a, greenp_a, optimize="optimal").astype(
            cfg.mixed_complex_dtype_testing
        )
        glgp_b = jnp.einsum("gpi,it->gpt", gl_b, greenp_b, optimize="optimal").astype(
            cfg.mixed_complex_dtype_testing
        )
        l2ci2_a = 0.5 * jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_a,
            glgp_a,
            c2aa_test,
            optimize="optimal",
        )
        l2ci2_b = 0.5 * jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_b,
            glgp_b,
            c2bb_test,
            optimize="optimal",
        )
        l2c2ab = jnp.einsum(
            "gpt,gqu,ptqu->g",
            glgp_a,
            glgp_b,
            c2ab_test,
            optimize="optimal",
        )
        e2_2_3 = l2ci2_a.sum() + l2ci2_b.sum() + l2c2ab.sum()

    e2_2_2 = e2_2_2_1 + e2_2_2_2
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    e2 = e2_0 + e2_1 + e2_2

    # triples contributions
    o3 = _triples_overlap(
        trial_data, green_occ_a, green_occ_b, low_memory=(cfg.memory_mode == "low")
    )
    overlap = 1.0 + ci1g + gci2g + o3

    e3_1 = _one_body_energy_triples(
        trial_data,
        green_a,
        green_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
        h1_a,
        h1_b,
        low_memory=(cfg.memory_mode == "low"),
    )
    e3_2 = _two_body_energy_triples(
        trial_data,
        green_a,
        green_b,
        green_occ_a,
        green_occ_b,
        greenp_a,
        greenp_b,
        chol_a,
        chol_b,
        low_memory=(cfg.memory_mode == "low"),
    )

    return (e1 + e2 + e3_1 + e3_2) / overlap + e0


# ---------------------------------------------------------------------------
# Context builder and factory
# ---------------------------------------------------------------------------


def build_meas_ctx(
    ham_data: HamChol, trial_data: UcisdtTrial, cfg: UcisdtMeasCfg = UcisdtMeasCfg()
) -> UcisdtMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("UCISDT MeasOps currently assumes HamChol.basis == 'restricted'.")
    n_oa, n_ob = trial_data.nocc
    cb = trial_data.mo_coeff_b
    cbH = trial_data.mo_coeff_b.conj().T
    h1_b = 0.5 * (cbH @ (ham_data.h1 + ham_data.h1.T) @ cb)
    chol_b = transform_cholesky(ham_data.chol, left=cbH, right=cb)
    rot_h1_a = ham_data.h1[:n_oa, :]
    rot_h1_b = ham_data.h1[:n_ob, :]
    rot_chol_a = ham_data.chol[:, :n_oa, :]
    rot_chol_b = chol_b[:, :n_ob, :]
    rot_chol_flat_a = rot_chol_a.reshape(rot_chol_a.shape[0], -1)
    rot_chol_flat_b = rot_chol_b.reshape(rot_chol_b.shape[0], -1)
    lci1_a = transform_cholesky(
        ham_data.chol, right=trial_data.c1a.T, column_slice=(n_oa, ham_data.chol.shape[2])
    )
    lci1_b = transform_cholesky(
        chol_b, right=trial_data.c1b.T, column_slice=(n_ob, chol_b.shape[2])
    )
    return UcisdtMeasCtx(
        h1_b=h1_b,
        chol_b=chol_b,
        rot_h1_a=rot_h1_a,
        rot_h1_b=rot_h1_b,
        rot_chol_a=rot_chol_a,
        rot_chol_b=rot_chol_b,
        rot_chol_flat_a=rot_chol_flat_a,
        rot_chol_flat_b=rot_chol_flat_b,
        lci1_a=lci1_a,
        lci1_b=lci1_b,
        cfg=cfg,
    )


def make_ucisdt_meas_ops(
    sys: System,
    memory_mode: str = "high",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    wk = sys.walker_kind.lower()

    cfg = UcisdtMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    if wk == "restricted":
        kernels = {
            k_force_bias: force_bias_kernel_rw_rh,
            k_energy: energy_kernel_rw_rh,
        }
        overlap_fn = overlap_r
    elif wk == "unrestricted":
        kernels = {
            k_force_bias: force_bias_kernel_uw_rh,
            k_energy: energy_kernel_uw_rh,
        }
        overlap_fn = overlap_u
    elif wk == "generalized":
        raise NotImplementedError("UCISDT does not support generalized walkers.")
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return MeasOps(
        overlap=overlap_fn,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels=kernels,
    )
