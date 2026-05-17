from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ham.chol import HamChol
from .chol_afqmc_ops import _build_exp_h1_half_from_h1, _get_h1_eff, _mf_shifts
from .chol_afqmc_ops import TrotterOps, CholAfqmcCtx
from .chol_afqmc_ops import _apply_trotter_r, _apply_trotter_u, _apply_trotter_g_from_restricted


def _build_prop_ctx_fp(
    ham_data: HamChol,
    rdm1: jax.Array,
    dt: float,
    ene0: float = 0.0,
    chol_flat_precision: jnp.dtype = jnp.float64,
) -> CholAfqmcCtx:
    dt_a = jnp.array(dt)
    sqrt_dt = jnp.sqrt(dt_a)
    mf = _mf_shifts(ham_data, rdm1)
    h0_prop = -ham_data.h0 - 0.5 * jnp.sum(mf**2) + ene0

    h1_eff = _get_h1_eff(ham_data, mf)
    exp_h1_half = _build_exp_h1_half_from_h1(h1_eff, dt_a)
    chol_flat = ham_data.chol.reshape(ham_data.chol.shape[0], -1).astype(chol_flat_precision)
    norb = ham_data.chol.shape[1]
    return CholAfqmcCtx(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half=exp_h1_half,
        mf_shifts=mf,
        h0_prop=h0_prop,
        chol_flat=chol_flat,
        norb=norb,
    )


def _make_vhs_split_flat_fp(*, chol_flat: jax.Array, x: jax.Array, n: int) -> jax.Array:
    # chol_flat: (n_fields, n*n) real
    v = x @ chol_flat  # (n*n,)
    return v.reshape(n, n)


def make_trotter_ops_fp(
    ham_basis: str, walker_kind: str, mixed_precision: bool = False
) -> TrotterOps:
    assert isinstance(ham_basis, str)
    assert isinstance(walker_kind, str)
    assert isinstance(mixed_precision, bool)

    walker_kind = walker_kind.lower()

    if mixed_precision:
        vhs_real_dtype = jnp.float32
    else:
        vhs_real_dtype = jnp.float64

    def make_vhs(field: jax.Array, ctx: CholAfqmcCtx) -> jax.Array:
        return _make_vhs_split_flat_fp(
            chol_flat=ctx.chol_flat,
            x=field.astype(vhs_real_dtype),
            n=ctx.norb,
        )

    if walker_kind not in ("restricted", "unrestricted", "generalized"):
        raise ValueError(f"unknown walker_kind: {walker_kind}")

    if ham_basis not in ("restricted", "generalized"):
        raise ValueError(f"unknown ham_basis: {ham_basis}")

    match ham_basis, walker_kind:
        case "restricted", "restricted":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_r(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case "restricted", "unrestricted":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_u(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case "restricted", "generalized":
            apply_trotter = (
                lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_g_from_restricted(
                    w, f, ctx, n_terms, make_vhs=mv
                )
            )
        case "generalized", "generalized":
            apply_trotter = lambda w, f, ctx, n_terms, mv=make_vhs: _apply_trotter_r(
                w, f, ctx, n_terms, make_vhs=mv
            )
        case _:
            raise NotImplementedError(
                f"Not implemented for ham_basis={ham_basis} and walker_kind={walker_kind}"
            )

    return TrotterOps(apply_trotter)
