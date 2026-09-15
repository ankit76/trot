from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util

from ..ham.chol_u import HamCholU
from ..walkers import split_rdm1_u
from .chol_afqmc_ops import _make_vhs_split_flat
from .utils import taylor_expm_action

# low level details of AFQMC chol propagation with unrestricted cholesky vectors.
#
# The companion of chol_afqmc_ops.py for HamCholU. The difference is that every orbital
# indexed quantity is per spin, while everything indexed by the auxiliary field g stays
# shared:
#
#   propagated per spin : exp_h1_half, chol_flat, vhs, walkers
#   shared across spins : the sampled fields, mf_shifts, h0_prop
#
# so norb_a and norb_b are free to differ. See trot/ham/chol_u.py.


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CholAfqmcCtxU:
    dt: jax.Array
    sqrt_dt: jax.Array
    exp_h1_half_a: jax.Array  # (norb_a, norb_a)
    exp_h1_half_b: jax.Array  # (norb_b, norb_b)
    mf_shifts: jax.Array  # (n_fields,)  shared
    h0_prop: jax.Array  # scalar
    chol_flat_a: jax.Array  # (n_fields, norb_a*norb_a)
    chol_flat_b: jax.Array  # (n_fields, norb_b*norb_b)
    norb_a: int
    norb_b: int

    @property
    def n_fields(self) -> int:
        return int(self.chol_flat_a.shape[0])

    def tree_flatten(self):
        return (
            self.dt,
            self.sqrt_dt,
            self.exp_h1_half_a,
            self.exp_h1_half_b,
            self.mf_shifts,
            self.h0_prop,
            self.chol_flat_a,
            self.chol_flat_b,
        ), (self.norb_a, self.norb_b)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            dt,
            sqrt_dt,
            exp_h1_half_a,
            exp_h1_half_b,
            mf_shifts,
            h0_prop,
            chol_flat_a,
            chol_flat_b,
        ) = children
        norb_a, norb_b = aux

        return cls(
            dt=dt,
            sqrt_dt=sqrt_dt,
            exp_h1_half_a=exp_h1_half_a,
            exp_h1_half_b=exp_h1_half_b,
            mf_shifts=mf_shifts,
            h0_prop=h0_prop,
            chol_flat_a=chol_flat_a,
            chol_flat_b=chol_flat_b,
            norb_a=norb_a,
            norb_b=norb_b,
        )


class TrotterOpsU(NamedTuple):
    apply_trotter: Callable[
        [Tuple[jax.Array, jax.Array], jax.Array, CholAfqmcCtxU, int],
        Tuple[jax.Array, jax.Array],
    ]  # (w_ud, field, ctx, n_terms) -> w_ud


def _mf_shifts(ham_data: HamCholU, rdm1: Any) -> jax.Array:
    """
    mf_g = i Tr[L^a_g rho_a] + i Tr[L^b_g rho_b]

    One length n_fields vector, summed over spin: the field is shared, so the mean field
    subtracted from v_g is the total expectation value.
    """
    dm_a, dm_b = split_rdm1_u(rdm1)
    mf_a = 1.0j * jnp.einsum("gij,ji->g", ham_data.chol_a, dm_a, optimize="optimal")
    mf_b = 1.0j * jnp.einsum("gij,ji->g", ham_data.chol_b, dm_b, optimize="optimal")
    return mf_a + mf_b


def _get_h1_eff_one_spin(h1: jax.Array, chol: jax.Array, mf: jax.Array) -> jax.Array:
    """
    h1_eff^sigma = h1^sigma - v0^sigma - v1^sigma

    v0 is spin diagonal (it contracts L^sigma with L^sigma), while v1 pairs the *total*
    mf_shifts with this spin's own cholesky vectors.
    """
    v0m = 0.5 * jnp.einsum("gik,gkj->ij", chol, chol, optimize="optimal")
    mf_r = (1.0j * mf).real
    v1m = jnp.einsum("g,gik->ik", mf_r, chol, optimize="optimal")
    return h1 - v0m - v1m


def _build_exp_h1_half_from_h1(h1: jax.Array, dt: jax.Array) -> jax.Array:
    return jax.scipy.linalg.expm(-0.5 * dt * h1)


def _build_prop_ctx_u(
    ham_data: HamCholU,
    rdm1: Any,
    dt: float,
    chol_flat_precision: jnp.dtype = jnp.float64,
) -> CholAfqmcCtxU:
    if ham_data.basis != "uchol":
        raise ValueError(f"Unknown Hamiltonian basis kind: {ham_data.basis}")

    dt_a = jnp.array(dt)
    sqrt_dt = jnp.sqrt(dt_a)

    mf = _mf_shifts(ham_data, rdm1)
    h0_prop = -ham_data.h0 - 0.5 * jnp.sum(mf**2)

    h1_eff_a = _get_h1_eff_one_spin(ham_data.h1_a, ham_data.chol_a, mf)
    h1_eff_b = _get_h1_eff_one_spin(ham_data.h1_b, ham_data.chol_b, mf)

    exp_h1_half_a = _build_exp_h1_half_from_h1(h1_eff_a, dt_a)
    exp_h1_half_b = _build_exp_h1_half_from_h1(h1_eff_b, dt_a)

    chol_a, chol_b = ham_data.chol_a, ham_data.chol_b
    chol_flat_a = chol_a.reshape(chol_a.shape[0], -1).astype(chol_flat_precision)
    chol_flat_b = chol_b.reshape(chol_b.shape[0], -1).astype(chol_flat_precision)

    return CholAfqmcCtxU(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half_a=exp_h1_half_a,
        exp_h1_half_b=exp_h1_half_b,
        mf_shifts=mf,
        h0_prop=h0_prop,
        chol_flat_a=chol_flat_a,
        chol_flat_b=chol_flat_b,
        norb_a=int(chol_a.shape[1]),
        norb_b=int(chol_b.shape[1]),
    )


def _apply_one_body_half_u(
    w_ud: Tuple[jax.Array, jax.Array], prop_ctx: CholAfqmcCtxU
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud
    return (prop_ctx.exp_h1_half_a @ wu, prop_ctx.exp_h1_half_b @ wd)


def _apply_two_body_u(
    w_ud: Tuple[jax.Array, jax.Array],
    field: jax.Array,
    prop_ctx: CholAfqmcCtxU,
    n_terms: int,
    *,
    make_vhs_a: Callable[[jax.Array, CholAfqmcCtxU], jax.Array],
    make_vhs_b: Callable[[jax.Array, CholAfqmcCtxU], jax.Array],
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud

    # one sampled field vector drives both spins, through different cholesky vectors
    vhs_a = make_vhs_a(field, prop_ctx).astype(wu.dtype)
    vhs_b = make_vhs_b(field, prop_ctx).astype(wd.dtype)

    a_u = (1.0j * prop_ctx.sqrt_dt).astype(wu.dtype)
    a_d = (1.0j * prop_ctx.sqrt_dt).astype(wd.dtype)

    return (
        taylor_expm_action(a_u, vhs_a, wu, n_terms),
        taylor_expm_action(a_d, vhs_b, wd, n_terms),
    )


def _apply_trotter_u(
    w_ud: Tuple[jax.Array, jax.Array],
    field: jax.Array,
    prop_ctx: CholAfqmcCtxU,
    n_terms: int,
    *,
    make_vhs_a: Callable[[jax.Array, CholAfqmcCtxU], jax.Array],
    make_vhs_b: Callable[[jax.Array, CholAfqmcCtxU], jax.Array],
) -> Tuple[jax.Array, jax.Array]:
    w1 = _apply_one_body_half_u(w_ud, prop_ctx)
    w2 = _apply_two_body_u(
        w1, field, prop_ctx, n_terms, make_vhs_a=make_vhs_a, make_vhs_b=make_vhs_b
    )
    return _apply_one_body_half_u(w2, prop_ctx)


def make_trotter_ops_u(
    ham_basis: str, walker_kind: str, mixed_precision: bool = False
) -> TrotterOpsU:
    assert isinstance(ham_basis, str)
    assert isinstance(walker_kind, str)
    assert isinstance(mixed_precision, bool)

    walker_kind = walker_kind.lower()

    if ham_basis != "uchol":
        raise ValueError(f"unknown ham_basis: {ham_basis}")

    if walker_kind not in ("restricted", "unrestricted", "generalized"):
        raise ValueError(f"unknown walker_kind: {walker_kind}")

    if walker_kind != "unrestricted":
        # alpha and beta live in different orbital spaces, so a shared or spin blocked
        # walker cannot represent them
        raise NotImplementedError(
            f"Not implemented for ham_basis={ham_basis} and walker_kind={walker_kind}"
        )

    if mixed_precision:
        vhs_complex_dtype = jnp.complex64
    else:
        vhs_complex_dtype = jnp.complex128

    def make_vhs_a(field: jax.Array, ctx: CholAfqmcCtxU) -> jax.Array:
        return _make_vhs_split_flat(
            chol_flat=ctx.chol_flat_a,
            x=field.astype(vhs_complex_dtype),
            n=ctx.norb_a,
        )

    def make_vhs_b(field: jax.Array, ctx: CholAfqmcCtxU) -> jax.Array:
        return _make_vhs_split_flat(
            chol_flat=ctx.chol_flat_b,
            x=field.astype(vhs_complex_dtype),
            n=ctx.norb_b,
        )

    apply_trotter = lambda w, f, ctx, n_terms, mva=make_vhs_a, mvb=make_vhs_b: _apply_trotter_u(
        w, f, ctx, n_terms, make_vhs_a=mva, make_vhs_b=mvb
    )

    return TrotterOpsU(apply_trotter)
