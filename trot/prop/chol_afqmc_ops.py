from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util
from jax.sharding import Mesh, PartitionSpec as P

from ..ham.chol import HamChol
from ..sharding import cholesky_model_mesh
from .utils import taylor_expm_action

# contains low level details of AFQMC chol propagation

_CHOLESKY_SQUARE_BATCH_SIZE = 256


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CholAfqmcCtx:
    dt: jax.Array
    sqrt_dt: jax.Array
    exp_h1_half: jax.Array  # (n,n) or (ns,ns)
    mf_shifts: jax.Array  # (n_fields,)
    h0_prop: jax.Array  # scalar
    # Full layout: (n_fields, n*n) or the shared Hamiltonian (n_fields, n, n).
    # Packed layout: (n_fields, n*(n+1)//2).
    chol_flat: jax.Array
    norb: int
    chol_packed: bool = False

    def tree_flatten(self):
        return (
            self.dt,
            self.sqrt_dt,
            self.exp_h1_half,
            self.mf_shifts,
            self.h0_prop,
            self.chol_flat,
        ), (self.norb, self.chol_packed)

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, sqrt_dt, exp_h1_half, mf_shifts, h0_prop, chol_flat = children
        norb, chol_packed = aux

        return cls(
            dt=dt,
            sqrt_dt=sqrt_dt,
            exp_h1_half=exp_h1_half,
            mf_shifts=mf_shifts,
            h0_prop=h0_prop,
            chol_flat=chol_flat,
            norb=norb,
            chol_packed=chol_packed,
        )


class TrotterOps(NamedTuple):
    apply_trotter: Callable[[Any, jax.Array, CholAfqmcCtx, int], Any]  # (w, field, ctx, n_terms)->w


def _as_total_rdm1_restricted(dm: jax.Array) -> jax.Array:
    if dm.ndim == 3 and dm.shape[0] == 2:
        return dm[0] + dm[1]
    return dm


def _get_dm(rdm1: jax.Array, ham_basis: str) -> jax.Array:
    match ham_basis:
        case "restricted":
            dm = _as_total_rdm1_restricted(rdm1)
        case "generalized":
            dm = rdm1
        case _:
            raise ValueError(f"Unknown Hamiltonian basis kind: {ham_basis}")
    return dm


@partial(
    jax.jit,
    # Setup profiling can copy the entire Cholesky input. Disable it for
    # this compilation without changing production-kernel autotuning.
    compiler_options={"xla_gpu_autotune_level": 0},
)
def _mf_shifts(ham_data: HamChol, rdm1: jax.Array) -> jax.Array:
    dm = _get_dm(rdm1, ham_data.basis)
    return 1.0j * jnp.einsum("gij,ji->g", ham_data.chol, dm, optimize="optimal")


@partial(jax.jit, compiler_options={"xla_gpu_autotune_level": 0})
def _weighted_chol_sum(weights: jax.Array, chol: jax.Array) -> jax.Array:
    """One-time mean-field potential without autotuner copies of full chol."""
    return jnp.einsum("g,gik->ik", weights, chol, optimize="optimal")


def _build_exp_h1_half_from_h1(h1: jax.Array, dt: jax.Array) -> jax.Array:
    return jax.scipy.linalg.expm(-0.5 * dt * h1)


def _packed_upper_size(n: int) -> int:
    return n * (n + 1) // 2


def _pack_symmetric_chol(chol: jax.Array) -> jax.Array:
    """Pack the upper triangle of each symmetric Cholesky matrix."""
    n = int(chol.shape[1])
    rows, cols = jnp.triu_indices(n)
    return chol[:, rows, cols]


def _unpack_symmetric_upper(packed: jax.Array, n: int) -> jax.Array:
    """Expand a row-major packed upper triangle into a symmetric matrix."""
    rows, cols = jnp.indices((n, n))
    upper_rows = jnp.minimum(rows, cols)
    upper_cols = jnp.maximum(rows, cols)
    packed_indices = upper_rows * n - upper_rows * (upper_rows + 1) // 2 + upper_cols
    return packed[packed_indices]


@partial(jax.jit, static_argnames=("dtype", "packed_cholesky"))
def _prepare_chol_for_vhs(
    chol: jax.Array,
    *,
    dtype: jnp.dtype,
    packed_cholesky: bool,
) -> jax.Array:
    # Fuse layout preparation with the cast; an eager FP64 reshape would
    # allocate another full Cholesky tensor before making the FP32 output.
    if packed_cholesky:
        chol_vhs = _pack_symmetric_chol(chol)
    else:
        chol_vhs = chol.reshape(chol.shape[0], -1)
    return chol_vhs.astype(dtype)


def _make_vhs_split_flat(
    *,
    chol_flat: jax.Array,
    x: jax.Array,
    n: int,
    chol_packed: bool = False,
) -> jax.Array:
    # Flatten a shared full tensor inside JIT so no second persistent device
    # array is needed. The packed and existing 2D representations are unchanged.
    if chol_flat.ndim == 3:
        chol_flat = chol_flat.reshape(chol_flat.shape[0], -1)
    v_re = jnp.real(x) @ chol_flat
    v_im = jnp.imag(x) @ chol_flat
    vhs = lax.complex(v_re, v_im)
    if chol_packed:
        return _unpack_symmetric_upper(vhs, n)
    return vhs.reshape(n, n)


@partial(
    jax.jit,
    static_argnames=("batch_size", "mesh"),
    # A slice/transpose fusion still takes the full chol tensor as input.
    # Profiling it duplicates that input even when its output is one batch.
    # Scope this option to the top-level, one-time setup compilation only.
    compiler_options={"xla_gpu_autotune_level": 0},
)
def _sum_chol_squares(
    chol: jax.Array,
    *,
    batch_size: int = _CHOLESKY_SQUARE_BATCH_SIZE,
    mesh: Mesh | None = None,
) -> jax.Array:
    """Sum L_g @ L_g with batch-sized transpose/GEMM intermediates.

    This one-time propagator setup contraction keeps the full input, but
    avoids transposing all Cholesky vectors into a second full-sized tensor.
    Slice inside the loop so an uneven last batch does not copy the prefix
    of the full input. The contraction is bilinear, including for complex L.
    With a model mesh, slice within each shard and reduce only the final matrix.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if chol.shape[0] == 0:
        return jnp.zeros(chol.shape[1:], dtype=chol.dtype)
    if mesh is not None:
        def local_sum(block):
            return lax.psum(_sum_chol_squares_local(block, batch_size), "model")

        return jax.shard_map(
            local_sum, mesh=mesh, in_specs=P("model"), out_specs=P()
        )(chol)
    return _sum_chol_squares_local(chol, batch_size)


def _sum_chol_squares_local(chol: jax.Array, batch_size: int) -> jax.Array:
    n_chol = chol.shape[0]
    # Preserve manual-axis variance when this loop runs inside shard_map.
    total = jnp.zeros_like(chol, shape=chol.shape[1:])
    if n_chol == 0:
        return total
    if n_chol <= batch_size:
        return jnp.einsum("gik,gkj->ij", chol, chol, optimize="optimal")

    n_batches, remainder = divmod(n_chol, batch_size)

    def accumulate(index, value):
        block = lax.dynamic_slice_in_dim(chol, index * batch_size, batch_size, axis=0)
        return value + jnp.einsum("gik,gkj->ij", block, block, optimize="optimal")

    total = lax.fori_loop(0, n_batches, accumulate, total)
    if remainder:
        tail = chol[n_batches * batch_size :]
        total = total + jnp.einsum("gik,gkj->ij", tail, tail, optimize="optimal")
    return total


def _get_h1_eff(ham_data: HamChol, mf: jax.Array) -> jax.Array:
    match ham_data.basis:
        case "restricted" | "generalized":
            v0m = 0.5 * _sum_chol_squares(
                ham_data.chol, mesh=cholesky_model_mesh(ham_data.chol)
            )
            mf_r = (1.0j * mf).real
            v1m = _weighted_chol_sum(mf_r, ham_data.chol)
            h1_eff = ham_data.h1 - v0m - v1m
        case _:
            raise ValueError(f"Unknown Hamiltonian basis kind: {ham_data.basis}")

    return h1_eff


def _build_prop_ctx(
    ham_data: HamChol,
    rdm1: jax.Array,
    dt: float,
    chol_flat_precision: jnp.dtype = jnp.float64,
    packed_cholesky: bool = False,
) -> CholAfqmcCtx:
    dt_a = jnp.array(dt)
    sqrt_dt = jnp.sqrt(dt_a)

    mf = _mf_shifts(ham_data, rdm1)
    h0_prop = -ham_data.h0 - 0.5 * jnp.sum(mf**2)
    h1_eff = _get_h1_eff(ham_data, mf)

    exp_h1_half = _build_exp_h1_half_from_h1(h1_eff, dt_a)
    norb = ham_data.chol.shape[1]
    chol_flat = _prepare_chol_for_vhs(
        ham_data.chol,
        dtype=chol_flat_precision,
        packed_cholesky=packed_cholesky,
    )
    return CholAfqmcCtx(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half=exp_h1_half,
        mf_shifts=mf,
        h0_prop=h0_prop,
        chol_flat=chol_flat,
        norb=norb,
        chol_packed=packed_cholesky,
    )


def _apply_one_body_half_array(w: jax.Array, prop_ctx: CholAfqmcCtx) -> jax.Array:
    return prop_ctx.exp_h1_half @ w


def _apply_one_body_half_unrestricted(
    w_ud: Tuple[jax.Array, jax.Array], prop_ctx: CholAfqmcCtx
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud
    e = prop_ctx.exp_h1_half
    return (e @ wu, e @ wd)


def _apply_one_body_half_generalized_from_restricted(
    w: jax.Array, prop_ctx: CholAfqmcCtx
) -> jax.Array:
    e = prop_ctx.exp_h1_half
    norb = w.shape[0] // 2
    top = e @ w[:norb, :]
    bot = e @ w[norb:, :]
    return jnp.vstack([top, bot])


def _apply_two_body_array(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> jax.Array:
    vhs = make_vhs(field, prop_ctx).astype(w.dtype)
    a = (1.0j * prop_ctx.sqrt_dt).astype(w.dtype)
    return taylor_expm_action(a, vhs, w, n_terms)


def _apply_two_body_unrestricted(
    w_ud: Tuple[jax.Array, jax.Array],
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> Tuple[jax.Array, jax.Array]:
    wu, wd = w_ud
    vhs = make_vhs(field, prop_ctx).astype(wu.dtype)
    a = (1.0j * prop_ctx.sqrt_dt).astype(wu.dtype)
    return (
        taylor_expm_action(a, vhs, wu, n_terms),
        taylor_expm_action(a, vhs, wd, n_terms),
    )


def _apply_two_body_generalized_from_restricted(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> jax.Array:
    vhs = make_vhs(field, prop_ctx).astype(w.dtype)
    a = (1.0j * prop_ctx.sqrt_dt).astype(w.dtype)
    norb = w.shape[0] // 2
    top = taylor_expm_action(a, vhs, w[:norb, :], n_terms)
    bot = taylor_expm_action(a, vhs, w[norb:, :], n_terms)
    return jnp.vstack([top, bot])


def _apply_trotter_r(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> jax.Array:
    w1 = _apply_one_body_half_array(w, prop_ctx)
    w2 = _apply_two_body_array(w1, field, prop_ctx, n_terms, make_vhs=make_vhs)
    return _apply_one_body_half_array(w2, prop_ctx)


def _apply_trotter_u(
    w_ud: Tuple[jax.Array, jax.Array],
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> Tuple[jax.Array, jax.Array]:
    w1 = _apply_one_body_half_unrestricted(w_ud, prop_ctx)
    w2 = _apply_two_body_unrestricted(w1, field, prop_ctx, n_terms, make_vhs=make_vhs)
    a = _apply_one_body_half_unrestricted(w2, prop_ctx)
    return a


def _apply_trotter_g_from_restricted(
    w: jax.Array,
    field: jax.Array,
    prop_ctx: CholAfqmcCtx,
    n_terms: int,
    *,
    make_vhs: Callable[[jax.Array, CholAfqmcCtx], jax.Array],
) -> jax.Array:
    w1 = _apply_one_body_half_generalized_from_restricted(w, prop_ctx)
    w2 = _apply_two_body_generalized_from_restricted(
        w1, field, prop_ctx, n_terms, make_vhs=make_vhs
    )
    return _apply_one_body_half_generalized_from_restricted(w2, prop_ctx)


def make_trotter_ops(ham_basis: str, walker_kind: str, mixed_precision: bool = False) -> TrotterOps:
    assert isinstance(ham_basis, str)
    assert isinstance(walker_kind, str)
    assert isinstance(mixed_precision, bool)

    walker_kind = walker_kind.lower()

    if mixed_precision:
        vhs_complex_dtype = jnp.complex64
    else:
        vhs_complex_dtype = jnp.complex128

    def make_vhs(field: jax.Array, ctx: CholAfqmcCtx) -> jax.Array:
        return _make_vhs_split_flat(
            chol_flat=ctx.chol_flat,
            x=field.astype(vhs_complex_dtype),
            n=ctx.norb,
            chol_packed=ctx.chol_packed,
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
