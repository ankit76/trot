"""Bounded-workspace Cholesky contractions used only during measurement setup."""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

from ..sharding import cholesky_model_mesh

SETUP_CHOL_BATCH_SIZE = 64


def transform_cholesky(
    chol: jax.Array,
    *,
    left: jax.Array | None = None,
    right: jax.Array | None = None,
    column_slice: tuple[int, int] | None = None,
) -> jax.Array:
    """Apply left/right orbital matrices without a full-Cholesky workspace.

    A column slice, when supplied, is taken inside each batch before applying
    the matrices. Input precision is preserved, including complex rotations.
    This setup policy is independent of production energy batching/precision.
    """
    return _transform_cholesky(
        chol, left, right, column_slice=column_slice, mesh=cholesky_model_mesh(chol)
    )


@partial(
    jax.jit,
    static_argnames=("column_slice", "batch_size", "mesh"),
    # As in restricted setup, do not profile copies of the full input tensor.
    # Production kernels keep their normal autotuning and precision policy.
    compiler_options={"xla_gpu_autotune_level": 0},
)
def _transform_cholesky(
    chol: jax.Array,
    left: jax.Array | None = None,
    right: jax.Array | None = None,
    *,
    column_slice: tuple[int, int] | None = None,
    batch_size: int = SETUP_CHOL_BATCH_SIZE,
    mesh: Mesh | None = None,
) -> jax.Array:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    local = partial(
        _transform_local, column_slice=column_slice, batch_size=batch_size,
        model_sharded=mesh is not None,
    )
    if mesh is not None:
        # Batch the local shard; global-index slices would gather the input.
        return jax.shard_map(
            local, mesh=mesh, in_specs=(P("model"), P(), P()), out_specs=P("model")
        )(chol, left, right)
    return local(chol, left, right)


def _transform_local(chol, left, right, *, column_slice, batch_size, model_sharded=False):
    def contract(block):
        if column_slice is not None:
            block = block[:, :, column_slice[0]:column_slice[1]]
        if left is not None:
            block = jnp.einsum("pi,gij->gpj", left, block, optimize="optimal")
        if right is not None:
            block = jnp.einsum("gpi,iq->gpq", block, right, optimize="optimal")
        return block

    n_chol = chol.shape[0]
    if n_chol <= batch_size:
        return contract(chol)
    n_rows = chol.shape[1] if left is None else left.shape[0]
    n_columns = chol.shape[2] if column_slice is None else column_slice[1]-column_slice[0]
    if right is not None:
        n_columns = right.shape[1]
    dtype = jnp.result_type(chol, *(x for x in (left, right) if x is not None))
    output = jnp.zeros((n_chol, n_rows, n_columns), dtype=dtype)
    if model_sharded:
        output = lax.pcast(output, ("model",), to="varying")
    n_batches, remainder = divmod(n_chol, batch_size)

    def body(index, output):
        block = lax.dynamic_slice_in_dim(chol, index * batch_size, batch_size, axis=0)
        return lax.dynamic_update_slice_in_dim(output, contract(block), index * batch_size, axis=0)

    # Carry only the final output, updated in place by XLA. Mapping batches and
    # concatenating a remainder could require a second full-sized output.
    output = lax.fori_loop(0, n_batches, body, output)
    if remainder:
        start = n_batches * batch_size
        output = lax.dynamic_update_slice_in_dim(output, contract(chol[start:]), start, axis=0)
    return output
