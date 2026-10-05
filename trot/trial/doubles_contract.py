"""Doubles applications without promoting a real four-index tensor to complex."""
import math

import jax.numpy as jnp
from jax import lax

# Used only when low-memory measurement must cast a large doubles tensor.
# Each slice has at most this many occupied-virtual pairs on the summed axis.
_DOUBLES_PAIR_BATCH_SIZE = 1024


def apply_doubles(tensor, vectors, *, transpose=False, dtype=None, low_memory=False):
    """Contract a pair of tensor indices with the last two vector dimensions.

    The default computes ``...pt,ptqu->...qu``; transpose=True computes
    ``ptqu,...qu->...pt``. There is no conjugation. Real/imaginary products
    preserve the input arithmetic precision and avoid a full complex tensor
    copy. When low_memory requires a dtype conversion, cast only a slice of
    the summed pair dimension at a time. Complex input tensors remain supported.
    """
    left, right = tensor.shape[:2], tensor.shape[2:]
    matrix = tensor.reshape(left[0] * left[1], right[0] * right[1])
    output_shape = left if transpose else right
    axis = 1 if transpose else 0
    contracted_size = matrix.shape[axis]
    output_size = matrix.shape[1 - axis]
    rows = vectors.reshape((math.prod(vectors.shape[:-2]), contracted_size))
    dtype = matrix.dtype if dtype is None else jnp.dtype(dtype)
    result_dtype = jnp.result_type(dtype, rows)

    def contract(rows, block):
        block = block.astype(dtype)
        if transpose:
            block = block.T
        if (jnp.issubdtype(block.dtype, jnp.floating)
                and jnp.issubdtype(rows.dtype, jnp.complexfloating)):
            real = (rows.real @ block).astype(result_dtype)
            imag = (rows.imag @ block).astype(result_dtype)
            return real + jnp.asarray(1j, dtype=result_dtype) * imag
        return rows @ block

    batch_size = _DOUBLES_PAIR_BATCH_SIZE
    if not low_memory or dtype == matrix.dtype or contracted_size <= batch_size:
        result = contract(rows, matrix)
    else:
        # Slice before casting: casting the original tensor at the call site
        # leaves a whole-tensor FP32 allocation live across the Cholesky loop.
        nfull, remainder = divmod(contracted_size, batch_size)
        def accumulate(index, total):
            start = index * batch_size
            block = lax.dynamic_slice_in_dim(matrix, start, batch_size, axis=axis)
            row_block = lax.dynamic_slice_in_dim(rows, start, batch_size, axis=1)
            return total + contract(row_block, block)
        result = lax.fori_loop(0, nfull, accumulate,
                              jnp.zeros((rows.shape[0], output_size), dtype=result_dtype))
        if remainder:
            start = nfull * batch_size
            block = lax.slice_in_dim(matrix, start, contracted_size, axis=axis)
            result = result + contract(rows[:, start:], block)
    return result.reshape(vectors.shape[:-2] + output_shape)
