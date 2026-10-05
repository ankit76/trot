"""Bounded-workspace real-Cholesky contractions used during propagation."""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P


def contract_cholesky(chol, matrix, cfg):
    """Compute L[g,i,j] M[i,j], casting only a batch of Cholesky vectors.

    Real/imaginary contractions retain the configured arithmetic precision.
    Batching applies independently of the doubles-energy memory mode. The
    concrete model mesh is recorded in the measurement config during setup.
    """
    if chol.shape[0] <= cfg.chol_batch_size:
        return _contract_local(chol, matrix, cfg=cfg)
    contract = partial(_contract_local, cfg=cfg, model_sharded=cfg.chol_mesh is not None)
    if cfg.chol_mesh is not None:
        return jax.shard_map(
            contract, mesh=cfg.chol_mesh, in_specs=(P("model"), P()), out_specs=P("model")
        )(chol, matrix)
    return contract(chol, matrix)


def _contract_local(chol, matrix, *, cfg, model_sharded=False):
    matrix_r = jnp.real(matrix).astype(cfg.mixed_real_dtype)
    matrix_i = jnp.imag(matrix).astype(cfg.mixed_real_dtype)
    imag_unit = jnp.asarray(1j, dtype=cfg.mixed_complex_dtype)

    def contract(block):
        block_r = block.astype(cfg.mixed_real_dtype)
        real = jnp.einsum("gij,ij->g", block_r, matrix_r, optimize="optimal")
        imag = jnp.einsum("gij,ij->g", block_r, matrix_i, optimize="optimal")
        return real.astype(cfg.mixed_complex_dtype) + imag_unit * imag.astype(cfg.mixed_complex_dtype)

    nchol, batch_size = chol.shape[0], cfg.chol_batch_size
    if nchol <= batch_size:
        return contract(chol)
    output = jnp.zeros((nchol,), dtype=cfg.mixed_complex_dtype)
    if model_sharded:
        output = lax.pcast(output, ("model",), to="varying")
    nfull, remainder = divmod(nchol, batch_size)

    def body(index, output):
        block = lax.dynamic_slice_in_dim(chol, index * batch_size, batch_size, axis=0)
        return lax.dynamic_update_slice_in_dim(output, contract(block), index * batch_size, axis=0)

    output = lax.fori_loop(0, nfull, body, output)
    if remainder:
        start = nfull * batch_size
        output = lax.dynamic_update_slice_in_dim(output, contract(chol[start:]), start, axis=0)
    return output
