"""Restricted PT half-rotation setup: batch boundaries, precision, and sharding."""
from types import SimpleNamespace

from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from trot.ham.chol import HamChol
from trot.meas.chol_setup import _transform_cholesky
from trot.meas.ptccsd_modes import build_ptccsd_thouless_mode_meas_ctx
from trot.sharding import replicate, shard_model_axis


@pytest.mark.parametrize("nchol", [0, 5, 256, 512, 518])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_pt_half_rotation_matches_dense_contraction(nchol, dtype):
    rng = np.random.default_rng(20261009)
    chol = rng.normal(size=(nchol, 7, 7)).astype(dtype)
    mo_t = rng.normal(size=(7, 3)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        chol += 1j * rng.normal(size=chol.shape)
        mo_t += 1j * rng.normal(size=mo_t.shape)
    # Nonsymmetric factors and complex orbitals expose transpose/conjugation errors.
    ham = HamChol(jnp.array(0.), jnp.eye(7), jnp.asarray(chol))
    trial = SimpleNamespace(mo_t=jnp.asarray(mo_t), mode_rank=9)
    ctx = build_ptccsd_thouless_mode_meas_ctx(ham, trial, n_mode_chunks=3)
    expected = np.stack([mo_t.conj().T @ x for x in chol]) if nchol else np.empty((0, 3, 7), dtype=dtype)
    assert ctx.rot_chol.dtype == expected.dtype
    assert ctx.n_mode_chunks == 3
    tolerance = 3e-6 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(ctx.rot_chol, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("n_data", [1, 2])
def test_pt_half_rotation_keeps_model_shards_local(n_data):
    if jax.local_device_count() < 2 * n_data:
        pytest.skip("Requires multiple logical CPU devices or GPUs.")
    mesh = Mesh(np.array(jax.local_devices()[:2*n_data]).reshape(n_data, 2), ("data", "model"))
    rng = np.random.default_rng(20261010)
    # Each model shard has two full batches and a six-vector remainder.
    chol = rng.normal(size=(1036, 7, 7))
    mo_t = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    ham = HamChol(replicate(0., mesh), replicate(np.eye(7), mesh), shard_model_axis(chol, mesh))
    trial = SimpleNamespace(mo_t=replicate(mo_t, mesh), mode_rank=9)
    ctx = build_ptccsd_thouless_mode_meas_ctx(ham, trial)
    assert ctx.rot_chol.sharding.spec == P("model")
    np.testing.assert_allclose(ctx.rot_chol, mo_t.conj().T @ chol, rtol=2e-12, atol=2e-12)
    compiled = _transform_cholesky.lower(
        ham.chol, trial.mo_t.conj().T, batch_size=256, mesh=mesh,
    ).compile()
    assert " all-gather(" not in compiled.as_text()


def test_pt_setup_workspace_is_bounded_by_cholesky_batch():
    # No arrays are allocated: only compile shape descriptors. A growing
    # full-input transpose or prefix/tail concatenation would fail this bound.
    sizes = []
    for nchol in (518, 5126):
        compiled = _transform_cholesky.lower(
            jax.ShapeDtypeStruct((nchol, 32, 32), jnp.float64),
            jax.ShapeDtypeStruct((12, 32), jnp.float64), batch_size=256,
        ).compile()
        sizes.append(compiled.memory_analysis().temp_size_in_bytes)
    assert sizes[1] < 5126 * 12 * 32 * 8 / 2
    assert sizes[1] <= sizes[0] + 256 * 32 * 32 * 8
