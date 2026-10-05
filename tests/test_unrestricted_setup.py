"""Setup rotations preserve dense algebra without full-sized temporary copies."""
from types import SimpleNamespace

from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from trot.ham.chol import HamChol
from trot.meas.chol_setup import _transform_cholesky, transform_cholesky
from trot.meas import ucisd, ucisdt, uhf, ptuccsd_thouless
from trot.sharding import replicate, shard_model_axis


@pytest.mark.parametrize("nchol", [0, 1, 64, 67, 128])
@pytest.mark.parametrize("complex_chol,complex_rotation", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("kind", ["rotation", "half", "singles"])
def test_setup_contractions_match_numpy(nchol, complex_chol, complex_rotation, kind):
    rng = np.random.default_rng(974)
    def array(shape, complex_):
        x = rng.normal(size=shape)
        return x + 1j*rng.normal(size=shape) if complex_ else x
    chol = array((nchol, 7, 7), complex_chol)
    c = array((7, 7), complex_rotation)
    if kind == "rotation":
        left, right, columns = c.conj().T, c, None
        expected = np.einsum("pi,gij,jq->gpq", left, chol, right)
    elif kind == "half":
        left, right, columns = c[:, :3].conj().T, None, None
        expected = np.einsum("pi,gij->gpj", left, chol)
    else:
        # FNO singles occupy a non-full virtual slice. Conjugation is not used.
        left, right, columns = None, c[:3, :2], (2, 5)
        expected = np.einsum("git,tp->gip", chol[:, :, 2:5], right)
    result = transform_cholesky(jnp.asarray(chol), left=left, right=right, column_slice=columns)
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, atol=2.e-12, rtol=2.e-12)


@pytest.mark.parametrize("kind", ["ucisd", "ucisdt", "ptuccsd", "uhf"])
def test_contexts_preserve_orbital_rotations_and_singles(kind):
    rng = np.random.default_rng(975)
    n = 7
    chol = rng.normal(size=(67, n, n))
    beta, _ = np.linalg.qr(rng.normal(size=(n, n)))
    ham = HamChol(jnp.array(0.), jnp.eye(n), jnp.asarray(chol))
    trial = SimpleNamespace(nocc=(3, 2), nvir=(3, 4), mo_coeff_b=jnp.asarray(beta),
        mo_coeff_a=jnp.eye(n, 3), c1a=jnp.asarray(rng.normal(size=(3, 3))),
        c1b=jnp.asarray(rng.normal(size=(2, 4))),
        mo_t_a=jnp.asarray(rng.normal(size=(n, 3))),
        mo_t_b=jnp.asarray(rng.normal(size=(n, 2))))
    expected_beta = np.einsum("pi,gij,jq->gpq", beta.T, chol, beta)
    if kind == "uhf":
        trial.mo_coeff_b = trial.mo_coeff_b[:, :2]
        ctx = uhf.build_meas_ctx(ham, trial)
        np.testing.assert_allclose(ctx.rot_chol_a, chol[:, :3], atol=2.e-12)
        np.testing.assert_allclose(ctx.rot_chol_b, np.einsum("pi,gij->gpj", beta[:, :2].T, chol), atol=2.e-12)
        return
    if kind == "ptuccsd":
        ctx = ptuccsd_thouless.build_ptuccsd_thouless_meas_ctx(ham, trial)
        np.testing.assert_allclose(ctx.rot_chol_a, np.einsum("pi,gpq->giq", trial.mo_t_a, chol), atol=2.e-12)
        np.testing.assert_allclose(ctx.rot_chol_b, np.einsum("pi,gpq->giq", trial.mo_t_b, expected_beta), atol=2.e-12)
    else:
        if kind == "ucisdt":  # This builder retains the full virtual space.
            trial.c1a = jnp.asarray(rng.normal(size=(3, 4)))
            trial.c1b = jnp.asarray(rng.normal(size=(2, 5)))
        ctx = (ucisd if kind == "ucisd" else ucisdt).build_meas_ctx(ham, trial)
        np.testing.assert_allclose(ctx.lci1_a, np.einsum("git,pt->gip", chol[:, :, 3:3+trial.c1a.shape[1]], trial.c1a), atol=2.e-12)
        np.testing.assert_allclose(ctx.lci1_b, np.einsum("git,pt->gip", expected_beta[:, :, 2:2+trial.c1b.shape[1]], trial.c1b), atol=2.e-12)
    np.testing.assert_allclose(ctx.chol_b, expected_beta, atol=2.e-12, rtol=2.e-12)


@pytest.mark.parametrize("kind", ["rotation", "half", "singles"])
def test_setup_batches_stay_local_to_model_shards(kind):
    if jax.local_device_count() < 2:
        pytest.skip("Requires two logical CPU devices or GPUs.")
    mesh = Mesh(np.array(jax.local_devices()[:2]), ("model",))
    rng = np.random.default_rng(976)
    chol = rng.normal(size=(134, 7, 7))
    c = rng.normal(size=(7, 7)) + 1j*rng.normal(size=(7, 7))
    left = None if kind == "singles" else (c.conj().T if kind == "rotation" else c[:, :3].conj().T)
    right = c if kind == "rotation" else (c[:3, :2] if kind == "singles" else None)
    columns = (2, 5) if kind == "singles" else None
    args = (shard_model_axis(chol, mesh), replicate(left, mesh) if left is not None else None,
            replicate(right, mesh) if right is not None else None)
    compiled = _transform_cholesky.lower(*args, column_slice=columns, mesh=mesh).compile()
    assert " all-gather(" not in compiled.as_text()
    result = compiled(*args)
    assert result.sharding.spec == P("model")
    expected = chol[:, :, columns[0]:columns[1]] if columns else chol
    if left is not None: expected = left @ expected
    if right is not None: expected = expected @ right
    np.testing.assert_allclose(result, expected, atol=2.e-12, rtol=2.e-12)


def test_setup_workspace_does_not_grow_like_full_rotated_tensor():
    # Compiler check: the tail must update the existing output, not concatenate
    # a separately materialized prefix of nearly the full Cholesky dimension.
    c = jax.ShapeDtypeStruct((32, 32), jnp.float64)
    sizes = []
    for nchol in (67, 1027):
        compiled = _transform_cholesky.lower(
            jax.ShapeDtypeStruct((nchol, 32, 32), jnp.float64), c, c
        ).compile()
        sizes.append(compiled.memory_analysis().temp_size_in_bytes)
    full_size = 1027 * 32 * 32 * 8
    assert sizes[1] < full_size / 2
    assert sizes[1] <= sizes[0] + 64 * 32 * 32 * 8
