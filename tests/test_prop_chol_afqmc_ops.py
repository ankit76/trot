import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from trot.ham.chol import HamChol
from trot.prop.chol_afqmc_ops import (
    _build_prop_ctx,
    _get_h1_eff,
    _make_vhs_split_flat,
    _mf_shifts,
    _packed_upper_size,
    _sum_chol_squares,
    make_trotter_ops,
)


def _make_small_ham(*, norb=4, n_fields=3, h0=0.0, seed=0):
    key = jax.random.PRNGKey(seed)

    a = jax.random.normal(key, (norb, norb))
    h1 = 0.1 * (a + a.T)

    key, sub = jax.random.split(key)
    chol = 0.05 * jax.random.normal(sub, (n_fields, norb, norb))
    chol = 0.5 * (chol + jnp.swapaxes(chol, -1, -2))

    return HamChol(basis="restricted", h0=jnp.asarray(h0), h1=h1, chol=chol)


def test_build_prop_ctx_shapes_and_nfields():
    norb, n_fields = 5, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=0.0)

    dm = jnp.zeros((norb, norb))
    dt = 0.2
    ctx = _build_prop_ctx(ham, dm, dt)

    assert ctx.mf_shifts.shape == (n_fields,)
    assert ctx.exp_h1_half.shape == (norb, norb)
    assert ctx.dt.shape == ()
    assert ctx.sqrt_dt.shape == ()
    assert ctx.h0_prop.shape == ()


@pytest.mark.parametrize("basis,spin_resolved", [
    ("restricted", False), ("restricted", True), ("generalized", False),
])
@pytest.mark.parametrize("complex_inputs", [False, True])
@pytest.mark.parametrize("n_fields", [0, 7])
def test_mean_field_setup_matches_dense_reference(basis, spin_resolved, complex_inputs, n_fields):
    rng = np.random.default_rng(317)
    chol = 0.1 * rng.normal(size=(n_fields, 4, 4))
    densities = rng.normal(size=(2, 4, 4))
    if complex_inputs:
        # Nonsymmetric complex inputs distinguish transposition from conjugation.
        chol = chol + 0.1j * rng.normal(size=chol.shape)
        densities = densities + 1j * rng.normal(size=densities.shape)
    dm = densities.sum(axis=0)
    h1 = np.diag(np.arange(4.0))
    ham = HamChol(jnp.asarray(1.0), jnp.asarray(h1), jnp.asarray(chol), basis=basis)
    rdm1 = densities if spin_resolved else dm
    shifts = 1j * np.array([np.trace(matrix @ dm) for matrix in chol])
    normal_ordering = sum((matrix @ matrix for matrix in chol), np.zeros((4, 4))) / 2
    potential = sum((weight * matrix for weight, matrix in zip((1j * shifts).real, chol)),
                    np.zeros((4, 4)))

    actual_shifts = _mf_shifts(ham, jnp.asarray(rdm1))
    np.testing.assert_allclose(actual_shifts, shifts, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(_get_h1_eff(ham, actual_shifts), h1 - normal_ordering - potential,
                               rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("n_fields", [0, 1, 256, 257, 513])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_chol_squares_matches_matrix_products_across_batch_boundaries(n_fields, dtype):
    rng = np.random.default_rng(117)
    chol = rng.normal(size=(n_fields, 5, 5))
    if np.issubdtype(dtype, np.complexfloating):
        chol = chol + 1.0j * rng.normal(size=chol.shape)
    chol = chol.astype(dtype)
    # Deliberately nonsymmetric/complex: no transpose or conjugation is
    # allowed in the mathematical product L_g @ L_g.
    reference_dtype = np.complex128 if np.iscomplexobj(chol) else np.float64
    expected = np.zeros((5, 5), dtype=reference_dtype)
    for matrix in chol.astype(reference_dtype):
        expected += matrix @ matrix
    actual = _sum_chol_squares(jnp.asarray(chol))
    # Compare against a double-precision reference with a normwise bound:
    # cancellation makes elementwise relative error misleading in float32.
    tolerance = 1.0e-5 if dtype in (np.float32, np.complex64) else 1.0e-12
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance * scale)
    assert actual.dtype == chol.dtype


def test_build_prop_ctx_with_multiple_cholesky_batches_matches_direct_exponential():
    from scipy.linalg import expm

    ham = _make_small_ham(norb=5, n_fields=513, h0=1.25, seed=18)
    dm = jnp.diag(jnp.asarray([2.0, 2.0, 0.0, 0.0, 0.0]))
    chol = np.asarray(ham.chol)
    shifts = 1.0j * np.einsum("gij,ji->g", chol, np.asarray(dm))
    v0 = 0.5 * sum((matrix @ matrix for matrix in chol), np.zeros((5, 5)))
    v1 = np.einsum("g,gij->ij", (1.0j * shifts).real, chol)
    expected = expm(-0.005 / 2 * (np.asarray(ham.h1) - v0 - v1))
    ctx = _build_prop_ctx(ham, dm, 0.005, packed_cholesky=True)
    np.testing.assert_allclose(ctx.exp_h1_half, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(ctx.mf_shifts, shifts, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(ctx.h0_prop, -ham.h0 - 0.5 * np.sum(shifts**2))


@pytest.mark.parametrize(
    ("chol_dtype", "complex_dtype", "rtol"),
    [
        (jnp.float32, jnp.complex64, 2.0e-6),
        (jnp.float64, jnp.complex128, 1.0e-12),
    ],
)
def test_packed_cholesky_reconstructs_full_vhs(chol_dtype, complex_dtype, rtol):
    norb, n_fields = 6, 9
    ham = _make_small_ham(norb=norb, n_fields=n_fields, seed=17)
    dm = jnp.zeros((norb, norb), dtype=jnp.float64)
    field = jax.random.normal(jax.random.PRNGKey(31), (n_fields,), dtype=jnp.float64)
    field = field + 1.0j * jax.random.normal(
        jax.random.PRNGKey(32), (n_fields,), dtype=jnp.float64
    )

    full_ctx = _build_prop_ctx(
        ham,
        dm,
        0.01,
        chol_flat_precision=chol_dtype,
        packed_cholesky=False,
    )
    packed_ctx = _build_prop_ctx(
        ham,
        dm,
        0.01,
        chol_flat_precision=chol_dtype,
        packed_cholesky=True,
    )

    assert full_ctx.chol_flat.shape == (n_fields, norb * norb)
    assert packed_ctx.chol_flat.shape == (n_fields, _packed_upper_size(norb))
    assert not full_ctx.chol_packed
    assert packed_ctx.chol_packed

    full_vhs = jax.jit(
        lambda x: _make_vhs_split_flat(
            chol_flat=full_ctx.chol_flat,
            x=x,
            n=norb,
        )
    )(field.astype(complex_dtype))
    packed_vhs = jax.jit(
        lambda x: _make_vhs_split_flat(
            chol_flat=packed_ctx.chol_flat,
            x=x,
            n=norb,
            chol_packed=True,
        )
    )(field.astype(complex_dtype))

    np.testing.assert_allclose(packed_vhs, full_vhs, rtol=rtol, atol=rtol)
    np.testing.assert_allclose(packed_vhs, packed_vhs.T, rtol=0.0, atol=0.0)


def test_packed_and_full_trotter_actions_match():
    norb, nocc, n_fields = 5, 2, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, seed=23)
    dm = jnp.zeros((norb, norb), dtype=jnp.float64)
    walker = jax.random.normal(
        jax.random.PRNGKey(41), (norb, nocc), dtype=jnp.float64
    ).astype(jnp.complex128)
    field = jax.random.normal(jax.random.PRNGKey(42), (n_fields,), dtype=jnp.float64)
    field = field + 1.0j * jax.random.normal(
        jax.random.PRNGKey(43), (n_fields,), dtype=jnp.float64
    )

    full_ctx = _build_prop_ctx(ham, dm, 0.005, packed_cholesky=False)
    packed_ctx = _build_prop_ctx(ham, dm, 0.005, packed_cholesky=True)
    trotter_ops = make_trotter_ops("restricted", "restricted", mixed_precision=False)

    full_walker = trotter_ops.apply_trotter(walker, field, full_ctx, 6)
    packed_walker = jax.jit(trotter_ops.apply_trotter, static_argnums=3)(
        walker, field, packed_ctx, 6
    )

    np.testing.assert_allclose(packed_walker, full_walker, rtol=1.0e-12, atol=1.0e-12)


if __name__ == "__main__":
    pytest.main([__file__])
