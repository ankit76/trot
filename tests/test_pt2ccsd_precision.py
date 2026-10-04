"""Precision-policy regressions for the dense PT2 component estimator."""
from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import combine_first_order_energy, make_pt2ccsd_estimator_ops
from trot.trial.pt2ccsd import Pt2ccsdTrial


def _case(complex_tensors=False, complex_walkers=True, nchol=17):
    rng = np.random.default_rng(593032)
    nocc, nvir = 3, 6
    norb = nocc + nvir

    def random(shape, scale):
        value = rng.normal(size=shape)
        if complex_tensors:
            value = value + 0.3j * rng.normal(size=shape)
        return scale * value

    h1 = random((norb, norb), 0.2)
    chol = random((nchol, norb, norb), 0.15)
    ham = HamChol(
        jnp.asarray(0.4),
        jnp.asarray((h1 + h1.T) / 2),
        jnp.asarray((chol + chol.transpose(0, 2, 1)) / 2),
        "restricted",
    )
    raw = random((nocc, nvir, nocc, nvir), 0.025)
    trial = Pt2ccsdTrial(
        jnp.asarray(np.vstack([np.eye(nocc), random((nvir, nocc), 0.08)])),
        jnp.asarray((raw + raw.transpose(2, 3, 0, 1)) / 2),
    )
    walkers = np.broadcast_to(np.eye(norb, nocc), (5, norb, nocc)).copy()
    walkers += rng.normal(size=walkers.shape) * 0.08
    if complex_walkers:
        walkers = walkers + 0.08j * rng.normal(size=walkers.shape)
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    return sys, ham, trial, jnp.asarray(walkers)


@pytest.mark.parametrize("complex_tensors", [False, True])
@pytest.mark.parametrize("complex_walkers", [False, True])
@pytest.mark.parametrize("nchol", [0, 1, 17])
def test_mixed_components_and_connected_energy_match_full_precision(
    complex_tensors, complex_walkers, nchol
):
    sys, ham, trial, walkers = _case(complex_tensors, complex_walkers, nchol)
    values = []
    for mixed in (False, True):
        ops = make_pt2ccsd_estimator_ops(sys, mixed_precision=mixed)
        ctx = ops.build_estimator_ctx(ham, trial)
        kernel = jax.jit(jax.vmap(ops.components, in_axes=(0, None, None, None)))
        values.append(np.asarray(kernel(walkers, ham, ctx, trial)))
    full, mixed = values
    assert mixed.dtype == full.dtype
    # Theta is unchanged: it still uses the original overlap/one-body precision.
    np.testing.assert_allclose(mixed[:, 0], full[:, 0], atol=1e-13, rtol=0)
    # HF exchange and Coulomb stay in double precision as well.
    np.testing.assert_allclose(mixed[:, 1], full[:, 1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(mixed, full, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        combine_first_order_energy(ham.h0, mixed),
        combine_first_order_energy(ham.h0, full),
        atol=2e-6,
        rtol=2e-6,
    )
    # The production PT estimator combines population-averaged components.
    weights = np.asarray([0.1, 0.2, 0.3, 0.15, 0.25])
    np.testing.assert_allclose(
        combine_first_order_energy(ham.h0, weights @ mixed),
        combine_first_order_energy(ham.h0, weights @ full),
        atol=2e-6,
        rtol=2e-6,
    )


def _dot_dtypes(value):
    """Collect dot operand dtypes through nested jit/scan jaxprs."""
    if hasattr(value, "jaxpr"):
        yield from _dot_dtypes(value.jaxpr)
    elif hasattr(value, "eqns"):
        for eqn in value.eqns:
            if eqn.primitive.name == "dot_general":
                yield from (str(v.aval.dtype) for v in eqn.invars[:2])
            yield from _dot_dtypes(eqn.params)
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _dot_dtypes(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _dot_dtypes(nested)


def test_mixed_precision_flag_changes_products_but_keeps_double_precision_work():
    sys, ham, trial, walkers = _case()
    traced_dtypes = []
    for mixed in (False, True):
        ops = make_pt2ccsd_estimator_ops(sys, mixed_precision=mixed)
        ctx = ops.build_estimator_ctx(ham, trial)
        traced = jax.make_jaxpr(ops.components)(walkers[0], ham, ctx, trial)
        traced_dtypes.append(set(_dot_dtypes(traced)))
    full, mixed = traced_dtypes
    assert "float32" not in full and "complex64" not in full
    assert "float32" in mixed
    assert "complex128" in mixed
