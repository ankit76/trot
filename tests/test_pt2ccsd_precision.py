"""Precision-policy regressions for the dense PT2 component estimator."""
from trot import config

config.configure_once()

from dataclasses import replace

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


def _full_space_reference(walker, ham, trial):
    """Independent NumPy evaluation of the original orbital-space formula."""
    c, t2 = np.asarray(trial.mo_t), np.asarray(trial.t2)
    h1, chol = np.asarray(ham.h1), np.asarray(ham.chol)
    nocc = trial.nocc
    g = (walker @ np.linalg.inv(c.T @ walker) @ c.T).T
    gp = (g - np.eye(g.shape[0]))[:, nocc:]
    gov = g[:nocc, nocc:]
    t2g = 2 * np.einsum("iajb,ia->jb", t2, gov)
    t2g -= np.einsum("iajb,ib->ja", t2, gov)
    correction = gp @ t2g.T @ g[:nocc]
    theta = np.sum(t2g * gov)
    e10 = 2 * np.sum(h1 * g)
    e12 = e10 * theta - 2 * np.sum(h1 * correction)
    gl = np.einsum("pr,gqr->gpq", g, chol)
    lt2 = np.einsum("gpr,qr->gpq", chol, correction)
    lg = np.einsum("gpp->g", gl)
    e20 = 2 * np.sum(lg**2) - np.einsum("gpq,gqp->", gl, gl)
    e221 = -np.einsum("gpq,pq,g->", chol, correction, lg)
    e222 = 0.5 * np.sum(gl * lt2)
    x = np.einsum("giq,qa->gia", gl[:, :nocc], gp)
    e23 = 2 * np.einsum("gia,gjb,iajb->", x, x, t2)
    e23 -= np.einsum("gib,gja,iajb->", x, x, t2)
    return np.asarray([theta, e10 + e20, e12 + e20 * theta + 4 * (e221 + e222) + e23])


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
    reference = np.asarray([_full_space_reference(np.asarray(w), ham, trial) for w in walkers])
    np.testing.assert_allclose(full, reference, atol=1e-12, rtol=1e-12)
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


@pytest.mark.parametrize("complex_tensors", [False, True])
def test_half_rotation_with_general_reference_and_nonsymmetric_factors(complex_tensors):
    sys, ham, trial, walkers = _case(complex_tensors)
    # Rotation cannot assume that the occupied reference block is identity,
    # or that Cholesky transposes can be interchanged.
    occupied_change = jnp.asarray([[.1, .2, 0], [0, .1, -.2], [.03, 0, .1]])
    trial = replace(trial, mo_t=trial.mo_t.at[:trial.nocc].add(occupied_change))
    ham = replace(ham, chol=ham.chol.at[:, 0, 1].add(.07))
    ops = make_pt2ccsd_estimator_ops(sys, mixed_precision=False)
    ctx = ops.build_estimator_ctx(ham, trial)
    got = jax.jit(jax.vmap(ops.components, in_axes=(0, None, None, None)))(walkers, ham, ctx, trial)
    expected = np.asarray([_full_space_reference(np.asarray(w), ham, trial) for w in walkers])
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mixed", [False, True])
def test_default_batches_all_choleskies_without_orbital_space_products(mixed):
    sys, ham, trial, walkers = _case()
    ops = make_pt2ccsd_estimator_ops(sys, mixed_precision=mixed)
    ctx = ops.build_estimator_ctx(ham, trial)
    assert ctx.cfg.memory_mode == "high"
    assert ctx.rot_chol.shape == (ham.chol.shape[0], trial.nocc, trial.norb)
    traced = jax.make_jaxpr(ops.components)(walkers[0], ham, ctx, trial)
    assert not any(eqn.primitive.name == "scan" for eqn in traced.jaxpr.eqns)
    dots = [eqn for eqn in traced.jaxpr.eqns if eqn.primitive.name == "dot_general"]
    shapes = [v.aval.shape for dot in dots for v in dot.outvars]
    assert (ham.chol.shape[0], trial.nocc, trial.norb) in shapes
    assert (ham.chol.shape[0], trial.norb, trial.norb) not in shapes


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("complex_tensors", [False, True])
@pytest.mark.parametrize("nchol,batch_size", [(0, 4), (1, 4), (16, 4), (17, 4), (17, 64)])
def test_bounded_batches_match_all_choleskies(mixed, complex_tensors, nchol, batch_size):
    sys, ham, trial, walkers = _case(complex_tensors, True, nchol)
    results = []
    for mode in ("high", "low"):
        ops = make_pt2ccsd_estimator_ops(
            sys, memory_mode=mode, mixed_precision=mixed, chol_batch_size=batch_size
        )
        ctx = ops.build_estimator_ctx(ham, trial)
        kernel = jax.jit(jax.vmap(ops.components, in_axes=(0, None, None, None)))
        results.append(np.asarray(kernel(walkers, ham, ctx, trial)))
    np.testing.assert_allclose(results[1], results[0], atol=1e-8 if mixed else 1e-12, rtol=0)


def test_low_memory_loop_contracts_batches_not_single_vectors():
    sys, ham, trial, walkers = _case(nchol=17)
    ops = make_pt2ccsd_estimator_ops(sys, memory_mode="low", chol_batch_size=4)
    ctx = ops.build_estimator_ctx(ham, trial)
    traced = jax.make_jaxpr(ops.components)(walkers[0], ham, ctx, trial)
    scans = [eqn for eqn in traced.jaxpr.eqns if eqn.primitive.name == "scan"]
    assert len(scans) == 1
    assert scans[0].params["length"] == 4
    body = scans[0].params["jaxpr"].jaxpr
    shapes = [
        v.aval.shape
        for eqn in body.eqns if eqn.primitive.name == "dot_general"
        for v in eqn.outvars
    ]
    assert (4, trial.nocc, trial.norb) in shapes
    assert (4, trial.norb, trial.norb) not in shapes


@pytest.mark.parametrize("size", [0, -1, 1.5])
def test_invalid_cholesky_batch_size_is_rejected(size):
    sys, _, _, _ = _case()
    with pytest.raises(ValueError, match="positive integer"):
        make_pt2ccsd_estimator_ops(sys, chol_batch_size=size)
