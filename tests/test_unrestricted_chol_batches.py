"""Dense spin-resolved Cholesky batching, including compiled walker batching."""

from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.ops import k_energy
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.pt2uccsd import make_pt2uccsd_estimator_ops, make_pt2uccsd_meas_ops
from trot.meas.ucisd import make_ucisd_meas_ops
from trot.trial.ptuccsd_thouless import PtuccsdThoulessTrial
from trot.trial.ucisd import UcisdTrial


def _case(kind, walker_kind="restricted", nchol=7):
    rng = np.random.default_rng(20261004)
    norb, noa, nob = 6, 3, 2
    sys = System(norb=norb, nelec=(noa, nob), walker_kind=walker_kind)
    beta, r = np.linalg.qr(np.eye(norb) + 0.04 * rng.standard_normal((norb, norb)))
    beta *= np.sign(np.diag(r))
    h1 = 0.1 * rng.standard_normal((norb, norb))
    chol = 0.1 * rng.standard_normal((nchol, norb, norb))
    ham = HamChol(
        h0=jnp.asarray(0.37), h1=jnp.asarray(h1 + h1.T),
        chol=jnp.asarray(chol + chol.transpose(0, 2, 1)), basis="restricted",
    )
    t1a = 0.04 * rng.standard_normal((noa, norb - noa))
    t1b = 0.04 * rng.standard_normal((nob, norb - nob))

    def same_spin(nocc):
        t = 0.02 * rng.standard_normal((nocc, norb - nocc, nocc, norb - nocc))
        return jnp.asarray(0.25 * (t - t.transpose(2, 1, 0, 3)
                                  - t.transpose(0, 3, 2, 1) + t.transpose(2, 3, 0, 1)))

    aa, bb = same_spin(noa), same_spin(nob)
    ab = jnp.asarray(0.02 * rng.standard_normal((noa, norb - noa, nob, norb - nob)))
    if kind == "ucisd":
        trial = UcisdTrial(
            mo_coeff_a=jnp.eye(norb), mo_coeff_b=jnp.asarray(beta),
            c1a=jnp.asarray(t1a), c1b=jnp.asarray(t1b), c2aa=aa, c2bb=bb, c2ab=ab,
        )
    else:
        trial = PtuccsdThoulessTrial(
            mo_t_a=jnp.asarray(np.vstack([np.eye(noa), t1a.T])),
            mo_t_b=jnp.asarray(np.vstack([np.eye(nob), t1b.T])),
            mo_coeff_b=jnp.asarray(beta), t2aa=aa, t2bb=bb, t2ab=ab,
        )

    def walkers(nocc):
        return jnp.asarray(np.eye(norb, nocc)[None, :, :] + 0.08 * (
            rng.standard_normal((2, norb, nocc))
            + 1j * rng.standard_normal((2, norb, nocc))))

    w = walkers(noa)
    if walker_kind == "unrestricted":
        w = (w, walkers(nob))
    return sys, ham, trial, w


def _ops(kind, sys, mode, mixed=True, batch_size=64):
    kwargs = dict(memory_mode=mode, mixed_precision=mixed,
                  testing=not mixed, chol_batch_size=batch_size)
    if kind == "ucisd":
        ops = make_ucisd_meas_ops(sys, **kwargs)
        return ops.require_kernel(k_energy), ops.build_meas_ctx
    if sys.walker_kind == "restricted":
        ops = make_pt2uccsd_estimator_ops(sys, **kwargs)
        return ops.components, ops.build_estimator_ctx
    ops = make_pt2uccsd_meas_ops(sys, **kwargs)
    return ops.observables["pt_components"], ops.build_meas_ctx


@pytest.mark.parametrize("kind", ["ucisd", "ptuccsd"])
@pytest.mark.parametrize("walker_kind", ["restricted", "unrestricted"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("nchol,batch_size", [(7, 1), (8, 4), (7, 3), (7, 7), (7, 64), (67, 64), (32, 16), (33, 16)])
def test_batched_doubles_match_all_choleskies(kind, walker_kind, mixed, nchol, batch_size):
    sys, ham, trial, walkers = _case(kind, walker_kind, nchol)
    results = []
    for mode in ("high", "low"):
        kernel, build_ctx = _ops(kind, sys, mode, mixed, batch_size)
        ctx = build_ctx(ham, trial)
        compiled = jax.jit(jax.vmap(kernel, in_axes=(0, None, None, None)))
        results.append(np.asarray(compiled(walkers, ham, ctx, trial)))
    # FP32 contractions/reductions change order between batch sizes.
    np.testing.assert_allclose(results[1], results[0], rtol=0, atol=1e-7 if mixed else 2e-12)


@pytest.mark.parametrize("kind", ["ucisd", "ptuccsd"])
def test_low_memory_uses_batched_products_and_only_pads_tail(kind):
    sys, ham, trial, walkers = _case(kind, nchol=67)
    kernel, build_ctx = _ops(kind, sys, "low")
    ctx = build_ctx(ham, trial)
    assert ctx.cfg.chol_batch_size == 64
    traced = jax.make_jaxpr(kernel)(walkers[0], ham, ctx, trial)
    scans = [e for e in traced.jaxpr.eqns if e.primitive.name == "scan"]
    # Residual L:M contractions have their own bounded loops. Locate the
    # doubles loop by its occupied-space products rather than the loop count.
    doubles_scans = []
    for scan in scans:
        body = scan.params["jaxpr"].jaxpr
        shapes = [v.aval.shape for e in body.eqns if e.primitive.name == "dot_general"
                  for v in e.outvars]
        if (64, sys.nup, sys.norb) in shapes and (64, sys.ndn, sys.norb) in shapes:
            doubles_scans.append(scan)
            assert (64, sys.norb, sys.norb) not in shapes
    assert len(doubles_scans) == 1
    assert doubles_scans[0].params["length"] == 1

    def nested_eqns(jaxpr):
        for eqn in jaxpr.eqns:
            yield eqn
            for value in eqn.params.values():
                if hasattr(value, "jaxpr"):
                    yield from nested_eqns(value.jaxpr)

    pads = [e for e in nested_eqns(traced.jaxpr) if e.primitive.name == "pad"]
    assert pads
    assert all(e.invars[0].aval.shape[0] == 3 for e in pads)
    assert all(e.outvars[0].aval.shape[0] == 64 for e in pads)


@pytest.mark.parametrize("factory", [make_ucisd_meas_ops, make_pt2uccsd_meas_ops,
                                    make_pt2uccsd_estimator_ops])
def test_factories_retain_high_memory_default(factory):
    sys, ham, trial, _ = _case("ucisd" if factory is make_ucisd_meas_ops else "ptuccsd")
    ops = factory(sys)
    build_ctx = (ops.build_estimator_ctx if factory is make_pt2uccsd_estimator_ops
                 else ops.build_meas_ctx)
    ctx = build_ctx(ham, trial)
    assert ctx.cfg.memory_mode == "high"
    assert ctx.cfg.chol_batch_size == 64


@pytest.mark.parametrize("factory", [make_ucisd_meas_ops, make_pt2uccsd_meas_ops,
                                    make_pt2uccsd_estimator_ops])
@pytest.mark.parametrize("size", [0, -1, 1.5])
def test_invalid_batch_size_is_rejected(factory, size):
    with pytest.raises(ValueError, match="positive integer"):
        factory(System(norb=6, nelec=(3, 2), walker_kind="restricted"), chol_batch_size=size)


@pytest.mark.parametrize("kind", ["ucisd", "ptuccsd"])
@pytest.mark.parametrize("n_chunks", [2, 3])
def test_sixteen_cholesky_batches_with_walker_remainder(kind, n_chunks):
    from trot.walkers import vmap_chunked
    sys, ham, trial, walkers = _case(kind, nchol=33)
    walkers = jnp.concatenate((walkers, walkers, walkers[:1]), axis=0)
    low_kernel, low_ctx = _ops(kind, sys, "low", batch_size=16)
    high_kernel, high_ctx = _ops(kind, sys, "high", batch_size=16)
    batched = jax.jit(vmap_chunked(low_kernel, n_chunks, in_axes=(0, None, None, None)))
    actual = batched(walkers, ham, low_ctx(ham, trial), trial)
    expected = jax.jit(jax.vmap(high_kernel, in_axes=(0, None, None, None)))(
        walkers, ham, high_ctx(ham, trial), trial)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
