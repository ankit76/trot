"""Default VHS layout, opt-out, and complex-factor policy across setup paths."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.prop.afqmc import make_prop_ops
from trot.prop.chol_afqmc_ops import (
    _resolve_packed_cholesky, _make_vhs_split_flat, make_trotter_ops,
)
from trot.prop.types import QmcParams
from trot.runtime_layout import _build_restricted_prop_ctx_from_host
from tests.test_prop_chol_afqmc_ops import _make_small_ham

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("input_dtype,requested,expected", [
    (np.float32, None, True), (np.float64, None, True),
    (np.float64, False, False), (np.complex64, None, False),
    (np.complex128, None, False), (np.complex128, True, False),
])
def test_packing_policy_needs_only_dtype(input_dtype, requested, expected):
    # No array contents or shape are available: default selection must not
    # inspect numerical symmetry or copy device data to the host.
    chol = SimpleNamespace(dtype=np.dtype(input_dtype))
    assert _resolve_packed_cholesky(chol, dtype=jnp.float32, packed_cholesky=requested) is expected


@pytest.mark.parametrize("mixed,requested,expected", [
    (True, None, True), (True, False, False), (True, True, True),
    (False, None, False), (False, True, True),
])
@pytest.mark.parametrize("sharded", [False, True])
def test_host_and_device_builders_agree(mixed, requested, expected, sharded):
    ham = _make_small_ham(norb=4, n_fields=35)
    dm = jnp.diag(jnp.array([2., 2., 0., 0.]))
    staged = SimpleNamespace(ham=ham)
    params = QmcParams(dt=.005)
    prop = make_prop_ops("restricted", "restricted", mixed_precision=mixed, packed_cholesky=requested)
    mesh = None
    device_ham, device_dm = ham, dm
    if sharded:
        if jax.local_device_count() < 4:
            pytest.skip("Requires four logical devices for data/model sharding")
        from jax.sharding import Mesh
        from trot.sharding import shard_ham_data, replicate
        mesh = Mesh(np.array(jax.local_devices()[:4]).reshape(2, 2), ("data", "model"))
        device_ham = shard_ham_data(ham, mesh)
        device_dm = replicate(dm, mesh)
    device = prop.build_prop_ctx(device_ham, device_dm, params)
    host = _build_restricted_prop_ctx_from_host(staged, trial_rdm1=dm, dt=params.dt,
        mixed_precision=mixed, mesh=mesh, packed_cholesky=requested)
    assert device.chol_packed is host.chol_packed is expected
    assert host.chol_flat.shape == device.chol_flat.shape == (36 if sharded else 35, 10 if expected else 16)
    if sharded:
        from jax.sharding import PartitionSpec as P
        assert host.chol_flat.sharding.spec == device.chol_flat.sharding.spec == P("model")
    assert host.chol_flat.dtype == (np.float32 if mixed else np.float64)
    np.testing.assert_array_equal(host.chol_flat, device.chol_flat)
    for name in ("exp_h1_half", "mf_shifts", "h0_prop"):
        np.testing.assert_allclose(getattr(host, name), getattr(device, name), rtol=2e-12, atol=2e-12)


def test_explicit_opt_out_preserves_nonsymmetric_vhs():
    ham = _make_small_ham()
    ham = replace(ham, chol=ham.chol.at[-1, 0, 1].add(.1))
    p = make_prop_ops("restricted", "restricted", mixed_precision=True, packed_cholesky=False)
    ctx = p.build_prop_ctx(ham, jnp.eye(4), QmcParams())
    assert not ctx.chol_packed
    field = jnp.array([.1 + .2j, -.3, .5j], dtype=jnp.complex64)
    actual = _make_vhs_split_flat(chol_flat=ctx.chol_flat, x=field, n=4, chol_packed=ctx.chol_packed)
    expected = np.einsum("g,gij->ij", np.asarray(field), np.asarray(ham.chol, dtype=np.float32))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("walker_kind", ["restricted", "unrestricted", "generalized"])
def test_default_fp32_propagation_matches_opt_out(walker_kind):
    ham = _make_small_ham(norb=4, n_fields=7)
    params = QmcParams(dt=.005)
    dm = jnp.eye(4)
    packed = make_prop_ops("restricted", walker_kind, mixed_precision=True).build_prop_ctx(ham, dm, params)
    full = make_prop_ops("restricted", walker_kind, mixed_precision=True, packed_cholesky=False).build_prop_ctx(ham, dm, params)
    rng = np.random.default_rng(519)
    fields = jnp.asarray(rng.normal(size=7) + .03j)
    walker = jnp.asarray(rng.normal(size=(4, 2)) + .2j * rng.normal(size=(4, 2)))
    if walker_kind == "unrestricted":
        walker = (walker, walker[:, :1])
    elif walker_kind == "generalized":
        walker = jnp.concatenate((walker, .5 * walker))
    apply = jax.jit(make_trotter_ops("restricted", walker_kind, mixed_precision=True).apply_trotter,
                    static_argnums=3)
    actual, expected = apply(walker, fields, packed, 6), apply(walker, fields, full, 6)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("kind", ["rhf", "cisd", "cisd_modes"])
@pytest.mark.parametrize("requested,expected", [(None, True), (False, False)])
def test_setup_routes_default_and_opt_out(kind, requested, expected):
    from trot.cisd_workflow import CisdWorkflowConfig, CisdModeConfig
    from trot.runtime_layout import DefaultRuntimeLayout, RhfHostRuntimeLayout, CisdHostRuntimeLayout
    from trot.setup import setup
    from trot.staging import TrialInput
    from tests.test_cisd_workflow import _staged_cisd

    staged = _staged_cisd()
    options = {}
    if kind == "rhf":
        staged = replace(staged, trial=TrialInput(kind="rhf", data={"mo": np.eye(4)},
            frozen=0, source_kind="mf"))
    elif kind == "cisd_modes":
        options["cisd_workflow"] = CisdWorkflowConfig(modes=CisdModeConfig(discarded_norm_target=.1))
    job = setup(staged, mixed_precision=True, params=QmcParams(n_walkers=2),
                prop_kwargs={"packed_cholesky": requested}, **options)
    layout_type = {"rhf": RhfHostRuntimeLayout, "cisd": CisdHostRuntimeLayout,
                   "cisd_modes": DefaultRuntimeLayout}[kind]
    assert isinstance(job.runtime_layout, layout_type)
    _, _, ctx = job._prepare_runtime()
    assert ctx.chol_packed is expected
    assert ctx.chol_flat.shape[-1] == (10 if expected else 16)

