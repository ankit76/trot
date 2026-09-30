import os
from functools import partial
from typing import cast

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import trot.walkers as wk
from trot import testing
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.ham.hubbard import HamHubbard
from trot.meas.auto import make_auto_meas_ops
from trot.meas.rhf import RhfMeasCfg, RhfMeasCtx, RhfMeasMemoryMode, make_rhf_meas_ops
from trot.meas.rhf import build_meas_ctx as build_rhf_meas_ctx
from trot.prop.afqmc import init_prop_state, make_prop_ops
from trot.prop.blocks import block
from trot.prop.chol_afqmc_ops import CholAfqmcCtx, _build_prop_ctx
from trot.prop.types import QmcParams
from trot.setup import setup
from trot.sharding import (
    make_data_mesh,
    make_data_model_mesh,
    shard_ham_data,
    shard_model_axis,
    shard_prop_state,
)
from trot.staging import HamInput, StagedInputs, TrialInput, dump
from trot.trial.rhf import RhfTrial, get_rdm1


def _assert_named_sharding_spec(a: object, spec: P) -> None:
    assert isinstance(a, jax.Array)
    sharding = a.sharding
    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == spec


@pytest.mark.parametrize("n_per_dev", [4])
def test_block_runs_under_sharding(n_per_dev):

    mesh = make_data_mesh()
    ndev = mesh.size

    norb, nocc, n_chol = 6, 3, 4
    nw = ndev * n_per_dev

    rng = jax.random.PRNGKey(42)
    k_ham, k_walk, k_walk2, k_wt = jax.random.split(rng, 4)

    ham = testing.make_random_ham_chol(k_ham, norb, n_chol)
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")

    trial_ops = testing.make_dummy_trial_ops()
    meas_ops = make_auto_meas_ops(sys, trial_ops)
    trial_data = {"rdm1": jnp.eye(norb, dtype=jnp.float64)}
    meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)

    params = QmcParams(
        dt=0.1,
        n_chunks=1,
        n_exp_terms=4,
        n_prop_steps=1,
        shift_ema=0.5,
        n_walkers=nw,
        seed=0,
        pop_control_damping=0.1,
    )

    prop_ops = make_prop_ops(ham.basis, sys.walker_kind)
    prop_ctx = _build_prop_ctx(ham, trial_data["rdm1"], params.dt)

    initial_walkers = jax.random.normal(
        k_walk, (nw, norb, nocc), dtype=jnp.float64
    ) + 1.0j * jax.random.normal(k_walk2, (nw, norb, nocc), dtype=jnp.float64)

    # unsharded reference
    state0_u = init_prop_state(
        sys=sys,
        ham_data=ham,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        params=params,
        initial_walkers=initial_walkers,
        mesh=None,
    )
    rand_weights = jax.random.uniform(k_wt, (nw,), dtype=jnp.float64, minval=0.5, maxval=2.0)
    state0_u = state0_u._replace(weights=rand_weights)

    # sharded version of the exact same state
    state0_s = shard_prop_state(state0_u, mesh)

    def _assert_sharded_first_axis(a):
        _assert_named_sharding_spec(a, P("data"))

    def _assert_replicated(a):
        _assert_named_sharding_spec(a, P())

    for leaf in tree_util.tree_leaves(state0_s.walkers):
        _assert_sharded_first_axis(leaf)
    _assert_sharded_first_axis(state0_s.weights)
    _assert_sharded_first_axis(state0_s.overlaps)
    _assert_replicated(state0_s.rng_key)
    _assert_replicated(state0_s.e_estimate)
    _assert_replicated(state0_s.pop_control_ene_shift)

    wsum_direct = jnp.sum(state0_s.weights)
    wsum_expected = float(jax.device_get(jnp.sum(rand_weights)))
    assert float(jax.device_get(wsum_direct)) == pytest.approx(wsum_expected, abs=1e-12)

    data_sh = NamedSharding(mesh, P("data"))
    sr_sharded = partial(wk.stochastic_reconfiguration, data_sharding=data_sh)
    sr_unsharded = partial(wk.stochastic_reconfiguration, data_sharding=None)

    def _call_block(st, *, sr_fn):
        return block(
            st,
            sys=sys,
            params=params,
            ham_data=ham,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ops=prop_ops,
            prop_ctx=prop_ctx,
            sr_fn=sr_fn,
        )

    run_block_u = jax.jit(partial(_call_block, sr_fn=sr_unsharded))
    run_block_s = jax.jit(partial(_call_block, sr_fn=sr_sharded))

    state1_u, obs_u = run_block_u(state0_u)
    state1_s, obs_s = run_block_s(state0_s)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(obs_s.scalars["weight"])),
        np.asarray(jax.device_get(obs_u.scalars["weight"])),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(obs_s.scalars["energy"])),
        np.asarray(jax.device_get(obs_u.scalars["energy"])),
        rtol=1e-12,
        atol=1e-12,
    )

    # walkers and weights must match after SR
    for leaf_s, leaf_u in zip(
        tree_util.tree_leaves(state1_s.walkers),
        tree_util.tree_leaves(state1_u.walkers),
    ):
        np.testing.assert_allclose(
            np.asarray(jax.device_get(leaf_s)),
            np.asarray(jax.device_get(leaf_u)),
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(state1_s.weights)),
        np.asarray(jax.device_get(state1_u.weights)),
        rtol=1e-12,
        atol=1e-12,
    )

    for leaf in tree_util.tree_leaves(state1_s.walkers):
        _assert_sharded_first_axis(leaf)
    _assert_sharded_first_axis(state1_s.weights)
    _assert_sharded_first_axis(state1_s.overlaps)
    _assert_replicated(state1_s.rng_key)
    _assert_replicated(state1_s.e_estimate)
    _assert_replicated(state1_s.pop_control_ene_shift)


@pytest.mark.parametrize("n_per_dev", [4])
def test_sr_sharded_matches_unsharded(n_per_dev):
    """Stochastic reconfiguration gives identical results with and without sharding."""

    mesh = make_data_mesh()
    ndev = mesh.size
    nw = ndev * n_per_dev
    norb, nocc = 6, 3

    rng = jax.random.PRNGKey(99)
    k1, k2, k3 = jax.random.split(rng, 3)
    walkers = jax.random.normal(k1, (nw, norb, nocc), dtype=jnp.float64) + 1.0j * jax.random.normal(
        k2, (nw, norb, nocc), dtype=jnp.float64
    )
    weights = jax.random.uniform(k3, (nw,), dtype=jnp.float64, minval=0.5, maxval=2.0)
    zeta = jax.random.uniform(jax.random.PRNGKey(7), (), dtype=jnp.float64)

    data_sh = NamedSharding(mesh, P("data"))
    walkers_s = jax.device_put(walkers, data_sh)
    weights_s = jax.device_put(weights, data_sh)

    sr = jax.jit(wk.stochastic_reconfiguration, static_argnames=("walker_kind", "data_sharding"))

    w_u, wt_u = sr(walkers, weights, zeta, "restricted", data_sharding=None)
    w_s, wt_s = sr(walkers_s, weights_s, zeta, "restricted", data_sharding=data_sh)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(w_s)),
        np.asarray(jax.device_get(w_u)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(wt_s)),
        np.asarray(jax.device_get(wt_u)),
        rtol=1e-12,
        atol=1e-12,
    )

    # output should retain sharding
    _assert_named_sharding_spec(w_s, P("data"))
    _assert_named_sharding_spec(wt_s, P("data"))


def test_hubbard_hamiltonian_rejects_model_axis_sharding():
    mesh = make_data_model_mesh(2, 2)
    ham = HamHubbard(h1=jnp.eye(4, dtype=jnp.float64), u=4.0)

    with pytest.raises(ValueError, match="Cannot shard Hubbard Hamiltonian"):
        shard_ham_data(ham, mesh)


def test_shard_ham_data_pads_chol_to_model_divisible(capsys):
    mesh = make_data_model_mesh(2, 2)
    norb, n_chol = 4, 5
    chol = np.arange(n_chol * norb * norb, dtype=np.float64).reshape(n_chol, norb, norb)
    ham = HamChol(
        h0=jnp.array(0.25, dtype=jnp.float64),
        h1=jnp.eye(norb, dtype=jnp.float64),
        chol=jnp.asarray(chol),
        basis="restricted",
    )

    ham_s = shard_ham_data(ham, mesh)
    out = capsys.readouterr().out

    assert "padding chol from 5 to 6" in out
    _assert_named_sharding_spec(ham_s.h0, P())
    _assert_named_sharding_spec(ham_s.h1, P())
    _assert_named_sharding_spec(ham_s.chol, P("model"))
    assert ham_s.chol.shape == (6, norb, norb)

    chol_s = np.asarray(jax.device_get(ham_s.chol))
    np.testing.assert_allclose(chol_s[:n_chol], chol)
    np.testing.assert_allclose(chol_s[n_chol], 0.0, atol=0.0)


def test_shard_model_axis_pads_numpy_chol_without_mutating_source(capsys):
    mesh = make_data_model_mesh(2, 2)
    norb, n_chol = 4, 5
    chol = np.arange(n_chol * norb * norb, dtype=np.float64).reshape(n_chol, norb, norb).copy()

    chol_s = shard_model_axis(chol, mesh)
    out = capsys.readouterr().out

    assert "padding chol from 5 to 6" in out
    assert chol.shape == (n_chol, norb, norb)
    assert chol_s.shape == (6, norb, norb)
    np.testing.assert_allclose(np.asarray(jax.device_get(chol_s[:n_chol])), chol)
    np.testing.assert_allclose(np.asarray(jax.device_get(chol_s[n_chol])), 0.0, atol=0.0)


@pytest.mark.parametrize("n_model", [1, 2])
@pytest.mark.parametrize("trailing_shape", [(12,), (3, 4)])
def test_shard_model_axis_casts_host_chol_with_replication_and_padding(n_model, trailing_shape):
    mesh = make_data_model_mesh(4 // n_model, n_model)
    shape = (5, *trailing_shape)
    chol = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 7
    original = chol.copy()

    chol_s = shard_model_axis(chol, mesh, dtype=np.float32)
    expected = np.pad(chol, [(0, (-shape[0]) % n_model), *[(0, 0)] * len(trailing_shape)])
    expected = expected.astype(np.float32)

    assert chol_s.dtype == np.float32
    _assert_named_sharding_spec(chol_s, P("model"))
    np.testing.assert_array_equal(np.asarray(chol_s), expected)
    np.testing.assert_array_equal(chol, original)
    assert len(chol_s.addressable_shards) == 4
    if n_model == 1:
        assert chol_s.is_fully_replicated
        for shard in chol_s.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), expected)


@pytest.mark.parametrize("memory_mode", ["high", "low"])
def test_job_prepare_runtime_compacts_hf_chol_and_reuses_cached_ctx(
    tmp_path, memory_mode: RhfMeasMemoryMode
):
    mesh = make_data_model_mesh(2, 2)
    norb, nocc, n_chol = 6, 3, 5
    padded_n_chol = 6
    chol = np.arange(n_chol * norb * norb, dtype=np.float64).reshape(n_chol, norb, norb)
    staged = StagedInputs(
        ham=HamInput(
            h0=0.25,
            h1=np.eye(norb, dtype=np.float64),
            chol=chol,
            nelec=(nocc, nocc),
            norb=norb,
            chol_cut=1.0e-5,
            frozen=0,
            source_kind="mf",
            basis="restricted",
        ),
        trial=TrialInput(
            kind="rhf",
            data={"mo": np.eye(norb, dtype=np.float64)},
            frozen=0,
            source_kind="mf",
        ),
        meta={"source_kind": "mf", "chol_cut": 1.0e-5},
    )
    path = tmp_path / "runtime_compact.h5"
    dump(staged, path)

    sys_override = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    meas_ops_override = make_rhf_meas_ops(sys_override, memory_mode=memory_mode)
    # These deliberately nonsymmetric dummy factors require the explicit opt-out.
    job = setup(path, mesh=mesh, meas_ops=meas_ops_override,
                prop_kwargs={"packed_cholesky": False})
    assert job.mesh is mesh
    _assert_named_sharding_spec(job.ham_data.h0, P())
    _assert_named_sharding_spec(job.ham_data.h1, P())
    _assert_named_sharding_spec(job.ham_data.chol, P("model"))
    assert job.ham_data.chol.shape == (0, 0, 0)
    assert job.ham_data.nchol == padded_n_chol
    assert job.staged.ham.chol.shape == chol.shape
    assert isinstance(job.staged.ham.chol, np.ndarray)

    ham_ref = shard_ham_data(
        HamChol(
            h0=jnp.array(0.25, dtype=jnp.float64),
            h1=jnp.eye(norb, dtype=jnp.float64),
            chol=jnp.asarray(chol),
            basis="restricted",
        ),
        mesh,
    )
    trial_ref = RhfTrial(mo_coeff=jnp.eye(norb, dtype=jnp.float64)[:, :nocc])
    prop_ctx_ref = _build_prop_ctx(ham_ref, get_rdm1(trial_ref), dt=job.params.dt)
    meas_ctx_ref = build_rhf_meas_ctx(ham_ref, trial_ref, cfg=RhfMeasCfg(memory_mode=memory_mode))
    state_ref = init_prop_state(
        sys=job.sys,
        ham_data=ham_ref,
        trial_ops=job.trial_ops,
        trial_data=trial_ref,
        meas_ops=job.meas_ops,
        params=job.params,
        mesh=mesh,
    )

    state0, meas_ctx0, prop_ctx0 = job._prepare_runtime()
    meas_ctx0 = cast(RhfMeasCtx, meas_ctx0)
    prop_ctx0 = cast(CholAfqmcCtx, prop_ctx0)

    _assert_named_sharding_spec(job.ham_data.chol, P("model"))
    assert job.ham_data.chol.shape == (0, 0, 0)
    assert job.ham_data.nchol == padded_n_chol
    _assert_named_sharding_spec(prop_ctx0.chol_flat, P("model"))
    _assert_named_sharding_spec(meas_ctx0.rot_chol, P("model"))
    assert prop_ctx0.chol_flat.shape[0] == padded_n_chol
    assert meas_ctx0.rot_chol.shape[0] == padded_n_chol
    assert meas_ctx0.cfg.memory_mode == memory_mode
    np.testing.assert_allclose(
        np.asarray(jax.device_get(prop_ctx0.mf_shifts)),
        np.asarray(jax.device_get(prop_ctx_ref.mf_shifts)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(prop_ctx0.exp_h1_half)),
        np.asarray(jax.device_get(prop_ctx_ref.exp_h1_half)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(prop_ctx0.chol_flat)),
        np.asarray(jax.device_get(prop_ctx_ref.chol_flat)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(meas_ctx0.rot_h1)),
        np.asarray(jax.device_get(meas_ctx_ref.rot_h1)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(meas_ctx0.rot_chol)),
        np.asarray(jax.device_get(meas_ctx_ref.rot_chol)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(meas_ctx0.rot_chol_flat)),
        np.asarray(jax.device_get(meas_ctx_ref.rot_chol_flat)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(state0.weights)),
        np.asarray(jax.device_get(state_ref.weights)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(state0.overlaps)),
        np.asarray(jax.device_get(state_ref.overlaps)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(state0.e_estimate)),
        np.asarray(jax.device_get(state_ref.e_estimate)),
        rtol=1e-12,
        atol=1e-12,
    )

    state1, meas_ctx1, prop_ctx1 = job._prepare_runtime()
    assert state1 is state0
    assert meas_ctx1 is meas_ctx0
    assert prop_ctx1 is prop_ctx0


if __name__ == "__main__":
    test_sr_sharded_matches_unsharded(n_per_dev=4)
    # test_block_runs_under_sharding(n_per_dev=4)
