from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import driver, testing
from trot.meas.rhf import make_rhf_meas_ops
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks import (
    load_all_prop_states_npz,
    load_prop_state_npz,
    make_block_state_logger,
)
from trot.prop.types import QmcParams
from trot.trial.rhf import RhfTrial, make_rhf_trial_ops


def _make_random_rhf_trial(key, norb, nocc):
    return RhfTrial(mo_coeff=testing.rand_orthonormal_cols(key, norb, nocc))


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype")
def test_block_state_logger_writes_each_block(tmp_path):
    key = jax.random.PRNGKey(0)
    norb = 4
    nocc = 2
    n_chol = 5
    walker_kind = "restricted"

    (
        sys,
        ham_data,
        trial_data,
        _meas_manual,
        _ctx_manual,
        _meas_auto,
        _ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_random_rhf_trial,
        make_trial_fn_kwargs=dict(norb=norb, nocc=nocc),
        make_trial_ops_fn=make_rhf_trial_ops,
        make_meas_ops_fn=make_rhf_meas_ops,
    )

    trial_ops = make_rhf_trial_ops(sys)
    meas_ops = make_rhf_meas_ops(sys)
    prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
    params = QmcParams(
        n_eql_blocks=2,
        n_blocks=3,
        n_walkers=6,
        n_prop_steps=2,
        n_chunks=1,
        dt=0.01,
        seed=11,
    )

    out = driver.run_qmc(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_data=trial_data,
        meas_ops=meas_ops,
        trial_ops=trial_ops,
        prop_ops=prop_ops,
        block_fn=make_block_state_logger(tmp_path),
        sample_dir=tmp_path,
    )

    files = sorted(tmp_path.glob("block_state_*.npz"))
    assert len(files) == params.n_eql_blocks + params.n_blocks

    state0 = load_prop_state_npz(files[0])
    assert isinstance(state0.walkers, jax.Array)
    assert state0.walkers.shape == (params.n_walkers, norb, nocc)
    assert state0.weights.shape == (params.n_walkers,)
    assert state0.overlaps.shape == (params.n_walkers,)

    state_last = load_prop_state_npz(files[-1])
    assert state_last.walkers.shape == (params.n_walkers, norb, nocc)
    all_states = load_all_prop_states_npz(tmp_path)
    assert len(all_states) == len(files)
    assert jnp.isfinite(out.mean_energy)

    archives = list(tmp_path.glob("samples_*.npz"))
    assert len(archives) == 1
    with np.load(archives[0], allow_pickle=False) as saved:
        assert saved["energies"].shape == (params.n_blocks,)
        assert saved["weights"].shape == (params.n_blocks,)
        assert saved["n_blocks_completed"] == params.n_blocks
        assert saved["sampling_complete"]
        np.testing.assert_allclose(saved["energies"], out.block_energies[-params.n_blocks:])
