from trot import config

config.configure_once()

import pytest
import jax
import jax.numpy as jnp

from pyscf import cc, gto, scf

import trot.setup
import trot.trial.ccsd
from trot.trial.rhf import overlap_r
from trot.meas.rhf import energy_kernel_rw_rh


def test_uccsd_walkers():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="6-31g",
        verbose=3,
    )

    mf = scf.RHF(mol)
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.kernel()

    job = trot.setup.setup(mf)
    job._prepare_runtime()

    trial_coeff = job.staged.trial.data["mo"]
    ham_data = job.ham_data
    trial_data = job.trial_data
    meas_ctx = job._runtime_meas_ctx
    key = jax.random.key(42)
    n_walkers = 20000
    hs_op = trot.trial.ccsd.build_hs_op(mycc.t2)  # type: ignore
    w = trot.trial.ccsd.init_walkers(trial_coeff, mycc.t1, hs_op, key, n_walkers)  # type: ignore

    o = jax.vmap(overlap_r, in_axes=(0, None))(w, trial_data)  # type: ignore
    e = jax.vmap(energy_kernel_rw_rh, in_axes=(0, None, None, None))(
        w, ham_data, meas_ctx, trial_data  # type: ignore
    )

    energy = jnp.sum(e * o) / jnp.sum(o)
    assert jnp.isclose(energy.real, mycc.e_tot, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
