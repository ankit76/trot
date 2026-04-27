from trot import config

config.configure_once()

import pytest
import jax
import jax.numpy as jnp

from pyscf import cc, gto, scf

import trot.setup
import trot.trial.uccsd
from trot.trial.uhf import overlap_u
from trot.meas.uhf import energy_kernel_uw_rh


def test_uccsd_walkers():
    mol = gto.M(
        atom="""
        N  -1.67119571   -1.44021737    0.00000000
        H  -2.12619571   -0.65213425    0.00000000
        H  -0.76119571   -1.44021737    0.00000000
        """,
        spin=1,
        basis="6-31g",
        verbose=3,
    )

    mf = scf.UHF(mol)
    mf.kernel()

    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability()

    mycc = cc.UCCSD(mf)
    mycc.kernel()

    job = trot.setup.setup(mf)
    job._prepare_runtime()

    trial_coeff = (
        job.staged.trial.data["mo_a"],
        job.staged.trial.data["mo_b"],
    )
    ham_data = job.ham_data
    trial_data = job.trial_data
    meas_ctx = job._runtime_meas_ctx
    key = jax.random.key(42)
    n_walkers = 20000
    hs_op = trot.trial.uccsd.build_hs_op(mycc.t2)
    w = trot.trial.uccsd.init_walkers(trial_coeff, mycc.t1, hs_op, key, n_walkers)

    o = jax.vmap(overlap_u, in_axes=(0, None))(w, trial_data)  # type: ignore
    e = jax.vmap(energy_kernel_uw_rh, in_axes=(0, None, None, None))(
        w, ham_data, meas_ctx, trial_data  # type: ignore
    )

    energy = jnp.sum(e * o) / jnp.sum(o)
    assert jnp.isclose(energy.real, mycc.e_tot, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
