import pytest
import numpy as np
import dataclasses
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcFp
import trot.trial.ccsd

import jax.numpy as jnp

e_ref = -76.1195848162


def test_ucisd_uccsd_fp():
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

    af = AfqmcFp(mycc)
    af.n_walkers = 20
    af.ene0 = mycc.e_tot
    af.seed = 5
    af.n_blocks = 1
    af.n_prop_steps = 100
    af.walker_kind = "restricted"
    af.build_job()

    job = af._job

    nmo = mf.mo_coeff.shape[-1]
    trial_coeff = np.identity(nmo)

    # Replace the function creating the initial state
    job.prop_ops = dataclasses.replace(
        job.prop_ops,
        init_prop_state=trot.trial.ccsd.make_init_prop_state(
            trial_coeff,
            mycc.t1,  # type: ignore
            mycc.t2,  # type: ignore
        ),
    )

    e, err = af.kernel()

    # Unless the SCF is kept fix the results will vary
    assert jnp.isclose(e[-1].real, e_ref, atol=1e-3), (e[-1].real, e_ref)


if __name__ == "__main__":
    pytest.main([__file__])
