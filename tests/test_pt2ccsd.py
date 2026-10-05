from trot import config

config.configure_once()

import jax.numpy as jnp
import pytest
from pyscf import cc, gto, scf

from trot.driver_mixed import run_mixed_qmc
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import (
    Pt2ccsdMeasCfg,
    build_meas_ctx,
    combine_first_order_energy,
    make_pt2ccsd_meas_ops,
)
from trot.meas.rhf import make_rhf_meas_ops
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks_mixed import block_mixed
from trot.prop.types import QmcParams
from trot.staging import StagedMfOrCc, _stage_pt2ccsd_input, stage
from trot.core.system import System
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot.trial.rhf import RhfTrial, make_rhf_trial_ops

# ---------------------------------------------------------------------------
# Module-level fixtures — built once for the whole test file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h8_system():
    """8 H2 molecules (nc=8, na=2) in sto-6g, well-separated clusters.

    Returns a dict containing the pyscf objects and all pre-built TROT
    inputs so each parametrised test can reuse them without re-running SCF/CC.
    """
    a = 2  # bond length inside each H2 dimer (Bohr)
    d = 100  # centre-to-centre distance between dimers (Bohr)
    unit = "b"
    na, nc = 2, 8
    elmt, basis = "H", "sto6g"

    atoms = ""
    for n in range(nc * na):
        shift = ((n - n % na) // na) * (d - a)
        atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

    mol = gto.M(atom=atoms, basis=basis, unit=unit, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.kernel()

    # Stage the guide (RHF) Hamiltonian
    staged_guide = stage(mycc._scf)
    ham = staged_guide.ham
    sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind="restricted")
    ham_data = HamChol(
        jnp.asarray(ham.h0),
        jnp.asarray(ham.h1),
        jnp.asarray(ham.chol),
        basis=ham.basis,
    )

    # Guide wavefunction (RHF MOs)
    guide_data = RhfTrial(
        mo_coeff=jnp.array(staged_guide.trial.data["mo"][:, : sys.nup]),
    )

    # Trial wavefunction (PT2-CCSD)
    trial_obj = StagedMfOrCc(mycc, frozen=None)
    staged_trial = _stage_pt2ccsd_input(trial_obj)
    trial_data = Pt2ccsdTrial(
        mo_t=jnp.array(staged_trial.data["mo_t"]),
        t2=jnp.array(staged_trial.data["t2"]),
    )

    # Measurement context (built once; seed-independent)
    trial_cfg = Pt2ccsdMeasCfg(memory_mode="low")
    trial_meas_ctx = build_meas_ctx(ham_data, trial_data, trial_cfg)

    # Operator factories
    guide_ops = make_rhf_trial_ops(sys)
    guide_meas_ops = make_rhf_meas_ops(sys)
    guide_prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
    trial_meas_ops = make_pt2ccsd_meas_ops(sys, mixed_precision=False)

    return dict(
        mycc=mycc,
        sys=sys,
        ham_data=ham_data,
        guide_data=guide_data,
        trial_data=trial_data,
        trial_meas_ctx=trial_meas_ctx,
        guide_ops=guide_ops,
        guide_meas_ops=guide_meas_ops,
        guide_prop_ops=guide_prop_ops,
        trial_meas_ops=trial_meas_ops,
    )


# ---------------------------------------------------------------------------
# Reference energies (dt=0.005, n_walkers=1, n_prop_steps=1, n_blocks=50,
# n_eql_blocks=1) - generated once from a known-good code version.
# Update only after a deliberate algorithmic change; never update silently.
# The errors of that version came from a delta-method estimate; the mixed driver now
# reports the jackknife blocking error of the component ratios, so only the energies
# (the same weighted component ratios) are compared.
# ---------------------------------------------------------------------------

_REFERENCES = {
    1: -8.771210912150202,
    2: -8.769501242545516,
    3: -8.769769708305297,
    4: -8.771395845249032,
}


# ---------------------------------------------------------------------------
# Parametrised test — one run per seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pt2ccsd_energy_matches_reference(h8_system, seed):
    """Run PT2-CCSD AFQMC with a fixed seed and check against a reference value.

    QmcParams: dt=0.005, n_walkers=1, n_prop_steps=1, n_blocks=100, n_eql_blocks=5.
    Reference energies and errors were obtained from a known-good run and are
    used as a regression tripwire — any change to propagation, trial, or
    measurement that alters the trajectory will fail this test.
    """
    s = h8_system
    e_ref = _REFERENCES[seed]

    params = QmcParams(
        dt=0.005,
        n_walkers=1,
        n_prop_steps=1,
        n_blocks=50,
        n_eql_blocks=1,
        seed=seed,
    )

    result = run_mixed_qmc(
        sys=s["sys"],
        params=params,
        ham_data=s["ham_data"],
        guide_data=s["guide_data"],
        guide_ops=s["guide_ops"],
        guide_prop_ops=s["guide_prop_ops"],
        guide_meas_ops=s["guide_meas_ops"],
        trial_data=s["trial_data"],
        trial_meas_ops=s["trial_meas_ops"],
        mix_block_fn=block_mixed,
        energy_fn=combine_first_order_energy,
    )

    e_mean = float(result.trial_mean_energy)
    e_err = float(result.trial_stderr_energy)

    print(f"seed={seed}  E={e_mean:.6f}  err={e_err:.6f}  ref={e_ref:.6f}")

    # Reference values are stored to 6 decimal places; match to that precision
    assert jnp.isclose(jnp.array(e_mean), jnp.array(e_ref), atol=1e-6), (
        f"seed={seed}: energy {e_mean:.6f} != reference {e_ref:.6f} " f"(diff={e_mean - e_ref:.2e})"
    )
    assert jnp.isfinite(e_err)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
