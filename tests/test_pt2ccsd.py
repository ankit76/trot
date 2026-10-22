from trot import config

config.configure_once()

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.driver import run_mixed_qmc
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import Pt2ccsdMeasCfg, build_meas_ctx, make_pt2ccsd_meas_ops
from trot.meas.rhf import make_rhf_meas_ops
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks import block_mixed
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
# Reference values (dt=0.005, n_walkers=1, n_prop_steps=1, n_blocks=100,
# n_eql_blocks=5) — generated once from a known-good code version.
# Update only after a deliberate algorithmic change; never update silently.
# ---------------------------------------------------------------------------

_REFERENCES = {
    1: (-8.771210912150202, 0.0002802273203235753),  # the errors are artifially small
    2: (-8.769501242545516, 0.0001545791982688015),  # because the prop_steps are small
    3: (-8.769769708305297, 0.0001204835908861157),  # the samples are hight correlated
    4: (-8.771395845249032, 0.0004370850562308357),  # just for testing
}


# ---------------------------------------------------------------------------
# Parametrised test — one run per seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pt2ccsd_energy_matches_reference(h8_system, seed, tmp_path):
    """Run PT2-CCSD AFQMC with a fixed seed and check against a reference value.

    QmcParams: dt=0.005, n_walkers=1, n_prop_steps=1, n_blocks=100, n_eql_blocks=5.
    Reference energies and errors were obtained from a known-good run and are
    used as a regression tripwire — any change to propagation, trial, or
    measurement that alters the trajectory will fail this test.
    """
    s = h8_system
    e_ref, err_ref = _REFERENCES[seed]

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
        sample_dir=tmp_path,
    )

    e_mean = float(result.trial_mean_energy.real)
    e_err = float(result.trial_stderr_energy.real)

    archives = list(tmp_path.glob("samples_*.npz"))
    assert len(archives) == 1
    with np.load(archives[0], allow_pickle=False) as saved:
        assert saved["sampling_complete"]
        assert saved["n_blocks_completed"] == params.n_blocks
        for key, values in (
            ("weights", result.trial_block_weights),
            ("t2", result.trial_block_t2s),
            ("e0", result.trial_block_e0s),
            ("e1", result.trial_block_e1s),
        ):
            np.testing.assert_array_equal(saved[key], values[-params.n_blocks:])

    print(f"seed={seed}  E={e_mean:.6f}  err={e_err:.6f}  ref={e_ref:.6f} +/- {err_ref:.6f}")

    # Reference values are stored to 6 decimal places; match to that precision
    assert jnp.isclose(jnp.array(e_mean), jnp.array(e_ref), atol=1e-6), (
        f"seed={seed}: energy {e_mean:.6f} != reference {e_ref:.6f} " f"(diff={e_mean - e_ref:.2e})"
    )
    assert jnp.isclose(jnp.array(e_err), jnp.array(err_ref), atol=1e-6), (
        f"seed={seed}: error {e_err:.6f} != reference {err_ref:.6f} "
        f"(diff={e_err - err_ref:.2e})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
