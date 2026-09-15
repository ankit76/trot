from trot import config

config.configure_once()

import io
import contextlib

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.driver import run_mixed_qmc
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import Pt2ccsdMeasCfg, build_meas_ctx, make_pt2ccsd_meas_ops
from trot.meas.rhf import make_rhf_meas_ops
from trot.mixed import available_mixed_recipes, get_mixed_recipe
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks import block_mixed
from trot.prop.types import QmcParams
from trot.staging import StagedMfOrCc, _stage_pt2ccsd_input, stage, stage_pt2ccsd_trial
from trot.stat_utils import pt2ccsd_blocking
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
    staged_trial = stage_pt2ccsd_trial(mycc)
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
        mf=mycc._scf,
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


# QmcParams shared by the reference test and the AfqmcMixed equivalence test.
# n_walkers=1 and n_prop_steps=1 make this a trajectory tripwire rather than a
# physically converged run: the samples are highly correlated and the quoted errors are
# artificially small. That is deliberate — it is cheap and extremely sensitive.
_DT = 0.005
_N_WALKERS = 1
_N_PROP_STEPS = 1
_N_BLOCKS = 20
_N_EQL_BLOCKS = 1
_PARAMS = dict(
    dt=_DT,
    n_walkers=_N_WALKERS,
    n_prop_steps=_N_PROP_STEPS,
    n_blocks=_N_BLOCKS,
    n_eql_blocks=_N_EQL_BLOCKS,
)


# ---------------------------------------------------------------------------
# Reference values for _PARAMS above, generated from a known-good code version.
# Update only after a deliberate algorithmic change; never update silently.
# ---------------------------------------------------------------------------

_REFERENCES = {
    1: (-8.769572413693957, 0.0002376533144792639),
    2: (-8.769077722260674, 0.00013848610297310015),
    3: (-8.769007694906517, 9.652256238191887e-05),
    4: (-8.769284062467616, 0.00018420402974092173),
}


def _run_manual(s, seed, **overrides):
    """The manual run_mixed_qmc setup, as in examples/pt2ccsd.py."""
    params = QmcParams(seed=seed, **{**_PARAMS, **overrides})
    with contextlib.redirect_stdout(io.StringIO()):
        return run_mixed_qmc(
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
        )


# ---------------------------------------------------------------------------
# Parametrised test — one run per seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pt2ccsd_energy_matches_reference(h8_system, seed):
    """Run PT2-CCSD AFQMC with a fixed seed and check against a reference value.

    Reference energies and errors were obtained from a known-good run and are
    used as a regression tripwire — any change to propagation, trial, or
    measurement that alters the trajectory will fail this test.
    """
    e_ref, err_ref = _REFERENCES[seed]
    result = _run_manual(h8_system, seed)

    e_mean = float(result.trial_mean_energy.real)
    e_err = float(result.trial_stderr_energy.real)

    print(f"seed={seed}  E={e_mean:.6f}  err={e_err:.6f}  ref={e_ref:.6f} +/- {err_ref:.6f}")

    # Reference values are stored to 6 decimal places; match to that precision
    assert jnp.isclose(jnp.array(e_mean), jnp.array(e_ref), atol=1e-6), (
        f"seed={seed}: energy {e_mean:.6f} != reference {e_ref:.6f} " f"(diff={e_mean - e_ref:.2e})"
    )
    assert jnp.isclose(jnp.array(e_err), jnp.array(err_ref), atol=1e-6), (
        f"seed={seed}: error {e_err:.6f} != reference {err_ref:.6f} "
        f"(diff={e_err - err_ref:.2e})"
    )


# ---------------------------------------------------------------------------
# The AfqmcMixed wrapper must reorder the setup without changing any math
# ---------------------------------------------------------------------------


def test_afqmc_mixed_matches_manual_setup(h8_system):
    """AfqmcMixed must reproduce the manual run_mixed_qmc numbers exactly.

    The wrapper only assembles the same pieces in a different order, so anything
    other than bit-identical means it is building a different calculation.
    """
    s = h8_system
    seed = 1
    ref = _run_manual(s, seed)

    with contextlib.redirect_stdout(io.StringIO()):
        af = AfqmcMixed(
            s["mycc"],
            seed=seed,
            dt=_DT,
            n_walkers=_N_WALKERS,
            n_prop_steps=_N_PROP_STEPS,
            n_blocks=_N_BLOCKS,
            n_eql_blocks=_N_EQL_BLOCKS,
        )
        e_tot, e_err = af.kernel()

    assert e_tot == pytest.approx(float(ref.trial_mean_energy.real), abs=1e-12)
    assert e_err == pytest.approx(float(ref.trial_stderr_energy.real), abs=1e-12)
    assert af.guide_e_tot == pytest.approx(float(ref.guide_mean_energy.real), abs=1e-12)
    assert af.guide_e_err == pytest.approx(float(ref.guide_stderr_energy.real), abs=1e-12)


def test_initial_energies_match_rhf_and_ccsd(h8_system):
    """
    At tau = 0 the walker is the guide determinant, so with no sampling involved the
    guide estimator must give RHF and the trial estimator must give CCSD, up to the
    cholesky truncation. This is the check the tau = 0 row of a run prints.
    """
    from trot.core.ops import k_energy
    from trot.meas.pt2ccsd import energy_kernel_rw_rh

    s = h8_system
    walker = jnp.asarray(s["guide_data"].mo_coeff) + 0.0j

    guide_meas_ctx = s["guide_meas_ops"].build_meas_ctx(s["ham_data"], s["guide_data"])
    guide_e = s["guide_meas_ops"].require_kernel(k_energy)(
        walker, s["ham_data"], guide_meas_ctx, s["guide_data"]
    )
    assert float(guide_e.real) == pytest.approx(float(s["mf"].e_tot), abs=1e-6)

    t2, e0, e1 = energy_kernel_rw_rh(walker, s["ham_data"], s["trial_meas_ctx"], s["trial_data"])
    trial_e = (s["ham_data"].h0 + e0 + e1 - t2 * e0).real
    assert float(trial_e) == pytest.approx(float(s["mycc"].e_tot), abs=1e-6)


# ---------------------------------------------------------------------------
# Recipe registry
# ---------------------------------------------------------------------------


def test_pt2ccsd_recipe_selects_its_own_blocking():
    """
    Choosing the trial chooses the whole pipeline, so a trial can never be paired with
    the wrong estimator.
    """
    rec = get_mixed_recipe("pt2ccsd")
    assert rec.guide == "rhf"
    assert rec.trial == "pt2ccsd"
    assert rec.walker_kind == "restricted"
    assert rec.mixed_block_fn is block_mixed
    assert rec.blocking_fn is pt2ccsd_blocking

    with pytest.raises(ValueError, match="unknown mixed recipe"):
        get_mixed_recipe("cisd")
    assert ("rhf", "pt2ccsd") in available_mixed_recipes()


def test_public_staging_matches_private(h8_system):
    """stage_pt2ccsd_trial is the public form of _stage_pt2ccsd_input."""
    mycc = h8_system["mycc"]
    new = stage_pt2ccsd_trial(mycc)
    old = _stage_pt2ccsd_input(StagedMfOrCc(mycc, frozen=None))
    assert new.kind == old.kind == "pt2ccsd"
    np.testing.assert_allclose(np.asarray(new.data["mo_t"]), np.asarray(old.data["mo_t"]))
    np.testing.assert_allclose(np.asarray(new.data["t2"]), np.asarray(old.data["t2"]))


# ---------------------------------------------------------------------------
# Blocking analysis: the `final` switch and the too-few-samples contract
# ---------------------------------------------------------------------------


def _synthetic_samples(n, seed=0):
    rng = np.random.default_rng(seed)
    w = jnp.asarray(np.abs(rng.normal(1.0, 0.05, n)))
    return (
        w,
        jnp.asarray(rng.normal(0.1, 0.01, n)),
        jnp.asarray(rng.normal(-1.0, 0.02, n)),
        jnp.asarray(rng.normal(-0.3, 0.02, n)),
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 10, 40])
def test_pt2ccsd_blocking_returns_none_instead_of_raising(n):
    """
    Too few samples must return None rather than blowing up on an empty reduction.
    final=False needs two samples; final=True needs enough for min_blocks blocks.
    """
    w, t2, e0, e1 = _synthetic_samples(max(n, 1))
    if n == 0:
        w = t2 = e0 = e1 = jnp.zeros(0)

    final = pt2ccsd_blocking(0.5, w, t2, e0, e1, final=True)
    cheap = pt2ccsd_blocking(0.5, w, t2, e0, e1, final=False)

    assert (cheap is None) == (n < 2)
    assert (final is None) == (n < 5)  # min_blocks defaults to 5

    if final is not None and cheap is not None:
        # the two paths differ only in how the error is estimated
        assert float(final[0]) == pytest.approx(float(cheap[0]), abs=1e-12)


def test_pt2ccsd_blocking_delta_method_matches_reference():
    """final=False uses a first-order (delta method) error; check it against the
    closed-form expression it is derived from."""
    w, t2, e0, e1 = _synthetic_samples(40, seed=3)
    got = pt2ccsd_blocking(0.5, w, t2, e0, e1, final=False)
    assert got is not None

    wn, t2n, e0n, e1n = (np.asarray(x) for x in (w, t2, e0, e1))
    agg_e0, agg_e1 = (wn * e0n).sum(), (wn * e1n).sum()
    agg_t2, agg_w = (wn * t2n).sum(), wn.sum()
    infl = (
        (1 / agg_w - agg_t2 / agg_w**2) * (wn * e0n)
        + (1 / agg_w) * (wn * e1n)
        + (-agg_e0 / agg_w**2) * (wn * t2n)
        + (-agg_e0 / agg_w**2 - agg_e1 / agg_w**2 + 2 * agg_t2 * agg_e0 / agg_w**3) * wn
    )
    ref = np.sqrt((infl**2).sum() * len(wn) / (len(wn) - 1))
    assert float(got[1]) == pytest.approx(ref, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
