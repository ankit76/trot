"""
Tests of trot.lnoafqmc on O2 (sto-3g, density fitted), two atomic fragments.

    pytest tests/test_lnoafqmc.py -q                 # fast checks
    pytest tests/test_lnoafqmc.py -q --run-slow      # + the end-to-end LNO-AFQMC runs

With a tight LNO threshold each fragment's local active space is the whole active space,
so the two fragments' projectors add up to the identity and their energies must add up
to full-space values: exactly at tau = 0 (the CCSD energy), within error bars after
sampling (the AFQMC/pt2CCSD energy of AfqmcMixed with the pt2ccsd_bar trial). Everything
below the propagator is deterministic and checked to machine precision.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import contextlib
import io
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf
from pyscf.data import elements

import trot.lnoafqmc  # noqa: F401  (allocator, before jax)
from trot import config

config.configure_once()

from trot.core.system import System
from trot.ham.chol import HamChol
from trot.lnoafqmc import LnoAfqmcMixed, LnoFragMixed, iao_fragment
from trot.lnoafqmc import integral as li
from trot.lnoafqmc import las, pipeline, solvers
from trot.lnoafqmc import staging as lst
from trot.lnoafqmc.meas import pt2ccsd_bar as lm
from trot.lnoafqmc.meas import pt2ccsd_fast as lf
from trot.lnoafqmc.mixed import available_mixed_recipes, get_mixed_recipe
from trot.lnoafqmc.trial.pt2ccsd import make_pt2ccsd_trial_data, overlap_r
from trot.lnoafqmc.trial.pt2ccsd_fast import make_pt2ccsd_fast_trial_data
from trot.meas.pt2ccsd_bar import build_meas_ctx as trot_build_ctx
from trot.meas.pt2ccsd_bar import energy_kernel_rw_rh_bar as trot_bar_kernel
from trot.meas.pt2ccsd_chunking import make_chunk_meas_cfg
from trot.stat_utils import blocking_analysis_components, component_estimator_outlier_mask
from trot.trial.pt2ccsd_bar import Pt2ccsdTrial as TrotPt2ccsdTrial

CHOL_CUT = 1e-6


@pytest.fixture(scope="module")
def o2():
    # max_memory: pyscf-forge budgets max_memory - RSS, which goes negative in a long session
    mol = gto.M(atom="O 0 0 0; O 0 0 1.208", basis="sto-3g", spin=0, verbose=0, max_memory=16000)
    mf: Any = scf.RHF(mol).density_fit()
    mf.kernel()
    nfrozen = int(elements.chemcore(mol))
    lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="atom")
    mlno = solvers.get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, 1e-12)
    mlno.verbose = 0
    eris = mlno.ao2mo()
    nfrag = len(frag_list)
    frags = [
        pipeline.cpu_stage(
            mlno,
            mf,
            lo_coeff,
            frag_list[i],
            mlno.lno_thresh,
            [None, None],
            [[None, None]] * nfrag,
            ["1h", "1h"],
            eris,
            i,
            i,
            frag_name[i],
            True,
            True,
            nfrozen,
        )
        for i in range(nfrag)
    ]
    mycc: Any = cc.CCSD(mf, frozen=nfrozen)
    mycc.kernel()
    return dict(
        mol=mol,
        mf=mf,
        nfrozen=nfrozen,
        lo_coeff=lo_coeff,
        frag_list=frag_list,
        frag_name=frag_name,
        frags=frags,
        e_ccsd=float(mycc.e_corr),
    )


@pytest.fixture(scope="module")
def frag0(o2):
    """Fragment 0's hamiltonian, trial and meas ctx on the device, in double precision."""
    mf, frag = o2["mf"], o2["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        ham = li.build_ham_lno_df(mf, frag.lno_coeff, frag.lno_frozen, chol_cut=CHOL_CUT)
    sys_ = System(norb=ham.norb, nelec=ham.nelec, walker_kind="restricted")
    ham_data = HamChol(
        jnp.asarray(ham.h0), jnp.asarray(ham.h1), jnp.asarray(ham.chol), basis="restricted"
    )
    tin = lst.stage_pt2ccsd_trial(frag)
    trial_data = make_pt2ccsd_trial_data(tin.data, sys_)
    ops = lm.make_pt2ccsd_meas_ops(sys_, mixed_precision=False, testing=True, nchol_chunk=3)
    ctx = ops.build_meas_ctx(ham_data, trial_data)
    t2_full = np.asarray(frag.t2).transpose(0, 2, 1, 3)  # unprojected, (i,a,j,b)
    return dict(
        ham=ham,
        sys=sys_,
        ham_data=ham_data,
        tin=tin,
        trial_data=trial_data,
        ops=ops,
        ctx=ctx,
        t2_full=t2_full,
    )


def _random_walkers(norb, nocc, n, seed=7):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield jnp.asarray(
            np.linalg.qr(rng.normal(size=(norb, nocc)) + 0.3j * rng.normal(size=(norb, nocc)))[0]
        )


def _e_hf_el(ham) -> float:
    nocc = ham.nelec[0]
    lo = ham.chol[:, :nocc, :nocc]
    e = 2 * np.trace(ham.h1[:nocc, :nocc]) + 2 * np.sum(np.trace(lo, axis1=1, axis2=2) ** 2)
    return float(e - np.einsum("gij,gji->", lo, lo))


# ----------------------------------------------------------------------------- CPU stage


def test_fragments_sum_to_canonical_mp2_ccsd(o2):
    frags = o2["frags"]
    assert o2["frag_name"] == ["O0", "O1"]
    assert all(f.nact == 8 for f in frags)
    e_mp = sum(f.efrag_mp for f in frags)
    e_cc = sum(f.efrag_cc for f in frags)
    from pyscf import mp

    mmp: Any = mp.MP2(o2["mf"], frozen=o2["nfrozen"])
    mmp.kernel()
    assert abs(e_mp - mmp.e_corr) < 1e-8
    assert abs(e_cc - o2["e_ccsd"]) < 1e-6


def test_check_span_passes_and_fails(o2):
    las.check_span(o2["mf"], o2["lo_coeff"], o2["nfrozen"])
    with pytest.raises(ValueError):
        las.check_span(o2["mf"], o2["lo_coeff"][:, :3], o2["nfrozen"])


# ----------------------------------------------------------------------------- integrals


def test_fragment_hamiltonian_reproduces_hf_energy(frag0, o2):
    """The full-space LAS with the DF-exact core: E_HF of the fragment ham is mf.e_tot."""
    ham = frag0["ham"]
    assert abs(ham.h0 + _e_hf_el(ham) - o2["mf"].e_tot) < 1e-6
    assert ham.norb == 8 and ham.nelec == (6, 6)


def test_restricted_core_matches_unrestricted_route(o2):
    """The restricted effective core (2J - K) is the unrestricted one at D^a = D^b = D."""
    mf, frag = o2["mf"], o2["frags"][0]
    ncore, _, ncas, actfrag = li.get_las_idx(mf, frag.lno_frozen)
    assert ncore > 0
    core = np.asarray(frag.lno_coeff)[:, :ncore]
    act = np.asarray(frag.lno_coeff)[:, actfrag]
    e_r, h1_r = li.lno_effective_core_r(mf, core, act)
    e_u, (h1_a, h1_b) = li.lno_effective_core_u(mf, (core, core), (act, act))
    assert abs(e_r - e_u) < 1e-10
    assert np.abs(h1_r - h1_a).max() < 1e-12 and np.abs(h1_r - h1_b).max() < 1e-12
    assert h1_r.shape == (ncas, ncas)
    e_d, h1_d = li.lno_effective_core(mf, core, act)
    assert e_d == e_r and np.array_equal(h1_d, h1_r)
    # the DF J and K on the device agree with pyscf's DF get_jk
    dm = core @ core.T
    vj, vk = mf.get_jk(mf.mol, dm, hermi=1)
    assert np.abs(li.core_rveff(mf, dm) - (2 * vj - vk)).max() < 1e-9


def test_las_ordering_is_asserted(o2):
    with pytest.raises(ValueError):
        li.get_las_idx(o2["mf"], np.array([0, 3]))


# ----------------------------------------------------------------------------- kernel


def test_kernel_prjlo_one_limit_matches_bar_kernel(frag0):
    """With the identity projector the fragment kernel is the branch's pt2ccsd_bar kernel:
    t2frg = theta, e1frg = h_t, e0 = electronic_0, and e0frg = electronic_0 - E_HF."""
    sys_, ham_data, tin, ops = frag0["sys"], frag0["ham_data"], frag0["tin"], frag0["ops"]
    nocc, norb = sys_.nup, sys_.norb
    t2_full = frag0["t2_full"]
    d = dict(tin.data)
    d["prjlo"] = np.eye(nocc)
    d["t2"] = t2_full
    td = make_pt2ccsd_trial_data(d, sys_)
    ctx = ops.build_meas_ctx(ham_data, td)
    ttrial = TrotPt2ccsdTrial(mo_t=td.mo_t, t2=jnp.asarray(t2_full))
    cfg = make_chunk_meas_cfg(mixed_precision=False, testing=True, nchol_chunk=3)
    tctx = trot_build_ctx(ham_data, ttrial, cfg)
    e_hf = _e_hf_el(frag0["ham"])
    for w in _random_walkers(norb, nocc, 4):
        out = np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, ctx, td))
        ref = np.asarray(trot_bar_kernel(w, ham_data, tctx, ttrial))  # [theta, e0, h_t]
        assert abs(out[0] - ref[0]) < 1e-9
        assert abs(out[3] - ref[1]) < 1e-9
        assert abs(out[2] - ref[2]) < 1e-9
        assert abs(out[1] - (ref[1] - e_hf)) < 1e-9


def test_kernel_partition_of_unity(frag0, o2):
    """The fragment projectors of O0 and O1 add up to 1: so do the fragment kernels."""
    sys_, ham_data, tin, ops = frag0["sys"], frag0["ham_data"], frag0["tin"], frag0["ops"]
    mf, frag, lo_coeff, frag_list = o2["mf"], o2["frags"][0], o2["lo_coeff"], o2["frag_list"]
    nocc, norb = sys_.nup, sys_.norb
    nocc_full = int(np.count_nonzero(mf.mo_occ))
    ncore = int(np.sum(np.asarray(frag.lno_frozen) < nocc_full))  # frozen occupied count
    s1e = mf.get_ovlp()
    actocc = frag.lno_coeff[:, ncore : ncore + nocc]
    prj = []
    for f in range(len(frag_list)):
        u = actocc.T @ s1e @ lo_coeff[:, frag_list[f]]
        prj.append(u @ u.T)
    assert np.abs(sum(prj) - np.eye(nocc)).max() < 1e-10

    t2_full = frag0["t2_full"]
    tds, ctxs = [], []
    for p in prj + [np.eye(nocc)]:
        d = dict(tin.data)
        d["prjlo"] = p
        d["t2"] = np.einsum("iajb,ik->kajb", t2_full, p)
        td = make_pt2ccsd_trial_data(d, sys_)
        tds.append(td)
        ctxs.append(ops.build_meas_ctx(ham_data, td))
    for w in _random_walkers(norb, nocc, 4, seed=11):
        parts = [
            np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, c, t)) for c, t in zip(ctxs, tds)
        ]
        full = parts[-1]
        s = parts[0] + parts[1]
        assert np.abs(s[:3] - full[:3]).max() < 1e-9  # t2frg, e0frg, e1frg are additive
        assert abs(parts[0][3] - full[3]) < 1e-9  # e0 is the same for every fragment


def test_kernel_chunking_and_precision_are_consistent(frag0):
    """The cholesky chunk does not change the numbers; mixed precision changes them a little."""
    sys_, ham_data, td = frag0["sys"], frag0["ham_data"], frag0["trial_data"]
    ref_ops = frag0["ops"]
    ref_ctx = frag0["ctx"]
    assert ref_ctx.nchol_chunk == 3
    ops1 = lm.make_pt2ccsd_meas_ops(sys_, mixed_precision=False, testing=True, nchol_chunk=None)
    ctx1 = ops1.build_meas_ctx(ham_data, td)
    assert ctx1.nchol_chunk == ham_data.chol.shape[0]  # the whole tensor in one step
    ops_mp = lm.make_pt2ccsd_meas_ops(sys_, mixed_precision=True, nchol_chunk=5)
    ctx_mp = ops_mp.build_meas_ctx(ham_data, td)
    cfg = lm.get_pt2ccsd_meas_cfg(ops_mp)
    assert cfg is not None and cfg.mixed_real_dtype == jnp.float32 and cfg.nchol_chunk == 5
    for w in _random_walkers(sys_.norb, sys_.nup, 3, seed=2):
        ref = np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, ref_ctx, td))
        out1 = np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, ctx1, td))
        out_mp = np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, ctx_mp, td))
        assert np.abs(out1 - ref).max() < 1e-10
        assert np.abs(out_mp - ref).max() < 1e-4
    del ref_ops


def test_overlap_ratio_is_trial_over_guide(frag0):
    """wp = w <T|phi>/<G|phi>: for the HF guide the ratio is afqmc's obar/o0."""
    sys_, td, ctx = frag0["sys"], frag0["trial_data"], frag0["ctx"]
    nocc, norb = sys_.nup, sys_.norb
    for w in _random_walkers(norb, nocc, 3, seed=3):
        obar = jnp.linalg.det((ctx.exp_t1 @ w)[:nocc, :]) ** 2
        o0 = jnp.linalg.det(w[:nocc, :]) ** 2
        assert abs(complex(overlap_r(w, td) / o0) - complex(obar / o0)) < 1e-10


def test_chunk_plan_sizes_the_fragment_kernel(frag0):
    sys_, ham_data, td = frag0["sys"], frag0["ham_data"], frag0["trial_data"]
    nchol = int(ham_data.chol.shape[0])
    big = lm.plan_chunking_for_run(
        sys_, ham_data, td, n_walkers=50, budget_bytes=4 * 1024**3, mixed_precision=False
    )
    assert big.nchol_chunk == nchol and big.n_chunks == 1
    small = lm.plan_chunking_for_run(
        sys_,
        ham_data,
        td,
        n_walkers=50,
        budget_bytes=big.model.resident
        + 50 * big.model.per_walker
        + 60 * big.model.per_walker_chol,
        mixed_precision=False,
    )
    assert 1 <= small.nchol_chunk < nchol
    fixed = lm.plan_chunking_for_run(
        sys_, ham_data, td, n_walkers=50, budget_bytes=4 * 1024**3, nchol_chunk=4
    )
    assert fixed.nchol_chunk == 4


# ----------------------------------------------------------------------------- pt2ccsd_fast


def test_fast_trial_staging(o2):
    """t2u = t2 U on the first index, U = <act_occ|lo> with nlo < nocc; prjlo = U U^T."""
    frag = o2["frags"][0]
    tin = lst.stage_pt2ccsd_fast_trial(frag)
    assert tin.kind == "pt2ccsd_fast"
    u = np.asarray(tin.data["u"])
    nocc, nlo = u.shape
    assert nlo < nocc and np.array_equal(u, np.asarray(frag.uocc_loc))
    t2 = np.asarray(frag.t2).transpose(0, 2, 1, 3)
    t2u = np.einsum("iajb,iI->Iajb", t2, u)
    assert tin.data["t2x"].shape == (nlo, t2.shape[1], nocc, t2.shape[3])
    np.testing.assert_allclose(tin.data["t2x"], 2 * t2u - t2u.transpose(0, 3, 2, 1), atol=1e-14)
    # the projected doubles of the pt2ccsd trial are t2u closed with the second factor
    tin_bar = lst.stage_pt2ccsd_trial(frag)
    np.testing.assert_allclose(np.einsum("Iajb,kI->kajb", t2u, u), tin_bar.data["t2"], atol=1e-12)
    np.testing.assert_allclose(u @ u.T, tin_bar.data["prjlo"], atol=1e-12)


def test_fast_kernel_matches_bar_kernel(frag0, o2):
    """The factored projector gives the same four components as the bar kernel, to roundoff."""
    sys_, ham_data, td, ctx = frag0["sys"], frag0["ham_data"], frag0["trial_data"], frag0["ctx"]
    frag = o2["frags"][0]
    tin = lst.stage_pt2ccsd_fast_trial(frag)
    tdf = make_pt2ccsd_fast_trial_data(tin.data, sys_)
    assert tdf.nocc == td.nocc and tdf.nvir == td.nvir and tdf.nlo < tdf.nocc
    ctxs = [
        lf.make_pt2ccsd_fast_meas_ops(
            sys_, mixed_precision=False, testing=True, nchol_chunk=k
        ).build_meas_ctx(ham_data, tdf)
        for k in (1, 3, None)
    ]
    assert ctxs[0].chol_ov_u.shape == (ham_data.chol.shape[0], tdf.nlo, tdf.nvir)
    for w in _random_walkers(sys_.norb, sys_.nup, 4, seed=13):
        ref = np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, ctx, td))
        for c in ctxs:
            out = np.asarray(lf.energy_kernel_rw_rh_fast(w, ham_data, c, tdf))
            assert np.abs(out - ref).max() < 1e-9
    # mixed precision: the T2 contractions in single precision
    ops_mp = lf.make_pt2ccsd_fast_meas_ops(sys_, mixed_precision=True, nchol_chunk=4)
    ctx_mp = ops_mp.build_meas_ctx(ham_data, tdf)
    assert lf.get_pt2ccsd_fast_meas_cfg(ops_mp).mixed_real_dtype == jnp.float32
    w = next(_random_walkers(sys_.norb, sys_.nup, 1, seed=4))
    ref = np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, ctx, td))
    assert (
        np.abs(np.asarray(lf.energy_kernel_rw_rh_fast(w, ham_data, ctx_mp, tdf)) - ref).max() < 1e-4
    )


def test_fast_recipes():
    assert ("rhf", "pt2ccsd_fast") in available_mixed_recipes()
    assert ("cisd", "pt2ccsd_fast") in available_mixed_recipes()
    rec = get_mixed_recipe("pt2ccsd_fast")
    assert rec.guide == "rhf" and rec.components == lm.TRIAL_COMPONENTS
    assert rec.energy_fn is lm.frag_pt2ccsd_energy_fn and rec.needs_amplitudes
    with pytest.raises(ValueError):
        get_mixed_recipe("pt2ccsd_fast", guide="uhf")


def test_fast_trial_run_reproduces_pt2ccsd_run(o2, tmp_path):
    """From one fragment file and seed, trial="pt2ccsd_fast" is the trial="pt2ccsd" run."""
    mf, frag = o2["mf"], o2["frags"][0]
    qmc: dict[str, Any] = dict(
        n_walkers=20, n_eql_blocks=2, n_blocks=20, dt=0.005, n_prop_steps=10, seed=3
    )
    with contextlib.redirect_stdout(io.StringIO()):
        path = LnoFragMixed(mf, frag, chol_cut=CHOL_CUT).save(tmp_path / "frag1.h5")
        fm_bar = LnoFragMixed.from_frag_data(path, trial="pt2ccsd", mixed_precision=False, **qmc)
        e_bar, err_bar = fm_bar.kernel()
        fm_fast = LnoFragMixed.from_frag_data(
            path, trial="pt2ccsd_fast", mixed_precision=False, **qmc
        )
        e_fast, err_fast = fm_fast.kernel()
    assert fm_fast.trial == "pt2ccsd_fast" and fm_fast.build_job().mix_trial_data.nlo < 6
    r_bar, r_fast = fm_bar.qmc_result, fm_fast.qmc_result
    assert abs(r_fast.frag_init_energy - r_bar.frag_init_energy) < 1e-10
    assert abs(e_fast - e_bar) < 1e-8 and abs(err_fast - err_bar) < 1e-8
    assert abs(fm_fast.guide_e_tot - fm_bar.guide_e_tot) < 1e-10


# ----------------------------------------------------------------------------- statistics


def test_frag_energy_fn_and_component_statistics():
    """
    E_F = <e0frg> + <e1frg> - <t2frg><e0> over the wp-weighted block averages: the
    branch's component blocking gives that mean, the proxy energies are the per block
    values, and the outlier mask drops an injected outlier.
    """
    rng = np.random.default_rng(3)
    n = 300
    w = rng.uniform(0.8, 1.2, n) + 0.05j * rng.normal(size=n)
    comps = np.column_stack(
        [
            0.1 + 0.02 * rng.normal(size=n),
            -0.3 + 0.03 * rng.normal(size=n),
            -0.05 + 0.03 * rng.normal(size=n),
            -75.0 + 0.3 * rng.normal(size=n),
        ]
    )
    avg = (w[:, None] * comps).sum(axis=0) / w.sum()
    e_ref = avg[1] + avg[2] - avg[0] * avg[3]
    assert abs(complex(lm.frag_pt2ccsd_energy_fn(0.0, avg)) - e_ref) < 1e-12
    assert abs(complex(lm.frag_pt2ccsd_energy_fn(123.0, avg)) - e_ref) < 1e-12  # h0 ignored
    stats = blocking_analysis_components(0.0, w, comps, lm.frag_pt2ccsd_energy_fn, print_q=False)
    assert abs(float(stats["mu"]) - e_ref.real) < 1e-12
    assert stats["se_star"] is not None and float(stats["se_star"]) > 0

    proxy, keep = component_estimator_outlier_mask(0.0, w, comps, lm.frag_pt2ccsd_energy_fn)
    per_block = comps[:, 1] + comps[:, 2] - comps[:, 0] * comps[:, 3]
    assert np.abs(proxy - per_block).max() < 1e-12 and keep.all()
    comps[17, 2] = 500.0
    _, keep = component_estimator_outlier_mask(0.0, w, comps, lm.frag_pt2ccsd_energy_fn)
    assert int(keep.sum()) == n - 1 and not bool(keep[17])


# ----------------------------------------------------------------------------- recipes and files


def test_recipes():
    assert ("rhf", "pt2ccsd") in available_mixed_recipes()
    assert ("cisd", "pt2ccsd") in available_mixed_recipes()
    rec = get_mixed_recipe("pt2ccsd")
    assert rec.guide == "rhf" and rec.needs_amplitudes and rec.components == lm.TRIAL_COMPONENTS
    assert getattr(rec.mixed_block_fn, "__name__", "") == "block_mixed"
    assert rec.energy_fn is lm.frag_pt2ccsd_energy_fn and rec.stage_guide is None
    with pytest.raises(ValueError):
        get_mixed_recipe("pt2ccsd", guide="uhf")
    with pytest.raises(ValueError):
        get_mixed_recipe("cisd")
    with pytest.raises(ValueError):
        get_mixed_recipe("pt2ccsd_bar")


def test_frag_file_round_trip(o2, tmp_path):
    mf, frag = o2["mf"], o2["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        fm = LnoFragMixed(mf, frag, trial="pt2ccsd", chol_cut=CHOL_CUT)
        path = fm.save(tmp_path / "frag1.h5")
        frag2, staged, attrs = lst.load_frag(path)
    assert frag2.frag_name == frag.frag_name and frag2.nact == frag.nact
    # the pt2ccsd trial stores the full doubles
    assert np.allclose(frag2.t2, frag.t2) and np.allclose(frag2.uocc_loc, frag.uocc_loc)
    assert frag2.t2u is None and frag2.has_full_amplitudes
    assert staged.ham.norb == 8 and staged.trial.kind == "rhf"
    assert abs(attrs["emf"] - mf.e_tot) < 1e-12
    # the guide staged by the branch in the fragment basis is the identity on the active LNOs
    assert np.allclose(staged.trial.data["mo"], np.eye(8), atol=1e-10)
    with contextlib.redirect_stdout(io.StringIO()):
        fm2 = LnoFragMixed.from_frag_data(path, trial="pt2ccsd")
        job = fm2.build_job()
    assert job.sys.norb == 8 and job.mix_trial_data.nocc == 6
    assert job.recipe.trial == "pt2ccsd" and job.mix_trial_meas_ops.has_kernel("energy")


def test_frag_file_projected_doubles(o2, tmp_path):
    """A file written for the fast trial carries t2u = t2 U instead of t2; every pt2CCSD trial
    re-runs from it, the CISD guide (which needs the full doubles) refuses it."""
    import h5py

    mf, frag = o2["mf"], o2["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        path = LnoFragMixed(mf, frag, trial="pt2ccsd_fast", chol_cut=CHOL_CUT).save(
            tmp_path / "f.h5"
        )
        frag2, _, _ = lst.load_frag(path)
    with h5py.File(path, "r") as f:
        assert "t2u" in f["amplitudes"] and "t2" not in f["amplitudes"]
        assert int(f.attrs["frag_file_version"]) == 2
    nlo, nocc, nvir = frag2.uocc_loc.shape[1], int(frag.nactocc), int(frag.nactvir)
    assert frag2.t2 is None and frag2.t2u.shape == (nlo, nvir, nocc, nvir)
    assert frag2.has_amplitudes and not frag2.has_full_amplitudes
    np.testing.assert_allclose(frag2.t2u, lst.projected_doubles(frag), atol=1e-14)
    # the same trial inputs as from the full amplitudes, for both pt2CCSD trials
    for stager in (lst.stage_pt2ccsd_trial, lst.stage_pt2ccsd_fast_trial):
        a, b = stager(frag), stager(frag2)
        for k in a.data:
            np.testing.assert_allclose(np.asarray(a.data[k]), np.asarray(b.data[k]), atol=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        for trial in ("pt2ccsd_fast", "pt2ccsd"):
            job = LnoFragMixed.from_frag_data(path, trial=trial).build_job()
            assert job.mix_trial_data.nocc == nocc
    with pytest.raises(ValueError, match="full fragment CCSD amplitudes"):
        LnoFragMixed.from_frag_data(path, guide="cisd").build_job()
    # a file written with the CISD guide carries that guide, so it re-runs with it
    with contextlib.redirect_stdout(io.StringIO()):
        path2 = LnoFragMixed(mf, frag, trial="pt2ccsd_fast", guide="cisd", chol_cut=CHOL_CUT).save(
            tmp_path / "f2.h5"
        )
        job2 = LnoFragMixed.from_frag_data(path2, trial="pt2ccsd_fast", guide="cisd").build_job()
    assert job2.staged.trial.kind == "cisd"
    # amplitudes="full" keeps the full doubles even for the fast trial
    with contextlib.redirect_stdout(io.StringIO()):
        path3 = LnoFragMixed(mf, frag, trial="pt2ccsd_fast", chol_cut=CHOL_CUT).save(
            tmp_path / "f3.h5", amplitudes="full"
        )
    assert lst.load_frag(path3)[0].has_full_amplitudes


def test_frag_mixed_options(o2):
    mf, frag = o2["mf"], o2["frags"][0]
    with pytest.raises(ValueError, match="tau_eql or n_eql_blocks"):
        LnoFragMixed(mf, frag, tau_eql=1.0, n_eql_blocks=3)
    with contextlib.redirect_stdout(io.StringIO()):
        fm = LnoFragMixed(mf, frag, tau_eql=1.0, dt=0.01, n_prop_steps=10, chol_cut=CHOL_CUT)
        job = fm.build_job()
    assert job.params.n_eql_blocks == 10 and fm.n_blocks == 300
    stripped = lst.LnoFragData(**{**frag.__dict__, "t1": None, "t2": None})
    with pytest.raises(ValueError, match="amplitudes"):
        LnoFragMixed(mf, stripped)
    # the guide and the trial precision can be set apart; mixed_precision sets both
    with contextlib.redirect_stdout(io.StringIO()):
        fm = LnoFragMixed(
            mf, frag, chol_cut=CHOL_CUT, mixed_precision=False, trial_mixed_precision=True
        )
        job = fm.build_job()
    assert not fm.guide_mixed_precision and fm.trial_mixed_precision
    assert lm.get_pt2ccsd_meas_cfg(job.mix_trial_meas_ops).mixed_real_dtype == jnp.float32
    ctx = job.prop_ops.build_prop_ctx(
        job.ham_data, job.trial_ops.get_rdm1(job.trial_data), job.params
    )
    assert ctx.chol_flat.dtype == jnp.float64
    lno = LnoAfqmcMixed(
        mf, o2["lo_coeff"], o2["frag_list"], nfrozen=o2["nfrozen"], guide_mixed_precision=False
    )
    assert lno.qmc_kwargs["guide_mixed_precision"] is False and lno.qmc_kwargs["mixed_precision"]


# ----------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_lno_afqmc_o2_end_to_end(o2, tmp_path, request):
    """
    Both O2 fragments: at tau = 0 the fragment energies add up to the (LNO-)CCSD
    energy exactly; after sampling they add up to the full-space AFQMC/pt2CCSD (bar)
    correlation energy within error bars; and a fragment re-run from its file with the
    same seed reproduces the loop.
    """
    if not request.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    from trot.afqmc import AfqmcMixed

    mf, nfrozen = o2["mf"], o2["nfrozen"]
    qmc: dict[str, Any] = dict(
        n_walkers=100, n_eql_blocks=10, n_blocks=60, dt=0.005, n_prop_steps=50
    )
    out = tmp_path
    with contextlib.redirect_stdout(io.StringIO()):
        lno = LnoAfqmcMixed(
            mf,
            o2["lo_coeff"],
            o2["frag_list"],
            frag_name=o2["frag_name"],
            lno_thresh=1e-12,
            nfrozen=nfrozen,
            trial="pt2ccsd",
            seed=17,
            chol_cut=CHOL_CUT,
            frag_output=str(out / "fragment.out"),
            lno_output=str(out / "lno_result.out"),
            save_frag_data=str(out / "frag_data"),
            keep_qmc_results=True,
            debug_memory=True,
            **qmc,
        )
        e_qmc, e_qmc_err = lno.kernel()
    init_sum = sum(r.frag_init_energy for r in lno.frag_qmc_results)
    assert abs(init_sum - lno.e_cc) < 1e-7  # LNO-CCSD conv_tol 1e-6, chol_cut
    assert abs(init_sum - o2["e_ccsd"]) < 1e-6
    assert (out / "fragment.out1").exists() and (out / "lno_result.out").exists()
    assert lno.lno_size == [8, 8] and lno.n_done == 2

    mycc = cc.CCSD(mf, frozen=nfrozen)
    mycc.kernel()
    with contextlib.redirect_stdout(io.StringIO()):
        af = AfqmcMixed(mycc, trial="pt2ccsd_bar", seed=17, chol_cut=CHOL_CUT, **qmc)
        e_ref, err_ref = af.kernel()
    e_ref_corr = e_ref - mf.e_tot
    assert abs(e_qmc - e_ref_corr) < 3 * np.hypot(e_qmc_err, err_ref)

    with contextlib.redirect_stdout(io.StringIO()):
        fr = LnoFragMixed.from_frag_data(
            out / "frag_data" / "frag1.h5", seed=int(lno._seeds[0]), **qmc
        )
        e1, err1 = fr.kernel()
    assert abs(e1 - lno.lno_eqmc[0]) < 1e-10 and abs(err1 - lno.lno_eqmc_err[0]) < 1e-10


def test_frag_memory_budget_and_chunk_plan(o2, tmp_path):
    """
    Without max_memory the fragment takes its budget from the device (also under the
    platform allocator, where jax reports none), so the trial gets a chunk plan; the plan
    and its memory lines are in the banner; an explicit max_memory sizes the chunk.
    """
    import jax

    from trot.lnoafqmc.afqmc import device_memory_budget_mb

    mf, frag = o2["mf"], o2["frags"][0]
    budget, source = device_memory_budget_mb()
    on_gpu = jax.devices()[0].platform == "gpu"
    assert (budget is not None) == on_gpu
    fm = LnoFragMixed(mf, frag, chol_cut=CHOL_CUT, trial="pt2ccsd_fast", n_walkers=8)
    assert fm.max_memory == budget and fm.max_memory_source == source
    small = LnoFragMixed(
        mf, frag, chol_cut=CHOL_CUT, trial="pt2ccsd_fast", n_walkers=8, max_memory=2
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        job = small.build_job()
        small.dump_flags(job)
    text = buf.getvalue()
    assert small.max_memory_source == "max_memory" and job.chunk_plan is not None
    nchol = int(job.ham_data.nchol)
    assert 1 <= job.chunk_plan.nchol_chunk <= nchol and job.chunk_plan.budget_bytes == 2 * 1024**2
    assert " max_memory      = 2 MB  (max_memory)" in text
    for key in (
        "chunk_plan",
        "nchol_chunk",
        "padded",
        "walkers_in_flight",
        "memory_resident",
        "memory_per_walker",
        "memory_walkers",
        "memory_used",
        "memory_budget",
    ):
        assert key in text, key
    if on_gpu:
        with contextlib.redirect_stdout(io.StringIO()):
            assert fm.build_job().chunk_plan is not None


def test_cpu_gpu_split_reproduces_one_machine_loop(o2, tmp_path):
    """
    run_qmc=False + save_frag_data writes the fragment files and runs no AFQMC; a second
    LnoAfqmcMixed(frag_data=...) runs the AFQMC from the files only and, with the same
    seed and settings, reproduces the one-machine loop fragment by fragment.
    """
    mf, nfrozen = o2["mf"], o2["nfrozen"]
    common: dict[str, Any] = dict(trial="pt2ccsd_fast", seed=17, mixed_precision=False)
    qmc: dict[str, Any] = dict(n_walkers=20, n_eql_blocks=2, n_blocks=20, dt=0.005, n_prop_steps=10)
    lno_args = (mf, o2["lo_coeff"], o2["frag_list"])
    lno_kw: dict[str, Any] = dict(
        frag_name=o2["frag_name"], lno_thresh=1e-12, nfrozen=nfrozen, chol_cut=CHOL_CUT
    )
    files = tmp_path / "frag_data"
    with contextlib.redirect_stdout(io.StringIO()):
        # CPU machine: LNO + MP2 + CCSD + the fragment integrals, no QMC
        cpu = LnoAfqmcMixed(*lno_args, run_qmc=False, save_frag_data=str(files), **lno_kw, **common)
        e0, err0 = cpu.kernel()
        # GPU machine: AFQMC from the files only
        gpu = LnoAfqmcMixed(
            frag_data=files,
            frag_output=str(tmp_path / "fragment.out"),
            lno_output=str(tmp_path / "lno_result.out"),
            keep_qmc_results=True,
            **common,
            **qmc,
        )
        e_files, err_files = gpu.kernel()
        # one machine
        one = LnoAfqmcMixed(*lno_args, keep_qmc_results=True, **lno_kw, **common, **qmc)
        e_one, err_one = one.kernel()
    assert (e0, err0) == (0.0, 0.0) and not np.any(cpu.lno_eqmc)
    assert sorted(p.name for p in files.glob("*.h5")) == ["frag1.h5", "frag2.h5"]
    meta = lst.frag_file_meta(files / "frag2.h5")
    assert meta["frag_idx"] == 1 and meta["nfrag_tot"] == 2 and not meta["unrestricted"]
    assert gpu._scf is None and gpu.nfrag_tot == 2 and gpu.run_frag == [0, 1]
    assert gpu.frag_name_all == one.frag_name_all and gpu.nfrozen == one.nfrozen
    assert gpu.lno_size == one.lno_size and gpu.lno_nocc == one.lno_nocc
    assert np.allclose(gpu.lno_emp, one.lno_emp) and np.allclose(gpu.lno_ecc, one.lno_ecc)
    assert np.allclose(gpu.lno_emp, cpu.lno_emp) and np.allclose(gpu.lno_ecc, cpu.lno_ecc)
    assert abs(gpu.e_cc - one.e_cc) < 1e-12
    for r_files, r_one in zip(gpu.frag_qmc_results, one.frag_qmc_results):
        assert abs(r_files.frag_init_energy - r_one.frag_init_energy) < 1e-10
    assert np.allclose(gpu.lno_eqmc, one.lno_eqmc, atol=1e-10)
    assert np.allclose(gpu.lno_eqmc_err, one.lno_eqmc_err, atol=1e-10)
    assert abs(e_files - e_one) < 1e-10 and abs(err_files - err_one) < 1e-10
    assert (tmp_path / "fragment.out2").exists() and (tmp_path / "lno_result.out").exists()

    # a subset of the fragments, and one file instead of the directory
    with contextlib.redirect_stdout(io.StringIO()):
        sub = LnoAfqmcMixed(frag_data=files, run_frag=[1], **common, **qmc)
        e_sub, _ = sub.kernel()
        one_file = LnoAfqmcMixed(frag_data=files / "frag2.h5", **common, **qmc)
        e_one_file, _ = one_file.kernel()
    assert sub.nfrag_tot == 2 and sub.run_frag == [1]
    assert abs(e_sub - gpu.lno_eqmc[1]) < 1e-10 and abs(e_one_file - gpu.lno_eqmc[1]) < 1e-10
    assert one_file.nfrag_tot == 2  # from the file, so the seeds are those of the loop

    # errors: both inputs, a missing fragment, the wrong trial for the files
    with pytest.raises(ValueError, match="replaces"):
        LnoAfqmcMixed(mf, o2["lo_coeff"], o2["frag_list"], frag_data=files, **lno_kw)
    with pytest.raises(ValueError, match="no fragment file"):
        LnoAfqmcMixed(frag_data=files / "frag2.h5", run_frag=[0], **common)
    with pytest.raises(ValueError, match="mean field"):
        LnoAfqmcMixed(frag_data=files, trial="upt2ccsd")
    with pytest.raises(ValueError, match="requires run_cc=True"):
        LnoAfqmcMixed(*lno_args, run_qmc=False, run_cc=False, save_frag_data=str(files), **lno_kw)


# ----------------------------------------------------------------------------- CISD guide


def test_cisd_guide_recipe():
    from trot.lnoafqmc.staging import stage_cisd_guide

    rec = get_mixed_recipe("pt2ccsd", guide="cisd")
    assert rec.guide == "cisd" and rec.walker_kind == "restricted"
    assert rec.needs_amplitudes and rec.stage_guide is stage_cisd_guide
    assert get_mixed_recipe("pt2ccsd").guide == "rhf"
    with pytest.raises(ValueError, match="no LNO recipe"):
        get_mixed_recipe("pt2ccsd", guide="ucisd")


def test_cisd_guide_staging_matches_trot(o2):
    """The fragment CISD guide is what the branch's _stage_cisd_input builds from the amplitudes."""
    from trot.lnoafqmc.staging import stage_cisd_guide

    frag = o2["frags"][0]
    tin = stage_cisd_guide(frag)
    assert tin.kind == "cisd"
    t1 = np.asarray(frag.t1)
    t2 = np.asarray(frag.t2)
    ci2 = (t2 + np.einsum("ia,jb->ijab", t1, t1)).transpose(0, 2, 1, 3)
    np.testing.assert_allclose(np.asarray(tin.data["ci1"]), t1)
    np.testing.assert_allclose(np.asarray(tin.data["ci2"]), ci2)


@pytest.mark.slow
def test_cisd_guide_fragments(o2, tmp_path, request):
    """
    Under the fragment CISD guide the walkers start from the reference determinant, so
    the tau = 0 fragment energies still add up to the LNO-CCSD energy; a short run gives
    finite numbers; and a fragment file written with one guide can be re-run with
    another.
    """
    if not request.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    from trot.trial.cisd import CisdTrial

    mf = o2["mf"]
    qmc: dict[str, Any] = dict(n_walkers=4, n_eql_blocks=2, n_blocks=12, dt=0.01, n_prop_steps=2)
    e_init = 0.0
    for frag in o2["frags"]:
        with contextlib.redirect_stdout(io.StringIO()):
            fm = LnoFragMixed(mf, frag, guide="cisd", chol_cut=CHOL_CUT, seed=5, **qmc)
            job = fm.build_job()
            assert isinstance(job.trial_data, CisdTrial)
            assert job.staged.trial.kind == "cisd"
            assert job.meas_ops.has_kernel("force_bias")
            e, err = fm.kernel()
        assert np.isfinite(e) and np.isfinite(fm.guide_e_tot)
        e_init += fm.qmc_result.frag_init_energy
    assert abs(e_init - o2["e_ccsd"]) < 1e-6

    # file written with the HF guide, re-run with the CISD guide rebuilt from its amplitudes
    frag = o2["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        path = LnoFragMixed(mf, frag, chol_cut=CHOL_CUT).save(tmp_path / "frag1.h5")
        fm2 = LnoFragMixed.from_frag_data(path, guide="cisd", **qmc)
        job2 = fm2.build_job()
    assert isinstance(job2.trial_data, CisdTrial)
    # and the other way round: a file carries the guide it was written with, so asking
    # for an HF guide from a CISD file is refused rather than silently changed
    with contextlib.redirect_stdout(io.StringIO()):
        path3 = LnoFragMixed(mf, frag, guide="cisd", chol_cut=CHOL_CUT).save(tmp_path / "frag1c.h5")
        with pytest.raises(ValueError, match="stages as 'cisd'"):
            LnoFragMixed.from_frag_data(path3, **qmc).build_job()
