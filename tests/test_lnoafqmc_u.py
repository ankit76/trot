"""
Tests of the unrestricted half of trot.lnoafqmc (trial="upt2ccsd"), on triplet O2 (UHF,
sto-3g, density fitted, two atomic fragments, full-space LAS). The restricted half and
the statistics are in test_lnoafqmc.py.

    pytest tests/test_lnoafqmc_u.py -q [--run-slow]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import contextlib
import io
from typing import Any, Iterator

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, mp, scf
from pyscf.data import elements

import trot.lnoafqmc  # noqa: F401  (allocator, before jax)
from trot import config

config.configure_once()

from trot.core.system import System_uh
from trot.ham.chol_u import HamCholU
from trot.lnoafqmc import LnoAfqmcMixed, LnoFragMixed, iao_fragment
from trot.lnoafqmc import integral as li
from trot.lnoafqmc import pipeline, solvers
from trot.lnoafqmc import staging as lst
from trot.lnoafqmc.meas import upt2ccsd_bar as lmu
from trot.lnoafqmc.meas import upt2ccsd_fast as lfu
from trot.lnoafqmc.mixed import available_mixed_recipes, get_mixed_recipe
from trot.lnoafqmc.trial.upt2ccsd import make_upt2ccsd_trial_data, overlap_u
from trot.lnoafqmc.trial.upt2ccsd_fast import make_upt2ccsd_fast_trial_data
from trot.meas.pt2ccsd_chunking import make_chunk_meas_cfg
from trot.meas.upt2ccsd_bar_uh import build_meas_ctx as trot_build_ctx
from trot.meas.upt2ccsd_bar_uh import energy_kernel_uw_uh_bar as trot_bar_kernel
from trot.trial.upt2ccsd_uh import Upt2ccsdTrial as TrotUpt2ccsdTrial

CHOL_CUT = 1e-6


@pytest.fixture(scope="module")
def o2t():
    # max_memory: pyscf-forge budgets max_memory - RSS, which goes negative in a long session
    mol = gto.M(atom="O 0 0 0; O 0 0 1.208", basis="sto-3g", spin=2, verbose=0, max_memory=16000)
    mf: Any = scf.UHF(mol).density_fit()
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
    mycc: Any = cc.UCCSD(mf, frozen=nfrozen)
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
def ufrag0(o2t):
    """Fragment 0's uchol hamiltonian, trial and meas ops on the device, in double precision."""
    mf, frag = o2t["mf"], o2t["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        ham = li.build_ham_ulno_df(mf, frag.lno_coeff, frag.lno_frozen, chol_cut=CHOL_CUT)
    sys_ = System_uh(norb=ham.norb, nelec=ham.nelec, walker_kind="unrestricted")
    ham_data = HamCholU(
        jnp.asarray(ham.h0),
        jnp.asarray(ham.h1_a),
        jnp.asarray(ham.h1_b),
        jnp.asarray(ham.chol_a),
        jnp.asarray(ham.chol_b),
        basis="uchol",
    )
    tin = lst.stage_upt2ccsd_trial(frag)
    ops = lmu.make_upt2ccsd_meas_ops(sys_, mixed_precision=False, testing=True, nchol_chunk=3)
    # unprojected amplitudes in the branch's (i,a,j,b) layout, same-spin antisymmetrized
    t2aa, t2ab, t2bb = (np.asarray(t) for t in frag.t2)
    t2aa = 0.5 * (t2aa - t2aa.transpose(0, 1, 3, 2))
    t2bb = 0.5 * (t2bb - t2bb.transpose(0, 1, 3, 2))
    t2_full = tuple(t.transpose(0, 2, 1, 3) for t in (t2aa, t2ab, t2bb))
    return dict(ham=ham, sys=sys_, ham_data=ham_data, tin=tin, ops=ops, t2_full=t2_full)


def _e_hf_el(ham) -> float:
    na, nb = ham.nelec
    loa, lob = ham.chol_a[:, :na, :na], ham.chol_b[:, :nb, :nb]
    tr = np.trace(loa, axis1=1, axis2=2) + np.trace(lob, axis1=1, axis2=2)
    return float(
        np.trace(ham.h1_a[:na, :na])
        + np.trace(ham.h1_b[:nb, :nb])
        + 0.5 * np.sum(tr**2)
        - 0.5 * np.einsum("gij,gji->", loa, loa)
        - 0.5 * np.einsum("gij,gji->", lob, lob)
    )


def _random_walkers(norb, nelec, n, seed=7) -> Iterator[Any]:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield tuple(
            jnp.asarray(
                np.linalg.qr(rng.normal(size=(no, ne)) + 0.3j * rng.normal(size=(no, ne)))[0]
            )
            for no, ne in zip(norb, nelec)
        )


def _projected_trial(ufrag0, prj_a, prj_b):
    t2aa, t2ab, t2bb = ufrag0["t2_full"]
    d = dict(ufrag0["tin"].data)
    d.update(
        prjlo_a=prj_a,
        prjlo_b=prj_b,
        t2aa=np.einsum("iajb,ik->kajb", t2aa, prj_a),
        t2ab=np.einsum("iajb,ik->kajb", t2ab, prj_a),
        t2ba=np.einsum("jbia,ik->kajb", t2ab, prj_b),
        t2bb=np.einsum("iajb,ik->kajb", t2bb, prj_b),
    )
    td = make_upt2ccsd_trial_data(d, ufrag0["sys"])
    return td, ufrag0["ops"].build_meas_ctx(ufrag0["ham_data"], td)


# ----------------------------------------------------------------------------- CPU stage, integrals


def test_u_fragments_sum_to_canonical_mp2_ccsd(o2t):
    frags = o2t["frags"]
    assert all(f.unrestricted and f.nact == (8, 8) and tuple(f.nactocc) == (7, 5) for f in frags)
    mmp: Any = mp.UMP2(o2t["mf"], frozen=o2t["nfrozen"])
    mmp.kernel()
    assert abs(sum(f.efrag_mp for f in frags) - mmp.e_corr) < 1e-8
    assert abs(sum(f.efrag_cc for f in frags) - o2t["e_ccsd"]) < 1e-6


def test_u_fragment_hamiltonian_reproduces_hf_energy(ufrag0, o2t):
    """Full-space LAS, DF-exact core, joint cholesky: E_UHF of the fragment ham is mf.e_tot."""
    ham = ufrag0["ham"]
    assert ham.norb == (8, 8) and ham.nelec == (7, 5) and ham.basis == "uchol"
    assert ham.frozen == (2, 2)  # the frozen core count per spin (chemcore of O2)
    assert ham.chol_a.shape[0] == ham.chol_b.shape[0]  # one shared cholesky index
    assert abs(ham.h0 + _e_hf_el(ham) - o2t["mf"].e_tot) < 1e-5


# ----------------------------------------------------------------------------- kernel


def test_u_kernel_prjlo_one_limit_matches_bar_kernel(ufrag0):
    """With identity projectors the fragment kernel is the branch's upt2ccsd_bar kernel; e0frg is e0 - E_HF."""
    ham, ham_data = ufrag0["ham"], ufrag0["ham_data"]
    na, nb = ham.nelec
    td, ctx = _projected_trial(ufrag0, np.eye(na), np.eye(nb))
    t2aa, t2ab, t2bb = ufrag0["t2_full"]
    ttrial = TrotUpt2ccsdTrial(
        mo_t_a=td.mo_t_a,
        mo_t_b=td.mo_t_b,
        t2aa=jnp.asarray(t2aa),
        t2ab=jnp.asarray(t2ab),
        t2bb=jnp.asarray(t2bb),
    )
    cfg = make_chunk_meas_cfg(mixed_precision=False, testing=True, nchol_chunk=3)
    tctx = trot_build_ctx(ham_data, ttrial, cfg)
    e_hf = _e_hf_el(ham)
    for w in _random_walkers(ham.norb, ham.nelec, 4):
        out = np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx, td))
        ref = np.asarray(trot_bar_kernel(w, ham_data, tctx, ttrial))  # [theta, e0, h_t]
        assert abs(out[0] - ref[0]) < 1e-9
        assert abs(out[3] - ref[1]) < 1e-9
        assert abs(out[2] - ref[2]) < 1e-9
        # e0 and E_HF are both ~50 Ha here, so their difference carries their roundoff
        assert abs(out[1] - (ref[1] - e_hf)) < 1e-8


def test_u_kernel_partition_of_unity(ufrag0, o2t):
    """The fragment projectors of O0 and O1 add up to 1 per spin: so do the fragment kernels."""
    mf, frag, lo_coeff, frag_list = o2t["mf"], o2t["frags"][0], o2t["lo_coeff"], o2t["frag_list"]
    ham, ham_data = ufrag0["ham"], ufrag0["ham_data"]
    ncore, nocc, _, _ = li.get_las_idx(mf, frag.lno_frozen)
    s1e = mf.get_ovlp()
    prj = []
    for f in range(len(frag_list)):
        pp = []
        for s in range(2):
            actocc = np.asarray(frag.lno_coeff[s])[:, ncore[s] : ncore[s] + nocc[s]]
            u = actocc.T @ s1e @ np.asarray(lo_coeff[s])[:, frag_list[f][s]]
            pp.append(u @ u.T)
        prj.append(pp)
    for s in range(2):
        assert np.abs(sum(p[s] for p in prj) - np.eye(nocc[s])).max() < 1e-10
    # the staged projector is fragment 0's
    assert np.abs(ufrag0["tin"].data["prjlo_a"] - prj[0][0]).max() < 1e-10
    assert np.abs(ufrag0["tin"].data["prjlo_b"] - prj[0][1]).max() < 1e-10

    parts = [_projected_trial(ufrag0, *p) for p in prj]
    td1, ctx1 = _projected_trial(ufrag0, np.eye(nocc[0]), np.eye(nocc[1]))
    for w in _random_walkers(ham.norb, ham.nelec, 4, seed=11):
        outs = [np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, c, t)) for t, c in parts]
        full = np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx1, td1))
        s = outs[0] + outs[1]
        assert np.abs(s[:3] - full[:3]).max() < 1e-9  # t2frg, e0frg, e1frg are additive
        assert abs(outs[0][3] - full[3]) < 1e-9  # e0 is the same for every fragment


def test_u_kernel_chunking_and_precision_are_consistent(ufrag0):
    ham, sys_, ham_data = ufrag0["ham"], ufrag0["sys"], ufrag0["ham_data"]
    td = make_upt2ccsd_trial_data(ufrag0["tin"].data, sys_)
    ctx3 = ufrag0["ops"].build_meas_ctx(ham_data, td)
    ops1 = lmu.make_upt2ccsd_meas_ops(sys_, mixed_precision=False, testing=True)
    ctx1 = ops1.build_meas_ctx(ham_data, td)
    assert ctx3.nchol_chunk == 3 and ctx1.nchol_chunk == ham_data.chol_a.shape[0]
    ops_mp = lmu.make_upt2ccsd_meas_ops(sys_, mixed_precision=True, nchol_chunk=5)
    ctx_mp = ops_mp.build_meas_ctx(ham_data, td)
    cfg = lmu.get_upt2ccsd_meas_cfg(ops_mp)
    assert cfg is not None and cfg.mixed_real_dtype == jnp.float32
    for w in _random_walkers(ham.norb, ham.nelec, 3, seed=2):
        ref = np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx3, td))
        assert (
            np.abs(np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx1, td)) - ref).max()
            < 1e-10
        )
        assert (
            np.abs(np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx_mp, td)) - ref).max()
            < 1e-4
        )
    plan = lmu.plan_chunking_for_run_u(
        sys_, ham_data, td, n_walkers=50, budget_bytes=4 * 1024**3, mixed_precision=False
    )
    assert plan.nchol_chunk == ham_data.chol_a.shape[0] and plan.n_chunks == 1


def test_u_overlap_is_exp_t1_reference(ufrag0):
    """<T|phi> = <HF|exp(T1) phi> per spin: with the UHF guide, wp/w is afqmc's obar/o0."""
    ham, sys_ = ufrag0["ham"], ufrag0["sys"]
    na, nb = ham.nelec
    td = make_upt2ccsd_trial_data(ufrag0["tin"].data, sys_)
    ctx = ufrag0["ops"].build_meas_ctx(ufrag0["ham_data"], td)
    for w in _random_walkers(ham.norb, ham.nelec, 3, seed=3):
        obar = jnp.linalg.det((ctx.exp_t1_a @ w[0])[:na, :]) * jnp.linalg.det(
            (ctx.exp_t1_b @ w[1])[:nb, :]
        )
        assert abs(complex(overlap_u(w, td)) - complex(obar)) < 1e-10


# ----------------------------------------------------------------------------- upt2ccsd_fast


def test_u_fast_trial_staging(o2t):
    """The factored doubles closed with the projector's second factor are the projected ones."""
    frag = o2t["frags"][0]
    tin = lst.stage_upt2ccsd_fast_trial(frag)
    tin_bar = lst.stage_upt2ccsd_trial(frag)
    assert tin.kind == "upt2ccsd_fast"
    ua, ub = np.asarray(tin.data["u_a"]), np.asarray(tin.data["u_b"])
    assert ua.shape[1] == ub.shape[1] and ua.shape[1] < ua.shape[0]
    for name, u in (("t2aa", ua), ("t2ab", ua), ("t2ba", ub), ("t2bb", ub)):
        t2u = np.asarray(tin.data[name + "_u"])
        assert t2u.shape[0] == u.shape[1]
        np.testing.assert_allclose(
            np.einsum("Iajb,kI->kajb", t2u, u), np.asarray(tin_bar.data[name]), atol=1e-12
        )
    # same-spin blocks stay antisymmetric in (a, b) after the projection on i
    for name in ("t2aa_u", "t2bb_u"):
        t = np.asarray(tin.data[name])
        assert np.abs(t + t.transpose(0, 3, 2, 1)).max() < 1e-12
    np.testing.assert_allclose(ua @ ua.T, tin_bar.data["prjlo_a"], atol=1e-12)


def test_u_fast_kernel_matches_bar_kernel(ufrag0, o2t):
    """The factored projectors give the same four components as the bar kernel, to roundoff."""
    ham, sys_, ham_data = ufrag0["ham"], ufrag0["sys"], ufrag0["ham_data"]
    frag = o2t["frags"][0]
    td = make_upt2ccsd_trial_data(ufrag0["tin"].data, sys_)
    ctx = ufrag0["ops"].build_meas_ctx(ham_data, td)
    tdf = make_upt2ccsd_fast_trial_data(lst.stage_upt2ccsd_fast_trial(frag).data, sys_)
    assert tdf.nocc == td.nocc and tdf.nvir == td.nvir and tdf.nlo <= min(tdf.nocc)
    ctxs = [
        lfu.make_upt2ccsd_fast_meas_ops(
            sys_, mixed_precision=False, testing=True, nchol_chunk=k
        ).build_meas_ctx(ham_data, tdf)
        for k in (1, 3, None)
    ]
    for w in _random_walkers(ham.norb, ham.nelec, 4, seed=13):
        ref = np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx, td))
        for c in ctxs:
            out = np.asarray(lfu.energy_kernel_uw_uh_fast(w, ham_data, c, tdf))
            assert np.abs(out - ref).max() < 1e-8  # e0 is ~150 Ha here
    ops_mp = lfu.make_upt2ccsd_fast_meas_ops(sys_, mixed_precision=True, nchol_chunk=4)
    ctx_mp = ops_mp.build_meas_ctx(ham_data, tdf)
    assert lfu.get_upt2ccsd_fast_meas_cfg(ops_mp).mixed_real_dtype == jnp.float32
    w = next(_random_walkers(ham.norb, ham.nelec, 1, seed=4))
    ref = np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx, td))
    assert (
        np.abs(np.asarray(lfu.energy_kernel_uw_uh_fast(w, ham_data, ctx_mp, tdf)) - ref).max()
        < 1e-4
    )


def test_u_fast_recipes():
    assert ("uhf", "upt2ccsd_fast") in available_mixed_recipes()
    assert ("ucisd", "upt2ccsd_fast") in available_mixed_recipes()
    rec = get_mixed_recipe("upt2ccsd_fast")
    assert rec.guide == "uhf" and rec.ham_basis == "uchol" and rec.needs_amplitudes
    with pytest.raises(ValueError):
        get_mixed_recipe("upt2ccsd_fast", guide="rhf")


def test_u_fast_trial_run_reproduces_upt2ccsd_run(o2t, tmp_path):
    """From one fragment file and seed, trial="upt2ccsd_fast" is the trial="upt2ccsd" run."""
    mf, frag = o2t["mf"], o2t["frags"][0]
    qmc: dict[str, Any] = dict(
        n_walkers=20, n_eql_blocks=2, n_blocks=20, dt=0.005, n_prop_steps=10, seed=3
    )
    with contextlib.redirect_stdout(io.StringIO()):
        path = LnoFragMixed(mf, frag, chol_cut=CHOL_CUT).save(tmp_path / "frag1.h5")
        fm_bar = LnoFragMixed.from_frag_data(path, trial="upt2ccsd", mixed_precision=False, **qmc)
        e_bar, err_bar = fm_bar.kernel()
        fm_fast = LnoFragMixed.from_frag_data(
            path, trial="upt2ccsd_fast", mixed_precision=False, **qmc
        )
        e_fast, err_fast = fm_fast.kernel()
    assert fm_fast.trial == "upt2ccsd_fast"
    r_bar, r_fast = fm_bar.qmc_result, fm_fast.qmc_result
    assert abs(r_fast.frag_init_energy - r_bar.frag_init_energy) < 1e-9
    assert abs(e_fast - e_bar) < 1e-8 and abs(err_fast - err_bar) < 1e-8
    assert abs(fm_fast.guide_e_tot - fm_bar.guide_e_tot) < 1e-10


def test_u_cpu_gpu_split_and_isolated_files(o2t, tmp_path):
    """
    Unrestricted CPU/GPU split: run_qmc=False + save_frag_data writes the files; the
    AFQMC from the files (here in a child process per fragment, isolate=True) reproduces
    the one-machine loop with the same seed.
    """
    mf, nfrozen = o2t["mf"], o2t["nfrozen"]
    common: dict[str, Any] = dict(trial="upt2ccsd_fast", seed=5, mixed_precision=False)
    qmc: dict[str, Any] = dict(n_walkers=20, n_eql_blocks=2, n_blocks=10, dt=0.005, n_prop_steps=10)
    lno_kw: dict[str, Any] = dict(
        frag_name=o2t["frag_name"], lno_thresh=1e-12, nfrozen=nfrozen, chol_cut=CHOL_CUT
    )
    files = tmp_path / "frag_data"
    with contextlib.redirect_stdout(io.StringIO()):
        cpu = LnoAfqmcMixed(
            mf,
            o2t["lo_coeff"],
            o2t["frag_list"],
            run_qmc=False,
            save_frag_data=str(files),
            **lno_kw,
            **common,
        )
        cpu.kernel()
        gpu = LnoAfqmcMixed(frag_data=files, run_frag=[1], isolate=True, **common, **qmc)
        e_files, err_files = gpu.kernel()
        one = LnoAfqmcMixed(
            mf, o2t["lo_coeff"], o2t["frag_list"], run_frag=[1], **lno_kw, **common, **qmc
        )
        e_one, err_one = one.kernel()
    assert gpu.unrestricted and gpu.nfrag_tot == 2 and gpu.lno_nocc[0] == one.lno_nocc[0]
    assert (files / "frag2.result.json").exists()
    assert abs(gpu.e_cc - one.e_cc) < 1e-12 and abs(gpu.e_mp - one.e_mp) < 1e-12
    assert abs(e_files - e_one) < 1e-8 and abs(err_files - err_one) < 1e-8
    with pytest.raises(ValueError, match="mean field"):
        LnoAfqmcMixed(frag_data=files, trial="pt2ccsd")


# ----------------------------------------------------------------------------- recipes and files


def test_u_recipes():
    assert ("uhf", "upt2ccsd") in available_mixed_recipes()
    rec = get_mixed_recipe("upt2ccsd")
    assert rec.guide == "uhf" and rec.ham_basis == "uchol" and rec.walker_kind == "unrestricted"
    assert rec.needs_amplitudes and rec.components == lmu.TRIAL_COMPONENTS
    assert rec.stage_guide is lst.stage_uhf_guide and rec.build_ham is li.build_ham_ulno_df
    with pytest.raises(ValueError):
        get_mixed_recipe("upt2ccsd", guide="rhf")


def test_u_trial_must_match_mean_field(o2t):
    with pytest.raises(ValueError):
        LnoAfqmcMixed(
            o2t["mf"], o2t["lo_coeff"], o2t["frag_list"], nfrozen=o2t["nfrozen"], trial="pt2ccsd"
        )
    with pytest.raises(ValueError):
        LnoFragMixed(o2t["mf"], o2t["frags"][0], trial="pt2ccsd")


def test_u_frag_file_round_trip(o2t, tmp_path):
    mf, frag = o2t["mf"], o2t["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        fm = LnoFragMixed(mf, frag, chol_cut=CHOL_CUT)  # trial defaults to upt2ccsd
        path = fm.save(tmp_path / "frag1.h5")
        ham = fm.stage().ham
        frag2, staged, attrs = lst.load_frag(path)
    assert fm.trial == "upt2ccsd" and fm.guide == "uhf"
    assert frag2.unrestricted and frag2.nact == frag.nact and frag2.frag_name == frag.frag_name
    for x, y in zip(frag2.t2, frag.t2):
        assert np.allclose(x, y)
    # written for the fast trial, the file carries the four projected blocks instead
    with contextlib.redirect_stdout(io.StringIO()):
        path_u = LnoFragMixed(mf, frag, trial="upt2ccsd_fast", chol_cut=CHOL_CUT).save(
            tmp_path / "frag1_fast.h5"
        )
        frag3, _, _ = lst.load_frag(path_u)
    assert (
        frag3.t2 is None
        and len(frag3.t2u) == 4
        and frag3.t2u[0].shape[0] == frag3.uocc_loc[0].shape[1]
    )
    for a, b in zip(frag3.t2u, lst.projected_doubles(frag)):
        np.testing.assert_allclose(a, b, atol=1e-14)
    for stager in (lst.stage_upt2ccsd_trial, lst.stage_upt2ccsd_fast_trial):
        x, y = stager(frag), stager(frag3)
        for k in x.data:
            np.testing.assert_allclose(np.asarray(x.data[k]), np.asarray(y.data[k]), atol=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        for trial in ("upt2ccsd_fast", "upt2ccsd"):
            assert LnoFragMixed.from_frag_data(
                path_u, trial=trial
            ).build_job().mix_trial_data.nocc == (7, 5)
    with pytest.raises(ValueError, match="full fragment CCSD amplitudes"):
        LnoFragMixed.from_frag_data(path_u, guide="ucisd").build_job()
    assert staged.ham.basis == "uchol" and staged.ham.norb == (8, 8) and staged.trial.kind == "uhf"
    assert np.array_equal(staged.ham.chol_a, ham.chol_a) and np.array_equal(
        staged.ham.h1_b, ham.h1_b
    )
    # the UHF guide is the identity reference of each spin's active LNO basis
    assert np.allclose(staged.trial.data["mo_a"], np.eye(8)[:, :7])
    assert np.allclose(staged.trial.data["mo_b"], np.eye(8)[:, :5])
    assert abs(attrs["emf"] - mf.e_tot) < 1e-12
    with contextlib.redirect_stdout(io.StringIO()):
        fm2 = LnoFragMixed.from_frag_data(path)
        job = fm2.build_job()
    assert fm2.trial == "upt2ccsd" and job.mix_trial_data.nocc == (7, 5)
    assert job.sys.norb == (8, 8) and job.sys.walker_kind == "unrestricted"


# ----------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_ulno_afqmc_o2_triplet_end_to_end(o2t, tmp_path, request):
    """
    Both triplet O2 fragments: at tau = 0 the fragment energies add up to the (LNO-)UCCSD
    energy exactly; after sampling they add up to the full-space AFQMC/upt2CCSD (bar)
    correlation energy within error bars; and a fragment re-run from its file with the
    same seed reproduces the loop.
    """
    if not request.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    from trot.afqmc import AfqmcMixed

    mf, nfrozen = o2t["mf"], o2t["nfrozen"]
    qmc: dict[str, Any] = dict(
        n_walkers=100, n_eql_blocks=10, n_blocks=60, dt=0.005, n_prop_steps=50
    )
    out = tmp_path
    with contextlib.redirect_stdout(io.StringIO()):
        lno = LnoAfqmcMixed(
            mf,
            o2t["lo_coeff"],
            o2t["frag_list"],
            frag_name=o2t["frag_name"],
            lno_thresh=1e-12,
            nfrozen=nfrozen,
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
    assert lno.trial == "upt2ccsd" and lno.guide == "uhf"
    init_sum = sum(r.frag_init_energy for r in lno.frag_qmc_results)
    assert abs(init_sum - lno.e_cc) < 1e-7  # LNO-CCSD conv_tol 1e-6, chol_cut
    assert abs(init_sum - o2t["e_ccsd"]) < 1e-6
    assert (out / "fragment.out1").exists() and (out / "lno_result.out").exists()
    assert lno.lno_size == [(8, 8), (8, 8)] and lno.n_done == 2

    mycc: Any = cc.UCCSD(mf, frozen=nfrozen)
    mycc.kernel()
    with contextlib.redirect_stdout(io.StringIO()):
        af = AfqmcMixed(mycc, trial="upt2ccsd_bar", seed=17, chol_cut=CHOL_CUT, **qmc)
        e_ref, err_ref = af.kernel()
    assert abs(e_qmc - (e_ref - mf.e_tot)) < 3 * np.hypot(e_qmc_err, err_ref)

    with contextlib.redirect_stdout(io.StringIO()):
        fr = LnoFragMixed.from_frag_data(
            out / "frag_data" / "frag1.h5", seed=int(lno._seeds[0]), **qmc
        )
        e1, err1 = fr.kernel()
    assert abs(e1 - lno.lno_eqmc[0]) < 1e-10 and abs(err1 - lno.lno_eqmc_err[0]) < 1e-10


# ----------------------------------------------------------------------------- UCISD guide


def test_ucisd_guide_recipe():
    from trot.lnoafqmc.staging import stage_ucisd_guide

    assert ("ucisd", "upt2ccsd") in available_mixed_recipes()
    rec = get_mixed_recipe("upt2ccsd", guide="ucisd")
    assert rec.guide == "ucisd" and rec.walker_kind == "unrestricted" and rec.ham_basis == "uchol"
    assert rec.needs_amplitudes and rec.stage_guide is stage_ucisd_guide


@pytest.mark.slow
def test_ucisd_guide_fragments(o2t, request):
    """
    The fragment UCISD guide on the uchol fragment hamiltonian: the guide bundle is the
    meas.ucisd_uh one, the walkers start from the reference determinant so the tau = 0
    fragment energies add up to the LNO-CCSD energy, and a short run is finite.
    """
    if not request.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    from trot.meas.ucisd_uh import force_bias_kernel_uw_uh, overlap_uw_uh
    from trot.trial.ucisd import UcisdTrial

    mf = o2t["mf"]
    qmc: dict[str, Any] = dict(n_walkers=4, n_eql_blocks=2, n_blocks=12, dt=0.01, n_prop_steps=2)
    e_init = 0.0
    for frag in o2t["frags"]:
        with contextlib.redirect_stdout(io.StringIO()):
            fm = LnoFragMixed(mf, frag, guide="ucisd", chol_cut=CHOL_CUT, seed=5, **qmc)
            job = fm.build_job()
            assert isinstance(job.trial_data, UcisdTrial)
            assert job.meas_ops.overlap is overlap_uw_uh
            assert job.meas_ops.kernels["force_bias"] is force_bias_kernel_uw_uh
            e, err = fm.kernel()
        assert np.isfinite(e) and np.isfinite(fm.guide_e_tot)
        e_init += fm.qmc_result.frag_init_energy
    assert abs(e_init - o2t["e_ccsd"]) < 1e-6
