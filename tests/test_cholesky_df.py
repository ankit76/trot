"""
The AO cholesky vectors of a density fitted mean field come from its DF tensor
(cholesky.ao_cholesky / df2chol), for the restricted hamiltonian of AfqmcMixed
(staging_u.stage_ham_input_df) and for the unrestricted one (staging_u.stage_uh); a mean
field without DF still goes through staging's modified cholesky of the exact ERIs.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot import config

config.configure_once()

from trot.cholesky import ao_cholesky, df2chol, df_cderi
from trot.staging import StagedMfOrCc, _stage_ham_input
from trot.staging import stage as stage_inputs
from trot.staging_u import stage_ham_input_df, stage_uh

CHOL_CUT = 1e-6


@pytest.fixture(scope="module")
def h2o():
    mol = gto.M(
        atom="O 0 0 0; H 0.957 0 0; H -0.24 0.927 0", basis="6-31g", verbose=0, max_memory=8000
    )
    mf: Any = scf.RHF(mol)
    mf.kernel()
    mf_df: Any = scf.RHF(mol).density_fit()
    mf_df.kernel()
    return dict(mol=mol, mf=mf, mf_df=mf_df)


def _e_hf(ham) -> float:
    """Closed-shell HF energy of a restricted HamInput with the reference in the leading columns."""
    nocc = ham.nelec[0]
    lo = ham.chol[:, :nocc, :nocc]
    e = ham.h0 + 2 * np.trace(ham.h1[:nocc, :nocc])
    e += 2 * np.sum(np.trace(lo, axis1=1, axis2=2) ** 2) - np.einsum("gij,gji->", lo, lo)
    return float(e)


def test_df2chol_factors_the_df_tensor(h2o):
    mf_df = h2o["mf_df"]
    cderi = df_cderi(mf_df)
    assert cderi is not None and df_cderi(h2o["mf"]) is None
    nao = h2o["mol"].nao
    chol = df2chol(cderi, max_error=1e-10)
    assert chol.shape[1:] == (nao, nao) and chol.shape[0] <= cderi.shape[0]
    eri_df = np.einsum("gp,gq->pq", cderi, cderi)  # (n_pair, n_pair) DF ERIs
    rows, cols = np.tril_indices(nao)
    lvec = chol[:, rows, cols]
    assert np.abs(np.einsum("gp,gq->pq", lvec, lvec) - eri_df).max() < 1e-8


def test_ao_cholesky_uses_df_tensor(h2o):
    with contextlib.redirect_stdout(io.StringIO()):
        chol_df = ao_cholesky(h2o["mf_df"], chol_cut=CHOL_CUT)
        chol_eri = ao_cholesky(h2o["mf"], chol_cut=CHOL_CUT)
    nao = h2o["mol"].nao
    assert chol_df.shape[1] == nao * nao and chol_eri.shape[1] == nao * nao
    assert chol_df.shape[0] <= df_cderi(h2o["mf_df"]).shape[0]
    # both factor an ERI tensor; DF and exact ERIs differ at the DF error level only
    eri_df = chol_df.T @ chol_df
    eri_ex = chol_eri.T @ chol_eri
    assert 1e-8 < np.abs(eri_df - eri_ex).max() < 1e-2


def test_stage_ham_input_df_matches_staging_without_df(h2o):
    with contextlib.redirect_stdout(io.StringIO()):
        ours = stage_ham_input_df(StagedMfOrCc(h2o["mf"], 1), chol_cut=CHOL_CUT)
        his = _stage_ham_input(StagedMfOrCc(h2o["mf"], 1), chol_cut=CHOL_CUT, verbose=False)
    assert ours.norb == his.norb and ours.nelec == his.nelec and ours.frozen == his.frozen == 1
    assert abs(ours.h0 - his.h0) < 1e-12
    assert np.array_equal(ours.h1, his.h1) and np.array_equal(ours.chol, his.chol)


def test_stage_ham_input_df_reproduces_the_df_hf_energy(h2o):
    mf_df = h2o["mf_df"]
    with contextlib.redirect_stdout(io.StringIO()):
        ham = stage_ham_input_df(StagedMfOrCc(mf_df, 0), chol_cut=1e-8)
        ham_fc = stage_ham_input_df(StagedMfOrCc(mf_df, 1), chol_cut=1e-8)
        ham_ex = stage_ham_input_df(StagedMfOrCc(h2o["mf"], 0), chol_cut=1e-8)
    assert ham.basis == "restricted" and ham.chol.shape[0] <= df_cderi(mf_df).shape[0]
    assert abs(_e_hf(ham) - mf_df.e_tot) < 1e-7
    assert ham_fc.norb == ham.norb - 1 and abs(_e_hf(ham_fc) - mf_df.e_tot) < 1e-7
    assert abs(_e_hf(ham_ex) - h2o["mf"].e_tot) < 1e-7
    # the DF and exact hamiltonians differ, since the DF mean field does
    assert abs(mf_df.e_tot - h2o["mf"].e_tot) > 1e-7


def test_afqmc_mixed_stages_the_df_hamiltonian(h2o):
    from trot.afqmc import AfqmcMixed

    mf_df = h2o["mf_df"]
    mycc: Any = cc.CCSD(mf_df, frozen=1)
    mycc.kernel()
    with contextlib.redirect_stdout(io.StringIO()):
        af = AfqmcMixed(mycc, trial="pt2ccsd_bar", chol_cut=1e-8)
        staged = af.stage()
        ham_df = stage_ham_input_df(StagedMfOrCc(mycc, 1), chol_cut=1e-8)
        plain = stage_inputs(mf_df, norb_frozen_core=1, chol_cut=1e-8)
    assert np.array_equal(staged.ham.chol, ham_df.chol) and staged.ham.norb == ham_df.norb
    assert abs(_e_hf(staged.ham) - mf_df.e_tot) < 1e-7
    # staging's own path decomposes the exact ERIs instead, so its vectors differ
    assert staged.ham.chol.shape != plain.ham.chol.shape or not np.allclose(
        staged.ham.chol, plain.ham.chol
    )
    assert abs(_e_hf(plain.ham) - h2o["mf"].e_tot) < 1e-7


def test_stage_uh_uses_df_tensor(h2o):
    mf_df = h2o["mf_df"]
    umf: Any = mf_df.to_uhf()
    with contextlib.redirect_stdout(io.StringIO()):
        staged = stage_uh(umf, chol_cut=1e-8)
        ham_r = stage_ham_input_df(StagedMfOrCc(mf_df, 0), chol_cut=1e-8)
    ham = staged.ham
    assert ham.basis == "uchol" and ham.chol_a.shape[0] == ham_r.chol.shape[0]
    assert np.allclose(ham.chol_a, ham_r.chol, atol=1e-10) and np.allclose(
        ham.chol_b, ham_r.chol, atol=1e-10
    )
