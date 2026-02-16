import copy
import os
import pickle
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import jax
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from jax import numpy as jnp
from pyscf import __config__, ao2mo, df, dft, lib, mcscf, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from ad_afqmc import Wigner_small_d, hamiltonian, propagation, sampling, wavefunctions
from ad_afqmc.config import mpi_print as print

from ad_afqmc.prep import PrepAfqmc

# modified cholesky for a given matrix
def modified_cholesky(mat: np.ndarray, max_error: float = 1e-6) -> np.ndarray:
    """Modified cholesky decomposition for a given matrix.

    Args:
        mat (np.ndarray): Matrix to decompose.
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        np.ndarray: Cholesky vectors.
    """
    diag = mat.diagonal()
    size = mat.shape[0]
    nchol_max = size
    chol_vecs = np.zeros((nchol_max, nchol_max))
    # ndiag = 0
    nu = np.argmax(diag)
    delta_max = diag[nu]
    Mapprox = np.zeros(size)
    chol_vecs[0] = np.copy(mat[nu]) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error and (nchol + 1) < nchol_max:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (mat[nu] - R) / (delta_max + 1e-10) ** 0.5
        nchol += 1

    return chol_vecs[:nchol]


# prepare phaseless afqmc with mf or cc trial
def prep_afqmc(
    mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, CCSD, UCCSD],
    basis_coeff: Optional[np.ndarray] = None,
    norb_frozen: int = 0,
    chol_cut: float = 1e-5,
    integrals: Optional[dict] = None,
    tmpdir: str = "./",
    write_to_disk: bool = False,
) -> dict:
    """Prepare AFQMC calculation with mean field or cc trial wavefunctions.

    Args:
        mf (Union[scf.uhf.UHF, scf.rhf.RHF, mcscf.mc1step.CASSCF]): pyscf mean field object. Used for generating integrals (if not provided) and trial.
        basis_coeff (np.ndarray, optional): Orthonormal basis used for afqmc, given in the basis of ao's. If not provided mo_coeff of mf is used as the basis.
        norb_frozen (int, optional): Number of frozen orbitals. Not supported for custom integrals.
        chol_cut (float, optional): Cholesky decomposition cutoff.
        integrals (dict, optional): Dictionary of integrals in an orthonormal basis, {"h0": enuc, "h1": h1e, "h2": eri}.
        tmpdir (str, optional): Directory to write integrals and mo coefficients. Defaults to "./".
        write_to_disk (bool, optional): Whether to write integrals and mo coefficients to disk. Defaults to False.

    Returns:
        dict: Dictionary containing integrals and trial wavefunction information.
    """

    print("#\n# Preparing AFQMC calculation")
    pyscf_prep = {}

    if isinstance(mf_or_cc, (CCSD, UCCSD)):
        if write_to_disk:
            write_pyscf_ccsd(mf_or_cc, tmpdir)
        else:
            amplitudes = get_ci_amplitudes_from_cc(mf_or_cc)
            pyscf_prep["amplitudes"] = amplitudes
        mf = mf_or_cc._scf
        if mf_or_cc.frozen is not None:
            assert (
                type(mf_or_cc.frozen) is int
            ), "Frozen orbitals should be given as an integer."
            norb_frozen = mf_or_cc.frozen
        else:
            norb_frozen = 0

    else:
        mf = mf_or_cc

    mol = mf.mol
    # choose the orbital basis
    if basis_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            basis_coeff = mf.mo_coeff[0]
        else:
            basis_coeff = mf.mo_coeff

    # calculate cholesky integrals
    h1e, chol, nelec, enuc, nbasis, _ = compute_cholesky_integrals(
        mol, mf, basis_coeff, integrals, norb_frozen, chol_cut
    )

    print("# Finished calculating Cholesky integrals\n#")

    nbasis = h1e.shape[-1]
    print("# Size of the correlation space:")
    print(f"# Number of electrons: {nelec}")
    print(f"# Number of basis functions: {nbasis}")
    print(f"# Number of Cholesky vectors: {chol.shape[0]}\n#")
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))

    # write trial mo coefficients
    if write_to_disk:
        trial_coeffs = write_trial_coeffs(
            mol, mf, basis_coeff, nbasis, norb_frozen, tmpdir
        )
    else:
        trial_coeffs = get_trial_coeffs(mol, mf, basis_coeff, nbasis, norb_frozen)

    if write_to_disk:
        write_dqmc(
            h1e,
            h1e_mod,
            chol,
            sum(nelec),
            nbasis,
            enuc,
            ms=mol.spin,
            filename=tmpdir + "/FCIDUMP_chol",
            mo_coeffs=trial_coeffs,
        )
    else:
        pyscf_prep["header"] = np.array([sum(nelec), nbasis, mol.spin, chol.shape[0]])
        pyscf_prep["hcore"] = h1e.flatten()
        pyscf_prep["hcore_mod"] = h1e_mod.flatten()
        pyscf_prep["chol"] = chol.flatten()
        pyscf_prep["energy_core"] = enuc
        if trial_coeffs is not None:
            pyscf_prep["trial_coeffs"] = trial_coeffs
    return pyscf_prep


def getCollocationMatrices(
    mol, grid_level=0, thc_eps=1.0e-4, mo1=None, mo2=None, alpha=0.25
):
    """Return the matrices X1 and X2 the equation
    Z(ab,r) = X1(a,P) X2(b,P) Xi(P,r)
    can be satisfied well. Note that in this code we do not evaluate Xi, for that call the least square solver.

    The two matrices X1 and X2 are the same when the space spanned by {a} and {b} is the same, e.g. all AOs.

    Args:
        mol: pysf Mol object
        grid_level (int, optional): The size of the Becke grid used in these calculations. Defaults to 0.
        thc_eps (double, optional): The threshold used in the THC calculation, the lower it is the more accurate the approximation. Defaults to 1.e-4.
        mo1 (array, optional): The orbital coefficient in terms of AOs mo1(ao, mo)
                Defaults to None, so this assumes mo1 is identity and all AOs are fit
                Other options might be all mos, all occupied, all virtual etc.
        mo2 (array, optional): The orbital coefficient in terms of AOs mo2(ao, mo)
                Defaults to None, so this assumes mo2 is identity and all AOs are fit
                Other options might be all mos, all occupied, all virtual etc.
        alpha (double, optional): the exponent on the weight in the collocation matrix
            X(a,P) = phi_a(r_P) w_p^\alpha
    """

    grids = dft.gen_grid.Grids(mol)  # type: ignore
    grids.level = 0
    grids.build()

    coords = grids.coords
    weights = grids.weights
    ao = mol.eval_gto("GTOval_sph", coords)  # aos on coords
    X1 = np.einsum("ri,r->ri", (ao @ mo1), abs(weights) ** alpha)  # mos on coords
    X2 = np.einsum("ri,r->ri", (ao @ mo2), abs(weights) ** alpha)  # mos on coords

    P = doISDF(X1, X2, thc_eps)
    return X1[P], X2[P]


def doISDF(X1: np.ndarray, X2: np.ndarray, thc_eps=1.0e-4):
    """Return the pivot points P such that the equation
    Z(ab,r) = X1(a,P) X2(b,P) Xi(P,r)
    can be satisfied to high accuracy.

    Args:
        X1 (np.ndarray): The first matrix
        X2 (np.ndarray): The second matrix
        thc_eps (_type_, optional): The threshold up to which the pivot points are retained. Defaults to 1.e-4.
    """

    S = np.einsum("ri,si->rs", X1, X1) * np.einsum("ri,si->rs", X2, X2)
    R, P, rankc, info = scipy.linalg.lapack.dpstrf(S, thc_eps**2)
    P = P[:rankc] - 1
    return P


def solveLS(T, X1, X2):
    """T is a three index quantity and we want to obtain Xi such that
    T(a,b,M) = X1(P,a) X2(P,b) Xi(P, M)
    is satisfied approximately using least square minimization.

    Args:
        T (array): The input three index quantity
        X1 (array): The two index collocation array, the size of the second index should be equal to the size of first index in T
        X2 (array): The two index collocation array, the size of the seonc idnex should be equal to the size of the second index in T
             and the size of the first index should be equal to the size of first index in X1

    Returns:
        Xi (array): The matrix that satisfies the equation above
    """

    S = np.einsum("Pa,Qa->PQ", X1, X1) * np.einsum("Pb,Qb->PQ", X2, X2)
    X12 = jnp.einsum("Pa,Pb->abP", X1, X2)
    Stilde = jnp.einsum("abP, abM->PM", X12, T)

    L = scipy.linalg.cholesky(S, lower=True)
    Xi = scipy.linalg.cho_solve((L, True), Stilde, overwrite_b=True)

    return Xi


def solveLS_twoSided(T, X1, X2):
    """T is a four index quantity and we want to obtain V such that
    T(a,b,c,d) = X1(P,a) X2(P,b) V(P, Q) X1(Q,c) X2(Q,d)
    is satisfied approximately using least square minimization.

    Args:
        T (array): The input four index quantity
        X1 (array): The two index collocation array, the size of the second index should be equal to the size of first and third index in T
        X2 (array): The two index collocation array, the size of the second idnex should be equal to the size of the second and fourth index in T
             and the size of the first index should be equal to the size of first index in X1

    Returns:
        V (array): The matrix that satisfies the equation above
    """

    S = np.einsum("Pa,Qa->PQ", X1, X1) * np.einsum("Pb,Qb->PQ", X2, X2)
    L = scipy.linalg.cholesky(S, lower=True)

    X12 = jnp.einsum("Pa,Pb->abP", X1, X2)
    E = jnp.einsum("abP, abcd->Pcd", X12, T).reshape(X1.shape[0], -1)

    E = scipy.linalg.cho_solve((L, True), E).reshape(-1, T.shape[2], T.shape[3])

    E = jnp.einsum("Pcd, cdQ->PQ", E, X12).T
    V = scipy.linalg.cho_solve((L, True), E)

    # symmetrize it
    V = 0.5 * (V + V.T)

    return V


# cholesky generation functions are from pauxy
def generate_integrals(mol, hcore, X, chol_cut=1e-5, verbose=False, DFbas=None):
    # Unpack SCF data.
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    if verbose:
        print(" # Transforming hcore and eri to ortho AO basis.")
    if len(X.shape) == 2:
        h1e = np.dot(X.T.conj(), np.dot(hcore, X))
    elif len(X.shape) == 3:
        h1e = np.dot(X[0].T.conj(), np.dot(hcore, X[0]))

    if DFbas is not None:
        chol_vecs = df.incore.cholesky_eri(mol, auxbasis=DFbas)
        chol_vecs = lib.unpack_tril(chol_vecs).reshape(chol_vecs.shape[0], -1)
    else:  # do cholesky
        # nbasis = h1e.shape[-1]
        # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
        if verbose:
            print(" # Performing modified Cholesky decomposition on ERI tensor.")
        chol_vecs = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)

    if verbose:
        print(" # Orthogonalising Cholesky vectors.")
    start = time.time()

    # Step 2.a Orthogonalise Cholesky vectors.
    if len(X.shape) == 2 and X.shape[0] != X.shape[1]:
        chol_vecs = ao2mo_chol_copy(chol_vecs, X)
    elif len(X.shape) == 2:
        ao2mo_chol(chol_vecs, X)
    elif len(X.shape) == 3:
        ao2mo_chol(chol_vecs, X[0])
    if verbose:
        print(" # Time to orthogonalise: %f" % (time.time() - start))
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec
    return h1e, chol_vecs, nelec, enuc


def ao2mo_chol(eri, C):
    nb = C.shape[-1]
    for i, cv in enumerate(eri):
        half = np.dot(cv.reshape(nb, nb), C)
        eri[i] = np.dot(C.conj().T, half).ravel()


def ao2mo_chol_copy(eri, C):
    nb = C.shape[0]
    nmo = C.shape[1]
    eri_copy = np.zeros((eri.shape[0], nmo * nmo))
    for i, cv in enumerate(eri):
        half = np.dot(cv.reshape(nb, nb), C)
        eri_copy[i] = np.dot(C.conj().T, half).ravel()
    return eri_copy


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs." % nchol_max)
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor(
        "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    )
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    return chol_vecs[:nchol]


# write cholesky integrals
def write_dqmc(
    hcore,
    hcore_mod,
    chol,
    nelec,
    nmo,
    enuc,
    ms=0,
    filename="FCIDUMP_chol",
    mo_coeffs=None,
):
    assert len(chol.shape) == 2
    with h5py.File(filename, "w") as fh5:
        fh5["header"] = np.array([nelec, nmo, ms, chol.shape[0]])
        fh5["hcore"] = hcore.flatten()
        fh5["hcore_mod"] = hcore_mod.flatten()
        fh5["chol"] = chol.flatten()
        fh5["energy_core"] = enuc
        if mo_coeffs is not None:
            fh5["mo_coeffs_up"] = mo_coeffs[0]
            fh5["mo_coeffs_dn"] = mo_coeffs[1]


def finite_difference_properties(
    mol,
    observable,
    observable_constant=0.0,
    epsilon=1.0e-5,
    norb_frozen=0,
    hf_type="rhf",
    relaxed=True,
    dm=None,
):
    from pyscf import cc

    print(
        f'#\n# Orbital {"" if relaxed else "un"}relaxed finite difference properties using {hf_type} reference'
    )
    print(f"# epsilon: {epsilon}")
    mf_coeff, mo_occ = None, None
    if hf_type == "rhf":
        mf = scf.RHF(mol)
    elif hf_type == "uhf":
        mf = scf.UHF(mol)
    if not relaxed:
        mf.verbose = 1
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        if hf_type == "uhf":
            mo1 = mf.stability(external=True)[0]
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
        mf_coeff = mf.mo_coeff.copy()  # type: ignore
        mf_occ = mf.mo_occ.copy()  # type: ignore

    h1e = mf.get_hcore() - epsilon * observable
    mf.get_hcore = lambda *args: h1e  # type: ignore
    mf.verbose = 1
    if relaxed:
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        if hf_type == "uhf":
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
    emf_m = mf.e_tot - epsilon * observable_constant
    mycc = cc.CCSD(mf)
    mycc.frozen = norb_frozen
    mycc.kernel()
    emp2_m = mycc.e_hf + mycc.emp2 - epsilon * observable_constant  # type: ignore
    eccsd_m = mycc.e_tot - epsilon * observable_constant
    et = mycc.ccsd_t()
    eccsdpt_m = mycc.e_tot + et - epsilon * observable_constant

    if hf_type == "rhf":
        mf = scf.RHF(mol)
    elif hf_type == "uhf":
        mf = scf.UHF(mol)
    if not relaxed:
        mf.verbose = 1
        mf.mo_coeff = mf_coeff  # type: ignore
        mf.mo_occ = mf_occ
    h1e = mf.get_hcore() + epsilon * observable
    mf.get_hcore = lambda *args: h1e  # type: ignore
    mf.verbose = 1
    if relaxed:
        if dm is None:
            mf.kernel()
        else:
            mf.kernel(dm)
        if hf_type == "uhf":
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
            mo1 = mf.stability(external=True)[0]  # type: ignore
            mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
    emf_p = mf.e_tot + epsilon * observable_constant
    mycc = cc.CCSD(mf)
    mycc.frozen = norb_frozen
    mycc.kernel()
    emp2_p = mycc.e_hf + mycc.emp2 + epsilon * observable_constant  # type: ignore
    eccsd_p = mycc.e_tot + epsilon * observable_constant
    et = mycc.ccsd_t()
    eccsdpt_p = mycc.e_tot + et + epsilon * observable_constant

    print("# FD single point energies:")
    print(f"# emf_m: {emf_m}, emf_p: {emf_p}")
    print(f"# emp2_m: {emp2_m}, emp2_p: {emp2_p}")
    print(f"# eccsd_m: {eccsd_m}, eccsd_p: {eccsd_p}")
    print(f"# eccsdpt_m: {eccsdpt_m}, eccsd_p: {eccsdpt_p}")

    obs_mf = (emf_p - emf_m) / 2 / epsilon
    obs_mp2 = (emp2_p - emp2_m) / 2 / epsilon
    obs_ccsd = (eccsd_p - eccsd_m) / 2 / epsilon
    obs_ccsdpt = (eccsdpt_p - eccsdpt_m) / 2 / epsilon

    print("# FD Observables:")
    if relaxed:
        print(f"HF observable: {obs_mf}")
    print(f"MP2 observable: {obs_mp2}")
    print(f"CCSD observable: {obs_ccsd}")
    print(f"CCSD(T) observable: {obs_ccsdpt}")
    return obs_mf, obs_mp2, obs_ccsd, obs_ccsdpt


def get_fci_state(
    fci: Any, ndets: Optional[int] = None, tol: float = 1.0e-4, root=0
) -> dict:
    """Get FCI state from a pyscf FCI object.

    Args:
        fci: FCI object.
        ndets: Number of determinants to include (sorted by coeffs).
        tol: Tolerance for including determinants.

    Returns:
        Dictionary with determinants as keys and coefficients as values.
    """
    ci_coeffs = fci.ci
    if isinstance(ci_coeffs, list):
        ci_coeffs = ci_coeffs[root]
    norb = fci.norb
    nelec = fci.nelec
    if ndets is None:
        ndets = int(ci_coeffs.size)
    coeffs, occ_a, occ_b = zip(
        *fci.large_ci(ci_coeffs, norb, nelec, tol=tol, return_strs=False)
    )
    coeffs, occ_a, occ_b = zip(
        *sorted(zip(coeffs, occ_a, occ_b), key=lambda x: -abs(x[0]))
    )
    state = {}
    for i in range(min(ndets, len(coeffs))):
        det = [[0 for _ in range(norb)], [0 for _ in range(norb)]]
        for j in range(nelec[0]):
            det[0][occ_a[i][j]] = 1
        for j in range(nelec[1]):
            det[1][occ_b[i][j]] = 1
        state[tuple(map(tuple, det))] = coeffs[i]
    return state


def get_excitations(
    state: Optional[dict] = None,
    num_core: int = 0,
    fname: str = "dets.bin",
    ndets: Optional[int] = None,
    max_excitation: int = 10,
) -> Tuple[dict, dict, dict, dict, dict, np.ndarray]:
    """Use given a state or read determinants from a binary file and return excitations.

    | psi_t > = sum_i coeff_i E_i | d_0 > (note that coeff_i differ from those in sum_i coeff_i_1 | d_i > by parity factors).

    Args:
        state: Dictionary with determinants as keys and coefficients as values.
        num_core: Number of core orbitals.
        fname: Binary file containing determinants.
        ndets: Number of determinants to read.
        max_excitation: Maximum excitation level (alpha + beta) to consider.

    Returns:
        Acre: Alpha creation indices.
        Ades: Alpha destruction indices.
        Bcre: Beta creation indices.
        Bdes: Beta destruction indices.
        coeff: Coefficients of the determinants.
        ref_det: Reference determinant.
    """
    if state is None:
        _, state, ndets_all = read_dets(fname, ndets)
        if ndets is None:
            ndets = ndets_all
    else:
        if ndets is None:
            ndets = len(state)

    dets = list(state.keys())[:ndets]
    Acre, Ades, Bcre, Bdes, coeff = {}, {}, {}, {}, {}
    d0 = dets[0]
    d0a, d0b = np.asarray(d0[0]), np.asarray(d0[1])
    for d in dets:
        dia, dib = np.asarray(d[0]), np.asarray(d[1])
        nex = (np.sum(abs(dia - d0a)) // 2, np.sum(abs(dib - d0b)) // 2)
        if nex[0] + nex[1] > max_excitation:
            continue
        coeff[nex] = coeff.get(nex, []) + [state[d]]
        if nex[0] > 0 and nex[1] > 0:
            occ_idx_a_rel = (
                np.arange(sum(d0a))[
                    (np.cumsum(d0a) - 1)[np.nonzero((d0a - dia) > 0)[0]]
                ],
            )
            occ_idx_b_rel = (
                np.arange(sum(d0b))[
                    (np.cumsum(d0b) - 1)[np.nonzero((d0b - dib) > 0)[0]]
                ],
            )
            Acre[nex], Ades[nex], Bcre[nex], Bdes[nex] = (
                Acre.get(nex, []) + [occ_idx_a_rel],
                Ades.get(nex, []) + [np.nonzero((d0a - dia) < 0)],
                Bcre.get(nex, []) + [occ_idx_b_rel],
                Bdes.get(nex, []) + [np.nonzero((d0b - dib) < 0)],
            )
            coeff[nex][-1] *= parity(
                d0a, np.nonzero((d0a - dia) > 0), Ades[nex][-1]
            ) * parity(d0b, np.nonzero((d0b - dib) > 0), Bdes[nex][-1])

        elif nex[0] > 0 and nex[1] == 0:
            occ_idx_a_rel = (
                np.arange(sum(d0a))[
                    (np.cumsum(d0a) - 1)[np.nonzero((d0a - dia) > 0)[0]]
                ],
            )
            Acre[nex], Ades[nex] = Acre.get(nex, []) + [occ_idx_a_rel], Ades.get(
                nex, []
            ) + [np.nonzero((d0a - dia) < 0)]
            coeff[nex][-1] *= parity(d0a, np.nonzero((d0a - dia) > 0), Ades[nex][-1])

        elif nex[0] == 0 and nex[1] > 0:
            occ_idx_b_rel = (
                np.arange(sum(d0b))[
                    (np.cumsum(d0b) - 1)[np.nonzero((d0b - dib) > 0)[0]]
                ],
            )
            Bcre[nex], Bdes[nex] = Bcre.get(nex, []) + [occ_idx_b_rel], Bdes.get(
                nex, []
            ) + [np.nonzero((d0b - dib) < 0)]
            coeff[nex][-1] *= parity(d0b, np.nonzero((d0b - dib) > 0), Bdes[nex][-1])

    coeff[(0, 0)] = np.asarray(coeff[(0, 0)]).reshape(
        -1,
    )

    # fill up the arrays up to max_excitation
    for i in range(1, max_excitation + 1):
        # singe alpha excitation
        if (i, 0) in Ades:
            Ades[(i, 0)] = np.asarray(Ades[(i, 0)]).reshape(-1, i) + num_core
            Acre[(i, 0)] = np.asarray(Acre[(i, 0)]).reshape(-1, i) + num_core
            coeff[(i, 0)] = np.asarray(coeff[(i, 0)]).reshape(
                -1,
            )
        else:
            Ades[(i, 0)] = np.zeros((1, i), dtype=int)
            Acre[(i, 0)] = np.zeros((1, i), dtype=int)
            coeff[(i, 0)] = np.zeros((1,))

        # singe beta excitation
        if (0, i) in Bdes:
            Bdes[(0, i)] = np.asarray(Bdes[(0, i)]).reshape(-1, i) + num_core
            Bcre[(0, i)] = np.asarray(Bcre[(0, i)]).reshape(-1, i) + num_core
            coeff[(0, i)] = np.asarray(coeff[(0, i)]).reshape(
                -1,
            )
        else:
            Bdes[(0, i)] = np.zeros((1, i), dtype=int)
            Bcre[(0, i)] = np.zeros((1, i), dtype=int)
            coeff[(0, i)] = np.zeros((1,))

        # alpha-beta
        if i != 0:
            for j in range(1, max_excitation + 1 - i):
                if (i, j) in Ades:
                    Ades[(i, j)] = np.asarray(Ades[(i, j)]).reshape(-1, i) + num_core
                    Acre[(i, j)] = np.asarray(Acre[(i, j)]).reshape(-1, i) + num_core
                    Bdes[(i, j)] = np.asarray(Bdes[(i, j)]).reshape(-1, j) + num_core
                    Bcre[(i, j)] = np.asarray(Bcre[(i, j)]).reshape(-1, j) + num_core
                    coeff[(i, j)] = np.asarray(coeff[(i, j)]).reshape(
                        -1,
                    )
                else:
                    Ades[(i, j)] = np.zeros((1, j), dtype=int)
                    Acre[(i, j)] = np.zeros((1, j), dtype=int)
                    Bdes[(i, j)] = np.zeros((1, j), dtype=int)
                    Bcre[(i, j)] = np.zeros((1, j), dtype=int)
                    coeff[(i, j)] = np.zeros((1,))
    ref_det = np.array([d0a, d0b])
    return Acre, Ades, Bcre, Bdes, coeff, ref_det


# a_i^dag a_j
def ci_parity(det, i, j):
    assert det[i] == 0 and det[j] == 1
    if i > j:
        return (-1.0) ** (det[j + 1 : i].count(1))
    else:
        return (-1.0) ** (det[i + 1 : j].count(1))


def calculate_ci_1rdm(norbs, state, ndets=100):
    norm_square = 0.0
    counter = 0
    rdm = [np.zeros((norbs, norbs)), np.zeros((norbs, norbs))]
    for det, coeff in state.items():
        counter += 1
        norm_square += coeff * coeff
        det_list = list(map(list, det))
        for sz in range(2):
            occ = [i for i, x in enumerate(det_list[sz]) if x == 1]
            empty = [i for i, x in enumerate(det_list[sz]) if x == 0]
            for j in occ:
                rdm[sz][j, j] += coeff * coeff
                for i in empty:
                    new_det = copy.deepcopy(det_list)
                    new_det[sz][i] = 1
                    new_det[sz][j] = 0
                    new_det = tuple(map(tuple, new_det))
                    if new_det in state:
                        rdm[sz][i, j] += (
                            ci_parity(det_list[sz], i, j) * state[new_det] * coeff
                        )
        if counter == ndets:
            break

    rdm[0] /= norm_square
    rdm[1] /= norm_square
    return rdm


# reading dets from dice
def read_dets(
    fname: str = "dets.bin", ndets: Optional[int] = None
) -> Tuple[int, dict, int]:
    """Read determinants from a binary file generated by Dice.

    Args:
        fname: Binary file containing determinants.
        ndets: Number of determinants to read.

    Returns:
        norbs: Number of orbitals.
        state: Dictionary with determinant (tuple of up and down occupation number tuples) keys and coefficient values.
        ndets_all: Total number of determinants in the file.
    """
    state = {}
    norbs = 0
    with open(fname, "rb") as f:
        ndets_all = struct.unpack("i", f.read(4))[0]
        norbs = struct.unpack("i", f.read(4))[0]
        if ndets is None:
            ndets = int(ndets_all)
        for _ in range(ndets):
            coeff = struct.unpack("d", f.read(8))[0]
            det = [[0 for _ in range(norbs)], [0 for _ in range(norbs)]]
            for j in range(norbs):
                occ = struct.unpack("c", f.read(1))[0]
                if occ == b"a":
                    det[0][j] = 1
                elif occ == b"b":
                    det[1][j] = 1
                elif occ == b"2":
                    det[0][j] = 1
                    det[1][j] = 1
            state[tuple(map(tuple, det))] = coeff

    return norbs, state, ndets_all


def write_dets(state: dict, norbs: int, fname: str = "dets.bin") -> None:
    """Write determinants to a binary file in Dice format.

    Args:
        state: Dictionary with determinant (tuple of up and down occupation number tuples) keys
              and coefficient values.
        norbs: Number of orbitals.
        fname: Output binary filename.

    The binary format is:
    - Number of determinants (4 bytes, integer)
    - Number of orbitals (4 bytes, integer)
    For each determinant:
    - Coefficient (8 bytes, double)
    - Orbital occupations (1 byte per orbital, character):
      '0' for empty, 'a' for alpha, 'b' for beta, '2' for double
    """
    import struct

    ndets = len(state)

    with open(fname, "wb") as f:
        # Write number of determinants and orbitals
        f.write(struct.pack("i", ndets))
        f.write(struct.pack("i", norbs))

        # Write each determinant
        for det, coeff in state.items():
            # Write coefficient
            f.write(struct.pack("d", coeff))

            # Convert occupation numbers to Dice format and write
            alpha, beta = det
            for i in range(norbs):
                if alpha[i] == 0 and beta[i] == 0:
                    f.write(struct.pack("c", b"0"))
                elif alpha[i] == 1 and beta[i] == 0:
                    f.write(struct.pack("c", b"a"))
                elif alpha[i] == 0 and beta[i] == 1:
                    f.write(struct.pack("c", b"b"))
                elif alpha[i] == 1 and beta[i] == 1:
                    f.write(struct.pack("c", b"2"))
                else:
                    raise ValueError(
                        f"Invalid occupation numbers at orbital {i}: "
                        f"alpha={alpha[i]}, beta={beta[i]}"
                    )


def parity(d0: np.ndarray, cre: Sequence, des: Sequence) -> float:
    """Compute the parity for an excitation.

    Args:
        d0: Reference determinant.
        cre: Creation indices.
        des: Destruction indices.

    Returns:
        Parity of the excitation.
    """
    d = 1.0 * d0
    parity = 1.0

    C = np.asarray(cre).flatten()
    D = np.asarray(des).flatten()
    for i in range(C.shape[0]):
        I, A = min(D[i], C[i]), max(D[i], C[i])
        parity *= 1.0 - 2.0 * ((np.sum(d[I + 1 : A])) % 2)
        d[C[i]] = 0
        d[D[i]] = 1
    return float(parity)


def get_ci_amplitudes_from_cc(cc):
    if isinstance(cc, UCCSD):
        ci2aa = cc.t2[0] + 2 * np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[0])
        ci2aa = (ci2aa - ci2aa.transpose(0, 1, 3, 2)) / 2
        ci2aa = ci2aa.transpose(0, 2, 1, 3)
        ci2bb = cc.t2[2] + 2 * np.einsum("ia,jb->ijab", cc.t1[1], cc.t1[1])
        ci2bb = (ci2bb - ci2bb.transpose(0, 1, 3, 2)) / 2
        ci2bb = ci2bb.transpose(0, 2, 1, 3)
        ci2ab = cc.t2[1] + np.einsum("ia,jb->ijab", cc.t1[0], cc.t1[1])
        ci2ab = ci2ab.transpose(0, 2, 1, 3)
        ci1a = np.array(cc.t1[0])
        ci1b = np.array(cc.t1[1])
        return {
            "ci1a": ci1a,
            "ci1b": ci1b,
            "ci2aa": ci2aa,
            "ci2ab": ci2ab,
            "ci2bb": ci2bb,
            "t1a": cc.t1[0],
            "t1b": cc.t1[1],
            "t2aa": cc.t2[0],
            "t2ab": cc.t2[1],
            "t2bb": cc.t2[2],
        }
    else:
        ci2 = cc.t2 + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
        ci2 = ci2.transpose(0, 2, 1, 3)
        ci1 = np.array(cc.t1)
        return {
            "ci1": ci1,
            "ci2": ci2,
            "t1": cc.t1,
            "t2": cc.t2,
        }


def write_pyscf_ccsd(cc, tmpdir):
    ci_amps = get_ci_amplitudes_from_cc(cc)
    if isinstance(cc, UCCSD):
        np.savez(
            tmpdir + "/amplitudes.npz",
            ci1a=ci_amps["ci1a"],
            ci1b=ci_amps["ci1b"],
            ci2aa=ci_amps["ci2aa"],
            ci2ab=ci_amps["ci2ab"],
            ci2bb=ci_amps["ci2bb"],
            t1a=cc.t1[0],
            t1b=cc.t1[1],
            t2aa=cc.t2[0],
            t2ab=cc.t2[1],
            t2bb=cc.t2[2],
        )
    else:
        ci2 = cc.t2 + np.einsum("ia,jb->ijab", np.array(cc.t1), np.array(cc.t1))
        ci2 = ci2.transpose(0, 2, 1, 3)
        ci1 = np.array(cc.t1)
        np.savez(
            tmpdir + "/amplitudes.npz",
            ci1=ci1,
            ci2=ci2,
            t1=cc.t1,
            t2=cc.t2,
        )


def compute_cholesky_integrals(mol, mf, basis_coeff, integrals, norb_frozen, chol_cut):
    print("# Calculating Cholesky integrals")
    assert basis_coeff.dtype == "float", "Only implemented for real-valued MOs"
    h1e, chol, nelec, enuc, nbasis, nchol = [None] * 6
    if integrals is not None:
        assert norb_frozen == 0, "Frozen orbitals not supported for custom integrals"
        enuc = integrals["h0"]
        h1e = integrals["h1"]
        eri = integrals["h2"]
        nelec = mol.nelec
        nbasis = h1e.shape[-1]
        norb = nbasis
        eri = ao2mo.restore(4, eri, norb)
        chol0 = modified_cholesky(eri, chol_cut)
        nchol = chol0.shape[0]
        chol = np.zeros((nchol, norb, norb))
        for i in range(nchol):
            for m in range(norb):
                for n in range(m + 1):
                    triind = m * (m + 1) // 2 + n
                    chol[i, m, n] = chol0[i, triind]
                    chol[i, n, m] = chol0[i, triind]

        # basis transformation
        h1e = basis_coeff.T @ h1e @ basis_coeff
        for gamma in range(nchol):
            chol[gamma] = basis_coeff.T @ chol[gamma] @ basis_coeff

    else:
        DFbas = None
        if getattr(mf, "with_df", None) is not None:
            DFbas = mf.with_df.auxmol.basis  # type: ignore
        h1e, chol, nelec, enuc = generate_integrals(
            mol, mf.get_hcore(), basis_coeff, chol_cut, DFbas=DFbas
        )
        nbasis = h1e.shape[-1]
        nelec = mol.nelec
        chol = chol.reshape((-1, nbasis, nbasis))
        if norb_frozen > 0:
            assert norb_frozen * 2 < sum(
                nelec
            ), "Frozen orbitals exceed number of electrons"
            mc = mcscf.CASSCF(
                mf, mol.nao - norb_frozen, mol.nelectron - 2 * norb_frozen
            )
            nelec = mc.nelecas  # type: ignore
            mc.mo_coeff = basis_coeff  # type: ignore
            h1e, enuc = mc.get_h1eff()  # type: ignore
            chol = chol[
                :,
                mc.ncore : mc.ncore + mc.ncas,  # type: ignore
                mc.ncore : mc.ncore + mc.ncas,  # type: ignore
            ]  # type: ignore
    return h1e, chol, nelec, enuc, nbasis, nchol


def get_trial_coeffs(mol, mf, basis_coeff, nbasis, norb_frozen):
    assert basis_coeff.dtype == "float", "Only implemented for real-valued MOs"
    trial_coeffs = np.empty((2, nbasis, nbasis))
    overlap = mf.get_ovlp(mol)
    if isinstance(mf, (scf.rohf.ROHF, scf.rhf.RHF)):
        uhfCoeffs = construct_uhf_coeffs_from_rhf(
            basis_coeff, mf.mo_coeff, overlap, norb_frozen, nbasis
        )
    elif isinstance(mf, scf.uhf.UHF):
        uhfCoeffs = construct_uhf_coeffs_from_uhf(
            basis_coeff, mf.mo_coeff, overlap, norb_frozen, nbasis
        )
    else:
        print("Cannot recognize type for mf object in write_trial")
        exit(1)

    trial_coeffs[0] = uhfCoeffs[:, :nbasis]
    trial_coeffs[1] = uhfCoeffs[:, nbasis:]
    return trial_coeffs


def write_trial_coeffs(mol, mf, basis_coeff, nbasis, norb_frozen, tmpdir):
    assert basis_coeff.dtype == "float", "Only implemented for real-valued MOs"
    trial_coeffs = get_trial_coeffs(mol, mf, basis_coeff, nbasis, norb_frozen)
    np.savez(tmpdir + "/mo_coeff.npz", mo_coeff=trial_coeffs)
    return trial_coeffs


def construct_uhf_coeffs_from_uhf(basis_coeff, mo_coeff, overlap, norb_frozen, nbasis):
    # Constructs UHF orbital coefficients relative to the
    # alpha basis (mo_coeff[0])
    uhfCoeffs = np.empty((nbasis, 2 * nbasis))

    q, r = np.linalg.qr(
        basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mo_coeff[0][:, norb_frozen:])
    )
    sgn = np.sign(r.diagonal())
    q = np.einsum("ij,j->ij", q, sgn)

    uhfCoeffs[:, :nbasis] = q
    q, r = np.linalg.qr(
        basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mo_coeff[1][:, norb_frozen:])
    )
    sgn = np.sign(r.diagonal())
    q = np.einsum("ij,j->ij", q, sgn)

    uhfCoeffs[:, nbasis:] = q

    return uhfCoeffs


def construct_uhf_coeffs_from_rhf(basis_coeff, mo_coeff, overlap, norb_frozen, nbasis):
    # Constructs UHF orbital coefficients (alpha = beta) relative to the
    # restricted basis
    uhfCoeffs = np.empty((nbasis, 2 * nbasis))

    q, r = np.linalg.qr(
        basis_coeff[:, norb_frozen:].T.dot(overlap).dot(mo_coeff[:, norb_frozen:])
    )
    sgn = np.sign(r.diagonal())
    q = np.einsum("ij,j->ij", q, sgn)
    uhfCoeffs[:, :nbasis] = q
    uhfCoeffs[:, nbasis:] = q

    return uhfCoeffs


def prep_afqmc_ghf_complex(mol, gmf: scf.ghf.GHF, tmpdir, chol_cut=1e-5):
    import scipy.linalg as la

    norb = np.shape(gmf.mo_coeff)[-1] // 2  # type: ignore
    mo_coeff = gmf.mo_coeff

    # Chol ao to mo
    chol_vecs = chunked_cholesky(mol, max_error=chol_cut)
    nchol = chol_vecs.shape[0]
    chol = np.zeros((nchol, 2 * norb, 2 * norb), dtype=complex)
    for i in range(nchol):
        chol_i = chol_vecs[i].reshape(norb, norb)
        chol_i = la.block_diag(chol_i, chol_i)
        chol[i] = mo_coeff.T.conj() @ chol_i @ mo_coeff

    # h ao to mo
    h = mo_coeff.T.conj() @ gmf.get_hcore() @ mo_coeff

    enuc = mol.energy_nuc()
    nbasis = h.shape[-1]
    print(f"nelec: {mol.nelec}")
    print(f"nbasis: {nbasis}")
    print(f"chol.shape: {chol.shape}")

    # Modified one-electron integrals
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("gik,gkj->ij", chol, chol, optimize="optimal")
    h_mod = h - v0
    chol = chol.reshape((chol.shape[0], -1))

    # Save
    write_dqmc(
        h,
        h_mod,
        chol,
        sum(mol.nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename=tmpdir + "/FCIDUMP_chol",
    )

    ovlp = gmf.get_ovlp(mol)
    q, r = np.linalg.qr(mo_coeff.T.conj() @ ovlp @ mo_coeff)
    sgn = np.sign(r.diagonal())
    q = np.einsum("ij,j->ij", q, sgn)
    np.savez(tmpdir + "/mo_coeff.npz", mo_coeff=[q, q])

    return h, h_mod, chol


def prep_afqmc_spinor(mol, mo_coeff, h_ao, n_ao, tmpdir, chol_cut=1e-5):
    import scipy.linalg as la

    try:
        from socutils.scf import spinor_hf  # type: ignore
    except ImportError:
        raise ImportError(
            "Please install socutils package to use spinor_hf module for AFQMC."
        )
    norb = n_ao

    # Chol ao to mo
    chol_vecs = chunked_cholesky(mol, max_error=chol_cut)
    nchol = chol_vecs.shape[0]
    chol = np.zeros((nchol, norb, norb), dtype=complex)
    for i in range(nchol):
        chol_i = chol_vecs[i].reshape(norb // 2, norb // 2)
        chol_i = spinor_hf.sph2spinor(mol, la.block_diag(chol_i, chol_i))
        chol[i] = mo_coeff.T.conj() @ chol_i @ mo_coeff

    # h ao to mo
    h = mo_coeff.T.conj() @ h_ao @ mo_coeff

    enuc = mol.energy_nuc()
    nbasis = h.shape[-1]
    print(f"nelec: {mol.nelec}")
    print(f"nbasis: {nbasis}")
    print(f"chol.shape: {chol.shape}")

    # Modified one-electron integrals
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("gik,gkj->ij", chol, chol, optimize="optimal")
    h_mod = h - v0
    chol = chol.reshape((chol.shape[0], -1))

    # Save
    write_dqmc(
        h,
        h_mod,
        chol,
        sum(mol.nelec),
        nbasis,
        enuc,
        ms=mol.spin,
        filename=tmpdir + "/FCIDUMP_chol",
    )

    ovlp = mol.intor("int1e_ovlp_spinor")
    q, r = np.linalg.qr(mo_coeff.T.conj() @ ovlp @ mo_coeff)
    sgn = np.sign(r.diagonal())
    q = np.einsum("ij,j->ij", q, sgn)
    np.savez(tmpdir + "/mo_coeff.npz", mo_coeff=[q, q])

    return h, h_mod, chol


def prep_afqmc_multislater(mf, state_dict, max_excitation, ndets, chol_cut=1e-5):
    Acre, Ades, Bcre, Bdes, coeff, ref_det = get_excitations(
        state=state_dict, max_excitation=max_excitation, ndets=ndets
    )  # this function reads the Dice dets.bin file if state is not provided

    prep_afqmc(mf, chol_cut=chol_cut)

    data = np.load("mo_coeff.npz")
    trial_coeffs = data["mo_coeff"]

    trial_coeffs[0] = trial_coeffs[0].T
    trial_coeffs[1] = trial_coeffs[1].T

    if isinstance(mf, (scf.rhf.RHF, scf.rohf.ROHF)):
        trial_coeffs = trial_coeffs[0]  # alpha = beta here

    wave_data = {
        "Acre": Acre,
        "Ades": Ades,
        "Bcre": Bcre,
        "Bdes": Bdes,
        "coeff": coeff,
        "ref_det": ref_det,
        "orbital_rotation": trial_coeffs,
    }

    import wavefunctions

    trial = wavefunctions.multislater(
        mf.mol.nao, mf.mol.nelec, max_excitation=max_excitation
    )

    import pickle

    with open("trial.pkl", "wb") as f:
        pickle.dump([trial, wave_data], f)


tmpdir = "."


def read_fcidump(tmp_dir: str = ".") -> Tuple:
    """
    Read the FCIDUMP_chol file containing Hamiltonian elements.

    Args:
        tmp_dir: Directory containing the FCIDUMP_chol file

    Returns:
        Tuple containing:
            h0: Core energy
            h1: One-body Hamiltonian matrix
            chol: Cholesky vectors
            norb: Number of molecular orbitals
            nelec_sp: Tuple of electrons per spin (alpha, beta)
    """
    directory = tmp_dir

    assert os.path.isfile(
        directory + "/FCIDUMP_chol"
    ), f"File '{directory}/FCIDUMP_chol' does not exist."
    with h5py.File(directory + "/FCIDUMP_chol", "r") as fh5:
        [nelec, norb, ms, nchol] = fh5["header"]
        h0 = jnp.array(fh5.get("energy_core"))
        h1 = jnp.array(fh5.get("hcore")).reshape(norb, norb)
        h1_mod = jnp.array(fh5.get("hcore_mod")).reshape(norb, norb)
        chol = jnp.array(fh5.get("chol")).reshape(-1, norb, norb)

    assert type(ms) is np.int64
    assert type(nelec) is np.int64
    assert type(norb) is np.int64
    ms, nelec, norb = int(ms), int(nelec), int(norb)

    # Calculate electrons per spin channel
    nelec_sp = ((nelec + abs(ms)) // 2, (nelec - abs(ms)) // 2)

    return h0, h1, h1_mod, chol, norb, nelec_sp


def read_options(options: Optional[Dict] = None, tmp_dir: str = ".") -> Dict:
    """
    Read calculation options from file or use provided options with defaults.

    Args:
        options: Dictionary of options (if None, tries to load from file)
        tmp_dir: Directory containing the options.bin file

    Returns:
        Dictionary of calculation options with defaults applied
    """
    directory = tmp_dir
    # Try to load options from file if not provided
    if options is None:
        try:
            with open(directory + "/options.bin", "rb") as f:
                options = pickle.load(f)
        except:
            options = {}

    options = get_options(options)

    return options


def get_options(options: Dict):

    # Set default values for options
    options["dt"] = options.get("dt", 0.01)
    options["n_walkers"] = options.get("n_walkers", 50)
    options["n_prop_steps"] = options.get("n_prop_steps", 50)
    options["n_ene_blocks"] = options.get("n_ene_blocks", 50)
    options["n_sr_blocks"] = options.get("n_sr_blocks", 1)
    options["n_qr_blocks"] = options.get("n_qr_blocks", 1)
    options["n_blocks"] = options.get("n_blocks", 50)
    options["n_ene_blocks_eql"] = options.get("n_ene_blocks_eql", 5)
    options["n_sr_blocks_eql"] = options.get("n_sr_blocks_eql", 10)
    options["seed"] = options.get("seed", np.random.randint(1, int(1e6)))
    options["n_eql"] = options.get("n_eql", 1)

    # AD mode options
    options["ad_mode"] = options.get("ad_mode", None)
    assert options["ad_mode"] in [None, "forward", "reverse", "2rdm", "nuc_grad"]

    # Wavefunction and algorithm options
    options["orbital_rotation"] = options.get("orbital_rotation", True)
    options["do_sr"] = options.get("do_sr", True)
    options["walker_type"] = options.get("walker_type", "restricted")
    options["symmetry_projector"] = options.get("symmetry_projector", None)
    options["trial_ket"] = options.get("trial_ket", None)

    # Handle backwards compatibility for walker types
    if options["walker_type"] == "rhf":
        options["walker_type"] = "restricted"
    elif options["walker_type"] == "uhf":
        options["walker_type"] = "unrestricted"
    assert options["walker_type"] in ["restricted", "unrestricted", "generalized"]

    options["symmetry"] = options.get("symmetry", False)
    options["save_walkers"] = options.get("save_walkers", False)
    options["trial"] = options.get("trial", None)
    assert options["trial"] in [
        None,
        "rhf",
        "uhf",
        "noci",
        "cisd",
        "ucisd",
        "ghf",
        "ghf_complex",
        "gcisd_complex",
        "UCISD",
    ]

    if options["trial"] is None:
        print(f"# No trial specified in options.")

    options["ene0"] = options.get("ene0", 0.0)
    options["free_projection"] = options.get("free_projection", False)

    # performance and memory options
    options["n_chunks"] = options.get("n_chunks", options.get("n_batch", 1))
    options["vhs_mixed_precision"] = options.get("vhs_mixed_precision", False)
    options["trial_mixed_precision"] = options.get("trial_mixed_precision", False)
    options["memory_mode"] = options.get("memory_mode", "low")

    # LNO options
    options["prjlo"] = options.get("prjlo", None)
    options["orbE"] = options.get("orbE", 0)
    options["maxError"] = options.get("maxError", 1e-3)

    assert options is not None, "Options dictionary cannot be None."
    return options


def read_observable(nmo: int, options: Dict, tmp_dir: str = ".") -> Optional[List]:
    """
    Read observable operator from file.

    Args:
        nmo: Number of molecular orbitals
        options: Dictionary of calculation options
        tmp_dir: Directory containing the observable.h5 file

    Returns:
        List containing observable operator and constant, or None if file not found
    """
    directory = tmp_dir

    try:
        with h5py.File(directory + "/observable.h5", "r") as fh5:
            [observable_constant] = fh5["constant"]
            observable_op = np.array(fh5.get("op")).reshape(nmo, nmo)
            if options["walker_type"] == "unrestricted":
                observable_op = jnp.array([observable_op, observable_op])
            observable = [observable_op, observable_constant]
    except:
        observable = None

    return observable


def set_ham(
    norb: int,
    h0: jnp.ndarray,
    h1: jnp.ndarray,
    chol: jnp.ndarray,
    ene0: float = 0.0,
) -> Tuple[Any, Dict]:
    """
    Set up the Hamiltonian for AFQMC calculation.

    Args:
        norb: Number of molecular orbitals
        h0: Core energy
        h1: One-body Hamiltonian matrix
        chol: Cholesky vectors
        ene0: Energy offset

    Returns:
        Tuple containing:
            ham: Hamiltonian object
            ham_data: Dictionary of Hamiltonian data
    """
    ham = hamiltonian.hamiltonian(norb)
    nchol = chol.shape[0]
    ham_data = {
        "h0": h0,
        "h1": jnp.array([h1, h1]),  # Replicate for up/down spins
        "chol": chol.reshape(nchol, -1),
        "ene0": ene0,
    }
    return ham, ham_data


def apply_symmetry_mask(ham_data: Dict, options: Dict) -> Dict:
    """
    Apply symmetry mask to Hamiltonian data based on options.

    Args:
        ham_data: Dictionary of Hamiltonian data
        options: Dictionary of calculation options

    Returns:
        Updated ham_data with mask applied
    """
    if options["symmetry"]:
        ham_data["mask"] = jnp.where(jnp.abs(ham_data["h1"]) > 1.0e-10, 1.0, 0.0)
    else:
        ham_data["mask"] = jnp.ones(ham_data["h1"].shape)

    return ham_data

def set_1rdm(
    wave_data,
    norb,
    nelec_sp,
    mo_coeff,
    options_trial,
    directory,
    ):
    # Try to read RDM1 from file
    try:
        rdm1 = jnp.array(np.load(directory + "/rdm1.npz")["rdm1"])
        assert rdm1.shape == (2, norb, norb)
        wave_data["rdm1"] = rdm1
        print(f"# Read RDM1 from disk")
    except:
        # Construct RDM1 from mo_coeff if file not found
        if options_trial in ["ghf_complex", "gcisd_complex"]:
            wave_data["rdm1"] = jnp.array(
                [
                    mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]]
                    @ mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]].T.conj(),
                    mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]]
                    @ mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]].T.conj(),
                ]
            )
        else:
            wave_data["rdm1"] = jnp.array(
                [
                    mo_coeff[0][:, : nelec_sp[0]] @ mo_coeff[0][:, : nelec_sp[0]].T,
                    mo_coeff[1][:, : nelec_sp[1]] @ mo_coeff[1][:, : nelec_sp[1]].T,
                ]
            )

    return wave_data

def set_symmetry_projector(wave_data, nelec_sp, options):
    if "s2_ghf" in options["symmetry_projector"]:
        # only singlet projection supported for now
        n_alpha = options.get("nalpha", 6)
        alpha_vals = jnp.pi * (jnp.arange(n_alpha) + 0.5) / n_alpha
        w_alpha = 1.0 / n_alpha
        wave_data["alpha"] = (alpha_vals, w_alpha)

        def make_beta_gl(n_beta: int):
            x, w = leggauss(int(n_beta))
            beta = np.arccos(x)
            order = np.argsort(beta)
            return beta[order], w[order]

        n_beta = options.get("nbeta", 6)
        beta_vals_np, w_beta_np = make_beta_gl(n_beta)
        beta_vals = jnp.asarray(beta_vals_np)
        w_beta = jnp.asarray(w_beta_np)
        wave_data["beta"] = (beta_vals, w_beta)

    elif "s2" in options["symmetry_projector"]:
        # Gauss-Legendre
        #
        # \int_0^\pi sin(\beta) f(\beta) \dd\beta
        # x = \cos(beta), \dd x = -\sin(beta)
        # = \int_{-1}^{1} f(\arccos(x)) \dd x
        # \approx \sum_{i=1}^n w_i f(\arccos(x_i))
        #
        S = options["target_spin"] / 2.0
        Sz = (nelec_sp[0] - nelec_sp[1]) / 2.0
        ngrid = options.get("s2_projector_ngrid", 4)

        # Gauss–Legendre
        x, w = leggauss(ngrid)
        betas = jnp.arccos(x)  # map [-1, 1] to [0, pi]

        # Wigner small-d matrix elements for each point
        w_betas = (
            jax.vmap(Wigner_small_d.wigner_small_d, (None, None, None, 0))(
                S, Sz, Sz, betas
            )
            * w
            * (2.0 * S + 1.0)
            / 2.0
        )

        # betas = np.linspace(0, np.pi, ngrid, endpoint=False)
        # w_betas = (
        #    jax.vmap(Wigner_small_d.wigner_small_d, (None, None, None, 0))(
        #        S, Sz, Sz, betas
        #    )
        #    * jnp.sin(betas)
        #    * (2 * S + 1)
        #    / 2.0
        #    * jnp.pi
        #    / ngrid
        # )

        wave_data["betas"] = (S, Sz, w_betas, betas)

    return wave_data

def set_trial(prep) -> Tuple[Any, Dict]:
    """
    Set up the trial wavefunction.

    Args:
        options: Dictionary of calculation options
        mo_coeff: Molecular orbital coefficients
        norb: Number of orbitals
        nelec_sp: Tuple of electrons per spin (alpha, beta)
        tmp_dir: Directory containing wavefunction files

    Returns:
        Tuple containing:
            trial: Trial wavefunction object
            wave_data: Dictionary of wavefunction data
    """
    norb = prep.mo_basis.norb
    nelec_sp = prep.mo_basis.nelec_sp
    mo_coeff = prep.mo_basis.trial_coeff
    options = prep.options
    options_trial = prep.options["trial"]
    if hasattr(prep.tmp, "trial"): # Dirty
         options_trial = prep.options["trial_ket"]
    if hasattr(prep.tmp, "amplitudes"):
        amplitudes = prep.tmp.amplitudes
    directory = prep.path.tmpdir

    wave_data = {}
    wave_data = set_1rdm(wave_data, norb, nelec_sp, mo_coeff, options_trial, directory)

    if options.get("symmetry_projector", None) is not None:
        wave_data = set_symmetry_projector(wave_data, nelec_sp, options)

    # Set up trial wavefunction based on specified type
    if options_trial == "rhf":
        trial = wavefunctions.rhf(
            norb,
            nelec_sp,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
        )
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

    elif options_trial == "uhf":
        trial = wavefunctions.uhf(
            norb,
            nelec_sp,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
        )
        wave_data["mo_coeff"] = [
            mo_coeff[0][:, : nelec_sp[0]],
            mo_coeff[1][:, : nelec_sp[1]],
        ]

    elif options_trial == "noci":
        with open(directory + "/dets.pkl", "rb") as f:
            ci_coeffs_dets = pickle.load(f)
        ci_coeffs_dets = [
            jnp.array(ci_coeffs_dets[0]),
            [jnp.array(ci_coeffs_dets[1][0]), jnp.array(ci_coeffs_dets[1][1])],
        ]
        wave_data["ci_coeffs_dets"] = ci_coeffs_dets
        trial = wavefunctions.noci(
            norb,
            nelec_sp,
            ci_coeffs_dets[0].size,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
        )

    elif options_trial == "cisd":
        ci1 = jnp.array(amplitudes["ci1"])
        ci2 = jnp.array(amplitudes["ci2"])
        trial_wave_data = {"ci1": ci1, "ci2": ci2}
        wave_data.update(trial_wave_data)

        if options["trial_mixed_precision"]:
            mixed_real_dtype = jnp.float32
            mixed_complex_dtype = jnp.complex64
        else:
            mixed_real_dtype = jnp.float64
            mixed_complex_dtype = jnp.complex128

        trial = wavefunctions.cisd(
            norb,
            nelec_sp,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
            mixed_real_dtype=mixed_real_dtype,
            mixed_complex_dtype=mixed_complex_dtype,
            memory_mode=options["memory_mode"],
        )

    elif options_trial == "ccsd":
        assert options["walker_type"] == "restricted"

        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])

        nocc, nvirt = t1.shape

        trial_wave_data = {
            "t1": t1,
        }

        wave_data.update(trial_wave_data)
        wave_data["mo_coeff"] = mo_coeff[0]

        if options["trial_mixed_precision"]:
            mixed_real_dtype = jnp.float32
            mixed_complex_dtype = jnp.complex64
        else:
            mixed_real_dtype = jnp.float64
            mixed_complex_dtype = jnp.complex128

        trial = wavefunctions.ccsd(
            norb,
            nelec_sp,
            nocc,
            nvirt,
            n_chunks=options["n_chunks"],
            mixed_real_dtype=mixed_real_dtype,
            mixed_complex_dtype=mixed_complex_dtype,
            memory_mode=options["memory_mode"],
        )
        wave_data = trial.hs_op(wave_data, t2)

    elif options_trial == "uccsd":
        assert options["walker_type"] == "unrestricted"

        t1a = jnp.array(amplitudes["t1a"])
        t1b = jnp.array(amplitudes["t1b"])
        t2aa = jnp.array(amplitudes["t2aa"])
        t2ab = jnp.array(amplitudes["t2ab"])
        t2bb = jnp.array(amplitudes["t2bb"])

        trial_wave_data = {
            "t1a": t1a,
            "t1b": t1b,
        }

        nOa, nVa = amplitudes["t1a"].shape
        nOb, nVb = amplitudes["t1b"].shape
        nocc = (nOa, nOb)
        nvir = (nVa, nVb)

        assert nocc == nelec_sp
        assert nvir == (norb - nOa, norb - nOb)

        wave_data.update(trial_wave_data)
        wave_data["mo_coeff"] = [mo_coeff[0], mo_coeff[1]]

        if options["trial_mixed_precision"]:
            mixed_real_dtype = jnp.float32
            mixed_complex_dtype = jnp.complex64
        else:
            mixed_real_dtype = jnp.float64
            mixed_complex_dtype = jnp.complex128

        trial = wavefunctions.uccsd(
            norb,
            nelec_sp,
            nocc,
            nvir,
            n_chunks=options["n_chunks"],
            mixed_real_dtype=mixed_real_dtype,
            mixed_complex_dtype=mixed_complex_dtype,
            memory_mode=options["memory_mode"],
        )

        wave_data = trial.hs_op(
            wave_data,
            t2aa,
            t2ab,
            t2bb,
        )

    elif options_trial == "ucisd" or options_trial == "UCISD":

        ci1a = jnp.array(amplitudes["ci1a"])
        ci1b = jnp.array(amplitudes["ci1b"])
        ci2aa = jnp.array(amplitudes["ci2aa"])
        ci2ab = jnp.array(amplitudes["ci2ab"])
        ci2bb = jnp.array(amplitudes["ci2bb"])
        trial_wave_data = {
            "ci1A": ci1a,
            "ci1B": ci1b,
            "ci2AA": ci2aa,
            "ci2AB": ci2ab,
            "ci2BB": ci2bb,
            "mo_coeff": mo_coeff,
        }
        wave_data.update(trial_wave_data)

        if options["trial_mixed_precision"]:
            mixed_real_dtype = jnp.float32
            mixed_complex_dtype = jnp.complex64
        else:
            mixed_real_dtype = jnp.float64
            mixed_complex_dtype = jnp.complex128

        trial = (
            wavefunctions.ucisd(
                norb,
                nelec_sp,
                n_chunks=options["n_chunks"],
                projector=options["symmetry_projector"],
                mixed_real_dtype=mixed_real_dtype,
                mixed_complex_dtype=mixed_complex_dtype,
                memory_mode=options["memory_mode"],
            )
            if options_trial == "ucisd"
            else wavefunctions.UCISD(
                norb,
                nelec_sp,
                n_chunks=options["n_chunks"],
                projector=options["symmetry_projector"],
            )
        )

    elif options_trial == "ghf":
        trial = wavefunctions.ghf(
            norb,
            nelec_sp,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
        )
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]]

    elif options_trial == "ghf_complex":
        trial = wavefunctions.ghf_complex(
            norb,
            nelec_sp,
            n_chunks=options["n_chunks"],
            projector=options["symmetry_projector"],
        )
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0] + nelec_sp[1]]

    elif options_trial == "gcisd_complex":

        t1 = jnp.array(amplitudes["t1"])
        t2 = jnp.array(amplitudes["t2"])

        ci1 = t1
        ci2 = (
            np.einsum("ijab->iajb", t2)
            + np.einsum("ia,jb->iajb", t1, t1)
            - np.einsum("ib,ja->iajb", t1, t1)
        )
        trial_wave_data = {
            "ci1": ci1,
            "ci2": ci2,
            "mo_coeff": mo_coeff,
        }
        wave_data.update(trial_wave_data)
        trial = wavefunctions.gcisd_complex(
            norb, nelec_sp, n_chunks=options["n_chunks"]
        )

    else:
        # Try to load trial from pickle file
        try:
            with open(directory + "/trial.pkl", "rb") as f:
                [trial, trial_wave_data] = pickle.load(f)
            wave_data.update(trial_wave_data)
            print(f"# Read trial of type {type(trial).__name__} from trial.pkl.")
        except:
            print("# trial.pkl not found, make sure to construct the trial separately.")
            trial = None

    if options["prjlo"] is not None:  # For LNO
        wave_data["prjlo"] = options["prjlo"]
        trial = wavefunctions.rhf_lno(norb, nelec_sp, n_chunks=options["n_chunks"])
        wave_data["mo_coeff"] = mo_coeff[0][:, : nelec_sp[0]]

    return trial, wave_data

def set_prop(options: Dict) -> Any:
    """
    Set up the propagator for AFQMC calculation.

    Args:
        options: Dictionary of calculation options

    Returns:
        Propagator object configured according to options
    """
    vhs_real_dtype = jnp.float32 if options["vhs_mixed_precision"] else jnp.float64
    vhs_complex_dtype = (
        jnp.complex64 if options["vhs_mixed_precision"] else jnp.complex128
    )
    n_exp_terms = 10 if options["free_projection"] else options.get("n_exp_terms", 6)
    return propagation.propagator_afqmc(
        options["dt"],
        options["n_walkers"],
        n_exp_terms=n_exp_terms,
        n_chunks=options["n_chunks"],
        vhs_real_dtype=vhs_real_dtype,
        vhs_complex_dtype=vhs_complex_dtype,
        walker_type=options["walker_type"],
    )


def set_sampler(options: Dict) -> Any:
    """
    Set up the sampler for AFQMC calculation.

    Args:
        options: Dictionary of calculation options

    Returns:
        Sampler object configured according to options
    """
    if options["prjlo"] is not None:
        # Use LNO sampler if prjlo is specified
        return sampling.sampler_lno(
            options["n_prop_steps"],
            options["n_ene_blocks"],
            options["n_sr_blocks"],
            options["n_blocks"],
        )
    else:
        return sampling.sampler(
            options["n_prop_steps"],
            options["n_ene_blocks"],
            options["n_sr_blocks"],
            options["n_qr_blocks"],
            options["n_blocks"],
        )


def load_mo_coefficients(tmp_dir: str = ".") -> jnp.ndarray:
    """
    Load molecular orbital coefficients from file.

    Args:
        tmp_dir: Directory containing mo_coeff.npz file

    Returns:
        Array of molecular orbital coefficients
    """
    directory = tmp_dir
    try:
        return jnp.array(np.load(directory + "/mo_coeff.npz")["mo_coeff"])
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find mo_coeff.npz in {directory}")
    except Exception as e:
        raise RuntimeError(f"Error loading molecular orbital coefficients: {str(e)}")


def setup_afqmc(
    options: Optional[Dict] = None,
    tmp_dir: str = ".",
) -> Tuple:
    """
    Prepare all components for an AFQMC calculation.

    Args:
        options: Dictionary of calculation options (optional)
        tmp_dir: Directory for input/output files (optional)

    Returns:
        Tuple containing all necessary components for AFQMC calculation:
            ham_data: Dictionary of Hamiltonian data
            ham: Hamiltonian object
            prop: Propagator object
            trial: Trial wavefunction object
            wave_data: Dictionary of wavefunction data
            sampler: Sampler object
            observable: Observable operator
            options: Dictionary of calculation options
    """
    directory = tmp_dir

    prep = PrepAfqmc()
    prep.options = options
    prep.set_tmpdir(directory)
    prep.io.set_read()
    if options is not None:
        prep.io.set_noio_options()
    prep.prep()
    prep.setup_afqmc()

    return (
        prep.tmp.ham_data,
        prep.tmp.ham,
        prep.tmp.prop,
        prep.tmp.trial,
        prep.tmp.wave_data,
        prep.tmp.sampler,
        prep.tmp.observable,
        prep.options,
    )


def setup_afqmc_ph(
    pyscf_prep: Optional[Dict] = None,
    options: Optional[Dict] = None,
    tmp_dir: str = ".",
) -> Tuple[
    Dict,
    hamiltonian.hamiltonian,
    propagation.propagator,
    wavefunctions.wave_function,
    Dict,
    sampling.sampler,
    Optional[Tuple],
    Dict,
]:
    """
    Prepare all components for an AFQMC calculation.

    Args:
        pyscf_prep: Dictionary containing PySCF preparation data (optional)
        options: Dictionary of calculation options (optional)
        tmp_dir: Directory for input/output files (optional)

    Returns:
        Tuple containing all necessary components for AFQMC calculation:
            ham_data: Dictionary of Hamiltonian data
            ham: Hamiltonian object
            prop: Propagator object
            trial: Trial wavefunction object
            wave_data: Dictionary of wavefunction data
            sampler: Sampler object
            observable: Observable operator
            options: Dictionary of calculation options
    """
    directory = tmp_dir

    prep = PrepAfqmc()
    prep.options = options
    prep.set_tmpdir(directory)
    prep.io.set_read()
    prep.prep()
    prep.setup_afqmc()

    # Ensure type-narrowing for the returned options
    assert prep.options is not None

    return (
        prep.tmp.ham_data,
        prep.tmp.ham,
        prep.tmp.prop,
        prep.tmp.trial,
        prep.tmp.wave_data,
        prep.tmp.sampler,
        prep.tmp.observable,
        prep.options,
    )


def setup_afqmc_fp(
    options: Optional[Dict] = None, tmp_dir: Optional[str] = None
) -> Tuple:
    """Prepare all components for a free projection AFQMC calculation.

    Args:
        options: Dictionary of calculation options (optional)
        tmp_dir: Directory for input/output files (optional)

    Returns:
        Tuple containing all necessary components for AFQMC free projection calculation:
            ham_data: Dictionary of Hamiltonian data
            ham: Hamiltonian object
            prop: Propagator object
            trial: Trial wavefunction object
            wave_data: Dictionary of wavefunction data
            trial_ket: Trial wavefunction object for ket
            wave_data_ket: Dictionary of wavefunction data for ket
            sampler: Sampler object
            observable: Observable operator
            options: Dictionary of calculation options
    """
    directory = tmp_dir if tmp_dir is not None else tmpdir

    prep = PrepAfqmc()
    prep.options = options
    prep.set_tmpdir(directory)
    prep.io.set_read()
    prep.prep()
    prep.setup_afqmc()

    # Ensure type-narrowing for the returned options
    assert prep.options is not None

    return (
        prep.tmp.ham_data,
        prep.tmp.ham,
        prep.tmp.prop,
        prep.tmp.trial,
        prep.tmp.wave_data,
        prep.tmp.trial_ket,
        prep.tmp.wave_data_ket,
        prep.tmp.sampler,
        prep.tmp.observable,
        prep.options,
    )
