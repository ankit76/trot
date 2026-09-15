from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple, TypeAlias, Union, cast

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

print = partial(print, flush=True)

from .ham.chol import HamBasis

# This file contains staging utilities to convert pyscf mf/cc objects
# into serializable data classes representing Hamiltonian and trial
# wavefunction inputs which can be used for building AFQMC objects.

Array: TypeAlias = NDArray[Any]

# to keep track of format versions when loading/saving staged inputs
STAGE_FORMAT_VERSION = 1


def _stage_begin(message: str) -> float:
    print(f"[stage] {message}...")
    return time.time()


def _stage_end(start: float, message: str, *, details: str | None = None) -> None:
    suffix = f" | {details}" if details else ""
    print(f"[stage] {message} in {time.time() - start:.2f}s{suffix}")


def _normalize_frozen_list(frozen: Any, *, nmo: int) -> NDArray:
    frozen_arr = np.asarray(frozen, dtype=np.int64)
    if frozen_arr.ndim != 1:
        raise ValueError("cc.frozen must be a one-dimensional list of MO indices.")
    if frozen_arr.size == 0:
        return frozen_arr

    frozen_arr = np.sort(frozen_arr)
    if np.unique(frozen_arr).size != frozen_arr.size:
        raise ValueError("cc.frozen contains duplicate MO indices.")
    if frozen_arr[0] < 0 or frozen_arr[-1] >= nmo:
        raise ValueError(f"cc.frozen indices must lie in [0, {nmo}).")

    return frozen_arr


def _mo_coeff_signature(mo_coeff: Any) -> tuple[tuple[int, ...], ...]:
    if isinstance(mo_coeff, (tuple, list)):
        return tuple(tuple(int(dim) for dim in np.asarray(block).shape) for block in mo_coeff)
    return (tuple(int(dim) for dim in np.asarray(mo_coeff).shape),)


def _mo_coeff_nmo(mo_coeff: Any) -> int:
    if isinstance(mo_coeff, (tuple, list)):
        shapes = [np.asarray(block).shape for block in mo_coeff]
        if len(shapes) == 0:
            raise ValueError("Spin-separated MO coefficients cannot be empty.")
        if any(len(shape) == 0 for shape in shapes):
            raise ValueError("MO coefficient blocks must have at least one dimension.")
        nmo = int(shapes[0][-1])
        if any(int(shape[-1]) != nmo for shape in shapes):
            raise ValueError(
                "Spin-separated MO coefficient blocks must have the same number of orbitals."
            )
        return nmo

    shape = np.asarray(mo_coeff).shape
    if len(shape) == 0:
        raise ValueError("MO coefficients must have at least one dimension.")
    return int(shape[-1])


def _copy_scf_with_cc_mo_coeff(cc: Any, mf: Any) -> Any:
    if not hasattr(cc, "mo_coeff") or cc.mo_coeff is None:
        return mf

    cc_sig = _mo_coeff_signature(cc.mo_coeff)
    mf_sig = _mo_coeff_signature(mf.mo_coeff)
    if cc_sig != mf_sig:
        raise ValueError(
            "CC object mo_coeff shape does not match the underlying SCF mo_coeff shape: "
            f"{cc_sig} != {mf_sig}."
        )

    mf_copy = copy.copy(mf)
    mf_copy.mo_coeff = cc.mo_coeff
    return mf_copy


def _freeze_core_from_mo_cholesky(
    *,
    h0: float,
    h1: NDArray,
    chol: NDArray,
    norb_frozen: int,
    nelec: Tuple[int, int],
) -> tuple[float, NDArray, NDArray, Tuple[int, int]]:
    nmo = int(h1.shape[0])
    if h1.shape != (nmo, nmo):
        raise ValueError(f"h1 must be square, got shape {h1.shape}.")
    if chol.ndim != 3 or chol.shape[1:] != (nmo, nmo):
        raise ValueError(f"chol must have shape (nchol, {nmo}, {nmo}), got {chol.shape}.")
    if norb_frozen < 0:
        raise ValueError(f"norb_frozen must be non-negative, got {norb_frozen}.")
    if norb_frozen > min(nelec):
        raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")
    if norb_frozen >= nmo:
        raise ValueError(f"norb_frozen={norb_frozen} leaves no active orbitals (nmo={nmo}).")
    if norb_frozen == 0:
        return float(h0), np.asarray(h1), np.asarray(chol), nelec

    nelec_active = (int(nelec[0] - norb_frozen), int(nelec[1] - norb_frozen))
    if nelec_active[0] < 0 or nelec_active[1] < 0:
        raise ValueError(
            f"norb_frozen={norb_frozen} leaves negative active electron count "
            f"{nelec_active} from nelec={nelec}."
        )
    if sum(nelec_active) <= 0:
        raise ValueError("Frozen core left no active electrons.")

    core = slice(0, norb_frozen)
    act = slice(norb_frozen, nmo)

    chol_core = np.asarray(chol[:, core, core])
    chol_act = np.asarray(chol[:, act, act])
    chol_act_core = np.asarray(chol[:, act, core])
    chol_core_act = np.asarray(chol[:, core, act])

    core_trace = np.trace(chol_core, axis1=1, axis2=2)
    vj = 2.0 * np.einsum("x,xpq->pq", core_trace, chol_act, optimize=True)
    vk = np.einsum("xpi,xiq->pq", chol_act_core, chol_core_act, optimize=True)

    h1_eff = np.asarray(h1[act, act]) + vj - vk

    e1_core = 2.0 * np.trace(np.asarray(h1[core, core]))
    ej_core = 2.0 * np.dot(core_trace, core_trace)
    ek_core = np.einsum("xij,xji->", chol_core, chol_core, optimize=True)
    ecore = float(np.real(h0 + e1_core + ej_core - ek_core))

    return ecore, np.asarray(h1_eff), np.array(chol_act, copy=True), nelec_active


def _infer_restricted_trial_freeze_from_cc(
    *,
    cc_frozen: Any,
    nmo_full: int,
    nocc_full: int,
    norb_frozen: int,
    t1_shape: tuple[int, int],
) -> tuple[int, int]:
    frozen = _normalize_frozen_list(cc_frozen, nmo=nmo_full)
    occ_frozen = frozen[frozen < nocc_full]
    vir_frozen = frozen[frozen >= nocc_full]

    nocc_cc_frozen = int(occ_frozen.size)
    nvir_cc_frozen = int(vir_frozen.size)

    if occ_frozen.size and not np.array_equal(
        occ_frozen, np.arange(nocc_cc_frozen, dtype=np.int64)
    ):
        raise ValueError(
            "Occupied orbitals in list-valued cc.frozen must form a contiguous prefix."
        )

    vir_expected = np.arange(nmo_full - nvir_cc_frozen, nmo_full, dtype=np.int64)
    if vir_frozen.size and not np.array_equal(vir_frozen, vir_expected):
        raise ValueError("Virtual orbitals in list-valued cc.frozen must form a contiguous suffix.")

    if norb_frozen > nocc_cc_frozen:
        raise ValueError("norb_frozen cannot exceed the number of occupied orbitals frozen in CC.")

    nocc_act_expected = nocc_full - nocc_cc_frozen
    nvir_act_expected = (nmo_full - nocc_full) - nvir_cc_frozen
    if t1_shape != (nocc_act_expected, nvir_act_expected):
        raise ValueError(
            "cc.frozen is inconsistent with the CC amplitudes in the restricted CISD trial."
        )

    nocc_t_core = nocc_cc_frozen - norb_frozen
    nvir_t_outer = nvir_cc_frozen
    return nocc_t_core, nvir_t_outer


def modified_cholesky(
    mat: Array,
    max_error: float = 1e-6,
) -> Array:
    """Modified cholesky decomposition for a given matrix.

    Args:
        mat (Array): Matrix to decompose.
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        Array: Cholesky vectors.
    """
    diag = mat.diagonal()
    norb = int(((-1 + (1 + 8 * mat.shape[0]) ** 0.5) / 2))
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

    chol0 = chol_vecs[:nchol]
    nchol = chol0.shape[0]
    chol = np.zeros((nchol, norb, norb))
    for i in range(nchol):
        for m in range(norb):
            for n in range(m + 1):
                triind = m * (m + 1) // 2 + n
                chol[i, m, n] = chol0[i, triind]
                chol[i, n, m] = chol0[i, triind]
    return chol


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10) -> NDArray:
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems. (copied from pauxy)

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
        print("# Generating Cholesky decomposition of ERIs.")
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
    eri_col = mol.intor("int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
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


def _rotate_chol_to_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """Rotate AO-space Cholesky into an MO basis."""
    C = np.asarray(basis_coeff)
    nao, norb = C.shape
    nchol = int(chol_vec.shape[0])
    out_dtype = np.result_type(chol_vec.dtype, C.dtype)

    reuse_storage = nao == norb and out_dtype == chol_vec.dtype
    if reuse_storage:
        chol = chol_vec.reshape(nchol, nao, nao)
    else:
        chol = np.empty((nchol, norb, norb), dtype=out_dtype)

    Cdag = np.asarray(C.conj().T)
    tmp = np.empty((nao, norb), dtype=out_dtype)
    for i in range(nchol):
        chol_i_ao = chol_vec[i].reshape(nao, nao)
        np.dot(chol_i_ao, C, out=tmp)
        np.dot(Cdag, tmp, out=chol[i])

    return chol


def _rotate_chol_to_ghf_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """Rotate spatial AO Cholesky factors into a generalized-spin MO basis."""
    C = np.asarray(basis_coeff)
    nao2, nmo = C.shape
    if nao2 % 2 != 0:
        raise ValueError(f"Expected even GHF AO dimension, got {nao2}")

    nao = nao2 // 2
    nchol = int(chol_vec.shape[0])
    out_dtype = np.result_type(chol_vec.dtype, C.dtype)
    chol = np.empty((nchol, nmo, nmo), dtype=out_dtype)

    Cdag = np.asarray(C.conj().T)
    chol_i_full = np.zeros((nao2, nao2), dtype=out_dtype)
    tmp = np.empty((nao2, nmo), dtype=out_dtype)
    for i in range(nchol):
        chol_i = chol_vec[i].reshape(nao, nao)
        chol_i_full.fill(0)
        chol_i_full[:nao, :nao] = chol_i
        chol_i_full[nao:, nao:] = chol_i
        np.dot(chol_i_full, C, out=tmp)
        np.dot(Cdag, tmp, out=chol[i])

    return chol


def _stage_frozen(frozen: int | ArrayLike | None) -> int | NDArray | None:
    if isinstance(frozen, (list, tuple, np.ndarray)):
        frozen = np.asarray(frozen, dtype=int)
    elif frozen is None:
        frozen = None
    elif isinstance(frozen, int):
        frozen = int(frozen)
    else:
        raise TypeError(f"Unsupported type '{type(frozen)}'.")

    return frozen


def _resolve_stage_frozen_arg(
    norb_frozen_core: int | None,
    norb_frozen: int | None,
    frozen_orbitals: ArrayLike | None,
) -> int | ArrayLike | None:
    if norb_frozen_core is not None and norb_frozen is not None and norb_frozen_core != norb_frozen:
        raise ValueError("norb_frozen_core and norb_frozen must match when both are passed.")
    core_frozen = norb_frozen_core if norb_frozen_core is not None else norb_frozen
    if core_frozen is not None and frozen_orbitals is not None:
        raise ValueError("Pass only one of norb_frozen_core/norb_frozen or frozen_orbitals.")
    if frozen_orbitals is not None:
        return frozen_orbitals
    return core_frozen


def _freeze_meta_value(frozen: int | NDArray | None) -> int | list[int] | None:
    if isinstance(frozen, np.ndarray):
        arr = np.asarray(frozen, dtype=np.int64).reshape(-1)
        return [int(x) for x in arr]
    if isinstance(frozen, int):
        return frozen
    if isinstance(frozen, np.integer):
        return int(frozen.item())
    if frozen is None:
        return None
    raise TypeError(f"Unsupported frozen metadata type: {type(frozen)}")


def _freeze_from_meta_value(frozen: Any) -> int | NDArray | None:
    if frozen is None:
        return None
    if isinstance(frozen, list):
        return np.asarray(frozen, dtype=np.int64)
    if isinstance(frozen, np.ndarray):
        return np.asarray(frozen, dtype=np.int64)
    if isinstance(frozen, (int, np.integer)):
        return int(frozen)
    raise TypeError(f"Unsupported frozen metadata type: {type(frozen)}")


def _dump_frozen(group: h5py.Group, frozen: int | NDArray, *, attr_name: str = "frozen") -> None:
    if isinstance(frozen, np.ndarray):
        group.create_dataset(attr_name, data=np.asarray(frozen, dtype=np.int64))
    else:
        group.attrs[attr_name] = int(frozen)


def _load_frozen(group: h5py.Group, *, attr_name: str = "frozen") -> int | NDArray:
    if attr_name in group:
        dataset = group[attr_name]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Expected dataset '{attr_name}', got {type(dataset)}")
        return np.asarray(dataset[...], dtype=np.int64)

    attr_value = group.attrs[attr_name]
    if isinstance(attr_value, np.ndarray):
        return int(np.asarray(attr_value).item())
    if isinstance(attr_value, (int, np.integer)):
        return int(attr_value)
    raise TypeError(f"Unsupported frozen attribute type: {type(attr_value)}")


@dataclass(frozen=True, slots=True)
class HamInput:
    """ham inputs in the chosen orthonormal one particle basis"""

    h0: float
    h1: Array  # (norb, norb)
    chol: Array  # (nchol, norb, norb)
    nelec: Tuple[int, int]
    norb: int
    chol_cut: float
    frozen: int | NDArray
    source_kind: str  # "mf" or "cc"
    basis: HamBasis  # "restricted" or "generalized"


@dataclass(frozen=True, slots=True)
class TrialInput:
    """trial inputs used to construct an afqmc trial object"""

    kind: str  # "slater", "cisd", "ucisd"
    data: Dict[str, Array]
    frozen: int | NDArray
    source_kind: str  # "mf" or "cc"


@dataclass(frozen=True, slots=True)
class StagedInputs:
    ham: HamInput
    trial: TrialInput
    meta: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class StagedCc:
    """Wrapper ensuring the validity of the CC object"""

    _delegate = {"t1", "t2", "_scf", "frozen"}
    kind: str  # "ccsd", "uccsd", "gccsd"
    cc: Any
    mf: Any
    trial_frozen: int | NDArray
    afqmc_frozen: int | NDArray

    def __init__(self, cc: Any, frozen: int | ArrayLike | None):
        from pyscf.cc.ccsd import CCSD
        from pyscf.cc.gccsd import GCCSD
        from pyscf.cc.uccsd import UCCSD

        if not isinstance(cc, (CCSD, UCCSD, GCCSD)):
            raise TypeError(f"Unsupported object type: {type(cc)}")

        if not hasattr(cc, "_scf"):
            raise TypeError("CC-like object missing _scf reference to underlying scf object.")
        else:
            mf = _copy_scf_with_cc_mo_coeff(cc, cc._scf)

        if not hasattr(cc, "t1") or not hasattr(cc, "t2"):
            raise ValueError("CC amplitudes not found; did you run cc.kernel()?")

        if isinstance(cc, CCSD):
            kind = "ccsd"
        elif isinstance(cc, UCCSD):
            kind = "uccsd"
        elif isinstance(cc, GCCSD):
            kind = "gccsd"

        frozen = _stage_frozen(frozen)
        cc_frozen = _stage_frozen(cc.frozen)

        if cc_frozen is None:
            if frozen is not None and not (isinstance(frozen, int) and frozen == 0):
                raise ValueError(
                    "Explicit AFQMC frozen orbitals are unsupported for CC objects without cc.frozen."
                )
            afqmc_frozen = 0
            trial_frozen = 0
        elif isinstance(cc_frozen, np.ndarray):
            if kind != "ccsd":
                raise NotImplementedError(
                    "List-valued cc.frozen is currently supported only for restricted CCSD staging."
                )
            trial_frozen = _normalize_frozen_list(cc_frozen, nmo=_mo_coeff_nmo(mf.mo_coeff))
            if frozen is None:
                afqmc_frozen = 0
            elif isinstance(frozen, int):
                afqmc_frozen = frozen
            else:
                raise TypeError(
                    "List-valued cc.frozen requires an integer AFQMC frozen-core count."
                )
        else:
            if frozen is None:
                afqmc_frozen = int(cc_frozen)
            elif isinstance(frozen, int):
                if int(cc_frozen) != frozen:
                    raise ValueError("cc.frozen and staging frozen must be equal.")
                afqmc_frozen = frozen
            else:
                raise TypeError(
                    "Integer cc.frozen is incompatible with list-valued staging frozen."
                )
            trial_frozen = int(cc_frozen)

        if isinstance(trial_frozen, np.ndarray) and kind != "ccsd":
            raise NotImplementedError(
                "List-valued cc.frozen is currently supported only for restricted CCSD staging."
            )

        mf = StagedMf(mf, afqmc_frozen)

        object.__setattr__(self, "cc", cc)
        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "afqmc_frozen", afqmc_frozen)
        object.__setattr__(self, "trial_frozen", trial_frozen)

    def __getattr__(self, name):
        if name in StagedCc._delegate:
            return getattr(self.cc, name)
        elif hasattr(self.cc, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the CC object but not in this wrapper."
            )
        elif hasattr(self.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF and CC objects or in this wrapper."
            )


@dataclass(frozen=True, slots=True)
class StagedMf:
    """Wrapper ensuring the validity of the SCF object"""

    _delegate = {"mo_coeff", "mo_occ", "mol", "nelec", "get_ovlp", "energy_nuc", "get_hcore"}
    kind: str  # "rhf", "rohf", "uhf", ghf
    mf: Any  # Python SCF object
    trial_frozen: int | NDArray
    afqmc_frozen: int | NDArray

    def __init__(self, mf: Any, frozen: int | ArrayLike | None):
        from pyscf.scf.ghf import GHF
        from pyscf.scf.hf import RHF
        from pyscf.scf.rohf import ROHF
        from pyscf.scf.uhf import UHF

        if not isinstance(mf, (RHF, ROHF, UHF, GHF)):
            raise TypeError(f"Unsupported object type: {type(mf)}")

        if not hasattr(mf, "mo_coeff"):
            raise ValueError("MO coefficients not found; did you run mf.kernel()?")

        if isinstance(mf, ROHF):
            kind = "rohf"
        elif isinstance(mf, RHF):
            kind = "rhf"
        elif isinstance(mf, UHF):
            kind = "uhf"
        elif isinstance(mf, GHF):
            kind = "ghf"

        frozen = _stage_frozen(frozen)
        nmo = _mo_coeff_nmo(mf.mo_coeff)

        if isinstance(frozen, np.ndarray):
            frozen_arr = _normalize_frozen_list(frozen, nmo=nmo)
            if frozen_arr.size > 0:
                assert frozen_arr.size < nmo
            frozen = frozen_arr
        elif isinstance(frozen, int):
            assert frozen < nmo
            assert frozen >= 0
        elif frozen is None:
            frozen = 0
        else:
            raise TypeError(
                f"Expected a type int | np.ndarray | None, but received '{type(frozen)}'."
            )

        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "afqmc_frozen", frozen)
        object.__setattr__(self, "trial_frozen", 0)

    @property
    def norb(self) -> int:
        return _mo_coeff_nmo(self.mf.mo_coeff)

    def __getattr__(self, name: str):
        if name in StagedMf._delegate:
            return getattr(self.mf, name)
        elif hasattr(self.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF object or in this wrapper."
            )


@dataclass(frozen=True, slots=True)
class StagedMfOrCc:
    """Wrapper ensuring the validity of the SCF/CC object"""

    _delegate_mf = StagedMf._delegate
    _delegate_cc = StagedCc._delegate
    kind: str  # StageCc.kind or StagedMf.kind
    source: str  # "cc", "mf"
    mf_or_cc: Any  # StagedMf or StagedCc
    mf: StagedMf
    afqmc_frozen: int | NDArray
    trial_frozen: int | NDArray

    def __init__(self, mf_or_cc: Any, frozen: int | ArrayLike | None):
        from pyscf.cc.ccsd import CCSD
        from pyscf.cc.gccsd import GCCSD
        from pyscf.cc.uccsd import UCCSD
        from pyscf.scf.ghf import GHF
        from pyscf.scf.hf import RHF
        from pyscf.scf.rohf import ROHF
        from pyscf.scf.uhf import UHF

        if isinstance(mf_or_cc, (CCSD, UCCSD, GCCSD)):
            mf_or_cc = StagedCc(mf_or_cc, frozen)
            mf = mf_or_cc.mf
            source = "cc"
        elif isinstance(mf_or_cc, (RHF, ROHF, UHF, GHF)):
            mf_or_cc = StagedMf(mf_or_cc, frozen)
            mf = mf_or_cc
            source = "mf"
        else:
            raise TypeError(f"Unreachable: '{type(mf_or_cc)}'")

        object.__setattr__(self, "mf_or_cc", mf_or_cc)
        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "kind", mf_or_cc.kind)
        object.__setattr__(self, "afqmc_frozen", mf_or_cc.afqmc_frozen)
        object.__setattr__(self, "trial_frozen", mf_or_cc.trial_frozen)

    @property
    def norb(self) -> int:
        return _mo_coeff_nmo(self.mf.mo_coeff)

    def __getattr__(self, name: str):
        if name in StagedMfOrCc._delegate_cc:
            return getattr(self.mf_or_cc, name)
        elif name in StagedMfOrCc._delegate_mf:
            return getattr(self.mf, name)
        elif self.source == "cc" and hasattr(self.mf_or_cc.cc, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the CC object but not in this wrapper."
            )
        elif hasattr(self.mf.mf, name):
            raise AttributeError(
                f"Attribute '{name}' exists in the SCF object but not in this wrapper."
            )
        else:
            raise AttributeError(
                f"Attribute '{name}' does not exist in the SCF and CC objects or in this wrapper."
            )


# public API
def stage(
    obj: Any,
    *,
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    frozen_orbitals: ArrayLike | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    ham: HamInput | None = None,
    trial: TrialInput | None = None,
) -> StagedInputs:
    """
    Stage inputs from a pyscf mf or cc object.

    Args:
        obj:
            pyscf mf object (RHF/ROHF/UHF) or cc object (CCSD/UCCSD).
        norb_frozen_core:
            Preferred name for the number of lowest occupied core orbitals removed from the
            AFQMC Hamiltonian.
        norb_frozen:
            Backward-compatible alias for ``norb_frozen_core``.
            For CC objects with integer ``cc.frozen``, this is inferred from ``cc.frozen``.
            For restricted CCSD objects with list-valued ``cc.frozen``, trial-space frozen
            occupied/virtual blocks are inferred from ``cc.frozen`` while
            ``norb_frozen_core``/``norb_frozen`` control the occupied core orbitals removed
            from the AFQMC Hamiltonian.
        frozen_orbitals:
            Explicit orbital list for LNO-style staging. Generic AFQMC/FNO staging should use
            ``norb_frozen_core`` instead.
        chol_cut:
            Cholesky decomposition cutoff.
        cache:
            Optional path to write on disk. If it exists and overwrite=False,
            loads it. Otherwise computes and writes it.
        overwrite:
            If True and cache is provided, recompute and overwrite cache.
        verbose:
            Print timing/info.
        ham:
            Optionally provide HamInput. If None, will be staged from obj.
        trial:
            Optionally provide TrialInput. If None, will be staged from obj.

    Returns:
        StagedInputs containing HamInput, TrialInput, and metadata.
    """
    cache_path = Path(cache).expanduser().resolve() if cache is not None else None
    if cache_path is not None and cache_path.exists() and not overwrite:
        return load(cache_path)

    t0 = time.time()

    resolved_frozen = _resolve_stage_frozen_arg(norb_frozen_core, norb_frozen, frozen_orbitals)
    obj = StagedMfOrCc(obj, resolved_frozen)
    mol = obj.mol

    if ham is None:
        t_ham = _stage_begin("building Hamiltonian")
        ham = _stage_ham_input(
            obj,
            chol_cut=chol_cut,
            verbose=verbose,
        )
        _stage_end(
            t_ham,
            "Hamiltonian ready",
            details=f"norb={ham.norb} nchol={ham.chol.shape[0]}",
        )

    if trial is None:
        t_trial = _stage_begin("building trial input")
        trial = _stage_trial_input(obj)
        _stage_end(t_trial, "trial input ready", details=f"kind={trial.kind}")

    meta: Dict[str, Any] = {
        "format_version": STAGE_FORMAT_VERSION,
        "timestamp_unix": time.time(),
        "source_kind": obj.source,
        "frozen": _freeze_meta_value(obj.afqmc_frozen),
        "chol_cut": ham.chol_cut if ham is not None else chol_cut,
        "mol": {
            "nao": int(mol.nao),
            "nelectron": int(mol.nelectron),
            "spin": int(mol.spin),
            "charge": int(mol.charge),
            "basis": getattr(mol, "basis", None),
        },
    }

    staged = StagedInputs(ham=ham, trial=trial, meta=meta)

    if cache_path is not None:
        dump(staged, cache_path)

    if verbose:
        dt = time.time() - t0
        print(f"[stage] done in {dt:.2f}s | norb={ham.norb} nchol={ham.chol.shape[0]}")

    return staged


def dump(staged: StagedInputs, path: Union[str, Path]) -> None:
    """
    Save staged inputs to a single h5 file.

    Args:
        staged: StagedInputs to serialize
        path: output file path
    """
    p = Path(path).expanduser().resolve()
    t_dump = _stage_begin(f"writing staged inputs to {p}")
    _dump_h5(staged, p)
    _stage_end(t_dump, "staged inputs written")


def load(path: Union[str, Path]) -> StagedInputs:
    """
    Load staged inputs from a single file written by dump().

    Args:
        path: input file path

    Returns:
        StagedInputs
    """
    p = Path(path).expanduser().resolve()
    t_load = _stage_begin(f"loading staged inputs from {p}")
    staged = _load_h5(p)
    _stage_end(
        t_load,
        "staged inputs loaded",
        details=f"norb={staged.ham.norb} nchol={staged.ham.chol.shape[0]} trial={staged.trial.kind}",
    )
    return staged


def _is_cc_like(obj: Any) -> bool:
    return hasattr(obj, "t1") and hasattr(obj, "t2")


def _stage_ham_input(obj: StagedMfOrCc, *, chol_cut: float, verbose: bool) -> HamInput:
    """
    Produce h0/h1/chol in a single orthonormal basis.
    For UHF, we use the alpha MO basis for integrals.
    """
    mol = obj.mol
    scf_obj = obj.mf

    match scf_obj.kind:
        case "rhf" | "rohf" | "ghf":
            basis_coeff = np.asarray(scf_obj.mo_coeff)
        case "uhf":
            basis_coeff = np.asarray(scf_obj.mo_coeff[0])
        case _:
            raise ValueError(f"Unreachable: '{scf_obj.kind}'.")

    match scf_obj.kind:
        case "rhf" | "rohf" | "uhf":
            ham_basis = "restricted"
        case "ghf":
            ham_basis = "generalized"
        case _:
            raise ValueError(f"Unreachable: '{scf_obj.kind}'.")

    # nuclear energy (without frozen core correction)
    h0 = float(scf_obj.energy_nuc())

    # one body
    hcore = scf_obj.get_hcore()
    h1 = basis_coeff.T.conj() @ hcore @ basis_coeff
    h1 = np.asarray(h1)

    # ao cholesky
    t0 = time.time()
    chol_vec = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print(f"[stage] AO cholesky: nchol={chol_vec.shape[0]} in {time.time() - t0:.2f}s")

    # full space electron count
    nelec: Tuple[int, int] = (int(mol.nelec[0]), int(mol.nelec[1]))
    norb_frozen = scf_obj.afqmc_frozen

    assert isinstance(norb_frozen, int)

    # mo Cholesky
    C = np.asarray(basis_coeff)
    if scf_obj.kind != "ghf":
        norb = int(basis_coeff.shape[1])
        chol = _rotate_chol_to_mo(chol_vec, C)
    else:
        norb = basis_coeff.shape[1] // 2
        chol = _rotate_chol_to_ghf_mo(chol_vec, C)

    # freeze core
    if norb_frozen > 0 and scf_obj.kind != "ghf":

        if norb_frozen > min(nelec):
            raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")

        ncas = basis_coeff.shape[1] - norb_frozen
        nelec_active = (nelec[0] - norb_frozen, nelec[1] - norb_frozen)
        if min(nelec_active) < 0 or sum(nelec_active) <= 0 or ncas <= 0:
            raise ValueError("Frozen core left no active electrons/orbitals.")

        h0, h1, chol, nelec = _freeze_core_from_mo_cholesky(
            h0=h0,
            h1=h1,
            chol=chol,
            norb_frozen=norb_frozen,
            nelec=nelec,
        )
        norb = int(ncas)
    elif norb_frozen > 0 and scf_obj.kind == "ghf":
        raise NotImplementedError(
            "Frozen core approximation not available for generalised integrals."
        )

    return HamInput(
        h0=h0,
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=norb_frozen,
        source_kind=obj.source,
        basis=ham_basis,
    )


def _load_fcidump_context(fcidump: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(fcidump, dict):
        ctx: Dict[str, Any] = dict(fcidump)
    elif isinstance(fcidump, (str, Path)):
        from pyscf.tools import fcidump as pyscf_fcidump

        ctx = pyscf_fcidump.read(str(Path(fcidump).expanduser()))
    else:
        raise TypeError(f"fcidump must be dict | str | Path, but received '{type(fcidump)}'.")

    required_keys = ("H1", "H2", "NORB", "NELEC")
    missing = [k for k in required_keys if k not in ctx]
    if missing:
        raise ValueError(f"fcidump is missing required keys: {missing}")
    return ctx


def _stage_ham_input_from_fcidump(
    obj: StagedMfOrCc,
    *,
    fcidump: Union[str, Path, Dict[str, Any]],
    chol_cut: float,
    verbose: bool,
) -> HamInput:
    """
    Build HamInput from FCIDUMP integrals while preserving the trial MO basis convention.
    """
    from pyscf import ao2mo

    scf_obj = obj.mf
    if scf_obj.kind == "ghf":
        raise NotImplementedError("FCIDUMP staging for stage_from_ccpy does not support GHF.")

    match scf_obj.kind:
        case "rhf" | "rohf":
            basis_coeff = np.asarray(scf_obj.mo_coeff)
        case "uhf":
            basis_coeff = np.asarray(scf_obj.mo_coeff[0])
        case _:
            raise ValueError(f"Unreachable: '{scf_obj.kind}'.")

    ctx = _load_fcidump_context(fcidump)
    norb = int(ctx["NORB"])
    if basis_coeff.shape[1] != norb:
        raise ValueError(
            "FCIDUMP NORB does not match mf orbital count: " f"{norb} != {basis_coeff.shape[1]}."
        )

    h0 = float(ctx.get("ECORE", 0.0))
    h1_ao = np.asarray(ctx["H1"])
    if h1_ao.shape != (norb, norb):
        raise ValueError(f"FCIDUMP H1 must have shape ({norb}, {norb}), got {h1_ao.shape}.")
    h1_ao = 0.5 * (h1_ao + h1_ao.T.conj())
    h1 = basis_coeff.T.conj() @ h1_ao @ basis_coeff
    h1 = np.asarray(h1)

    h2_raw = np.asarray(ctx["H2"])
    eri_ao = ao2mo.restore(1, h2_raw, norb)
    eri_mo = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl",
        basis_coeff.conj(),
        basis_coeff.conj(),
        basis_coeff,
        basis_coeff,
        eri_ao,
        optimize=True,
    )

    nelec_tot = int(ctx["NELEC"])
    ms2 = int(ctx.get("MS2", 0))
    if (nelec_tot + ms2) % 2 != 0 or (nelec_tot - ms2) % 2 != 0:
        raise ValueError(f"Inconsistent FCIDUMP NELEC/MS2 pair: NELEC={nelec_tot}, MS2={ms2}.")
    nelec: Tuple[int, int] = ((nelec_tot + ms2) // 2, (nelec_tot - ms2) // 2)

    norb_frozen = scf_obj.afqmc_frozen
    assert isinstance(norb_frozen, int)
    if norb_frozen < 0:
        raise ValueError(f"norb_frozen must be non-negative, got {norb_frozen}.")
    if norb_frozen > min(nelec):
        raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")
    if norb_frozen >= norb:
        raise ValueError(f"norb_frozen={norb_frozen} leaves no active orbitals (norb={norb}).")
    if norb_frozen > 0:
        raise NotImplementedError(
            "FCIDUMP staging currently supports only norb_frozen=0. "
            "Use stage_from_ccpy(..., fcidump=...) which falls back to the existing MF frozen-core path."
        )

    t0 = time.time()
    eri_s4 = ao2mo.restore(4, np.asarray(eri_mo), norb)
    chol = modified_cholesky(eri_s4, max_error=chol_cut)
    if verbose:
        print(f"[stage] FCIDUMP cholesky: nchol={chol.shape[0]} in {time.time() - t0:.2f}s")

    return HamInput(
        h0=float(h0),
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=norb_frozen,
        source_kind=obj.source,
        basis="restricted",
    )


def _stage_trial_input(obj: StagedMfOrCc) -> TrialInput:
    """
    Produce TrialInput consistent with the Hamiltonian basis and frozen core choice
    """

    match obj.kind:
        case "rhf" | "rohf" | "uhf" | "ghf":
            stage_tr_fun = _stage_mf_input
        case "ccsd":
            stage_tr_fun = _stage_cisd_input
        case "uccsd":
            stage_tr_fun = _stage_ucisd_input
        case "gccsd":
            stage_tr_fun = _stage_gcisd_input
        case "pt2ccsd":
            stage_tr_fun = _stage_pt2ccsd_input
        case _:
            raise ValueError(f"Unreachable: '{obj.kind}'.")

    return stage_tr_fun(obj)


def _active_orbital_indices(norb: int, frozen: int | NDArray) -> NDArray:
    if isinstance(frozen, int):
        return np.arange(frozen, norb, dtype=np.int64)
    if isinstance(frozen, np.ndarray):
        return np.delete(np.arange(norb, dtype=np.int64), frozen)
    raise TypeError(f"frozen must be an integer or a np.ndarray, but received '{type(frozen)}'.")


def _apply_frozen_mask(vec: NDArray, frozen: int | NDArray) -> NDArray:
    return vec[_active_orbital_indices(vec.shape[0], frozen)]


def _stage_mf_input(obj: StagedMfOrCc) -> TrialInput:

    mol = obj.mol
    S = obj.get_ovlp(mol)
    frozen = obj.afqmc_frozen

    match obj.mf.kind:
        case "rhf" | "ghf":
            Ca = np.asarray(obj.mo_coeff)
            mo = _mf_coeff_helper(Ca, Ca, S, frozen)
            data = {"mo": np.asarray(mo)}

        case "rohf":
            Ca = np.asarray(obj.mo_coeff)
            mo = _mf_coeff_helper(Ca, Ca, S, frozen)
            mo_occ = _apply_frozen_mask(np.asarray(obj.mf.mo_occ), frozen)
            data = {
                "mo_a": np.asarray(mo[:, mo_occ > 0.0]),
                "mo_b": np.asarray(mo[:, mo_occ > 1.0]),
            }

        case "uhf":
            Ca = np.asarray(obj.mo_coeff[0])
            Cb = np.asarray(obj.mo_coeff[1])

            # basis is alpha MOs, represent alpha and beta orbitals in this basis
            moa = _mf_coeff_helper(Ca, Ca, S, frozen)
            mob = _mf_coeff_helper(Ca, Cb, S, frozen)
            data = {"mo_a": np.asarray(moa), "mo_b": np.asarray(mob)}
        case _:
            raise ValueError(f"Unreachable: '{obj.kind}'.")

    return TrialInput(
        kind=obj.kind,
        data=data,
        frozen=frozen,
        source_kind=obj.source,
    )


def _mf_coeff_helper(
    Ca: NDArray,
    Cb: NDArray,
    S: NDArray,
    frozen: int | NDArray,
) -> NDArray:
    q, r = np.linalg.qr(Ca.T @ S @ Cb)
    sgn = np.sign(np.diag(r))
    q = q * sgn[None, :]
    idx = _active_orbital_indices(len(q), frozen)
    return q[np.ix_(idx, idx)]


def _stage_cisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "ccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1_arr = np.asarray(obj.t1)
    t2_arr = np.asarray(obj.t2)
    nocc_t_core = 0
    nvir_t_outer = 0

    if isinstance(obj.trial_frozen, (np.ndarray)):
        if obj.mol.nelec[0] != obj.mol.nelec[1]:
            raise ValueError(
                "List-valued cc.frozen is currently supported only for closed-shell restricted CCSD."
            )

        assert isinstance(obj.afqmc_frozen, int)

        nocc_t_core, nvir_t_outer = _infer_restricted_trial_freeze_from_cc(
            cc_frozen=obj.trial_frozen,
            nmo_full=_mo_coeff_nmo(obj.mf.mo_coeff),
            nocc_full=int(obj.mol.nelectron // 2),
            norb_frozen=int(obj.afqmc_frozen),
            t1_shape=(int(t1_arr.shape[0]), int(t1_arr.shape[1])),
        )

    ci2 = t2_arr + np.einsum("ia,jb->ijab", t1_arr, t1_arr)
    ci2 = ci2.transpose(0, 2, 1, 3)  # (i,a,j,b) -> (i,j,a,b)
    ci1 = t1_arr

    data = {
        "ci1": ci1,
        "ci2": ci2,
        "nocc_t_core": np.array(nocc_t_core, dtype=np.int64),
        "nvir_t_outer": np.array(nvir_t_outer, dtype=np.int64),
    }
    return TrialInput(
        kind="cisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _stage_ucisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "uccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1a, t1b = obj.t1
    t2aa, t2ab, t2bb = obj.t2

    ci2aa = np.asarray(t2aa) + 2.0 * np.einsum("ia,jb->ijab", np.asarray(t1a), np.asarray(t1a))
    ci2aa = 0.5 * (ci2aa - ci2aa.transpose(0, 1, 3, 2))
    ci2aa = ci2aa.transpose(0, 2, 1, 3)

    ci2bb = np.asarray(t2bb) + 2.0 * np.einsum("ia,jb->ijab", np.asarray(t1b), np.asarray(t1b))
    ci2bb = 0.5 * (ci2bb - ci2bb.transpose(0, 1, 3, 2))
    ci2bb = ci2bb.transpose(0, 2, 1, 3)

    ci2ab = np.asarray(t2ab) + np.einsum("ia,jb->ijab", np.asarray(t1a), np.asarray(t1b))
    ci2ab = ci2ab.transpose(0, 2, 1, 3)

    _uhf_input = _stage_mf_input(obj)
    moa = _uhf_input.data["mo_a"]
    mob = _uhf_input.data["mo_b"]

    data = {
        "mo_coeff_a": np.asarray(moa),
        "mo_coeff_b": np.asarray(mob),
        "ci1a": np.asarray(t1a),
        "ci1b": np.asarray(t1b),
        "ci2aa": np.asarray(ci2aa),
        "ci2ab": np.asarray(ci2ab),
        "ci2bb": np.asarray(ci2bb),
    }

    return TrialInput(
        kind="ucisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def _stage_gcisd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "gccsd":
        raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1 = obj.t1
    t2 = obj.t2

    ci2 = (
        np.einsum("ijab->iajb", t2)
        + np.einsum("ia,jb->iajb", t1, t1)
        - np.einsum("ib,ja->iajb", t1, t1)
    )
    ci1 = np.asarray(t1)

    _ghf_input = _stage_mf_input(obj)
    mo = _ghf_input.data["mo"]

    data = {"mo_coeff": mo, "ci1": ci1, "ci2": ci2}

    return TrialInput(
        kind="gcisd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def stage_pt2ccsd_trial(cc: Any, *, frozen: int | ArrayLike | None = None) -> TrialInput:
    """
    Stage a pt2CCSD trial from a pyscf CC object.

    A CCSD object is ambiguous: it can serve as a CISD trial (what stage() builds) or as
    a pt2CCSD trial. Type dispatch cannot tell those apart, so this is a separate entry
    point rather than another case in _stage_trial_input.

    Returns a TrialInput with data {"mo_t", "t2"}: mo_t is the Thouless-transformed
    reference exp(T1)|HF> in the MO basis, t2 the doubles amplitudes as (i, a, j, b).
    """
    return _stage_pt2ccsd_input(StagedMfOrCc(cc, frozen))


def _stage_pt2ccsd_input(obj):
    # TODO obj.kind is frozen... figure out how to assign more flexible trial
    # if obj.kind != "pt2ccsd":
    #     raise ValueError(f"Unreachable: '{obj.kind}'.")

    t1 = obj.t1
    t2 = obj.t2
    nocc, nvir = t1.shape
    norb = nocc + nvir

    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    t2 = t2.transpose(0, 2, 1, 3)  # (i,j,a,b) -> (i,a,j,b)

    def _thouless(init_slater, t1):
        # Thouless transformation: |psi'> = exp(t1_ia a+ i)|psi>
        # init slater: mo_coeff of psi (in mo basis)
        # return mo_coeff of psi' (in mo basis)
        norb, nvir = t1.shape
        norb = nocc + nvir
        exp_t1 = np.eye(norb, dtype=np.float64)
        exp_t1[:nocc, nocc:] = t1
        # exp_t1 = jsp.linalg.expm(t1_full)
        return exp_t1.T @ init_slater

    mo_coeff = np.eye(norb, dtype=np.float64)[:, :nocc]
    mo_t = _thouless(mo_coeff, t1)

    data = {"mo_t": mo_t, "t2": t2}
    return TrialInput(
        kind="pt2ccsd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


def stage_upt2ccsd_trial(cc: Any, *, frozen: int | ArrayLike | None = None) -> TrialInput:
    """
    Stage an unrestricted pt2CCSD trial from a pyscf UCCSD object, in the conventions of
    afqmc's upt2ccsd trials (integral.save_cc_amplitude, slater_tools.uthouless).

    Each spin is expressed in its own MO basis -- the basis build_ham_uchol builds that
    spin's hamiltonian in -- so its reference occupies the leading nocc orbitals.

    Returns a TrialInput with data {"mo_t_a", "mo_t_b", "t2aa", "t2ab", "t2bb"}: mo_t_s is
    the Thouless-transformed reference exp(T1s)|HF_s>, and the doubles are (i, a, j, b)
    with the same spin blocks antisymmetrized.
    """
    return _stage_upt2ccsd_input(StagedMfOrCc(cc, frozen))


def _stage_upt2ccsd_input(obj: StagedMfOrCc) -> TrialInput:
    if obj.kind != "uccsd":
        raise ValueError(f"unrestricted pt2CCSD needs a UCCSD object, got kind {obj.kind!r}.")

    t1a, t1b = (np.asarray(t) for t in obj.t1)
    t2aa, t2ab, t2bb = (np.asarray(t) for t in obj.t2)

    # antisymmetrize the same spin blocks (a no-op on pyscf's, which already are), then
    # (i,j,a,b) -> (i,a,j,b)
    t2aa = 0.5 * (t2aa - t2aa.transpose(0, 1, 3, 2))
    t2bb = 0.5 * (t2bb - t2bb.transpose(0, 1, 3, 2))
    t2aa = t2aa.transpose(0, 2, 1, 3)
    t2ab = t2ab.transpose(0, 2, 1, 3)
    t2bb = t2bb.transpose(0, 2, 1, 3)

    def _thouless(t1):
        # |psi'> = exp(t1_ia a+ i)|psi>, with |psi> the leading nocc orbitals of the MO
        # basis; returns the mo_coeff of psi' in that basis
        nocc, nvir = t1.shape
        exp_t1 = np.eye(nocc + nvir, dtype=np.float64)
        exp_t1[:nocc, nocc:] = t1
        return exp_t1.T[:, :nocc]

    data = {
        "mo_t_a": _thouless(t1a),
        "mo_t_b": _thouless(t1b),
        "t2aa": t2aa,
        "t2ab": t2ab,
        "t2bb": t2bb,
    }
    return TrialInput(
        kind="upt2ccsd",
        data=data,
        frozen=obj.trial_frozen,
        source_kind=obj.source,
    )


# ---------------------------------------------------------------------------
# ccpy interface helpers
# ---------------------------------------------------------------------------


def _ccpy_t_to_c_amplitudes(driver: Any, order: int, order_cc: int) -> dict:
    from .staging_ccpy import _ccpy_t_to_c_amplitudes as _impl

    return _impl(driver, order=order, order_cc=order_cc)


def stage_from_ccpy(
    driver: Any,
    mf: Any,
    *,
    order: int = -1,
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    fcidump: Union[str, Path, Dict[str, Any], None] = None,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> StagedInputs:
    from .staging_ccpy import stage_from_ccpy as _impl

    return _impl(
        driver,
        mf,
        order=order,
        norb_frozen_core=norb_frozen_core,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        fcidump=fcidump,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
    )


def _dump_h5(staged: StagedInputs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(staged.meta)

        gham = f.create_group("ham")
        gham.create_dataset("h0", data=np.array(staged.ham.h0))
        gham.create_dataset("h1", data=staged.ham.h1)
        gham.create_dataset("chol", data=staged.ham.chol)
        gham.create_dataset("nelec", data=np.array(staged.ham.nelec, dtype=np.int64))
        gham.attrs["norb"] = staged.ham.norb
        gham.attrs["chol_cut"] = staged.ham.chol_cut
        _dump_frozen(gham, staged.ham.frozen)
        gham.attrs["source_kind"] = staged.ham.source_kind
        gham.attrs["basis"] = staged.ham.basis

        gtr = f.create_group("trial")
        gtr.attrs["kind"] = staged.trial.kind
        _dump_frozen(gtr, staged.trial.frozen)
        gtr.attrs["source_kind"] = staged.trial.source_kind
        gdata = gtr.create_group("data")
        for k, v in staged.trial.data.items():
            gdata.create_dataset(k, data=np.asarray(v))


def _to_json_str(x: Any) -> str:
    # np scalar -> python scalar
    if isinstance(x, np.ndarray):
        x = x.item()
    # bytes like -> decode
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return bytes(x).decode("utf-8")
    return str(x)


def _load_h5(path: Path) -> StagedInputs:
    with h5py.File(path, "r") as f:
        meta = json.loads(_to_json_str(f.attrs["meta_json"]))
        if "frozen" in meta:
            meta["frozen"] = _freeze_from_meta_value(meta["frozen"])

        t_ham = _stage_begin("reading Hamiltonian from cache")
        gham: Any = f["ham"]
        ham = HamInput(
            h0=float(np.array(gham["h0"]).item()),
            h1=np.array(gham["h1"]),
            chol=np.array(gham["chol"]),
            nelec=(int(np.array(gham["nelec"])[0]), int(np.array(gham["nelec"])[1])),
            norb=int(gham.attrs["norb"]),
            chol_cut=float(gham.attrs["chol_cut"]),
            frozen=_load_frozen(gham),
            source_kind=str(gham.attrs["source_kind"]),
            basis=cast(HamBasis, str(gham.attrs["basis"])),
        )
        _stage_end(
            t_ham, "Hamiltonian loaded", details=f"norb={ham.norb} nchol={ham.chol.shape[0]}"
        )

        t_trial = _stage_begin("reading trial input from cache")
        gtr: Any = f["trial"]
        gdata = gtr["data"]
        trial_data = {k: np.array(gdata[k]) for k in gdata.keys()}
        trial = TrialInput(
            kind=str(gtr.attrs["kind"]),
            data=trial_data,
            frozen=_load_frozen(gtr),
            source_kind=str(gtr.attrs["source_kind"]),
        )
        _stage_end(t_trial, "trial input loaded", details=f"kind={trial.kind}")

        return StagedInputs(ham=ham, trial=trial, meta=meta)


def build_ham_lno(
    obj: Any,
    *,
    frozen_orbitals: ArrayLike,
    chol_cut: float,
) -> HamInput:
    from pyscf import ao2mo, mcscf

    obj = StagedMfOrCc(obj, 0)
    mf = obj.mf.mf
    mol = mf.mol

    norb = obj.norb
    frozen = _normalize_frozen_list(frozen_orbitals, nmo=norb)
    basis_coeff = mf.mo_coeff

    nelec_frozen = 2 * np.sum(frozen < mol.nelec[0])
    nact = basis_coeff.shape[1] - frozen.size
    nelec_act = mol.nelectron - nelec_frozen
    mc = mcscf.CASSCF(mf, nact, nelec_act)
    mc.frozen = frozen  # type: ignore
    nelec = mc.nelecas  # type: ignore
    h1, h0 = mc.get_h1eff()  # type: ignore
    act = _active_orbital_indices(norb, frozen)
    e = np.asarray(ao2mo.kernel(mf.mol, mf.mo_coeff[:, act]))  # , compact=False)
    chol = modified_cholesky(e, max_error=chol_cut)
    chol = chol.reshape((-1, nact, nact))

    ham = HamInput(
        h0=float(h0),
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=nact,
        chol_cut=float(chol_cut),
        frozen=frozen,
        source_kind=obj.source,
        basis="restricted",
    )

    return ham


# ======================================================================================
# unrestricted (uchol) hamiltonian
#
# alpha and beta keep their own orbital basis, so h1 and chol are carried per spin. One
# common set of AO cholesky vectors is projected into each basis,
#
#     L^sigma_g = C_sigma^T L_g C_sigma,
#
# which keeps the auxiliary field index g shared while the orbital indices are per spin,
# and lets norb_a differ from norb_b.
# ======================================================================================


@dataclass(frozen=True, slots=True)
class HamInputU:
    """Unrestricted ham inputs, each spin in its own orthonormal one particle basis."""

    h0: float
    h1_a: Array  # (norb_a, norb_a)
    h1_b: Array  # (norb_b, norb_b)
    chol_a: Array  # (nchol, norb_a, norb_a)
    chol_b: Array  # (nchol, norb_b, norb_b)
    nelec: Tuple[int, int]
    norb: Tuple[int, int]
    chol_cut: float
    frozen: int | NDArray
    source_kind: str  # "mf" or "cc"
    basis: str = "uchol"


def df2chol(dferi: NDArray, max_error: float = 1e-6) -> NDArray:
    """
    Modified cholesky decomposition of a density fitting tensor.

    Args:
        dferi: packed 3-index DF integrals, shape (n_aux, n_pair) with n_pair the lower
            triangle of the AO pair index (pyscf lib.pack_tril ordering).
        max_error: stop when the residual diagonal falls below this.

    Returns:
        (n_chol, norb, norb) cholesky vectors.
    """
    dferi = np.asarray(dferi)
    n_aux, n_pair = dferi.shape
    norb = int(round((-1 + (1 + 8 * n_pair) ** 0.5) / 2))
    if norb * (norb + 1) // 2 != n_pair:
        raise ValueError(f"n_pair={n_pair} is not a valid packed lower triangle size")

    diag = (dferi**2).sum(axis=0)
    chol_vecs = np.zeros((n_aux, n_pair))
    m_approx = np.zeros(n_pair)
    diag_residual = diag.copy()

    nchol = 0
    while nchol < n_aux:
        nu = int(np.argmax(diag_residual))
        delta_max = diag_residual[nu]
        if delta_max < max_error:
            break

        row_nu = dferi.T @ dferi[:, nu]
        if nchol == 0:
            chol_vecs[nchol] = row_nu / delta_max**0.5
        else:
            r = chol_vecs[:nchol, nu] @ chol_vecs[:nchol, :]
            chol_vecs[nchol] = (row_nu - r) / delta_max**0.5

        m_approx += chol_vecs[nchol] ** 2
        diag_residual = np.abs(diag - m_approx)
        nchol += 1

    chol = np.zeros((nchol, norb, norb))
    row_idx, col_idx = np.tril_indices(norb)
    chol[:, row_idx, col_idx] = chol_vecs[:nchol]
    chol[:, col_idx, row_idx] = chol_vecs[:nchol]
    return chol


def _df_cderi(mf: Any) -> NDArray | None:
    """Packed (n_aux, n_pair) DF tensor if this mean field is density fitted, else None."""
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        return None
    blocks = [np.asarray(b) for b in with_df.loop()]
    if not blocks:
        return None
    return np.vstack(blocks)


def ao_cholesky(mf: Any, mol: Any, *, chol_cut: float, verbose: bool = False) -> NDArray:
    """
    AO cholesky vectors, flattened to (n_chol, nao*nao).

    Uses the density fitting tensor when the mean field carries one, otherwise falls back
    to the modified cholesky decomposition of the AO ERIs.
    """
    t0 = time.time()
    cderi = _df_cderi(mf)

    if cderi is not None:
        print(
            f"[stage] cholesky from density fitting (n_aux={cderi.shape[0]}), "
            f"max_error={chol_cut:g} ..."
        )
        chol = df2chol(cderi, max_error=chol_cut)
        nao = int(chol.shape[1])
        chol = chol.reshape(chol.shape[0], nao * nao)
        source = "density fitting"
    else:
        print(f"[stage] AO modified cholesky, max_error={chol_cut:g} ...")
        chol = np.asarray(chunked_cholesky(mol, max_error=chol_cut, verbose=verbose))
        source = "AO modified cholesky"

    print(f"[stage] {source}: nchol={chol.shape[0]} in {time.time() - t0:.2f}s")
    return chol


def _freeze_core_from_mo_cholesky_uh(
    *,
    h0: float,
    h1_a: NDArray,
    h1_b: NDArray,
    chol_a: NDArray,
    chol_b: NDArray,
    norb_frozen: int,
    nelec: Tuple[int, int],
) -> tuple[float, NDArray, NDArray, NDArray, NDArray, Tuple[int, int]]:
    """
    Unrestricted frozen core.

    _freeze_core_from_mo_cholesky is closed shell: it carries factors of 2 that assume
    both spins occupy the same spatial core orbitals. Here the two spins have their own
    basis and their own core, so:

        E_core   = sum_sigma tr h^sigma_cc
                   + 1/2 sum_g [ T_g^2 - sum_sigma tr(L^sigma_cc L^sigma_cc) ]
        h1_eff^s = h^s_aa + sum_g T_g L^s_aa - sum_g L^s_ac L^s_ca

    with T_g = tr L^a_g,cc + tr L^b_g,cc the core trace summed over spin. The coulomb
    term sees both spins (T_g), the exchange term only its own.

    Setting alpha == beta reduces to the restricted expressions.
    """
    if norb_frozen < 0:
        raise ValueError(f"norb_frozen must be non-negative, got {norb_frozen}.")
    if norb_frozen == 0:
        return float(h0), h1_a, h1_b, chol_a, chol_b, nelec
    if norb_frozen > min(nelec):
        raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")

    nmo_a, nmo_b = int(h1_a.shape[0]), int(h1_b.shape[0])
    if norb_frozen >= min(nmo_a, nmo_b):
        raise ValueError(
            f"norb_frozen={norb_frozen} leaves no active orbitals "
            f"(norb_a={nmo_a}, norb_b={nmo_b})."
        )

    nelec_active = (int(nelec[0] - norb_frozen), int(nelec[1] - norb_frozen))
    if min(nelec_active) < 0 or sum(nelec_active) <= 0:
        raise ValueError("Frozen core left no active electrons.")

    core = slice(0, norb_frozen)

    def blocks(h1, chol, nmo):
        act = slice(norb_frozen, nmo)
        return (
            np.asarray(h1[core, core]),
            np.asarray(h1[act, act]),
            np.asarray(chol[:, core, core]),
            np.asarray(chol[:, act, act]),
            np.asarray(chol[:, act, core]),
            np.asarray(chol[:, core, act]),
        )

    h1_cc_a, h1_aa_a, l_cc_a, l_aa_a, l_ac_a, l_ca_a = blocks(h1_a, chol_a, nmo_a)
    h1_cc_b, h1_aa_b, l_cc_b, l_aa_b, l_ac_b, l_ca_b = blocks(h1_b, chol_b, nmo_b)

    # core trace summed over spin: the coulomb field the active space sees
    t_g = np.trace(l_cc_a, axis1=1, axis2=2) + np.trace(l_cc_b, axis1=1, axis2=2)

    vj_a = np.einsum("x,xpq->pq", t_g, l_aa_a, optimize=True)
    vj_b = np.einsum("x,xpq->pq", t_g, l_aa_b, optimize=True)
    vk_a = np.einsum("xpi,xiq->pq", l_ac_a, l_ca_a, optimize=True)
    vk_b = np.einsum("xpi,xiq->pq", l_ac_b, l_ca_b, optimize=True)

    h1_eff_a = h1_aa_a + vj_a - vk_a
    h1_eff_b = h1_aa_b + vj_b - vk_b

    e1_core = np.trace(h1_cc_a) + np.trace(h1_cc_b)
    ej_core = 0.5 * float(np.dot(t_g, t_g))
    ek_core = 0.5 * (
        np.einsum("xij,xji->", l_cc_a, l_cc_a, optimize=True)
        + np.einsum("xij,xji->", l_cc_b, l_cc_b, optimize=True)
    )
    ecore = float(np.real(h0 + e1_core + ej_core - ek_core))

    return (
        ecore,
        np.asarray(h1_eff_a),
        np.asarray(h1_eff_b),
        np.array(l_aa_a, copy=True),
        np.array(l_aa_b, copy=True),
        nelec_active,
    )


def build_ham_uchol(
    obj: Any,
    *,
    chol_cut: float = 1e-5,
    basis_a: NDArray | None = None,
    basis_b: NDArray | None = None,
    norb_frozen_core: int = 0,
    verbose: bool = False,
) -> HamInputU:
    """
    Build an unrestricted cholesky hamiltonian from a pyscf UHF (or CC) object.

    basis_a / basis_b default to the UHF alpha and beta coefficients. Pass them
    explicitly to use two independently chosen active spaces, which is what unrestricted
    LNO does and where norb_a != norb_b comes from.
    """
    staged = StagedMfOrCc(obj, norb_frozen_core)
    mf = staged.mf.mf
    mol = mf.mol

    if basis_a is None or basis_b is None:
        mo = staged.mo_coeff
        # pyscf gives UHF coefficients either as a (2, nao, nmo) array or as a pair
        if not isinstance(mo, (tuple, list)):
            mo_arr = np.asarray(mo)
            if mo_arr.ndim != 3 or mo_arr.shape[0] != 2:
                raise ValueError(
                    "build_ham_uchol needs a UHF-like object with (mo_a, mo_b) "
                    "coefficients, or explicit basis_a / basis_b."
                )
            mo = (mo_arr[0], mo_arr[1])
        elif len(mo) != 2:
            raise ValueError(
                "build_ham_uchol needs a UHF-like object with (mo_a, mo_b) coefficients, "
                "or explicit basis_a / basis_b."
            )
        basis_a = np.asarray(mo[0]) if basis_a is None else np.asarray(basis_a)
        basis_b = np.asarray(mo[1]) if basis_b is None else np.asarray(basis_b)
    basis_a = np.asarray(basis_a)
    basis_b = np.asarray(basis_b)

    h0 = float(mf.energy_nuc())
    hcore = np.asarray(mf.get_hcore())
    chol_ao = ao_cholesky(mf, mol, chol_cut=chol_cut, verbose=verbose)

    t_proj = time.time()
    h1_a = basis_a.conj().T @ hcore @ basis_a
    h1_b = basis_b.conj().T @ hcore @ basis_b
    # _rotate_chol_to_mo rotates in place when nao == norb, so the alpha call would
    # otherwise clobber chol_ao and the beta call would rotate it a second time. Hand
    # alpha a copy and let beta consume the original (its last use).
    chol_a = _rotate_chol_to_mo(np.array(chol_ao, copy=True), basis_a)
    chol_b = _rotate_chol_to_mo(chol_ao, basis_b)
    print(
        f"[stage] projected chol into the alpha/beta bases "
        f"({basis_a.shape[1]}, {basis_b.shape[1]} orbitals) in {time.time() - t_proj:.2f}s"
    )

    nelec: Tuple[int, int] = (int(mol.nelec[0]), int(mol.nelec[1]))

    norb_frozen = int(norb_frozen_core)
    if norb_frozen > 0:
        h0, h1_a, h1_b, chol_a, chol_b, nelec = _freeze_core_from_mo_cholesky_uh(
            h0=h0,
            h1_a=h1_a,
            h1_b=h1_b,
            chol_a=chol_a,
            chol_b=chol_b,
            norb_frozen=norb_frozen,
            nelec=nelec,
        )

    norb = (int(h1_a.shape[0]), int(h1_b.shape[0]))
    print(
        f"[stage] uchol ham ready: norb={norb} nchol={chol_a.shape[0]} "
        f"nelec={nelec} frozen={norb_frozen} h0={h0:.10f}"
    )

    return HamInputU(
        h0=h0,
        h1_a=np.asarray(h1_a),
        h1_b=np.asarray(h1_b),
        chol_a=np.asarray(chol_a),
        chol_b=np.asarray(chol_b),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=norb_frozen,
        source_kind=staged.source,
        basis="uchol",
    )
