"""
Staging for the unrestricted (uchol) hamiltonian.

The companion of staging.py for the case where alpha and beta keep their own orbital
basis. Nothing in staging.py is changed: this module builds a HamInputU, stages the
trial half through staging's own functions, and carries the result in a StagedInputs
whose ham slot holds the HamInputU.

    HamInputU          the unrestricted inputs, each spin in its own basis
    build_ham_uchol    HamInputU from a pyscf UHF mean field (or an RHF via to_uhf), or
                       from a UCCSD object (bases = the CC orbitals, core = cc.frozen)
    stage_uh           StagedInputs (HamInputU + TrialInput + meta), with optional cache;
                       the trial is the UHF determinant for a mean field and the
                       CC-derived UCISD for a UCCSD object (trial_kind picks one)
    stage_upt2ccsd_trial_uh
                       the unrestricted pt2CCSD trial of a UCCSD object, for the mixed runs
    dump_uh / load_uh  the h5 layout, ham/{h1_a,h1_b,chol_a,chol_b} with basis "uchol"
    is_uchol_file      whether a staged file was written by dump_uh
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import h5py
import numpy as np
from numpy.typing import NDArray

from .cholesky import (
    ao_cholesky,
    df_cderi,
    freeze_core_from_mo_cholesky_uh,
    normalize_frozen_core_uh,
    rotate_chol_to_mo,
)
from .staging import (
    STAGE_FORMAT_VERSION,
    HamInput,
    StagedInputs,
    StagedMfOrCc,
    TrialInput,
    _freeze_core_from_mo_cholesky,
    _freeze_from_meta_value,
    _is_cc_like,
    _stage_begin,
    _stage_end,
    _stage_ham_input,
    _stage_trial_input,
    _to_json_str,
)

Array = Any


@dataclass(frozen=True, slots=True)
class HamInputU:
    """
    Unrestricted ham inputs, each spin in its own orthonormal one particle basis.

    The reference determinant of spin s occupies the leading nelec[s] orbitals of that
    spin's basis (the basis is that spin's MOs, or the columns the caller passed as
    basis_s, in that order), which is what the uchol trials and walkers assume.
    """

    h0: float
    h1_a: Array  # (norb_a, norb_a)
    h1_b: Array  # (norb_b, norb_b)
    chol_a: Array  # (nchol, norb_a, norb_a)
    chol_b: Array  # (nchol, norb_b, norb_b)
    nelec: Tuple[int, int]  # active electrons
    norb: Tuple[int, int]  # active orbitals per spin
    chol_cut: float
    frozen: Tuple[int, int]  # frozen core orbitals per spin
    source_kind: str  # "mf" or "cc"
    basis: str = "uchol"

    @property
    def nchol(self) -> int:
        return int(self.chol_a.shape[0])


def _uhf_bases(staged: StagedMfOrCc) -> tuple[NDArray, NDArray]:
    mo = staged.mo_coeff
    # pyscf gives UHF coefficients either as a (2, nao, nmo) array or as a pair
    if not isinstance(mo, (tuple, list)):
        mo_arr = np.asarray(mo)
        if mo_arr.ndim != 3 or mo_arr.shape[0] != 2:
            raise ValueError(
                "build_ham_uchol needs a UHF-like object with (mo_a, mo_b) coefficients "
                "(convert an RHF with mf.to_uhf()), or explicit basis_a / basis_b."
            )
        return np.asarray(mo_arr[0]), np.asarray(mo_arr[1])
    if len(mo) != 2:
        raise ValueError(
            "build_ham_uchol needs a UHF-like object with (mo_a, mo_b) coefficients, "
            "or explicit basis_a / basis_b."
        )
    return np.asarray(mo[0]), np.asarray(mo[1])


def _staged_obj_uh(obj: Any, norb_frozen_core: Any) -> tuple[StagedMfOrCc, Tuple[int, int]]:
    """
    The validated pyscf object and the per spin frozen core.

    For a CC object the core is cc.frozen (an int, the same for both spins, as pyscf's
    UCCSD amplitudes require); norb_frozen_core may repeat it but not contradict it. For a
    mean field it is norb_frozen_core, an int or a pair.
    """
    if _is_cc_like(obj):
        cc_frozen = getattr(obj, "frozen", None)
        if cc_frozen is not None and not isinstance(cc_frozen, (int, np.integer)):
            raise NotImplementedError(
                "list-valued cc.frozen is not supported on the unrestricted hamiltonian; "
                "freeze an integer core in the UCCSD."
            )
        cc_core = int(cc_frozen or 0)
        n_core = normalize_frozen_core_uh(norb_frozen_core)
        if norb_frozen_core is not None and n_core != (cc_core, cc_core):
            raise ValueError(
                f"norb_frozen_core={n_core} contradicts cc.frozen={cc_core}; the trial "
                "amplitudes fix the frozen core of a CC object."
            )
        n_core = (cc_core, cc_core)
        # StagedCc checks the count against cc.frozen and copies the CC orbitals onto mf
        return StagedMfOrCc(obj, cc_core), n_core

    n_core = normalize_frozen_core_uh(norb_frozen_core)
    # StagedMf validates the object; an int core keeps its checks, a pair is validated
    # against each spin by the frozen core routine
    return StagedMfOrCc(obj, max(n_core) if n_core[0] != n_core[1] else n_core[0]), n_core


def build_ham_uchol(
    obj: Any,
    *,
    chol_cut: float = 1e-5,
    basis_a: NDArray | None = None,
    basis_b: NDArray | None = None,
    norb_frozen_core: int | Tuple[int, int] | None = None,
    verbose: bool = False,
) -> HamInputU:
    """
    Build an unrestricted cholesky hamiltonian from a pyscf UHF (or UCCSD) object.

    The AO two-electron integrals are cholesky decomposed once (cholesky.ao_cholesky:
    from the DF tensor of a density fitted mean field, otherwise staging's
    chunked_cholesky of the exact ERIs, with the same cutoff semantics as the restricted
    hamiltonian) and the vectors are projected into the alpha and beta bases,

        L^s_g = C_s^dag L_g C_s,     h1^s = C_s^dag hcore C_s,

    so the field index g is shared and the orbital dimensions are per spin.

    basis_a / basis_b default to the object's UHF alpha and beta coefficients. Pass them
    explicitly to use two independently chosen orbital sets, which may differ in size;
    the leading columns of each must be that spin's occupied orbitals.

    norb_frozen_core is an int (same core for both spins) or a pair (n_a, n_b); for a
    CC object it is cc.frozen. The core energy and potential are built from the
    projected cholesky vectors, as staging does for the restricted hamiltonian
    (cholesky.freeze_core_from_mo_cholesky_uh).
    """
    staged, n_core = _staged_obj_uh(obj, norb_frozen_core)
    mf = staged.mf.mf
    mol = mf.mol

    if basis_a is None or basis_b is None:
        mo_a, mo_b = _uhf_bases(staged)
        basis_a = mo_a if basis_a is None else np.asarray(basis_a)
        basis_b = mo_b if basis_b is None else np.asarray(basis_b)
    basis_a = np.asarray(basis_a)
    basis_b = np.asarray(basis_b)
    if basis_a.ndim != 2 or basis_b.ndim != 2 or basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError(
            "basis_a and basis_b must be (nao, norb_s) coefficient matrices over the same "
            f"AO basis, got {basis_a.shape} and {basis_b.shape}."
        )

    h0 = float(mf.energy_nuc())
    hcore = np.asarray(mf.get_hcore())
    chol_ao = ao_cholesky(mf, chol_cut=chol_cut, verbose=verbose)

    t_proj = time.time()
    h1_a = np.asarray(basis_a.conj().T @ hcore @ basis_a)
    h1_b = np.asarray(basis_b.conj().T @ hcore @ basis_b)
    # the rotation reuses its input storage when nao == norb, so the alpha call would
    # otherwise clobber chol_ao and the beta call would rotate it a second time. Hand
    # alpha a copy and let beta consume the original (its last use).
    chol_a = rotate_chol_to_mo(np.array(chol_ao, copy=True), basis_a)
    chol_b = rotate_chol_to_mo(chol_ao, basis_b)
    del chol_ao
    print(
        f"[stage] projected chol into the alpha/beta bases "
        f"({basis_a.shape[1]}, {basis_b.shape[1]} orbitals) in {time.time() - t_proj:.2f}s"
    )

    nelec: Tuple[int, int] = (int(mol.nelec[0]), int(mol.nelec[1]))

    h0, h1_a, h1_b, chol_a, chol_b, nelec = freeze_core_from_mo_cholesky_uh(
        h0=h0,
        h1_a=h1_a,
        h1_b=h1_b,
        chol_a=chol_a,
        chol_b=chol_b,
        norb_frozen=n_core,
        nelec=nelec,
    )

    norb = (int(h1_a.shape[0]), int(h1_b.shape[0]))
    print(
        f"[stage] uchol ham ready: norb={norb} nchol={chol_a.shape[0]} "
        f"nelec={nelec} frozen={n_core} h0={h0:.10f}"
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
        frozen=n_core,
        source_kind=staged.source,
        basis="uchol",
    )


def stage_ham_input_df(obj: Any, *, chol_cut: float = 1e-5, verbose: bool = False) -> HamInput:
    """
    staging._stage_ham_input with the AO cholesky vectors of cholesky.ao_cholesky: from
    the DF tensor when the mean field is density fitted, else the same modified cholesky
    of the exact ERIs (then this simply defers to staging's own builder). obj is a pyscf
    mf/cc object or a StagedMfOrCc; the frozen core is its afqmc_frozen. Hand the result
    to staging.stage(..., ham=) to get the restricted hamiltonian of a DF mean field on
    the same footing as the mean field, as AfqmcMixed does.
    """
    staged = obj if isinstance(obj, StagedMfOrCc) else StagedMfOrCc(obj, None)
    scf_obj = staged.mf
    mf = scf_obj.mf
    if df_cderi(mf) is None:
        return _stage_ham_input(staged, chol_cut=chol_cut, verbose=verbose)
    if scf_obj.kind == "ghf":
        raise NotImplementedError("DF cholesky staging is not wired for GHF objects.")

    mol = staged.mol
    basis_coeff = np.asarray(scf_obj.mo_coeff[0] if scf_obj.kind == "uhf" else scf_obj.mo_coeff)
    h0 = float(scf_obj.energy_nuc())
    h1 = np.asarray(basis_coeff.T.conj() @ scf_obj.get_hcore() @ basis_coeff)
    chol_vec = ao_cholesky(mf, chol_cut=chol_cut, verbose=verbose)
    chol = rotate_chol_to_mo(chol_vec, basis_coeff)
    norb = int(basis_coeff.shape[1])

    nelec: Tuple[int, int] = (int(mol.nelec[0]), int(mol.nelec[1]))
    norb_frozen = scf_obj.afqmc_frozen
    if not isinstance(norb_frozen, (int, np.integer)):
        raise ValueError("stage_ham_input_df takes an integer frozen core count.")
    norb_frozen = int(norb_frozen)
    if norb_frozen > 0:
        h0, h1, chol, nelec = _freeze_core_from_mo_cholesky(
            h0=h0, h1=h1, chol=chol, norb_frozen=norb_frozen, nelec=nelec
        )
        norb = norb - norb_frozen

    return HamInput(
        h0=h0,
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        frozen=norb_frozen,
        source_kind=staged.source,
        basis="restricted",
    )


def _trial_input_uh(
    staged: StagedMfOrCc, ham: HamInputU, trial_kind: str | None = None
) -> TrialInput:
    """
    The trial for the uchol layout: the UCISD built from the CC amplitudes of a UCCSD
    object (staging._stage_ucisd_input; its coefficients are already in each spin's own
    MO basis, and the rotations it also stages are ignored by the uchol trial), or the
    UHF determinant of a mean field. trial_kind="uhf" asks for the determinant even from
    a CC object (a mixed run with the UHF guide); "ucisd" needs a CC object.
    """
    if trial_kind is not None and trial_kind not in ("uhf", "ucisd"):
        raise ValueError(f"trial_kind must be 'uhf' or 'ucisd', got {trial_kind!r}")
    if trial_kind == "ucisd" and staged.source != "cc":
        raise ValueError("the UCISD trial needs a UCCSD object.")
    if staged.source == "cc" and trial_kind != "uhf":
        if staged.kind != "uccsd":
            raise ValueError(
                f"the unrestricted hamiltonian takes a UCCSD object, got kind {staged.kind!r}; "
                "convert a restricted CCSD with pyscf.cc.addons.convert_to_uccsd."
            )
        tin = _stage_trial_input(staged)
        return TrialInput(
            kind=tin.kind,
            data=tin.data,
            frozen=np.asarray(ham.frozen, dtype=np.int64),
            source_kind=tin.source_kind,
        )
    return _uhf_trial_input(ham)


def _uhf_trial_input(ham: HamInputU) -> TrialInput:
    """
    The UHF trial in the uchol layout: each spin's reference determinant is the leading
    nelec[s] columns of the identity in that spin's active basis.
    """
    norb_a, norb_b = ham.norb
    data = {
        "mo_a": np.eye(norb_a)[:, : ham.nelec[0]],
        "mo_b": np.eye(norb_b)[:, : ham.nelec[1]],
    }
    return TrialInput(
        kind="uhf",
        data=data,
        frozen=np.asarray(ham.frozen, dtype=np.int64),
        source_kind=ham.source_kind,
    )


def stage_uh(
    obj: Any,
    *,
    norb_frozen_core: int | Tuple[int, int] | None = None,
    chol_cut: float = 1e-5,
    basis_a: NDArray | None = None,
    basis_b: NDArray | None = None,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    trial_kind: str | None = None,
) -> StagedInputs:
    """
    Stage the unrestricted hamiltonian and the trial: the UHF determinant from a pyscf
    mean field, the CC-derived UCISD from a UCCSD object (or its UHF determinant with
    trial_kind="uhf").

    Mirrors staging.stage for the uchol layout. The returned StagedInputs carries a
    HamInputU in its ham slot (StagedInputs.ham is typed as the restricted HamInput).
    With cache set, an existing file is loaded unless overwrite=True, and a fresh
    staging is written there.
    """
    cache_path = Path(cache).expanduser().resolve() if cache is not None else None
    if cache_path is not None and cache_path.exists() and not overwrite:
        return load_uh(cache_path)

    t0 = time.time()
    t_ham = _stage_begin("building unrestricted Hamiltonian")
    ham = build_ham_uchol(
        obj,
        chol_cut=chol_cut,
        basis_a=basis_a,
        basis_b=basis_b,
        norb_frozen_core=norb_frozen_core,
        verbose=verbose,
    )
    _stage_end(t_ham, "Hamiltonian ready", details=f"norb={ham.norb} nchol={ham.nchol}")

    staged_obj, _ = _staged_obj_uh(obj, norb_frozen_core)
    t_trial = _stage_begin("building trial input")
    trial = _trial_input_uh(staged_obj, ham, trial_kind)
    _stage_end(t_trial, "trial input ready", details=f"kind={trial.kind}")

    mol = staged_obj.mol
    meta: Dict[str, Any] = {
        "format_version": STAGE_FORMAT_VERSION,
        "timestamp_unix": time.time(),
        "source_kind": ham.source_kind,
        "frozen": [int(n) for n in ham.frozen],
        "chol_cut": ham.chol_cut,
        "ham_basis": "uchol",
        "mol": {
            "nao": int(mol.nao),
            "nelectron": int(mol.nelectron),
            "spin": int(mol.spin),
            "charge": int(mol.charge),
            "basis": getattr(mol, "basis", None),
        },
    }

    staged = StagedInputs(ham=ham, trial=trial, meta=meta)  # type: ignore[arg-type]

    if cache_path is not None:
        dump_uh(staged, cache_path)

    if verbose:
        print(f"[stage] done in {time.time() - t0:.2f}s | norb={ham.norb} nchol={ham.nchol}")

    return staged


def _thouless(t1: NDArray) -> NDArray:
    # |psi'> = exp(t1_ia a+_a a_i)|psi>, |psi> the leading nocc orbitals; the generator is
    # nilpotent so exp(X) = 1 + X exactly. Returns the mo_coeff of psi' in the MO basis.
    nocc, nvir = t1.shape
    exp_t1 = np.eye(nocc + nvir, dtype=np.float64)
    exp_t1[:nocc, nocc:] = t1
    return exp_t1.T[:, :nocc]


def stage_upt2ccsd_trial_uh(cc: Any, *, frozen: Any = None) -> TrialInput:
    """
    The unrestricted pt2CCSD trial of a pyscf UCCSD object, for the uchol layout.

    Each spin is expressed in its own MO basis, the basis build_ham_uchol builds that
    spin's hamiltonian in, so its reference occupies the leading nocc orbitals. Returns a
    TrialInput of kind "upt2ccsd" with data {"mo_t_a", "mo_t_b", "t2aa", "t2ab", "t2bb"}:
    mo_t_s is the Thouless reference exp(T1_s)|HF_s>, and the doubles are (i, a, j, b)
    with the same spin blocks antisymmetrized. frozen may repeat cc.frozen; the core is
    always the CC object's.
    """
    staged, n_core = _staged_obj_uh(cc, frozen)
    if staged.kind != "uccsd":
        raise ValueError(
            f"the unrestricted pt2CCSD trial needs a UCCSD object, got kind {staged.kind!r}; "
            "convert a restricted CCSD with pyscf.cc.addons.convert_to_uccsd."
        )
    t1a, t1b = (np.asarray(t, dtype=np.float64) for t in staged.t1)
    t2aa, t2ab, t2bb = (np.asarray(t, dtype=np.float64) for t in staged.t2)

    # antisymmetrize the same spin blocks (a no-op on pyscf's, which already are), then
    # (i,j,a,b) -> (i,a,j,b)
    t2aa = 0.5 * (t2aa - t2aa.transpose(0, 1, 3, 2))
    t2bb = 0.5 * (t2bb - t2bb.transpose(0, 1, 3, 2))
    data = {
        "mo_t_a": _thouless(t1a),
        "mo_t_b": _thouless(t1b),
        "t2aa": t2aa.transpose(0, 2, 1, 3),
        "t2ab": t2ab.transpose(0, 2, 1, 3),
        "t2bb": t2bb.transpose(0, 2, 1, 3),
    }
    return TrialInput(
        kind="upt2ccsd",
        data=data,
        frozen=np.asarray(n_core, dtype=np.int64),
        source_kind="cc",
    )


# ======================================================================================
# h5 layout
# ======================================================================================


def is_uchol_file(path: Union[str, Path]) -> bool:
    p = Path(path).expanduser().resolve()
    with h5py.File(p, "r") as f:
        return str(f["ham"].attrs.get("basis", "")) == "uchol"


def dump_uh(staged: StagedInputs, path: Union[str, Path]) -> None:
    """
    Write staged uchol inputs to a single h5 file: staging's layout with the ham group
    holding h1_a/h1_b/chol_a/chol_b, norb and frozen as integer pairs, and
    attrs["basis"] == "uchol".
    """
    ham: Any = staged.ham
    if getattr(ham, "basis", None) != "uchol":
        raise ValueError("dump_uh expects a StagedInputs carrying a HamInputU.")
    p = Path(path).expanduser().resolve()
    t_dump = _stage_begin(f"writing staged uchol inputs to {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        f.attrs["meta_json"] = json.dumps(staged.meta)

        gham = f.create_group("ham")
        gham.create_dataset("h0", data=np.array(ham.h0))
        gham.create_dataset("h1_a", data=ham.h1_a)
        gham.create_dataset("h1_b", data=ham.h1_b)
        gham.create_dataset("chol_a", data=ham.chol_a)
        gham.create_dataset("chol_b", data=ham.chol_b)
        gham.create_dataset("nelec", data=np.array(ham.nelec, dtype=np.int64))
        gham.create_dataset("norb", data=np.array(ham.norb, dtype=np.int64))
        gham.create_dataset("frozen", data=np.array(ham.frozen, dtype=np.int64))
        gham.attrs["chol_cut"] = ham.chol_cut
        gham.attrs["source_kind"] = ham.source_kind
        gham.attrs["basis"] = "uchol"

        gtr = f.create_group("trial")
        gtr.attrs["kind"] = staged.trial.kind
        frozen = staged.trial.frozen
        if isinstance(frozen, np.ndarray):
            gtr.create_dataset("frozen", data=np.asarray(frozen, dtype=np.int64))
        else:
            gtr.attrs["frozen"] = int(frozen)
        gtr.attrs["source_kind"] = staged.trial.source_kind
        gdata = gtr.create_group("data")
        for k, v in staged.trial.data.items():
            gdata.create_dataset(k, data=np.asarray(v))
    _stage_end(t_dump, "staged uchol inputs written")


def load_uh(path: Union[str, Path]) -> StagedInputs:
    """Load staged uchol inputs written by dump_uh."""
    p = Path(path).expanduser().resolve()
    t_load = _stage_begin(f"loading staged uchol inputs from {p}")
    with h5py.File(p, "r") as f:
        meta = json.loads(_to_json_str(f.attrs["meta_json"]))
        gham: Any = f["ham"]
        if str(gham.attrs.get("basis", "")) != "uchol":
            raise ValueError(f"{p} is not a uchol staged file; use staging.load.")
        nelec = np.array(gham["nelec"])
        norb = np.array(gham["norb"])
        frozen = np.array(gham["frozen"])
        ham = HamInputU(
            h0=float(np.array(gham["h0"]).item()),
            h1_a=np.array(gham["h1_a"]),
            h1_b=np.array(gham["h1_b"]),
            chol_a=np.array(gham["chol_a"]),
            chol_b=np.array(gham["chol_b"]),
            nelec=(int(nelec[0]), int(nelec[1])),
            norb=(int(norb[0]), int(norb[1])),
            chol_cut=float(gham.attrs["chol_cut"]),
            frozen=(int(frozen[0]), int(frozen[1])),
            source_kind=str(gham.attrs["source_kind"]),
            basis="uchol",
        )

        gtr: Any = f["trial"]
        gdata = gtr["data"]
        trial_data = {k: np.array(gdata[k]) for k in gdata.keys()}
        if "frozen" in gtr:
            trial_frozen: Any = np.asarray(gtr["frozen"][...], dtype=np.int64)
        else:
            trial_frozen = int(gtr.attrs["frozen"])
        trial = TrialInput(
            kind=str(gtr.attrs["kind"]),
            data=trial_data,
            frozen=trial_frozen,
            source_kind=str(gtr.attrs["source_kind"]),
        )
    if "frozen" in meta and not isinstance(meta["frozen"], list):
        meta["frozen"] = _freeze_from_meta_value(meta["frozen"])
    _stage_end(
        t_load,
        "staged uchol inputs loaded",
        details=f"norb={ham.norb} nchol={ham.nchol} trial={trial.kind}",
    )
    return StagedInputs(ham=ham, trial=trial, meta=meta)  # type: ignore[arg-type]
