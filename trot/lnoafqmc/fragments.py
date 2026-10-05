from __future__ import annotations

import os
import re
from collections import defaultdict
from functools import partial

import h5py
from typing import Any

import numpy as np
from pyscf import gto, lo, scf
from pyscf.data import elements
from pyscf.scf import atom_hf

try:
    from pyscf.lno import tools as lno_tools  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError("trot.lnoafqmc needs pyscf-forge for the LNO machinery.") from e

print = partial(print, flush=True)

# Fragment definitions for LNO: IAO local orbitals grouped by atom (or heavy atom plus its
# hydrogens), ported from afqmc's lno_afqmc/tools.py. iao_fragment is the entry point:
#
#     lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen)
#
# lo_coeff   (nao, nlo) orthonormal IAOs, or a pair for UHF
# frag_list  one list of LO indices per fragment (a pair of lists per fragment for UHF)
# frag_name  one label per fragment, e.g. "O0H1H2"
#
# For a basis outside the cc-pVXZ family the IAO reference basis is built from free-atom
# HF orbitals (free_atom_minao), since pyscf's 'minao' does not cover it.


def _fix_ecpbas(mol):
    """
    Repair `_ecpbas` on a mol restored from a chkfile.

    gto.mole.dumps() serializes an empty `_ecpbas` with .tolist(), which collapses it to a
    bare []; loads() then rebuilds it as shape (0,), and anything doing `_ecpbas[:, 0]`
    (e.g. pyscf.scf.atom_hf) raises IndexError. Only bites ECP-free molecules.
    """
    if np.asarray(mol._ecpbas).ndim != 2:
        mol._ecpbas = np.zeros((0, gto.BAS_SLOTS), dtype=np.int32)
    return mol


# plain all-electron Dunning cc-pVXZ, incl. aug- and core-valence variants; anchored so
# relativistic/PP suffixes (-dk, -pp, -f12) deliberately fail
_CCPVXZ = re.compile(r"^(daug|aug)?ccp(w?c)?v[dtq56]z$")

# relativistically recontracted basis families: Douglas-Kroll, DKH, X2C, ANO-RCC, ZORA, Dyall
_RELATIVISTIC = re.compile(r"(dk\d?$|dkh\d?|^x2c|anorcc|^zora|^dyall)")


def _is_ccpvxz_family(basis):
    """True only if every element uses a cc-pVXZ-family basis name."""

    def is_cc(name):
        if not isinstance(name, str):
            return False
        return bool(_CCPVXZ.match(re.sub(r"[\s\-_]", "", name.lower())))

    if isinstance(basis, str):
        return is_cc(basis)
    if isinstance(basis, dict):
        return bool(basis) and all(is_cc(v) for v in basis.values())
    return False


def _is_relativistic_basis(basis):
    """True if any element uses a relativistically recontracted basis."""

    def is_rel(name):
        if not isinstance(name, str):
            return False
        return bool(_RELATIVISTIC.search(re.sub(r"[\s\-_]", "", name.lower())))

    if isinstance(basis, str):
        return is_rel(basis)
    if isinstance(basis, dict):
        return any(is_rel(v) for v in basis.values())
    return False


def free_atom_minao(mol, occ_tol=1e-6, sv_tol=1e-8, x2c=None):
    """
    Free-atom occupied HF orbitals (core+valence) in the uncontracted working basis,
    packaged as a pyscf basis dict for use as `minao`.

    x2c : None -> enable scalar relativity iff the working basis is a relativistically
          recontracted one; True/False to force it on or off.
    """
    _fix_ecpbas(mol)
    if x2c is None:
        x2c = _is_relativistic_basis(mol.basis)

    elems = set(mol.elements)
    unc = {sym: gto.uncontract(mol._basis[sym]) for sym in elems}

    # An ECP basis has no core functions, so the free atom must carry the same
    # pseudopotential as the molecule
    ecp = getattr(mol, "_ecp", None) or {}
    first_ia = {}
    for ia, sym in enumerate(mol.elements):
        first_ia.setdefault(sym, ia)

    ref_basis = {}
    for sym in elems:
        atom_ecp = {sym: ecp[sym]} if sym in ecp else {}
        nelec = gto.charge(sym) - mol.atom_nelec_core(first_ia[sym])
        a1 = gto.M(
            atom=f"{sym} 0 0 0",
            basis={sym: unc[sym]},
            ecp=atom_ecp,
            spin=nelec % 2,
            verbose=0,
        )
        if a1.nelectron != nelec:
            raise RuntimeError(
                f"free-atom {sym} has {a1.nelectron} electrons but the molecule leaves it "
                f"{nelec}: the ECP did not carry over. Check that mol._ecp is keyed by "
                f"element symbol (found {list(ecp)})."
            )
        ao_loc = a1.ao_loc_nr()
        shells_by_l = defaultdict(list)  # l -> [(exp, ao_start), ...]
        for ib in range(a1.nbas):
            shells_by_l[a1.bas_angular(ib)].append((float(a1.bas_exp(ib)[0]), ao_loc[ib]))

        amf: Any = atom_hf.AtomHF1e(a1) if a1.nelectron == 1 else atom_hf.AtomSphAverageRHF(a1)
        if x2c:
            amf = amf.sfx2c1e()
        amf.run()

        c, occ = amf.mo_coeff, amf.mo_occ
        occ_cols = c[:, occ > occ_tol]

        shells = []
        for l in sorted(shells_by_l):
            exps = np.array([e for e, _ in shells_by_l[l]])
            starts = [s for _, s in shells_by_l[l]]
            p, ncomp = len(exps), 2 * l + 1
            a = np.stack([occ_cols[s : s + ncomp, :] for s in starts])  # (p, ncomp, nocc)
            rrad = a.reshape(p, -1)
            u, sv, _ = np.linalg.svd(rrad, full_matrices=False)
            for i in np.where(sv > sv_tol * sv.max())[0]:  # distinct radial functions
                d = u[:, i]
                shells.append([l] + [[float(exps[k]), float(d[k])] for k in range(p)])
        ref_basis[sym] = shells
    return ref_basis


def name_fragments(frag_atmlist, elems, sep=""):
    """One label per fragment: each atom tagged with its element symbol and index."""
    return [sep.join(f"{elems[i]}{i}" for i in frag) for frag in frag_atmlist]


def _localize(mol, lo_coeff, s1e, more_loc):
    if more_loc is None:
        return lo_coeff
    if more_loc == "boys":
        lo_coeff = lo.Boys(mol, lo_coeff).kernel()
    elif more_loc == "pm":
        lo_coeff = lo.PM(mol, lo_coeff).kernel()
    else:
        raise ValueError(f"Unsupported lo type {more_loc!r}")
    return lo.orth.vec_lowdin(lo_coeff, s1e)


def _assert_orthonormal(lo_coeff, s1e):
    ortho = lo_coeff.conj().T @ s1e @ lo_coeff
    dev = np.abs(ortho - np.eye(ortho.shape[1])).max()
    assert dev < 1e-8, f"IAOs not orthonormal: max dev {dev:.2e}"


def riao_fragment(mf, nfrozen, frag_type="atom", more_loc=None, minao: Any = "minao"):
    mol = mf.mol
    s1e = mf.get_ovlp()
    moliao = lo.iao.reference_mol(mol, minao)
    nocc = np.count_nonzero(mf.mo_occ)
    orbocc = mf.mo_coeff[:, nfrozen:nocc]
    lo_coeff = lo.iao.iao(mol, orbocc, minao=minao)
    lo_coeff = lo.orth.vec_lowdin(lo_coeff, s1e)

    if frag_type == "atom":
        frag_atmlist = lno_tools.autofrag_atom(moliao, H2heavy=False)
    elif frag_type == "h2heavy":
        frag_atmlist = lno_tools.autofrag_atom(moliao, H2heavy=True)
    else:
        raise ValueError(f"Unsupported fragment type {frag_type!r}")

    if more_loc is None:
        frag_list = lno_tools.autofrag_iao(moliao, "atom", frag_atmlist)
    else:
        lo_coeff = _localize(mol, lo_coeff, s1e, more_loc)
        frag_list = lno_tools.map_lo_to_frag(mol, lo_coeff, frag_atmlist)

    frag_name = name_fragments(frag_atmlist, moliao.elements, sep="")
    _assert_orthonormal(lo_coeff, s1e)
    return lo_coeff, frag_list, frag_name


def uiao_fragment(mf, nfrozen, frag_type="atom", more_loc=None, minao: Any = "minao"):
    mol = mf.mol
    s1e = mf.get_ovlp()
    moliao = lo.iao.reference_mol(mol, minao)
    lo_coeff = []
    for s in range(2):
        nocc = np.count_nonzero(mf.mo_occ[s])
        orbocc = mf.mo_coeff[s][:, nfrozen:nocc]
        c = lo.iao.iao(mol, orbocc, minao)
        lo_coeff.append(lo.orth.vec_lowdin(c, s1e))

    if frag_type == "atom":
        frag_atmlist = lno_tools.autofrag_atom(moliao, H2heavy=False)
    elif frag_type == "h2heavy":
        frag_atmlist = lno_tools.autofrag_atom(moliao, H2heavy=True)
    else:
        raise ValueError(f"Unsupported fragment type {frag_type!r}")

    if more_loc is None:
        frag_list = lno_tools.autofrag_iao(moliao, "atom", frag_atmlist)
        frag_list = [[i, i] for i in frag_list]
    else:
        lo_coeff = [_localize(mol, c, s1e, more_loc) for c in lo_coeff]
        frag_list = lno_tools.map_lo_to_frag(mol, lo_coeff, frag_atmlist)

    frag_name = name_fragments(frag_atmlist, moliao.elements, sep="")
    for c in lo_coeff:
        _assert_orthonormal(c, s1e)
    return lo_coeff, frag_list, frag_name


def iao_fragment(
    mf,
    nfrozen=None,
    frag_type="h2heavy",
    more_loc=None,
    minao: Any = "minao",
    x2c=None,
    save2=None,
    read_from=None,
):
    """
    Build (or reuse) the IAO fragment input for an LNO calculation.

    nfrozen   : number of frozen core orbitals (default: chemcore)
    frag_type : "atom" (one fragment per atom) or "h2heavy" (hydrogens join their heavy atom)
    more_loc  : None, "boys" or "pm": further localize the IAOs before mapping to fragments
    save2     : path of an HDF5 file to write (lo_coeff, frag_list, frag_name) to
    read_from : path of such a file to read instead of rebuilding the IAOs; the stored
                molecule must match mf.mol
    """
    mol = mf.mol
    want = "unrestricted" if isinstance(mf, scf.uhf.UHF) else "restricted"

    if read_from is not None:
        if not os.path.isfile(read_from):
            raise FileNotFoundError(
                f"IAO fragment file {read_from} not found. Run the calculation once with "
                "save2=<file> to create it."
            )
        print(f"Reading IAO fragments from {read_from}")
        lo_coeff, frag_list, frag_name, meta = load_iao_fragment(
            read_from, mol=mol, s1e=mf.get_ovlp()
        )
        kind = _iao_spin_kind(lo_coeff)
        if kind != want:
            raise ValueError(
                f"{read_from} holds {kind} IAOs but {type(mf).__name__} needs {want} ones"
            )
        for key, val in (("frag_type", frag_type), ("more_loc", more_loc)):
            if key in meta and meta[key] != str(val):
                print(
                    f"Warning: {read_from} was built with {key}={meta[key]}, but {key}={val} "
                    "was requested. Using the stored fragments."
                )
        print(f"Loaded {len(frag_name)} IAO fragments: {frag_name}")
        if save2 is not None and os.path.abspath(save2) != os.path.abspath(read_from):
            save_iao_fragment(save2, lo_coeff, frag_list, frag_name, mol=mol, meta=meta)
            print(f"IAO fragments copied to {save2}")
        return lo_coeff, frag_list, frag_name

    if nfrozen is None:
        nfrozen = elements.chemcore(mol)

    if not _is_ccpvxz_family(mol.basis):
        print(
            "Detected basis set not in the cc-pVXZ family. "
            "Run free atom scf to generate reference basis."
        )
        if x2c is None:
            x2c = _is_relativistic_basis(mol.basis)
        if x2c:
            print(
                "Detected relativistic basis set. "
                "Run free atom scf with scalar relativistic (sfX2C1e) effects."
            )
        minao = free_atom_minao(mol, occ_tol=1e-6, sv_tol=1e-8, x2c=x2c)

    if isinstance(mf, scf.uhf.UHF):
        lo_coeff, frag_list, frag_name = uiao_fragment(mf, nfrozen, frag_type, more_loc, minao)
    elif isinstance(mf, scf.rhf.RHF):
        lo_coeff, frag_list, frag_name = riao_fragment(mf, nfrozen, frag_type, more_loc, minao)
    else:
        raise TypeError(f"Unsupported mf type {type(mf)}")

    if save2 is not None:
        meta = {
            "frag_type": frag_type,
            "more_loc": more_loc,
            "nfrozen": nfrozen,
            "minao": minao if isinstance(minao, str) else "free_atom_minao",
        }
        save_iao_fragment(save2, lo_coeff, frag_list, frag_name, mol=mol, meta=meta)
        print(f"IAO fragments saved to {save2}")

    return lo_coeff, frag_list, frag_name


# --------------------------------------------------------------------------- file IO

IAO_FILE_VERSION = 1


def _iao_spin_kind(lo_coeff):
    """'restricted' for a single (nao, nlo) array, 'unrestricted' for a pair."""
    if isinstance(lo_coeff, np.ndarray) and lo_coeff.ndim == 2:
        return "restricted"
    if (
        isinstance(lo_coeff, (list, tuple))
        and len(lo_coeff) == 2
        and all(isinstance(c, np.ndarray) and c.ndim == 2 for c in lo_coeff)
    ):
        return "unrestricted"
    raise TypeError(
        "lo_coeff must be a 2D array (restricted) or a pair of 2D arrays (unrestricted), "
        f"got {type(lo_coeff)}"
    )


def save_iao_fragment(filename, lo_coeff, frag_list, frag_name, mol=None, meta=None):
    """
    Dump the output of iao_fragment to an HDF5 file, with a fingerprint of the molecule so
    load_iao_fragment can refuse to hand the IAOs to a different one.
    """
    kind = _iao_spin_kind(lo_coeff)
    if len(frag_list) != len(frag_name):
        raise ValueError(
            f"frag_list ({len(frag_list)}) and frag_name ({len(frag_name)}) have different lengths"
        )

    frag0 = frag_list[0] if kind == "restricted" else frag_list[0][0]
    idx_kind = "array" if isinstance(frag0, np.ndarray) else "list"

    with h5py.File(filename, "w") as fh5:
        fh5.attrs["version"] = IAO_FILE_VERSION
        fh5.attrs["spin"] = kind
        if kind == "restricted":
            fh5["lo_coeff"] = np.asarray(lo_coeff)
        else:
            fh5["lo_coeff_a"] = np.asarray(lo_coeff[0])
            fh5["lo_coeff_b"] = np.asarray(lo_coeff[1])
        fh5["frag_name"] = np.array(list(frag_name), dtype=h5py.string_dtype())

        grp = fh5.create_group("frag_list")
        grp.attrs["nfrag"] = len(frag_list)
        grp.attrs["idx_kind"] = idx_kind
        for i, frag in enumerate(frag_list):
            if kind == "restricted":
                grp[f"{i}"] = np.asarray(frag, dtype=np.int64)
            else:
                grp[f"{i}_a"] = np.asarray(frag[0], dtype=np.int64)
                grp[f"{i}_b"] = np.asarray(frag[1], dtype=np.int64)

        if mol is not None:
            mgrp = fh5.create_group("mol")
            mgrp["atom_charges"] = np.asarray(mol.atom_charges())
            mgrp["atom_coords"] = np.asarray(mol.atom_coords())  # bohr
            mgrp.attrs["natm"] = mol.natm
            mgrp.attrs["nao"] = mol.nao
            mgrp.attrs["basis"] = str(mol.basis)[:1024]
            mgrp.attrs["charge"] = mol.charge
            mgrp.attrs["spin"] = mol.spin

        mtgrp = fh5.create_group("meta")
        for key, val in (meta or {}).items():
            mtgrp.attrs[key] = str(val)[:1024]

    return filename


def load_iao_fragment(filename, mol=None, s1e=None, coord_tol=1e-6, ortho_tol=1e-8):
    """
    Read back what save_iao_fragment wrote.

    mol : if given, the stored molecular fingerprint must match it.
    s1e : if given, the loaded IAOs must be orthonormal w.r.t. it.
    """
    lo_coeff: Any
    with h5py.File(filename, "r") as h5file:
        fh5: Any = h5file  # h5py's item types are unions pyright cannot narrow
        version = int(fh5.attrs.get("version", -1))
        if version != IAO_FILE_VERSION:
            raise ValueError(
                f"{filename}: IAO file version {version} is not readable by this code "
                f"(expected {IAO_FILE_VERSION})"
            )
        kind = fh5.attrs["spin"]
        if kind == "restricted":
            lo_coeff = np.asarray(fh5["lo_coeff"][()])
            nao = lo_coeff.shape[0]
        else:
            lo_coeff = [np.asarray(fh5["lo_coeff_a"][()]), np.asarray(fh5["lo_coeff_b"][()])]
            nao = lo_coeff[0].shape[0]

        frag_name = [n.decode() if isinstance(n, bytes) else str(n) for n in fh5["frag_name"][()]]

        grp = fh5["frag_list"]
        nfrag = int(grp.attrs["nfrag"])
        as_list = grp.attrs.get("idx_kind", "array") == "list"

        def _idx(dset):
            idx = np.asarray(dset[()], dtype=np.int64)
            return idx.tolist() if as_list else idx

        if kind == "restricted":
            frag_list = [_idx(grp[f"{i}"]) for i in range(nfrag)]
        else:
            frag_list = [[_idx(grp[f"{i}_a"]), _idx(grp[f"{i}_b"])] for i in range(nfrag)]

        stored_mol = dict(fh5["mol"].attrs) if "mol" in fh5 else None
        if stored_mol is not None:
            stored_mol["atom_charges"] = np.asarray(fh5["mol/atom_charges"][()])
            stored_mol["atom_coords"] = np.asarray(fh5["mol/atom_coords"][()])
        meta = dict(fh5["meta"].attrs) if "meta" in fh5 else {}

    if len(frag_name) != nfrag:
        raise ValueError(f"{filename}: {nfrag} fragments but {len(frag_name)} names")

    if mol is not None:
        if mol.nao != nao:
            raise ValueError(
                f"{filename}: IAOs were built in a basis with {nao} AOs but the current mol "
                f"has {mol.nao}"
            )
        if stored_mol is None:
            print(
                f"Warning: {filename} carries no molecular fingerprint; cannot verify that it "
                "belongs to this molecule."
            )
        else:
            if not np.array_equal(stored_mol["atom_charges"], mol.atom_charges()):
                raise ValueError(f"{filename}: nuclear charges differ from the current molecule")
            dev = np.abs(stored_mol["atom_coords"] - mol.atom_coords()).max()
            if dev > coord_tol:
                raise ValueError(
                    f"{filename}: geometry differs from the current molecule (max deviation "
                    f"{dev:.2e} bohr > {coord_tol:.1e})"
                )

    if s1e is not None:
        for c in [lo_coeff] if kind == "restricted" else lo_coeff:
            ortho = c.conj().T @ s1e @ c
            dev = np.abs(ortho - np.eye(ortho.shape[1])).max()
            if dev > ortho_tol:
                raise ValueError(
                    f"{filename}: loaded IAOs are not orthonormal w.r.t. the current overlap "
                    f"(max dev {dev:.2e}). They most likely belong to another molecule or "
                    "another basis set."
                )

    return lo_coeff, frag_list, frag_name, meta
