from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Union

import h5py
import numpy as np
from numpy.typing import NDArray
from pyscf import lib

from ..staging import StagedInputs, TrialInput
from ..staging import _load_h5 as _load_staged_h5
from ..staging import dump as dump_staged
from ..staging_u import dump_uh, is_uchol_file, load_uh

# The handoff between the CPU half of a fragment (LNO + MP2 + CCSD) and its AFQMC half,
# and the fragment trial staging. Mirrors trot/staging.py in role: plain numpy in, a
# TrialInput of arrays out, nothing jax.
#
#   LnoFragData          what cpu_stage produces for one fragment
#   frag_mf              the mean field in the fragment's LNO basis, for trot's guide staging
#   stage_pt2ccsd_trial  LnoFragData -> TrialInput {mo_t, t2 (projected), prjlo, t1}
#   stage_pt2ccsd_fast_trial   the same with the projector factored: {mo_t, t2x, u, t1}
#   stage_upt2ccsd_trial the unrestricted counterpart
#   stage_upt2ccsd_fast_trial  the unrestricted counterpart with the projectors factored
#   projected_doubles    the doubles contracted with U on the first occupied index
#   dump_frag / load_frag  the self-contained frag{i}.h5 a fragment can be re-run from


def _as_frozen_array(frozen: Any) -> NDArray:
    """make_las returns 0 when nothing is frozen; keep a (possibly empty) int array."""
    if isinstance(frozen, (int, np.integer)):
        return np.zeros((0,), dtype=np.int64) if int(frozen) == 0 else np.arange(int(frozen))
    return np.asarray(frozen, dtype=np.int64).reshape(-1)


@dataclass(frozen=True)
class LnoFragData:
    """
    One fragment's LNO data, as produced by pipeline.cpu_stage.

    Restricted: lno_coeff (nao, nmo), lno_frozen (nfrozen,), uocc_loc (nactocc, nlo),
    t1 (nactocc, nactvir), t2 (nactocc, nactocc, nactvir, nactvir) in pyscf's (i,j,a,b)
    layout, all in the LNO basis. Unrestricted: each is a pair (alpha, beta), and t2 is
    (t2aa, t2ab, t2bb).

    t2u holds the doubles contracted with U = uocc_loc on their first occupied index,
    t2u_{Iajb} = sum_i t2_{iajb} U_{iI} in the (I, a, j, b) layout (the four blocks
    t2aa_u, t2ab_u, t2ba_u, t2bb_u for an unrestricted fragment; see
    projected_doubles). It is what the pt2CCSD trials need and nlo/nocc the size of t2,
    so a fragment file written for a fast trial carries t2u instead of t2. Either one
    makes has_amplitudes true; the CISD guides need t2.

    lno_coeff is ordered [frz_occ | act_occ | act_vir | frz_vir], so the core is the
    leading columns and the active block is contiguous; integral.py relies on that.
    """

    frag_idx: int
    frag_name: str
    lno_coeff: Any
    lno_frozen: Any
    uocc_loc: Any
    nactocc: Any
    nactvir: Any
    t1: Any = None
    t2: Any = None
    t2u: Any = None
    efrag_mp: float = 0.0
    efrag_cc: float = 0.0
    lno_thresh: tuple = (None, None)
    nfrozen: int = 0
    t_las: float = 0.0
    t_mp: float = 0.0
    t_cc: float = 0.0
    t_cpu: float = 0.0
    log: str = field(default="", repr=False)

    @property
    def unrestricted(self) -> bool:
        return isinstance(self.lno_coeff, (tuple, list))

    @property
    def has_amplitudes(self) -> bool:
        return self.t1 is not None and (self.t2 is not None or self.t2u is not None)

    @property
    def has_full_amplitudes(self) -> bool:
        return self.t1 is not None and self.t2 is not None

    @property
    def nact(self):
        """Active-space size, an int or an (alpha, beta) pair."""
        if self.unrestricted:
            return (
                int(self.nactocc[0]) + int(self.nactvir[0]),
                int(self.nactocc[1]) + int(self.nactvir[1]),
            )
        return int(self.nactocc) + int(self.nactvir)

    @staticmethod
    def _split_frozen(frozen: Any) -> tuple[int, int]:
        # the LAS ordering [frz_occ | act_occ | act_vir | frz_vir] puts the frozen occupied
        # at the leading indices 0, 1, ..., so they are the initial run of the sorted list
        idx = np.sort(_as_frozen_array(frozen))
        nfrzocc = int(np.sum(idx == np.arange(idx.size)))
        return nfrzocc, int(idx.size - nfrzocc)

    @property
    def nfrzocc(self):
        """Frozen occupied count, an int or an (alpha, beta) pair."""
        if self.unrestricted:
            return tuple(self._split_frozen(f)[0] for f in self.lno_frozen)
        return self._split_frozen(self.lno_frozen)[0]

    @property
    def nfrzvir(self):
        """Frozen virtual count, an int or an (alpha, beta) pair."""
        if self.unrestricted:
            return tuple(self._split_frozen(f)[1] for f in self.lno_frozen)
        return self._split_frozen(self.lno_frozen)[1]


def frag_mf(mf: Any, frag: LnoFragData) -> Any:
    """
    A shallow copy of mf whose orbitals are the fragment's LNOs.

    trot.staging.stage builds the guide from mf.mo_coeff, so handing it this copy (and
    the frozen LNO indices) stages the guide in the fragment basis: for HF that is the
    identity on the active LNOs, with the reference in the leading columns. mo_occ is
    left as it is, since the LAS ordering keeps occupied before virtual.
    """
    mf2 = copy.copy(mf)
    if frag.unrestricted:
        mf2.mo_coeff = np.array([np.asarray(frag.lno_coeff[0]), np.asarray(frag.lno_coeff[1])])
    else:
        mf2.mo_coeff = np.asarray(frag.lno_coeff)
    return mf2


def _thouless(t1: NDArray) -> NDArray:
    # |psi'> = exp(t1_ia a+_a a_i)|psi>, |psi> the leading nocc orbitals; the generator is
    # nilpotent so exp(X) = 1 + X exactly. Returns the mo_coeff of psi' in the MO basis.
    nocc, nvir = t1.shape
    exp_t1 = np.eye(nocc + nvir, dtype=np.float64)
    exp_t1[:nocc, nocc:] = t1
    return exp_t1.T[:, :nocc]


def projected_doubles(frag: LnoFragData) -> Any:
    """
    The doubles contracted with U = uocc_loc on their first occupied index, in the
    (I, a, j, b) layout: t2u for a restricted fragment, (t2aa_u, t2ab_u, t2ba_u, t2bb_u)
    for an unrestricted one (same-spin blocks antisymmetrized in (a, b), t2ba the
    transpose of t2ab projected on its beta index). frag.t2u when it is there, else built
    from frag.t2.
    """
    if frag.t2u is not None:
        return frag.t2u
    if frag.t2 is None:
        raise ValueError("the fragment data carries no doubles (run_cc=True?).")
    if not frag.unrestricted:
        t2 = np.asarray(frag.t2, dtype=np.float64).transpose(0, 2, 1, 3)  # (i,j,a,b) -> (i,a,j,b)
        u = np.asarray(frag.uocc_loc)
        return lib.einsum("iajb,iI->Iajb", t2, u, optimize="optimal")
    t2aa, t2ab, t2bb = (np.asarray(t, dtype=np.float64) for t in frag.t2)
    t2aa = 0.5 * (t2aa - t2aa.transpose(0, 1, 3, 2))
    t2bb = 0.5 * (t2bb - t2bb.transpose(0, 1, 3, 2))
    t2aa = t2aa.transpose(0, 2, 1, 3)
    t2ab = t2ab.transpose(0, 2, 1, 3)
    t2bb = t2bb.transpose(0, 2, 1, 3)
    ua, ub = (np.asarray(u) for u in frag.uocc_loc)
    return (
        lib.einsum("iajb,iI->Iajb", t2aa, ua, optimize="optimal"),
        lib.einsum("iajb,iI->Iajb", t2ab, ua, optimize="optimal"),
        lib.einsum("jbia,iI->Iajb", t2ab, ub, optimize="optimal"),
        lib.einsum("iajb,iI->Iajb", t2bb, ub, optimize="optimal"),
    )


def stage_pt2ccsd_trial(frag: LnoFragData) -> TrialInput:
    """
    The fragment pt2CCSD trial.

    As trot's stage_pt2ccsd_trial, plus the fragment projector prjlo = U U^T with
    U = <act_occ|lo>, and t2 projected on its first occupied index,
    t2_kajb = sum_i t2_iajb prjlo_ik (afqmc's prep.proj_cc_amplitude). t1 is kept
    unprojected: the meas ctx needs it for e0t1orb.
    """
    if frag.unrestricted:
        raise ValueError(
            "stage_pt2ccsd_trial needs restricted fragment data; use stage_upt2ccsd_trial."
        )
    if not frag.has_amplitudes:
        raise ValueError("the pt2CCSD trial needs the fragment CCSD amplitudes (run_cc=True).")

    t1 = np.asarray(frag.t1, dtype=np.float64)
    uocc = np.asarray(frag.uocc_loc)
    prjlo = uocc @ uocc.T.conj()
    # t2_kajb = sum_i t2_iajb prjlo_ik = sum_I t2u_Iajb U*_kI
    t2 = lib.einsum("Iajb,kI->kajb", projected_doubles(frag), uocc.conj(), optimize="optimal")

    data = {"mo_t": _thouless(t1), "t2": t2, "prjlo": prjlo, "t1": t1}
    return TrialInput(
        kind="pt2ccsd", data=data, frozen=_as_frozen_array(frag.lno_frozen), source_kind="mf"
    )


def stage_pt2ccsd_fast_trial(frag: LnoFragData) -> TrialInput:
    """
    The fragment pt2CCSD trial with the projector in factored form (trial/pt2ccsd_fast.py):
    the doubles contracted with U = <act_occ|lo> on their first occupied index,
    t2u_Iajb = sum_i t2_iajb U_iI, stored with the exchange folded in,
    t2x_Iajb = 2 t2u_Iajb - t2u_Ibja, and U itself; the kernel applies the second factor.
    """
    if frag.unrestricted:
        raise ValueError("stage_pt2ccsd_fast_trial needs restricted fragment data.")
    if not frag.has_amplitudes:
        raise ValueError("the pt2CCSD trial needs the fragment CCSD amplitudes (run_cc=True).")

    t1 = np.asarray(frag.t1, dtype=np.float64)
    u = np.asarray(frag.uocc_loc)
    t2u = np.asarray(projected_doubles(frag), dtype=np.float64)
    t2x = 2.0 * t2u - t2u.transpose(0, 3, 2, 1)

    data = {"mo_t": _thouless(t1), "t2x": t2x, "u": u, "t1": t1}
    return TrialInput(
        kind="pt2ccsd_fast", data=data, frozen=_as_frozen_array(frag.lno_frozen), source_kind="mf"
    )


def stage_cisd_guide(frag: LnoFragData) -> TrialInput:
    """
    The fragment CISD as a guide: trot's _stage_cisd_input on the fragment CCSD
    amplitudes, unprojected. ci1 = t1, ci2 = t2 + t1 t1 in (i, a, j, b), with the
    reference in the leading nactocc orbitals of the active LNO basis, which is where
    build_ham_lno_df puts it. The frozen LNOs are already out of the hamiltonian, so
    there is no trial core or outer block.
    """
    if frag.unrestricted:
        raise ValueError("stage_cisd_guide needs restricted fragment data; use stage_ucisd_guide.")
    if not frag.has_full_amplitudes:
        raise ValueError(
            "the CISD guide needs the full fragment CCSD amplitudes (run_cc=True); a fragment "
            "file written for a fast trial carries only the projected doubles, so re-run "
            "such a file with the guide it was written with, or write it with the CISD guide."
        )
    t1 = np.asarray(frag.t1, dtype=np.float64)
    t2 = np.asarray(frag.t2, dtype=np.float64)
    ci2 = (t2 + lib.einsum("ia,jb->ijab", t1, t1)).transpose(0, 2, 1, 3)
    data = {
        "ci1": t1,
        "ci2": ci2,
        "nocc_t_core": np.array(0, dtype=np.int64),
        "nvir_t_outer": np.array(0, dtype=np.int64),
    }
    return TrialInput(
        kind="cisd", data=data, frozen=_as_frozen_array(frag.lno_frozen), source_kind="cc"
    )


def stage_uhf_guide(frag: LnoFragData) -> TrialInput:
    """
    The UHF guide on the uchol fragment hamiltonian, in the layout of staging_u's
    _uhf_trial_input: each spin's reference determinant is the leading nactocc columns of
    the identity in that spin's active LNO basis (build_ham_ulno_df puts the active
    occupied LNOs first). The branch's mean-field staging cannot be used here, since it
    expresses both spins in the alpha basis.
    """
    if not frag.unrestricted:
        raise ValueError("stage_uhf_guide needs unrestricted fragment data.")
    nocc_a, nocc_b = (int(n) for n in frag.nactocc)
    nvir_a, nvir_b = (int(n) for n in frag.nactvir)
    data = {
        "mo_a": np.eye(nocc_a + nvir_a)[:, :nocc_a],
        "mo_b": np.eye(nocc_b + nvir_b)[:, :nocc_b],
    }
    frozen = np.concatenate([_as_frozen_array(f) for f in frag.lno_frozen])
    return TrialInput(kind="uhf", data=data, frozen=frozen, source_kind="mf")


def stage_ucisd_guide(frag: LnoFragData) -> TrialInput:
    """
    The fragment UCISD as a guide, in the conventions of trot's _stage_ucisd_input: same
    spin doubles antisymmetrized, everything (i, a, j, b), each spin in its own LNO
    basis with the reference in its leading nactocc orbitals (build_ham_ulno_df's
    layout). mo_coeff_s is that identity reference; the uchol kernels never rotate.
    """
    if not frag.unrestricted:
        raise ValueError("stage_ucisd_guide needs unrestricted fragment data.")
    if not frag.has_full_amplitudes:
        raise ValueError(
            "the UCISD guide needs the full fragment CCSD amplitudes (run_cc=True); a fragment "
            "file written for a fast trial carries only the projected doubles, so re-run "
            "such a file with the guide it was written with, or write it with the UCISD guide."
        )
    t1a, t1b = (np.asarray(t, dtype=np.float64) for t in frag.t1)
    t2aa, t2ab, t2bb = (np.asarray(t, dtype=np.float64) for t in frag.t2)

    ci2aa = t2aa + 2.0 * lib.einsum("ia,jb->ijab", t1a, t1a)
    ci2aa = 0.5 * (ci2aa - ci2aa.transpose(0, 1, 3, 2))
    ci2bb = t2bb + 2.0 * lib.einsum("ia,jb->ijab", t1b, t1b)
    ci2bb = 0.5 * (ci2bb - ci2bb.transpose(0, 1, 3, 2))
    ci2ab = t2ab + lib.einsum("ia,jb->ijab", t1a, t1b)

    nocc_a, nvir_a = t1a.shape
    nocc_b, nvir_b = t1b.shape
    data = {
        "mo_coeff_a": np.eye(nocc_a + nvir_a)[:, :nocc_a],
        "mo_coeff_b": np.eye(nocc_b + nvir_b)[:, :nocc_b],
        "ci1a": t1a,
        "ci1b": t1b,
        "ci2aa": ci2aa.transpose(0, 2, 1, 3),
        "ci2ab": ci2ab.transpose(0, 2, 1, 3),
        "ci2bb": ci2bb.transpose(0, 2, 1, 3),
    }
    frozen = np.concatenate([_as_frozen_array(f) for f in frag.lno_frozen])
    return TrialInput(kind="ucisd", data=data, frozen=frozen, source_kind="cc")


def stage_upt2ccsd_trial(frag: LnoFragData) -> TrialInput:
    """
    The unrestricted fragment pt2CCSD trial, in the conventions of trot's
    stage_upt2ccsd_trial_uh: same-spin blocks antisymmetrized, everything (i,a,j,b), each
    spin in its own LNO basis. The projection on the first index makes t2ab and t2ba
    distinct, so both are staged.
    """
    if not frag.unrestricted:
        raise ValueError("stage_upt2ccsd_trial needs unrestricted fragment data.")
    if not frag.has_amplitudes:
        raise ValueError("the pt2CCSD trial needs the fragment CCSD amplitudes (run_cc=True).")

    t1a, t1b = (np.asarray(t, dtype=np.float64) for t in frag.t1)
    t2aa_u, t2ab_u, t2ba_u, t2bb_u = (
        np.asarray(t, dtype=np.float64) for t in projected_doubles(frag)
    )
    ua, ub = (np.asarray(u) for u in frag.uocc_loc)
    prjlo_a = ua @ ua.T.conj()
    prjlo_b = ub @ ub.T.conj()

    # t2_kajb = sum_i t2_iajb prjlo_ik = sum_I t2u_Iajb U*_kI, per block
    data = {
        "mo_t_a": _thouless(t1a),
        "mo_t_b": _thouless(t1b),
        "t2aa": lib.einsum("Iajb,kI->kajb", t2aa_u, ua.conj(), optimize="optimal"),
        "t2ab": lib.einsum("Iajb,kI->kajb", t2ab_u, ua.conj(), optimize="optimal"),
        "t2ba": lib.einsum("Iajb,kI->kajb", t2ba_u, ub.conj(), optimize="optimal"),
        "t2bb": lib.einsum("Iajb,kI->kajb", t2bb_u, ub.conj(), optimize="optimal"),
        "prjlo_a": prjlo_a,
        "prjlo_b": prjlo_b,
        "t1a": t1a,
        "t1b": t1b,
    }
    frozen = tuple(_as_frozen_array(f) for f in frag.lno_frozen)
    return TrialInput(kind="upt2ccsd", data=data, frozen=frozen[0], source_kind="mf")


def stage_upt2ccsd_fast_trial(frag: LnoFragData) -> TrialInput:
    """
    The unrestricted fragment pt2CCSD trial with the projectors in factored form
    (trial/upt2ccsd_fast.py): the conventions of stage_upt2ccsd_trial, the doubles
    contracted with U_s = <act_occ_s|lo_s> on their first occupied index instead of
    being projected with U_s U_s^H.
    """
    if not frag.unrestricted:
        raise ValueError("stage_upt2ccsd_fast_trial needs unrestricted fragment data.")
    if not frag.has_amplitudes:
        raise ValueError("the pt2CCSD trial needs the fragment CCSD amplitudes (run_cc=True).")

    t1a, t1b = (np.asarray(t, dtype=np.float64) for t in frag.t1)
    t2aa_u, t2ab_u, t2ba_u, t2bb_u = (
        np.asarray(t, dtype=np.float64) for t in projected_doubles(frag)
    )
    ua, ub = (np.asarray(u) for u in frag.uocc_loc)

    data = {
        "mo_t_a": _thouless(t1a),
        "mo_t_b": _thouless(t1b),
        "t2aa_u": t2aa_u,
        "t2ab_u": t2ab_u,
        "t2ba_u": t2ba_u,
        "t2bb_u": t2bb_u,
        "u_a": ua,
        "u_b": ub,
        "t1a": t1a,
        "t1b": t1b,
    }
    frozen = tuple(_as_frozen_array(f) for f in frag.lno_frozen)
    return TrialInput(kind="upt2ccsd_fast", data=data, frozen=frozen[0], source_kind="mf")


# --------------------------------------------------------------------------- frag{i}.h5

FRAG_FILE_VERSION = 2  # 2: the doubles may be stored projected (amplitudes/t2u or the four blocks)
_FRAG_FILE_VERSIONS = (1, 2)


def _fingerprint(mf: Any) -> dict[str, Any]:
    mol = mf.mol
    with_df = getattr(mf, "with_df", None)
    return {
        "atom_charges": [int(c) for c in mol.atom_charges()],
        "atom_coords": np.asarray(mol.atom_coords()).tolist(),
        "basis": str(mol.basis)[:1024],
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "nao": int(mol.nao),
        "naux": int(with_df.get_naoaux()) if with_df is not None else None,
        "e_tot": float(mf.e_tot),
    }


def _write_pair_or_array(group: h5py.Group, name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (tuple, list)):
        for s, tag in enumerate(("a", "b")):
            group.create_dataset(f"{name}_{tag}", data=np.asarray(value[s]))
    else:
        group.create_dataset(name, data=np.asarray(value))


def _read_pair_or_array(group: h5py.Group, name: str) -> Any:
    if name in group:
        return np.array(group[name])
    if f"{name}_a" in group:
        return (np.array(group[f"{name}_a"]), np.array(group[f"{name}_b"]))
    return None


def dump_frag(
    path: Union[str, Path],
    frag: LnoFragData,
    staged: StagedInputs,
    mf: Any,
    *,
    emf: float | None = None,
    amplitudes: str = "full",
    nfrag_tot: int | None = None,
) -> Path:
    """
    Write the self-contained fragment file: the LNO data and amplitudes, the fragment
    hamiltonian and the staged guide (in trot's staged-inputs layout, so trot.staging.load
    reads the restricted ones and staging_u.load_uh the uchol ones), and a fingerprint
    of the system. A re-run needs nothing else.

    amplitudes="full" stores t2 as the CCSD produced it; "projected" stores the doubles
    contracted with U on their first occupied index instead (projected_doubles), nlo/nocc
    the size, which is all the pt2CCSD trials need (the CISD guides need the full ones).
    nfrag_tot, the number of fragments of the whole fragmentation, is stored when given so
    that LnoAfqmcMixed(frag_data=...) draws the same per fragment seeds as the loop that
    wrote the files.
    """
    if amplitudes not in ("full", "projected"):
        raise ValueError(f"amplitudes must be 'full' or 'projected', got {amplitudes!r}")
    path = Path(path)
    if getattr(staged.ham, "basis", None) == "uchol":
        dump_uh(staged, path)
    else:
        dump_staged(staged, path)  # ham/, trial/ (= guide), meta_json
    with h5py.File(path, "a") as f:
        f.attrs["frag_file_version"] = FRAG_FILE_VERSION
        f.attrs["emf"] = float(mf.e_tot if emf is None else emf)
        f.attrs["fingerprint_json"] = json.dumps(_fingerprint(mf))
        f.attrs["timestamp_unix"] = time.time()

        g = f.create_group("frag")
        g.attrs["frag_idx"] = int(frag.frag_idx)
        g.attrs["frag_name"] = str(frag.frag_name)
        g.attrs["unrestricted"] = bool(frag.unrestricted)
        g.attrs["nfrozen"] = int(frag.nfrozen)
        if nfrag_tot is not None:
            g.attrs["nfrag_tot"] = int(nfrag_tot)
        g.attrs["lno_thresh_json"] = json.dumps(
            [None if x is None else float(x) for x in frag.lno_thresh]
        )
        g.attrs["efrag_mp"] = float(frag.efrag_mp)
        g.attrs["efrag_cc"] = float(frag.efrag_cc)
        for k in ("t_las", "t_mp", "t_cc", "t_cpu"):
            g.attrs[k] = float(getattr(frag, k))
        _write_pair_or_array(g, "lno_coeff", frag.lno_coeff)
        _write_pair_or_array(g, "lno_frozen", frag.lno_frozen)
        _write_pair_or_array(g, "uocc_loc", frag.uocc_loc)
        g.create_dataset("nactocc", data=np.asarray(frag.nactocc, dtype=np.int64))
        g.create_dataset("nactvir", data=np.asarray(frag.nactvir, dtype=np.int64))

        if frag.has_amplitudes:
            ga = f.create_group("amplitudes")
            _write_pair_or_array(ga, "t1", frag.t1)
            if amplitudes == "full" and frag.t2 is None:
                raise ValueError(
                    "amplitudes='full' but the fragment data carries only the projected doubles"
                )
            if amplitudes == "projected":
                t2u = projected_doubles(frag)
                if frag.unrestricted:
                    for name, t in zip(("t2aa_u", "t2ab_u", "t2ba_u", "t2bb_u"), t2u):
                        ga.create_dataset(name, data=np.asarray(t))
                else:
                    ga.create_dataset("t2u", data=np.asarray(t2u))
            elif frag.unrestricted:
                for name, t in zip(("t2aa", "t2ab", "t2bb"), frag.t2):
                    ga.create_dataset(name, data=np.asarray(t))
            else:
                ga.create_dataset("t2", data=np.asarray(frag.t2))
    return path


def load_frag(path: Union[str, Path]) -> tuple[LnoFragData, StagedInputs, dict[str, Any]]:
    """Read back what dump_frag wrote: (frag data, staged ham + guide, file attributes)."""
    path = Path(path)
    staged = load_uh(path) if is_uchol_file(path) else _load_staged_h5(path)
    with h5py.File(path, "r") as h5file:
        f: Any = h5file  # h5py's item types are unions pyright cannot narrow
        version = int(f.attrs.get("frag_file_version", -1))
        if version not in _FRAG_FILE_VERSIONS:
            raise ValueError(
                f"{path}: fragment file version {version}, expected one of {_FRAG_FILE_VERSIONS}"
            )
        attrs = {
            "emf": float(f.attrs["emf"]),
            "fingerprint": json.loads(str(f.attrs["fingerprint_json"])),
            "timestamp_unix": float(f.attrs["timestamp_unix"]),
        }
        g = f["frag"]
        unrestricted = bool(g.attrs["unrestricted"])
        nactocc = np.array(g["nactocc"])
        nactvir = np.array(g["nactvir"])
        kw: dict[str, Any] = dict(
            frag_idx=int(g.attrs["frag_idx"]),
            frag_name=str(g.attrs["frag_name"]),
            lno_coeff=_read_pair_or_array(g, "lno_coeff"),
            lno_frozen=_read_pair_or_array(g, "lno_frozen"),
            uocc_loc=_read_pair_or_array(g, "uocc_loc"),
            nactocc=tuple(int(x) for x in nactocc) if unrestricted else int(nactocc),
            nactvir=tuple(int(x) for x in nactvir) if unrestricted else int(nactvir),
            efrag_mp=float(g.attrs["efrag_mp"]),
            efrag_cc=float(g.attrs["efrag_cc"]),
            lno_thresh=tuple(json.loads(str(g.attrs["lno_thresh_json"]))),
            nfrozen=int(g.attrs["nfrozen"]),
        )
        for k in ("t_las", "t_mp", "t_cc", "t_cpu"):
            kw[k] = float(g.attrs[k])
        if "amplitudes" in f:
            ga = f["amplitudes"]
            kw["t1"] = _read_pair_or_array(ga, "t1")
            if "t2u" in ga or "t2aa_u" in ga:
                # written for a fast trial: the projected doubles only
                if unrestricted:
                    kw["t2u"] = tuple(
                        np.array(ga[n]) for n in ("t2aa_u", "t2ab_u", "t2ba_u", "t2bb_u")
                    )
                else:
                    kw["t2u"] = np.array(ga["t2u"])
            elif unrestricted:
                kw["t2"] = tuple(np.array(ga[n]) for n in ("t2aa", "t2ab", "t2bb"))
            else:
                kw["t2"] = np.array(ga["t2"])
    return LnoFragData(**kw), staged, attrs


def frag_file_meta(path: Union[str, Path]) -> dict[str, Any]:
    """
    The metadata of a fragment file without its arrays: frag_idx, frag_name,
    unrestricted, nactocc, nactvir, nact, efrag_mp, efrag_cc, lno_thresh, nfrozen, emf,
    the fragment hamiltonian's basis and nfrag_tot when the loop that wrote the file
    stored it (else None).
    """
    path = Path(path)
    with h5py.File(path, "r") as h5file:
        f: Any = h5file
        version = int(f.attrs.get("frag_file_version", -1))
        if version not in _FRAG_FILE_VERSIONS:
            raise ValueError(
                f"{path}: fragment file version {version}, expected one of {_FRAG_FILE_VERSIONS}"
            )
        g = f["frag"]
        unrestricted = bool(g.attrs["unrestricted"])
        occ = np.atleast_1d(np.array(g["nactocc"])).astype(int)
        vir = np.atleast_1d(np.array(g["nactvir"])).astype(int)
        nactocc: Any = tuple(int(x) for x in occ) if unrestricted else int(occ[0])
        nactvir: Any = tuple(int(x) for x in vir) if unrestricted else int(vir[0])
        nact: Any = tuple(int(x) for x in occ + vir) if unrestricted else int(occ[0] + vir[0])
        return dict(
            path=path,
            frag_idx=int(g.attrs["frag_idx"]),
            frag_name=str(g.attrs["frag_name"]),
            unrestricted=unrestricted,
            nactocc=nactocc,
            nactvir=nactvir,
            nact=nact,
            efrag_mp=float(g.attrs["efrag_mp"]),
            efrag_cc=float(g.attrs["efrag_cc"]),
            lno_thresh=tuple(json.loads(str(g.attrs["lno_thresh_json"]))),
            nfrozen=int(g.attrs["nfrozen"]),
            nfrag_tot=int(g.attrs["nfrag_tot"]) if "nfrag_tot" in g.attrs else None,
            t_cpu=float(g.attrs["t_cpu"]),
            emf=float(f.attrs["emf"]),
            basis=str(f["ham"].attrs.get("basis", "restricted")),
        )


def frag_data_fields() -> tuple[str, ...]:
    return tuple(f.name for f in fields(LnoFragData))
