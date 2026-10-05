from __future__ import annotations

from functools import partial, reduce

from typing import Any

import numpy as np
from pyscf import lib, scf
from pyscf.lib import logger

try:
    from pyscf.lno import lno, ulno  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError("trot.lnoafqmc needs pyscf-forge for the LNO machinery.") from e

print = partial(print, flush=True)

# Local active space (LAS) construction and the orbital bookkeeping around it, ported
# from afqmc's lno_afqmc/tools.py. Everything here is pyscf/numpy; nothing touches jax.
#
#   make_las / make_rlas / make_ulas    LNOs of one fragment from the LNO object
#   split_lno                           [frz_occ | act_occ | act_vir | frz_vir] blocks
#   can2lno_amplitude, rot_amplitude    rotate t1/t2 between two orbital sets
#   check_span, mo_span                 do the LOs span the occupied MOs?
#
# Orbital order matters downstream: make_las returns lno_coeff as
# hstack([frz_occ, act_occ, act_vir, frz_vir]) and lno_frozen as the indices of the
# first and the last block, so the core is always the leading columns. integral.py
# relies on that and asserts it.


# --------------------------------------------------------------------------- overlaps


def mo_olp(mf: Any, mo1: Any, mo2: Any) -> Any:
    """<mo1|mo2> in the AO metric, per spin for a UHF mean field."""
    s1e = mf.get_ovlp()
    if isinstance(mf, scf.uhf.UHF):
        return [mo1[0].conj().T @ s1e @ mo2[0], mo1[1].conj().T @ s1e @ mo2[1]]
    return mo1.conj().T @ s1e @ mo2


def mo_span(mo1, s1e, mo2):
    """
    Subspace containment between mo1 and mo2, both orthonormal in the s1e metric.

    Returns (span12, span21):
      span12  max-abs residual of  span(mo2) ⊆ span(mo1)   ("mo1 spans mo2")
      span21  max-abs residual of  span(mo1) ⊆ span(mo2)   ("mo2 spans mo1")
    A small residual means the containment holds.
    """
    olp11 = mo1.T.conj() @ s1e @ mo1
    olp12 = mo1.T.conj() @ s1e @ mo2
    olp22 = mo2.T.conj() @ s1e @ mo2
    span12 = np.abs(olp12.T.conj() @ olp12 - olp22).max()
    span21 = np.abs(olp12 @ olp12.T.conj() - olp11).max()
    return span12, span21


def check_span(mf, lo_coeff_occ, frozen=0, thresh=1e-6):
    """
    The LOs have to span the occupied MOs outside the frozen core; the converse is not
    required. Raises if they do not.
    """
    s1e = mf.get_ovlp()

    if isinstance(mf, scf.uhf.UHF):
        if isinstance(frozen, int):
            frozen = (frozen, frozen)
        nocc = (np.count_nonzero(mf.mo_occ[0]), np.count_nonzero(mf.mo_occ[1]))
        mo_occ = (
            mf.mo_coeff[0][:, frozen[0] : nocc[0]],
            mf.mo_coeff[1][:, frozen[1] : nocc[1]],
        )
        pa = mo_span(lo_coeff_occ[0], s1e, mo_occ[0])
        pb = mo_span(lo_coeff_occ[1], s1e, mo_occ[1])
        p12, p21 = (pa[0], pb[0]), (pa[1], pb[1])
        span12 = p12[0] < thresh and p12[1] < thresh
        span21 = p21[0] < thresh and p21[1] < thresh
    elif isinstance(mf, scf.rhf.RHF):
        nocc = np.count_nonzero(np.asarray(mf.mo_occ))
        mo_occ = mf.mo_coeff[:, frozen:nocc]
        p12, p21 = mo_span(lo_coeff_occ, s1e, mo_occ)
        span12 = p12 < thresh
        span21 = p21 < thresh
    else:
        raise TypeError(f"unsupported mean-field type: {type(mf)}")

    print(f"LO span the MO occ space - {span12}.\n" f"MO occ span the LO space - {span21}.")
    if not span12:
        raise ValueError(
            "the local orbitals do not span the occupied orbitals; check the localization "
            f"(projection losses {p12}, {p21})"
        )


# --------------------------------------------------------------------------- amplitudes


def rot_amplitude(mf, t1, t2, mo1occ, mo2occ, mo1vir, mo2vir):
    """Rotate t1/t2 (pyscf (i,j,a,b) layout) from the orbitals mo1 to the orbitals mo2."""
    u12occ = mo_olp(mf, mo1occ, mo2occ)
    u12vir = mo_olp(mf, mo1vir, mo2vir)

    if isinstance(mf, scf.uhf.UHF):
        t1rot = [
            lib.einsum("ij,ab,jb->ia", u12occ[s].T, u12vir[s].conj().T, t1[s], optimize="optimal")
            for s in range(2)
        ]
        pairs = ((0, 0), (0, 1), (1, 1))
        t2rot = [
            lib.einsum(
                "ik,jl,ac,bd,klcd->ijab",
                u12occ[s].T,
                u12occ[t].T,
                u12vir[s].conj().T,
                u12vir[t].conj().T,
                t2[n],
                optimize="optimal",
            )
            for n, (s, t) in enumerate(pairs)
        ]
        return t1rot, t2rot

    t1rot = lib.einsum("ij,ab,jb->ia", u12occ.T, u12vir.conj().T, t1, optimize="optimal")
    t2rot = lib.einsum(
        "ik,jl,ac,bd,klcd->ijab",
        u12occ.T,
        u12occ.T,
        u12vir.conj().T,
        u12vir.conj().T,
        t2,
        optimize="optimal",
    )
    return t1rot, t2rot


def can2lno_amplitude(mf, t1, t2, can_split, lno_split):
    """Amplitudes solved in the canonicalized LAS, rotated to the LNO (natural) orbitals."""
    if isinstance(mf, scf.uhf.UHF):
        mo1occ = [can_split[s][1] for s in range(2)]
        mo1vir = [can_split[s][2] for s in range(2)]
        mo2occ = [lno_split[s][1] for s in range(2)]
        mo2vir = [lno_split[s][2] for s in range(2)]
    else:
        mo1occ, mo1vir = can_split[1:3]
        mo2occ, mo2vir = lno_split[1:3]
    return rot_amplitude(mf, t1, t2, mo1occ, mo2occ, mo1vir, mo2vir)


# --------------------------------------------------------------------------- LAS


def make_rlas(mlno, eris, orbloc, lno_type, lno_param):
    log = logger.new_logger(mlno)
    cput1 = (logger.process_clock(), logger.perf_counter())

    s1e = mlno.s1e

    orboccfrz_core, orbocc, orbvir, orbvirfrz_core = mlno.split_mo_coeff()
    moeocc, moevir = mlno.split_mo_energy()[1:3]

    # projection of the LOs onto occ and vir
    uocc_loc = reduce(np.dot, (orbloc.T.conj(), s1e, orbocc))  # <loc|mo_occ>
    uocc_loc, uocc_std, uocc_orth = lno.projection_construction(
        uocc_loc, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active
    )
    if uocc_loc.shape[1] == 0:
        log.error(
            "LOs do not overlap with occupied space. This could be caused by either a bad "
            "fragment choice or too high of `lo_proj_thresh_active` (current value: %s).",
            mlno.lo_proj_thresh_active,
        )
        raise RuntimeError
    log.info(
        "LO occ proj: %d active | %d standby | %d orthogonal",
        *[u.shape[1] for u in [uocc_loc, uocc_std, uocc_orth]],
    )

    uvir_loc: Any
    uvir_std: Any
    uvir_orth: Any
    uvir_loc = reduce(np.dot, (orbloc.T.conj(), s1e, orbvir))
    uvir_loc, uvir_std, uvir_orth = lno.projection_construction(
        uvir_loc, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active
    )
    log.info(
        "LO vir proj: %d active | %d standby | %d orthogonal",
        *[u.shape[1] for u in [uvir_loc, uvir_std, uvir_orth]],
    )
    if uvir_loc.shape[1] == 0:
        uvir_loc = uvir_std = uvir_orth = None

    # LNO construction, occupied
    dmoo = mlno.make_lo_rdm1_occ(eris, moeocc, moevir, uocc_loc, uvir_loc, lno_type[0])
    if mlno._match_oldcode:
        dmoo *= 0.5
    dmoo = reduce(np.dot, (uocc_orth.T.conj(), dmoo, uocc_orth))
    if lno_param[0]["norb"] is not None:
        lno_param[0]["norb"] -= uocc_loc.shape[1] + uocc_std.shape[1]
    uoccact_orth, uoccfrz_orth = lno.natorb_select(dmoo, uocc_orth, **lno_param[0])
    # for occ, flip the NOs so they are in the order small -> large eigenvalue
    uoccact_orth = uoccact_orth[:, ::-1]
    uoccfrz_orth = uoccfrz_orth[:, ::-1]
    orboccfrz = np.hstack((orboccfrz_core, np.dot(orbocc, uoccfrz_orth)))
    uoccact = np.hstack((uoccact_orth, uocc_std, uocc_loc))
    orboccact = np.dot(orbocc, uoccact)
    uoccact_loc = np.linalg.multi_dot((orboccact.T.conj(), s1e, orbloc))
    can_uoccact = lno.subspace_eigh(np.diag(moeocc), uoccact)[1]
    can_orboccact = np.dot(orbocc, can_uoccact)
    can_uoccact_loc = np.linalg.multi_dot((can_orboccact.T.conj(), s1e, orbloc))
    cput1 = log.timer_debug1("make_lo_rdm1_occ", *cput1)

    # LNO construction, virtual
    dmvv = mlno.make_lo_rdm1_vir(eris, moeocc, moevir, uocc_loc, uvir_loc, lno_type[1])
    if mlno._match_oldcode:
        dmvv *= 0.5
    if uvir_orth is not None:
        dmvv = reduce(np.dot, (uvir_orth.T.conj(), dmvv, uvir_orth))
        if lno_param[1]["norb"] is not None:
            lno_param[1]["norb"] -= uvir_loc.shape[1] + uvir_std.shape[1]
        uviract_orth, uvirfrz_orth = lno.natorb_select(dmvv, uvir_orth, **lno_param[1])
        orbvirfrz = np.hstack((np.dot(orbvir, uvirfrz_orth), orbvirfrz_core))
        # vir in decreasing eigenvalue order
        uviract = np.hstack((uvir_loc, uvir_std, uviract_orth))
        orbviract = np.dot(orbvir, uviract)
        can_uviract = lno.subspace_eigh(np.diag(moevir), uviract)[1]
        can_orbviract = np.dot(orbvir, can_uviract)
    else:
        orbviract, orbvirfrz = lno.natorb_select(dmvv, orbvir, **lno_param[1])
        orbvirfrz = np.hstack((orbvirfrz, orbvirfrz_core))
        uviract = reduce(np.dot, (orbvir.T.conj(), s1e, orbviract))
        orbviract = np.dot(orbvir, uviract)
        can_uviract = lno.subspace_eigh(np.diag(moevir), uviract)[1]
        can_orbviract = np.dot(orbvir, can_uviract)
    cput1 = log.timer_debug1("make_lo_rdm1_vir", *cput1)

    # LAS construction
    orbfragall = [orboccfrz, orboccact, orbviract, orbvirfrz]
    can_orbfragall = [orboccfrz, can_orboccact, can_orbviract, orbvirfrz]
    orbfrag = np.hstack(orbfragall)
    can_orbfrag = np.hstack(can_orbfragall)
    norbfragall = np.asarray([x.shape[1] for x in orbfragall])
    locfragall = np.cumsum([0] + norbfragall.tolist()).astype(int)
    frzfrag = np.concatenate(
        (np.arange(locfragall[0], locfragall[1]), np.arange(locfragall[3], locfragall[4]))
    ).astype(int)
    frag_msg = "%d/%d Occ | %d/%d Vir | %d/%d MOs" % (
        norbfragall[1],
        sum(norbfragall[:2]),
        norbfragall[2],
        sum(norbfragall[2:4]),
        sum(norbfragall[1:3]),
        sum(norbfragall),
    )
    if len(frzfrag) == 0:
        frzfrag = 0

    return orbfrag, can_orbfrag, frzfrag, uoccact_loc, can_uoccact_loc, frag_msg


def make_ulas(mlno, eris, orbloc, lno_type, lno_param):
    """Local active spaces of one fragment, one per spin."""
    log = logger.new_logger(mlno)
    s1e = mlno.s1e

    orboccfrz_core: list[Any] = [None] * 2
    orbocc: list[Any] = [None] * 2
    orbvir: list[Any] = [None] * 2
    orbvirfrz_core: list[Any] = [None] * 2
    moeocc: list[Any] = [None] * 2
    moevir: list[Any] = [None] * 2
    uocc_loc: list[Any] = [None] * 2
    uocc_std: list[Any] = [None] * 2
    uocc_orth: list[Any] = [None] * 2

    mo_splits = mlno.split_mo_coeff()
    moe_splits = mlno.split_mo_energy()
    for s in range(2):
        orboccfrz_core[s], orbocc[s], orbvir[s], orbvirfrz_core[s] = mo_splits[s]
        moeocc[s], moevir[s] = moe_splits[s][1:3]
        # projection of the LOs onto occ
        ovlp = orbloc[s].T.conj() @ s1e @ orbocc[s]
        uocc_loc[s], uocc_std[s], uocc_orth[s] = lno.projection_construction(
            ovlp, mlno.lo_proj_thresh, mlno.lo_proj_thresh_active
        )
        log.info(
            "LO occ proj: %d active | %d standby | %d orthogonal",
            *[u.shape[1] for u in [uocc_loc[s], uocc_std[s], uocc_orth[s]]],
        )

    if lno_type[0] == lno_type[1] == "1h":
        # uvir_loc is not used in 1h/1h
        if getattr(mlno, "with_df", None):
            dmoo, dmvv = ulno.make_lo_rdm1_1h_df(eris, moeocc, moevir, uocc_loc)
        else:
            dmoo, dmvv = ulno.make_lo_rdm1_1h(eris, moeocc, moevir, uocc_loc)
    else:
        raise NotImplementedError("Unsupported LNO type")

    lno_orbfrag: list[Any] = [None] * 2
    frzfrag: list[Any] = [None] * 2
    uoccact_loc: list[Any] = [None] * 2
    can_orbfrag: list[Any] = [None] * 2
    can_uoccact_loc: list[Any] = [None] * 2
    frag_msg = ""

    for s in range(2):
        dmoo[s] = uocc_orth[s].T.conj() @ dmoo[s] @ uocc_orth[s]

        _param = lno_param[s][0]
        if _param["norb"] is not None:
            _param["norb"] -= uocc_loc[s].shape[1] + uocc_std[s].shape[1]

        uoccact_orth, uoccfrz_orth = lno.natorb_select(dmoo[s], uocc_orth[s], **_param)
        uoccact_orth = uoccact_orth[:, ::-1]
        uoccfrz_orth = uoccfrz_orth[:, ::-1]
        orboccfrz = np.hstack((orboccfrz_core[s], np.dot(orbocc[s], uoccfrz_orth)))
        uoccact = np.hstack((uoccact_orth, uocc_std[s], uocc_loc[s]))
        orboccact = np.dot(orbocc[s], uoccact)
        uoccact_loc[s] = np.linalg.multi_dot((orboccact.T.conj(), s1e, orbloc[s]))
        can_uoccact = lno.subspace_eigh(np.diag(moeocc[s]), uoccact)[1]
        can_orboccact = np.dot(orbocc[s], can_uoccact)
        can_uoccact_loc[s] = np.linalg.multi_dot((can_orboccact.T.conj(), s1e, orbloc[s]))

        orbviract, orbvirfrz = lno.natorb_select(dmvv[s], orbvir[s], **(lno_param[s][1]))
        orbvirfrz = np.hstack((orbvirfrz, orbvirfrz_core[s]))
        uviract = orbvir[s].T.conj() @ s1e @ orbviract
        orbviract = np.dot(orbvir[s], uviract)
        can_uviract = lno.subspace_eigh(np.diag(moevir[s]), uviract)[1]
        can_orbviract = np.dot(orbvir[s], can_uviract)

        orbfragall = [orboccfrz, orboccact, orbviract, orbvirfrz]
        can_orbfragall = [orboccfrz, can_orboccact, can_orbviract, orbvirfrz]
        lno_orbfrag[s] = np.hstack(orbfragall)
        can_orbfrag[s] = np.hstack(can_orbfragall)
        norbfragall = np.asarray([x.shape[1] for x in orbfragall])
        locfragall = np.cumsum([0] + norbfragall.tolist()).astype(int)
        frzfrag[s] = np.concatenate(
            (np.arange(locfragall[0], locfragall[1]), np.arange(locfragall[3], locfragall[4]))
        ).astype(int)
        frag_msg += "\nSpin channel %d: %d/%d Occ | %d/%d Vir | %d/%d MOs" % (
            s,
            norbfragall[1],
            sum(norbfragall[:2]),
            norbfragall[2],
            sum(norbfragall[2:4]),
            sum(norbfragall[1:3]),
            sum(norbfragall),
        )
        if len(frzfrag[s]) == 0:
            frzfrag[s] = 0

    return lno_orbfrag, can_orbfrag, frzfrag, uoccact_loc, can_uoccact_loc, frag_msg


def make_las(mlno, eris, orbloc, lno_type, lno_param):
    """
    LNOs of one fragment.

    Returns (lno_coeff, can_coeff, lno_frozen, uocc_loc, can_uocc_loc, frag_msg): the LNO
    and the canonicalized-LAS coefficients, both ordered [frz_occ | act_occ | act_vir |
    frz_vir]; the frozen indices; <act_occ|lo> in each of the two orbital sets; a summary.
    """
    if isinstance(mlno._scf, scf.uhf.UHF):
        return make_ulas(mlno, eris, orbloc, lno_type, lno_param)
    if isinstance(mlno._scf, scf.rhf.RHF):
        return make_rlas(mlno, eris, orbloc, lno_type, lno_param)
    raise TypeError(f"unsupported mean-field type: {type(mlno._scf)}")


def _split_one(coeff, nocc, nao, lno_frozen):
    idx_act = np.array([i for i in range(nao) if i not in lno_frozen], dtype=int)
    idx_frzocc = np.array([i for i in range(nocc) if i not in idx_act], dtype=int)
    idx_actocc = np.array([i for i in range(nocc) if i in idx_act], dtype=int)
    idx_actvir = np.array([i for i in range(nocc, nao) if i in idx_act], dtype=int)
    idx_frzvir = np.array([i for i in range(nocc, nao) if i not in idx_act], dtype=int)
    split = [coeff[:, idx_frzocc], coeff[:, idx_actocc], coeff[:, idx_actvir], coeff[:, idx_frzvir]]
    sizes = (len(idx_frzocc), len(idx_actocc), len(idx_actvir), len(idx_frzvir))
    return split, sizes


def split_lno(mlno, lno_coeff, lno_frozen):
    """
    Split lno_coeff into [frz_occ, act_occ, act_vir, frz_vir] blocks.

    Returns (lno_split, nfrzocc, nactocc, nactvir, nfrzvir); for a UHF mean field each is
    a pair (alpha, beta).
    """
    mf = mlno._scf
    nao = mf.mol.nao
    mo_occ = mlno.mo_occ

    if isinstance(mf, scf.uhf.UHF):
        splits, sizes = [], []
        for s in range(2):
            frozen = lno_frozen[s] if not isinstance(lno_frozen[s], int) else []
            sp, sz = _split_one(lno_coeff[s], np.count_nonzero(mo_occ[s]), nao, frozen)
            splits.append(sp)
            sizes.append(sz)
        nfrzocc, nactocc, nactvir, nfrzvir = ([sizes[0][k], sizes[1][k]] for k in range(4))
        return splits, nfrzocc, nactocc, nactvir, nfrzvir

    frozen = lno_frozen if not isinstance(lno_frozen, int) else []
    split, (nfrzocc, nactocc, nactvir, nfrzvir) = _split_one(
        lno_coeff, np.count_nonzero(mo_occ), nao, frozen
    )
    return split, nfrzocc, nactocc, nactvir, nfrzvir
