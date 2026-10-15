from __future__ import annotations

from typing import Any
from functools import partial

import numpy as np
from pyscf import lib, scf

try:
    from pyscf.lno import lnoccsd, ulnoccsd  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError("trot.lnoafqmc needs pyscf-forge for the LNO machinery.") from e

print = partial(print, flush=True)

# The LNO object and the per-fragment MP2 / CCSD impurity solves, ported from afqmc's
# lno_afqmc.py (get_lnoccsd, get_lnoparam, lnomp2_kernel, lnoccsd_kernel) and
# mod_lnoccsd.py (the solvers, a modified copy of pyscf-forge's lnoccsd impurity solver
# that also hands back the amplitudes).
#
# The solves run in the canonicalized LAS (can_coeff of las.make_las), where CCSD
# converges faster; las.can2lno_amplitude then rotates t1/t2 to the LNOs.


def get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, thresh, verbose=3):
    """The pyscf-forge LNO object for the whole calculation."""
    if isinstance(mf, scf.uhf.UHF):
        mlno = ulnoccsd.ULNOCCSD(mf, lo_coeff, frag_list, frozen=nfrozen).set(verbose=verbose)
    elif isinstance(mf, scf.rhf.RHF):
        mlno = lnoccsd.LNOCCSD(mf, lo_coeff, frag_list, frozen=nfrozen).set(verbose=verbose)
    else:
        raise NotImplementedError("LNO only supports restricted and unrestricted mean fields")

    if isinstance(thresh, float):
        mlno.lno_thresh = [thresh * 10, thresh]
    elif isinstance(thresh, (list, tuple)):
        assert len(thresh) == 2
        mlno.lno_thresh = [thresh[0], thresh[1]]
    else:
        raise TypeError(f"lno_thresh must be a float or a pair, got {type(thresh)}")

    return mlno


def get_lnoparam(mf, lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag):
    """The LOs of one fragment and the natorb_select parameters for its occ and vir."""
    if isinstance(mf, scf.uhf.UHF):
        orbloc = [lo_coeff[0][:, loidx[0]], lo_coeff[1][:, loidx[1]]]

        def _pick(x: Any, s: int) -> Any:
            return x[s] if isinstance(x, (list, tuple, np.ndarray)) else x

        lno_param = [
            [
                {
                    "thresh": _pick(lno_thresh[i], s),
                    "pct_occ": _pick(lno_pct_occ[i], s),
                    "norb": _pick(lno_norb[ifrag][i], s),
                }
                for i in (0, 1)
            ]
            for s in range(2)
        ]
    elif isinstance(mf, scf.rhf.RHF):
        orbloc = lo_coeff[:, loidx]
        lno_param = [
            {"thresh": lno_thresh[i], "pct_occ": lno_pct_occ[i], "norb": lno_norb[ifrag][i]}
            for i in (0, 1)
        ]
    else:
        raise NotImplementedError("LNO only supports restricted and unrestricted mean fields")

    return orbloc, lno_param


def get_maskact(mf, frozen_idx, mo_occ):
    """(frozen, maskact) as pyscf-forge's get_maskact, per spin for UHF."""
    if isinstance(mf, scf.uhf.UHF):
        return ulnoccsd.get_maskact(frozen_idx, [mo_occ[0].size, mo_occ[1].size])
    if isinstance(mf, scf.rhf.RHF):
        return lnoccsd.get_maskact(frozen_idx, mo_occ.size)
    raise TypeError(f"unsupported mean-field type: {type(mf)}")


def _make_cc(mf, mo_coeff, frozen, verbose):
    if isinstance(mf, scf.uhf.UHF):
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=verbose)
    elif isinstance(mf, scf.rhf.RHF):
        mcc = lnoccsd.CCSD(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=verbose)
    else:
        raise NotImplementedError("LNO only supports restricted and unrestricted orbitals")
    return mcc


def lnomp2_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    """LNO-MP2 fragment energy (canonical orbitals only)."""
    mf = mlno._scf
    mcc = _make_cc(mf, lno_coeff, lno_frozen, verbose)
    if isinstance(mf, scf.uhf.UHF):
        return float(ulnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact))
    return float(rlnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact))


def lnoccsd_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    """LNO-CCSD fragment energy and the (t1, t2) of the impurity solve."""
    mf = mlno._scf
    mcc = _make_cc(mf, lno_coeff, lno_frozen, verbose)
    mcc.conv_tol = 1e-6
    mcc.conv_tol_normt = 3e-5
    if isinstance(mf, scf.uhf.UHF):
        ecc, t1, t2 = ulnoccsd_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    else:
        ecc, t1, t2 = rlnoccsd_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    return float(ecc), t1, t2


# --------------------------------------------------------------------------- solvers


def _rsplit(mo_coeff, mo_occ, maskact):
    maskocc = mo_occ > 1e-10
    orbs = [
        mo_coeff[:, ~maskact & maskocc],
        mo_coeff[:, maskact & maskocc],
        mo_coeff[:, maskact & ~maskocc],
        mo_coeff[:, ~maskact & ~maskocc],
    ]
    return [orb.shape[1] for orb in orbs]


def _usplit(mo_coeff, mo_occ, maskact):
    return [_rsplit(mo_coeff[s], mo_occ[s], maskact[s]) for s in range(2)]


def rlnomp2_solver(mcc, mo_coeff, uocc_loc, mo_occ, maskact):
    nfrzocc, nactocc, nactvir, nfrzvir = _rsplit(mo_coeff, mo_occ, maskact)
    if nactocc == 0 or nactvir == 0:
        return lib.tag_array(0.0, spin_comp=np.array((0.0, 0.0)))

    imp_eris = mcc.ao2mo()
    ovov = imp_eris.ovov if isinstance(imp_eris.ovov, np.ndarray) else imp_eris.ovov[()]
    oovv = ovov.reshape(nactocc, nactvir, nactocc, nactvir).transpose(0, 2, 1, 3)
    ovov = None
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    return lnoccsd.get_fragment_energy(oovv, t2, uocc_loc).real


def ulnomp2_solver(mcc, mo_coeff, uocc_loc, mo_occ, maskact):
    (_, nactocca, nactvira, _), (_, nactoccb, nactvirb, _) = _usplit(mo_coeff, mo_occ, maskact)
    prjlo = [uocc_loc[0].T.conj(), uocc_loc[1].T.conj()]
    if nactocca * nactvira == 0 and nactoccb * nactvirb == 0:
        return lib.tag_array(0.0, spin_comp=np.array((0.0, 0.0)))

    imp_eris = mcc.ao2mo()
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    return ulnoccsd.get_fragment_energy(imp_eris, t1, t2, prjlo)


def rlnoccsd_solver(mcc, mo_coeff, uocc_loc, mo_occ, maskact):
    nfrzocc, nactocc, nactvir, nfrzvir = _rsplit(mo_coeff, mo_occ, maskact)
    if nactocc == 0 or nactvir == 0:
        return lib.tag_array(0.0, spin_comp=np.array((0.0, 0.0))), None, None

    imp_eris = mcc.ao2mo()
    ovov = imp_eris.ovov if isinstance(imp_eris.ovov, np.ndarray) else imp_eris.ovov[()]
    oovv = ovov.reshape(nactocc, nactvir, nactocc, nactvir).transpose(0, 2, 1, 3)
    ovov = None

    t1, t2 = mcc.kernel(eris=imp_eris)[1:]
    if not mcc.converged:
        print("# Impurity CCSD did not converge!")

    t2 += lib.einsum("ia,jb->ijab", t1, t1)
    ecc_frag = lnoccsd.get_fragment_energy(oovv, t2, uocc_loc)
    t2 -= lib.einsum("ia,jb->ijab", t1, t1)
    return ecc_frag, t1, t2


def ulnoccsd_solver(mcc, mo_coeff, uocc_loc, mo_occ, maskact):
    (_, nactocca, nactvira, _), (_, nactoccb, nactvirb, _) = _usplit(mo_coeff, mo_occ, maskact)
    prjlo = [uocc_loc[0].T.conj(), uocc_loc[1].T.conj()]
    if nactocca * nactvira == 0 and nactoccb * nactvirb == 0:
        return lib.tag_array(0.0, spin_comp=np.array((0.0, 0.0))), None, None

    imp_eris = mcc.ao2mo()
    t1, t2 = mcc.kernel(eris=imp_eris)[1:]
    if not mcc.converged:
        print("# Impurity CCSD did not converge!")
    ecc_frag = ulnoccsd.get_fragment_energy(imp_eris, t1, t2, prjlo)
    return ecc_frag, t1, t2
