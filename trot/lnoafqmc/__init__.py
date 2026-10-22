"""
LNO-AFQMC on trot's mixed guide/trial pipeline.

    from trot.lnoafqmc import LnoAfqmcMixed, iao_fragment

    lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen)
    lno = LnoAfqmcMixed(mf, lo_coeff, frag_list, frag_name=frag_name, lno_thresh=1e-5,
                        trial="pt2ccsd", target_error=1e-3)
    e_qmc, e_qmc_err = lno.kernel()

Per fragment, kernel() runs make_las -> LNO-MP2 -> LNO-CCSD -> LNO-AFQMC. The AFQMC of one
fragment is an AfqmcMixed run (LnoFragMixed) on the fragment hamiltonian, with the guide built
by trot's own staging and the fragment energy measured against the pt2CCSD trial, the bar
estimator on the similarity transformed hamiltonian (meas/pt2ccsd_bar.py). An RHF mean
field runs trial="pt2ccsd" (restricted walkers, RHF or CISD guide).

The package only imports from core trot (the mixed guide/trial pipeline: trot/mixed.py,
setup_mixed.py, driver_mixed.py, prop/blocks_mixed.py and the chunked pt2ccsd_bar
kernel); it does not touch the RHF-trial LNO code in trot/lno.py, trot/afqmc.py
(AfqmcLno, AfqmcLnoFrag) or trot/meas/rhf.py.

Fragments run one after another in this process, so what one fragment put on the device
has to be handed back before the next one starts. The platform allocator returns freed
buffers to the driver instead of keeping them in a pool; it has to be chosen before jax
initializes, which is why it is set here rather than in LnoAfqmcMixed.
"""

from __future__ import annotations

import os
import sys as _sys

_ALLOCATOR = "XLA_PYTHON_CLIENT_ALLOCATOR"

# only effective if jax has not been imported yet; LnoAfqmcMixed warns otherwise
if "jax" not in _sys.modules:
    os.environ.setdefault(_ALLOCATOR, "platform")


# the heavy modules (jax, pyscf-forge) are imported on first use
_LAZY = {
    "LnoAfqmcMixed": ".afqmc",
    "LnoFragMixed": ".afqmc",
    "LnoFragData": ".staging",
    "iao_fragment": ".fragments",
    "save_iao_fragment": ".fragments",
    "load_iao_fragment": ".fragments",
    "check_span": ".las",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
