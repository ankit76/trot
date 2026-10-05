from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable

from pyscf import scf

from . import las, solvers
from .staging import LnoFragData

print = partial(print, flush=True)

# The CPU half of one fragment (make_las -> LNO-MP2 -> LNO-CCSD) and the thread that runs
# it ahead of the AFQMC loop, ported from afqmc's lno_afqmc.py.
#
# cpu_stage is pure with respect to shared state: make_las only reads the LNO object and
# the eris, and every solver builds its own CC object. So the CPU stage of fragment i+1
# can run in a background thread while fragment i's AFQMC occupies the device; pyscf
# (numpy/BLAS) releases the GIL in the heavy parts, and so does jax dispatch. The pyscf
# log of the worker is captured and replayed by the main thread so the printed output
# stays ordered by fragment.
#
# A thread rather than a process on purpose: the stage needs the LNO object and the
# (potentially huge) eris, which a forked worker would have to re-create or pickle.


@contextmanager
def _capture_pyscf_log(obj: Any):
    """
    Redirect the pyscf logger of obj into a string buffer. logger.new_logger(obj) writes
    to obj.stdout, looked up at call time, so swapping it collects the worker's log
    without touching the global sys.stdout shared with the main thread.
    """
    buf = io.StringIO()
    old = getattr(obj, "stdout", None)
    obj.stdout = buf
    try:
        yield buf
    finally:
        obj.stdout = old


def cpu_stage(
    mlno: Any,
    mf: Any,
    lo_coeff: Any,
    loidx: Any,
    lno_thresh: Any,
    lno_pct_occ: Any,
    lno_norb: Any,
    lno_type: Any,
    eris: Any,
    ifrag: int,
    frag_idx: int,
    frag_name: str,
    run_mp: bool,
    run_cc: bool,
    nfrozen: int = 0,
) -> LnoFragData:
    """
    Everything that runs on the CPU for one fragment: the LAS, LNO-MP2 and LNO-CCSD.

    Output is buffered in LnoFragData.log instead of printed, so the main thread can
    replay it in order.
    """
    log = io.StringIO()
    _log = partial(print, file=log)
    t_start = time.perf_counter()

    orbloc, lno_param = solvers.get_lnoparam(
        mf, lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag
    )

    with _capture_pyscf_log(mlno) as pyscf_log:
        lno_coeff, can_coeff, frozen_idx, lno_loc, can_loc, frag_msg = las.make_las(
            mlno, eris, orbloc, lno_type, lno_param
        )
    if pyscf_log.getvalue():
        log.write(pyscf_log.getvalue())
    _log(f"LNO-LAS: {frag_msg}")

    frozen_idx, maskact = solvers.get_maskact(mf, frozen_idx, mlno.mo_occ)
    nactocc: Any
    nactvir: Any
    lno_split, nfrzocc, nactocc, nactvir, nfrzvir = las.split_lno(mlno, lno_coeff, frozen_idx)
    can_split = las.split_lno(mlno, can_coeff, frozen_idx)[0]
    t_las = time.perf_counter() - t_start

    time0 = time.perf_counter()
    if run_mp:
        # fragment MP2 only supports canonical orbitals
        efrag_mp = solvers.lnomp2_kernel(mlno, can_coeff, frozen_idx, can_loc, maskact, verbose=0)
    else:
        efrag_mp = 0.0
    t_mp = time.perf_counter() - time0

    time0 = time.perf_counter()
    if run_cc:
        # fragment CCSD converges faster in canonical orbitals; rotate the amplitudes to
        # the LNOs afterwards
        efrag_cc, t1, t2 = solvers.lnoccsd_kernel(
            mlno, can_coeff, frozen_idx, can_loc, maskact, verbose=0
        )
        if t1 is not None:
            t1, t2 = las.can2lno_amplitude(mf, t1, t2, can_split, lno_split)
    else:
        efrag_cc, t1, t2 = 0.0, None, None
    t_cc = time.perf_counter() - time0

    if isinstance(mf, scf.uhf.UHF):
        lno_coeff = (lno_coeff[0], lno_coeff[1])
        frozen_idx = (frozen_idx[0], frozen_idx[1])
        lno_loc = (lno_loc[0], lno_loc[1])
        nactocc = (int(nactocc[0]), int(nactocc[1]))
        nactvir = (int(nactvir[0]), int(nactvir[1]))
        if t1 is not None:
            t1 = (t1[0], t1[1])
            t2 = (t2[0], t2[1], t2[2])
    else:
        nactocc = int(nactocc)
        nactvir = int(nactvir)

    return LnoFragData(
        frag_idx=int(frag_idx),
        frag_name=str(frag_name),
        lno_coeff=lno_coeff,
        lno_frozen=frozen_idx,
        uocc_loc=lno_loc,
        nactocc=nactocc,
        nactvir=nactvir,
        t1=t1,
        t2=t2,
        efrag_mp=float(efrag_mp),
        efrag_cc=float(efrag_cc),
        lno_thresh=tuple(mlno.lno_thresh),
        nfrozen=int(nfrozen),
        t_las=t_las,
        t_mp=t_mp,
        t_cc=t_cc,
        t_cpu=time.perf_counter() - t_start,
        log=log.getvalue(),
    )


class CpuPipeline:
    """
    Runs fn(i) `depth` fragments ahead of the consumer.

    depth == 0 executes inline in the calling thread (the serial reference). A single
    worker keeps the CPU stages in their original order and only depth + 1 fragments'
    amplitudes alive at once.
    """

    def __init__(self, fn: Callable[[int], LnoFragData], nfrag: int, depth: int = 1):
        self.fn = fn
        self.nfrag = nfrag
        self.depth = int(depth)
        self.futures: dict[int, Any] = {}
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="lno-cpu")
            if self.depth > 0
            else None
        )

    def schedule(self, i: int) -> None:
        if self.executor is None or i >= self.nfrag or i in self.futures:
            return
        self.futures[i] = self.executor.submit(self.fn, i)

    def fill(self, i: int) -> None:
        """Keep fragments [i, i+depth] queued."""
        for k in range(i, min(i + self.depth + 1, self.nfrag)):
            self.schedule(k)

    def get(self, i: int) -> LnoFragData:
        if self.executor is None:
            return self.fn(i)
        self.schedule(i)
        return self.futures.pop(i).result()

    def shutdown(self, cancel: bool = False) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=not cancel, cancel_futures=cancel)
