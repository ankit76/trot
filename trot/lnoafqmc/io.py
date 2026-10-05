from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

# Optional output files of LnoAfqmcMixed, in the format of afqmc's lno_afqmc.py so
# existing analysis scripts keep reading them:
#
#   frag_output="./fragment.out"  ->  ./fragment.out{i}   (i = frag_idx + 1)
#       the fragment's AFQMC log (stdout is tee'd while it runs) + a summary block
#   lno_output="./lno_result.out"
#       the results table, rewritten after every fragment


class _Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def tee_to_file(path: str | Path | None, mode: str = "a") -> Iterator[None]:
    """Duplicate sys.stdout into a file for the duration of the block (no-op for None)."""
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as fh:
        old = sys.stdout
        sys.stdout = _Tee(old, fh)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stdout = old


def frag_output_path(frag_output: str | Path, frag_idx: int) -> Path:
    """./fragment.out -> ./fragment.out{frag_idx + 1}"""
    p = Path(frag_output)
    return p.with_name(f"{p.name}{frag_idx + 1}")


def write_frag_summary(
    path: str | Path,
    *,
    frag_idx: int,
    frag_name: str,
    nactocc: Any,
    norb: Any,
    efrag_mp: float,
    efrag_cc: float,
    efrag_qmc: float,
    efrag_qmc_err: float,
    t_cc: float,
    t_wait: float,
    t_qmc: float,
) -> None:
    header = f" Fragment{frag_idx + 1} Results "
    width = 80
    with open(path, "a") as f:
        f.write("\n")
        f.write(f"{header:=^{width}}\n")
        f.write("\t LNO Fragment " + str(frag_name) + "\n")
        f.write("-" * width + "\n")
        f.write(f"\t LNO-Active Space electrons: {np.array(nactocc)} | orbitals: {norb} \n")
        f.write(f"\t LNO-MP2 Fragment Energy:    {efrag_mp:.8f} \n")
        f.write(f"\t LNO-CCSD Fragment Energy:   {efrag_cc:.8f} \n")
        f.write(f"\t LNO-AFQMC Fragment Energy:  {efrag_qmc:.5f} +/- {efrag_qmc_err:.5f} \n")
        f.write(f"\t LNO-CCSD Fragment Time:     {t_cc:.2f} \n")
        f.write(f"\t LNO-CCSD Fragment Wait:     {t_wait:.2f} \n")
        f.write(f"\t LNO-AFQMC Fragment Time:    {t_qmc:.2f} \n")
        f.write("=" * width + "\n")


def write_lno_result(
    path: str | Path,
    *,
    run_frag: Sequence[int],
    frag_name: Sequence[str],
    lno_size: Sequence[Any],
    lno_emp: Any,
    lno_ecc: Any,
    lno_eqmc: Any,
    lno_eqmc_err: Any,
    lno_cc_time: Any,
    lno_wait_time: Any,
    lno_qmc_time: Any,
    lno_thresh: Sequence[Any],
    depth: int,
    loop_time: float,
) -> None:
    n = len(run_frag)
    sizes = [
        f"{np.array(s)}" if not isinstance(s, (int, np.integer)) else f"{int(s)}" for s in lno_size
    ]
    lno_max = max((int(np.max(s)) for s in lno_size), default=0)
    e_mp = float(np.sum(lno_emp))
    e_cc = float(np.sum(lno_ecc))
    e_qmc = float(np.sum(lno_eqmc))
    e_qmc_err = float(np.sqrt(np.sum(np.asarray(lno_eqmc_err) ** 2)))
    tot_cc = float(np.sum(lno_cc_time))
    tot_qmc = float(np.sum(lno_qmc_time))
    tot_wait = float(np.sum(lno_wait_time))
    serial = tot_cc + tot_qmc

    width = 120
    with open(path, "w") as f:
        f.write("=" * width + "\n")
        f.write(f'{"LNO-AFQMC Results":^{width}}\n')
        f.write("=" * width + "\n")
        f.write(
            f'{"Num":>4s}  {"Fragment":>16s}  {"LAS SIZE":>10s}  {"E(MP2)":>10s}  {"E(CCSD)":>10s}  '
            f'{"E(AFQMC)":>10s}  {"Error":>8s}  {"t(CCSD)":>8s}  {"t(wait)":>8s}  {"t(AFQMC)":>8s}\n'
        )
        f.write("-" * width + "\n")
        for k in range(n):
            f.write(
                f"{run_frag[k] + 1:4d}  {frag_name[k]:>16s}  {sizes[k]:10s}  "
                f"{lno_emp[k]:10.8f}  {lno_ecc[k]:10.8f}  {lno_eqmc[k]:10.5f}  {lno_eqmc_err[k]:8.5f}  "
                f"{lno_cc_time[k]:8.2f}  {lno_wait_time[k]:8.2f}  {lno_qmc_time[k]:8.2f}\n"
            )
        f.write("-" * width + "\n")
        f.write(f'{"Summarize Fragments":^{width}}\n')
        f.write("-" * width + "\n")
        thresh_str = "[" + ", ".join(f"{x:.2e}" for x in lno_thresh) + "]"
        f.write(
            f'{"LNO-Thresh":<20} {"Max LAS":>8} {"E[MP2]":>12} {"E[CCSD]":>12} '
            f'{"E[AFQMC]":>10} {"Err[AFQMC]":>10} {"CCSD-Time":>10} {"AFQMC-Time":>10}\n'
        )
        f.write(
            f"{thresh_str:<20} {lno_max:>8} {e_mp:>12.8f} {e_cc:>12.8f} "
            f"{e_qmc:>10.5f} {e_qmc_err:>10.5f} {tot_cc:>10.2f} {tot_qmc:>10.2f}\n"
        )
        f.write("-" * width + "\n")
        f.write(f'{"Pipeline (prefetch depth " + str(depth) + ")":^{width}}\n')
        f.write("-" * width + "\n")
        f.write(f'{"Loop wall time":<28} {loop_time:>10.2f} s\n')
        f.write(f'{"Serial equivalent (CPU+GPU)":<28} {serial:>10.2f} s\n')
        hidden = tot_cc - tot_wait
        f.write(
            f'{"CPU time hidden behind GPU":<28} {hidden:>10.2f} s '
            f"({100.0 * hidden / tot_cc if tot_cc > 0 else 0.0:.1f}%)\n"
        )
        f.write(
            f'{"Speedup vs serial":<28} {serial / loop_time if loop_time > 0 else 0.0:>10.2f} x\n'
        )
        f.write("=" * width + "\n\n")
