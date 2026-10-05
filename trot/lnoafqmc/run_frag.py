"""
Run one LNO fragment's AFQMC from its self-contained file, in its own process:

    python -m trot.lnoafqmc.run_frag frag3.h5 --options opts.json --out res.json

opts.json holds the LnoFragMixed keyword arguments (trial, guide, max_error, seed, ...);
res.json gets the fragment energy and error. LnoAfqmcMixed(isolate=True) drives this so
that the OS frees everything the fragment held on the device when the process exits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frag_file")
    parser.add_argument("--options", default=None, help="json file with LnoFragMixed kwargs")
    parser.add_argument("--out", default=None, help="json file to write the result to")
    args = parser.parse_args(argv)

    from . import LnoFragMixed  # noqa: E402  (sets the allocator before jax loads)

    opts = json.loads(Path(args.options).read_text()) if args.options else {}
    frag = LnoFragMixed.from_frag_data(args.frag_file, **opts)
    e, err = frag.kernel()
    res = {
        "e_frag": e,
        "e_frag_err": err,
        "guide_e": frag.guide_e_tot,
        "guide_e_err": frag.guide_e_err,
        "n_blocks_run": frag.qmc_result.n_blocks_run if frag.qmc_result is not None else None,
        "frag_idx": int(frag.frag.frag_idx),
    }
    text = json.dumps(res)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
