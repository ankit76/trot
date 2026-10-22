"""
Re-running one LNO fragment's AFQMC from its saved file
=======================================================

LnoAfqmcMixed(save_frag_data=DIR) writes DIR/frag{i}.h5 for every fragment it runs: the
fragment hamiltonian in the active LNO basis (h0, h1, the cholesky vectors), the staged
guide, the LNO data (lno_coeff, the frozen LNO indices, the overlap U = <act occ|lo> the
fragment projector is built from) and the CCSD amplitudes the trial needs: the full t2
for the pt2ccsd / upt2ccsd trials, the projected t2u = t2 U (nlo/nocc the size) for the
_fast ones. A re-run needs nothing else: no mean field, no LNO, CCSD or integral work.

Part 1 below produces such files on a small water tetramer (skipped when they exist);
part 2 re-runs one fragment from its file with LnoFragMixed.from_frag_data; part 3 does
the same from the command line. Run from the repository root:

    python examples/lnoafqmc_frag_from_file.py
"""

import os
from typing import Any

from pyscf import gto, scf
from pyscf.data.elements import chemcore

# the first trot import: importing trot.lnoafqmc before jax initialises selects the
# allocator that hands device memory back between fragments (see trot/lnoafqmc/__init__.py)
from trot.lnoafqmc import LnoAfqmcMixed, LnoFragMixed

FRAG_DIR = "./frag_data"

# ---------------------------------------------------------------------- part 1: write the files
if not os.path.exists(os.path.join(FRAG_DIR, "frag1.h5")):
    atoms = """
    O     1.370000     1.370000     0.000000
    H     0.400000     1.370000     0.000000
    H     1.612869     1.370000     0.939103
    O    -1.370000     1.370000     0.000000
    H    -1.370000     0.400000     0.000000
    H    -1.370000     1.612869    -0.939103
    O    -1.370000    -1.370000     0.000000
    H    -0.400000    -1.370000     0.000000
    H    -1.612869    -1.370000     0.939103
    O     1.370000    -1.370000     0.000000
    H     1.370000    -0.400000     0.000000
    H     1.370000    -1.612869    -0.939103
    """
    mol = gto.M(atom=atoms, basis="6-31g", spin=0, verbose=3, max_memory=16000)
    mf: Any = scf.RHF(mol).density_fit()
    mf.kernel()
    nfrozen = int(chemcore(mol))
    from trot.lnoafqmc import iao_fragment

    lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="h2heavy", more_loc="pm")
    # a short loop: the point here is the files, not the numbers
    lno = LnoAfqmcMixed(
        mf,
        lo_coeff,
        frag_list,
        frag_name=frag_name,
        lno_thresh=1e-5,
        nfrozen=nfrozen,
        trial="pt2ccsd_fast",  # the files then carry t2u instead of t2
        n_walkers=50,
        n_eql_blocks=4,
        n_blocks=20,
        seed=27,
        save_frag_data=FRAG_DIR,
    )
    lno.kernel()

# ---------------------------------------------------------------------- part 2: re-run a fragment
# Every LnoFragMixed keyword can be given: the trial ("pt2ccsd" or "pt2ccsd_fast" both
# work from a file with t2u; "upt2ccsd(_fast)" for a UHF fragment), the guide (the file's
# own by default; "cisd" needs a file with the full t2 or one written with that guide),
# the run length, the seed, the early stop, the precision of the guide and the trial.
frag = LnoFragMixed.from_frag_data(
    os.path.join(FRAG_DIR, "frag1.h5"),
    trial="pt2ccsd_fast",
    n_walkers=200,
    n_eql_blocks=40,
    n_blocks=200,
    seed=11,
    max_error=None,  # e.g. 1e-4 to stop once the fragment error is below 0.7 * 1e-4
    mixed_precision=True,  # both sides; guide_mixed_precision / trial_mixed_precision set them apart
)
e_frag, e_frag_err = frag.kernel()

r = frag.qmc_result
print(
    f"\nfragment {frag.frag.frag_idx + 1} [{frag.frag.frag_name}], "
    f"nactocc={frag.frag.nactocc} nactvir={frag.frag.nactvir}"
)
print(f"LNO-MP2  fragment energy: {frag.frag.efrag_mp:.8f}")
print(f"LNO-CCSD fragment energy: {frag.frag.efrag_cc:.8f}")
print(f"tau = 0  fragment energy: {r.frag_init_energy:.8f}   (equals LNO-CCSD)")
print(
    f"AFQMC    fragment energy: {e_frag:.6f} +/- {e_frag_err:.6f}  ({r.n_blocks_run} blocks, {r.n_outliers} outliers dropped)"
)
print(f"guide energy:             {frag.guide_e_tot:.6f} +/- {frag.guide_e_err:.6f}")
print(f"HF energy stored with the file: {frag.emf:.8f}")
# the block data, if you want your own statistics
# r.frag_block_weights, r.frag_block_components["t2frg"|"e0frg"|"e1frg"|"e0"], r.guide_block_energies

# ---------------------------------------------------------------------- part 3: the command line
# The same re-run in its own process (LnoAfqmcMixed(isolate=True) does this per fragment):
#
#     echo '{"trial": "pt2ccsd_fast", "n_walkers": 200, "n_eql_blocks": 40, "n_blocks": 200, "seed": 11}' > opts.json
#     python -m trot.lnoafqmc.run_frag frag_data/frag1.h5 --options opts.json --out res.json
#
# res.json then holds e_frag, e_frag_err, guide_e, guide_e_err, n_blocks_run and frag_idx.
