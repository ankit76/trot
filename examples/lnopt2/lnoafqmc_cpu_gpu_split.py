"""
LNO-AFQMC/pt2CCSD on two machines: the CPU stage, then the AFQMC
=================================================================

The LNO stage of every fragment (the local active space, LNO-MP2, LNO-CCSD, the
fragment integrals) runs on the CPU and needs the mean field; the AFQMC runs on the GPU
and needs only the fragment files. LnoAfqmcMixed splits the two:

  step 1, CPU machine:  run_qmc=False, save_frag_data=DIR writes DIR/frag{i}.h5 for every
                        fragment (its hamiltonian, guide, LNO data and amplitudes) and stops;
  step 2, GPU machine:  LnoAfqmcMixed(frag_data=DIR, ...) runs the AFQMC of every file
                        (or of run_frag) and fills the usual results; the LNO-MP2/CCSD
                        energies and fragment sizes come from the files.

With the same seed, trial and QMC settings step 2 reproduces the one-machine loop
fragment by fragment (the per fragment seeds are drawn from seed and the number of
fragments stored in the files). Run from the repository root, once per step:

    python examples/lnopt2/lnoafqmc_cpu_gpu_split.py cpu
    python examples/lnopt2/lnoafqmc_cpu_gpu_split.py gpu
"""

import sys

from pyscf import gto, scf
from pyscf.data.elements import chemcore

# the first trot import: importing trot.lnoafqmc before jax initialises selects the
# allocator that hands device memory back between fragments (see trot/lnoafqmc/__init__.py)
from trot.lnoafqmc import LnoAfqmcMixed

FRAG_DIR = "./frag_data"
STEP = sys.argv[1] if len(sys.argv) > 1 else "cpu"
TRIAL = "pt2ccsd_fast"

if STEP == "cpu":
    from trot.lnoafqmc import iao_fragment

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
    mol = gto.M(atom=atoms, basis="6-31g", verbose=4)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    nfrozen = chemcore(mol)
    lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="h2heavy")

    lno = LnoAfqmcMixed(
        mf,
        lo_coeff,
        frag_list,
        frag_name=frag_name,
        lno_thresh=3e-5,
        nfrozen=nfrozen,
        trial=TRIAL,  # decides what the files carry (the _fast trials: the projected t2)
        run_qmc=False,
        save_frag_data=FRAG_DIR,
        lno_output="lno_cpu.out",
    )
    lno.kernel()
    print(f"LNO-MP2 {lno.e_mp:.8f}  LNO-CCSD {lno.e_cc:.8f}; fragment files in {FRAG_DIR}")

elif STEP == "gpu":
    lno = LnoAfqmcMixed(
        frag_data=FRAG_DIR,  # a directory of frag*.h5, one file, or a list of files
        trial=TRIAL,
        # run_frag=[0, 1],   # a subset of the fragments (0-based, as in the one-machine run)
        target_error=1e-3,
        seed=17,
        n_walkers=200,
        n_blocks=200,
        frag_output="fragment.out",
        lno_output="lno_result.out",
        # isolate=True,      # one child process per fragment
    )
    e_qmc, e_qmc_err = lno.kernel()
    print(f"LNO-AFQMC correlation energy {e_qmc:.6f} +/- {e_qmc_err:.6f}")

else:
    raise SystemExit(f"usage: {sys.argv[0]} cpu|gpu")
