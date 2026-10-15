import os

os.environ["OMP_NUM_THREADS"] = "1"  # LNO orbitals are thread sensitive; keeps runs reproducible

import trot.lnoafqmc  # noqa: F401  before jax: chooses the allocator that frees memory between fragments
from pyscf import gto, scf
from pyscf.data.elements import chemcore

from trot.lnoafqmc import LnoAfqmcMixed, iao_fragment

a = 1.20577  # O-O bond length (Angstrom)
d = 3        # centre-to-centre distance between the two molecules (Angstrom)
na, nc = 2, 8  # atoms per monomer, number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"O {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="ccpvdz", spin=2 * nc, verbose=3)
mf = scf.UHF(mol).density_fit()  # the fragment integrals come from the DF tensor
mf.kernel()

# follow the UHF solution down to a stable one
for _ in range(5):
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        break
    mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
print(f"UHF energy: {mf.e_tot:.10f}")

nfrozen = int(chemcore(mol))

# for a UHF mean field lo_coeff is an (alpha, beta) pair and each fragment a pair of LO lists
lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="atom")

lno = LnoAfqmcMixed(
    mf,
    lo_coeff,
    frag_list,
    frag_name=frag_name,
    run_frag=[0],
    lno_thresh=1e-6,
    nfrozen=nfrozen,
    trial="upt2ccsd_fast",  # the default for a UHF mf; guide=None -> the UHF guide, or guide="ucisd"
    target_error=1e-5,
    n_walkers=300,
    n_eql_blocks=80,
    n_blocks=400,
    dt=0.005,
    seed=27,
    guide_mixed_precision=False,
    trial_mixed_precision=True, # specify mixed precision separately
)
e_qmc, e_qmc_err = lno.kernel()
