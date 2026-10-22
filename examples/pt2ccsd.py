"""
Example: a manual setup of AFQMC/pt2CCSD energy for 8 non-interacting H2 dimers
================================================================================

This script demonstrates how to run an AFQMC calculation using pt2CCSD
trial wavefunction manually, without a high-level driver object.
AFQMC/pt2CCSD uses a perturbative CCSD wavefunction as the trial while the
guide (propagation) wavefunction remains at the RHF wavefunction.

workflow on setup AFQMC/pt2CCSD (Manually)
---------------------------------------------------------------------------------
1. Run a PySCF RHF + CCSD calculation to obtain MOs and amplitudes.
2. Stage the guide (RHF) Hamiltonian and the pt2CCSD trial separately.
3. Build operator factories (trial, measurement, propagation).
4. Run the mixed AFQMC via run_mixed_qmc().
5. Post-process the raw block data with clean_pt2ccsd() and blocking analysis.

Note: trot does not yet have a single high-level interface for
pt2CCSD (unlike Afqmc for CISD).  All setup steps are therefore explicit here.
"""

from trot import config

config.configure_once()

from pyscf import cc, gto, scf
from trot.afqmc import AfqmcMixed

a = 2  # intra-dimer bond length (Bohr)
d = 100  # centre-to-centre distance between dimers (Bohr)
na = 2  # atoms per monomer (H2)
nc = 8  # number of monomers
elmt = "H"
unit = "b"  # length unit: Bohr
basis = "sto6g"

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis=basis, unit=unit, verbose=4)

mf = scf.RHF(mol)
mf.kernel()
print(f"RHF  energy: {mf.e_tot:.10f} Ha")

mycc = cc.CCSD(mf)
mycc.kernel()
print(f"CCSD energy: {mycc.e_tot:.10f} Ha")

af = AfqmcMixed(
    mycc,
    guide="rhf",
    trial="pt2ccsd",
    n_walkers=300,
    n_blocks=600,
    seed=7,
    mixed_precision=False,
)

e_rhf, err_rhf = af.kernel()
print(f"AFQMC/pt2CCSD (RHF guide)  energy: {e_rhf:.6f} +/- {err_rhf:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")
