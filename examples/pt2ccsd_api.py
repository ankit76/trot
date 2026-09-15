"""
Example: AFQMC/pt2CCSD for 8 non-interacting H2 dimers, through the AfqmcMixed driver
=====================================================================================

AFQMC/pt2CCSD is a *mixed* calculation: the walkers propagate under the RHF guide while
the energy is measured against a perturbative CCSD trial.

This is the same calculation as examples/pt2ccsd.py, which sets every piece up by hand.
Keep that one for reference on what the pieces are; use this one to actually run.

Choosing trial="pt2ccsd" (the default) selects the whole pipeline that goes with it --
staging, measurement ops, block function and blocking analysis -- so the trial can never
be paired with the wrong estimator. trot.mixed.available_mixed_recipes() lists what is
registered.
"""

from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed

# =============================================================================
# Molecular system: nc H2 dimers, far enough apart to be non-interacting
# =============================================================================

a = 2  # intra-dimer bond length (Bohr)
d = 100  # centre-to-centre distance between dimers (Bohr)
na = 2  # atoms per monomer (H2)
nc = 8  # number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"H {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", unit="b", verbose=4)

# =============================================================================
# RHF gives the guide and the integrals; CCSD gives the t2 amplitudes
# =============================================================================

mf = scf.RHF(mol)
mf.kernel()
print(f"RHF  energy: {mf.e_tot:.10f} Ha")

mycc = cc.CCSD(mf)
mycc.kernel()
print(f"CCSD energy: {mycc.e_tot:.10f} Ha")

# =============================================================================
# AFQMC/pt2CCSD
# =============================================================================
# The guide is taken from mycc._scf and the trial from mycc itself, so the CC object is
# the only argument needed. kernel() returns the trial (pt2CCSD) energy; the guide
# result is kept on the object alongside it.

af = AfqmcMixed(mycc, dt=0.005, n_walkers=200, n_blocks=200, n_eql_blocks=40, seed=17)
mean, err = af.kernel()

print(f"\nGuide  (AFQMC/RHF)    : {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f} Ha")
print(f"Trial  (AFQMC/pt2CCSD): {mean:.6f} +/- {err:.6f} Ha")
print(f"Reference (CCSD)      : {mycc.e_tot:.6f} Ha")
