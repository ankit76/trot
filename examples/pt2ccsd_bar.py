"""
Example: AFQMC/pt2CCSD with exp(T1) applied to the right (trial="pt2ccsd_bar")
==============================================================================

The pt2CCSD trial is exp(T1)|HF> plus a perturbative T2. trial="pt2ccsd_bar" moves
exp(T1) off the trial and onto the hamiltonian and the walker:

    exp_t1     = 1 + X,  X[:nocc, nocc:] = t1     (X**2 = 0, so this is exact)
    h1_bar     = exp_t1 @ h1   @ exp_mt1
    chol_bar   = exp_t1 @ chol @ exp_mt1
    walker_bar = exp_t1 @ walker

The trial becomes the bare reference determinant, so the greens function only has nocc
nonzero rows and every chunk intermediate shrinks with it. The cost is chol_bar, a second
copy of the cholesky tensor. The energy is the same as trial="pt2ccsd".

Only the trial side is transformed: the RHF guide still propagates on the bare hamiltonian.

See examples/pt2ccsd_api.py for the plain run and examples/pt2ccsd_chunked.py for how
max_memory sizes the chunking.
"""

from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed

# =============================================================================
# Molecular system: 8 H2 dimers, far enough apart to be non-interacting
# =============================================================================

a = 2  # intra-dimer bond length (Bohr)
d = 100  # centre-to-centre distance between dimers (Bohr)
na, nc = 2, 8  # atoms per monomer, number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"H {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", unit="b", verbose=0)

mf = scf.RHF(mol)
mf.kernel()
print(f"RHF  energy: {mf.e_tot:.10f} Ha")

mycc = cc.CCSD(mf)
mycc.kernel()
print(f"CCSD energy: {mycc.e_tot:.10f} Ha")

# =============================================================================
# AFQMC/pt2CCSD with the bar kernel
# =============================================================================
# The trial name is the whole switch. Other options worth trying:
#
#   max_memory=20          MB per device; sizes the cholesky chunk (and n_chunks if needed)
#   nchol_chunk=8          fix the cholesky chunk size directly
#   mixed_precision=True   single precision for the heavy two-body contractions

af = AfqmcMixed(
    mycc,
    trial="pt2ccsd_bar",
    guide="rhf",
    dt=0.005,
    n_walkers=200,
    n_blocks=200,
    n_eql_blocks=40,
    seed=17,
)
mean, err = af.kernel()

print(f"\nGuide  (AFQMC/RHF)    : {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f} Ha")
print(f"Trial  (AFQMC/pt2CCSD): {mean:.6f} +/- {err:.6f} Ha")
print(f"Reference (CCSD)      : {mycc.e_tot:.6f} Ha")
