from pyscf import gto, scf

from trot.afqmc import AfqmcUh

# AFQMC with an unrestricted hamiltonian: alpha and beta each keep their own orbital
# basis, so h1 and the cholesky vectors are carried per spin and the two orbital spaces
# may differ. Contrast with examples/uhf.py, which uses unrestricted walkers against a
# hamiltonian built in the alpha MO basis alone.
#
# The AO ERIs are cholesky decomposed once and the vectors are projected into the alpha
# and beta MO bases, so the auxiliary field index is shared between the spins.

mol = gto.M(
    atom="""
    N  -1.67119571   -1.44021737    0.00000000
    H  -2.12619571   -0.65213425    0.00000000
    H  -0.76119571   -1.44021737    0.00000000
    """,
    spin=1,
    basis="6-31g",
    verbose=3,
)

mf = scf.UHF(mol)
mf.kernel()

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

# norb_frozen takes an int, or a pair (n_core_a, n_core_b)
af = AfqmcUh(mf, norb_frozen=1, n_walkers=100, n_blocks=100, seed=7)
mean, err = af.kernel()
print(f"AFQMC/UHF (uchol) energy: {mean:.6f} +/- {err:.6f}")
