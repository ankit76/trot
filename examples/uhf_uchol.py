from pyscf import gto, scf

from trot.afqmc import AfqmcUh

# AFQMC with an unrestricted hamiltonian: alpha and beta each keep their own orbital
# basis, so h1 and the cholesky vectors are carried per spin. Contrast with examples/
# uhf.py, which uses unrestricted walkers against a hamiltonian built in the alpha MO
# basis alone.
#
# The cholesky vectors come from the density fitting tensor when mf carries one
# (mf.density_fit()), otherwise from the modified cholesky decomposition of the AO ERIs.

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

af = AfqmcUh(mf)
mean, err = af.kernel()
