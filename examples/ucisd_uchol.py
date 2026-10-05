from pyscf import cc, gto, scf

from trot.afqmc import AfqmcUh

# AFQMC with an unrestricted hamiltonian and the UCISD trial built from UCCSD amplitudes.
# Alpha and beta each keep their own MO basis, so the CC amplitudes are used exactly in
# the basis they were computed in; no rotation of the beta reference into the alpha
# basis is involved. Contrast with examples/ucisd.py on the alpha-basis hamiltonian.

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

# the frozen core of the hamiltonian follows cc.frozen
mycc = cc.UCCSD(mf, frozen=1)
mycc.kernel()

af = AfqmcUh(mycc, n_walkers=100, n_blocks=100, seed=7)
mean, err = af.kernel()


print(f"AFQMC/UCISD (uham) energy: {mean:.6f} +/- {err:.6f}")
