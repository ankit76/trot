from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed

# Mixed guide/trial AFQMC with the pt2CCSD estimator on the restricted hamiltonian: the
# walkers propagate under the guide (RHF, or the CC-derived CISD) and the energy is
# measured against the pt2CCSD trial built from the CCSD amplitudes.
#
# trial="pt2ccsd_bar" applies exp(T1) to the right, onto the hamiltonian and the walker,
# so the bra is the bare reference determinant: the same energy, faster, and chunked
# over the cholesky index. The chunk is sized from the device memory by default; pass
# nchol_chunk to set it, or max_memory (MB) to size it against a budget of your own.
#
# mixed_precision is True by default: the T2 contractions of the estimator and the
# propagator's cholesky products run in single precision, the sums in double. Set it to
# False for a fully double precision run.

mol = gto.M(
    atom="""
    C       0.00000000       0.00000000       0.00000000
    N       0.00000000       0.00000000       1.16739000
    """,
    basis="ccpvdz",
    spin=1,
    verbose=3,
)

mf = scf.UHF(mol)
mf.kernel()

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf.kernel(dm0=dm1)
mf.stability()

# the frozen core of the hamiltonian follows cc.frozen
mycc = cc.CCSD(mf).set_frozen()
mycc.kernel()

# RHF guide, bar estimator
af = AfqmcMixed(
    mycc, guide="uhf", trial="upt2ccsd_bar", n_walkers=300, n_blocks=600, tau_eql=25, seed=7
)
e_rhf, err_rhf = af.kernel()
print(f"AFQMC/pt2CCSD (RHF guide)  energy: {e_rhf:.6f} +/- {err_rhf:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")

# CISD guide, bar estimator: the guide energy is then the CISD one and the trial
# energy the same pt2CCSD estimate with a different constraint
af = AfqmcMixed(
    mycc,
    guide="ucisd",
    trial="upt2ccsd_bar",
    n_walkers=300,
    n_blocks=600,
    tau_eql=25,
    # n_eql_blocks=100,
    seed=7,
    mixed_precision=False,
)
e_cisd, err_cisd = af.kernel()
print(f"AFQMC/pt2CCSD (CISD guide) energy: {e_cisd:.6f} +/- {err_cisd:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")
