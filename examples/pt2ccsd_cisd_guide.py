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
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="6-31g",
    verbose=3,
)

mf = scf.RHF(mol)
mf.kernel()

# the frozen core of the hamiltonian follows cc.frozen
mycc = cc.CCSD(mf, frozen=1)
mycc.kernel()

# RHF guide, bar estimator
af = AfqmcMixed(mycc, guide="rhf", trial="pt2ccsd_bar", n_walkers=200, n_blocks=200, seed=7)
e_rhf, err_rhf = af.kernel()
print(f"AFQMC/pt2CCSD (RHF guide)  energy: {e_rhf:.6f} +/- {err_rhf:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")

# CISD guide, bar estimator: the guide energy is then the CISD one and the trial
# energy the same pt2CCSD estimate with a different constraint
af = AfqmcMixed(mycc, guide="cisd", trial="pt2ccsd_bar", n_walkers=200, n_blocks=200, seed=7)
e_cisd, err_cisd = af.kernel()
print(f"AFQMC/pt2CCSD (CISD guide) energy: {e_cisd:.6f} +/- {err_cisd:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")
