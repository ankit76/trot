"""
Example: AFQMC/pt2CCSD for an open-shell molecule, through the AfqmcMixed driver
================================================================================

The unrestricted counterpart of examples/pt2ccsd_api.py. The walkers propagate under a
UHF guide on the unrestricted hamiltonian of AfqmcUh -- alpha and beta each in their own
MO basis -- and the energy is measured against a perturbative UCCSD trial.

Passing a UCCSD object is what selects the unrestricted family. The trial name then picks
the energy kernel, and all three compute the same energy:

  "upt2ccsd"           chunked over the cholesky index (the default for a UCCSD object)
  "upt2ccsd_bar"       exp(T1) moved onto the hamiltonian and the walker
  "upt2ccsd_sto_chol"  the bar kernel with a semistochastic cholesky sum

See examples/pt2ccsd_bar.py and examples/pt2ccsd_sto_chol.py for what the bar and
sto_chol kernels trade; the unrestricted ones trade the same things.
"""

from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed

# =============================================================================
# Molecular system: nc triplet O2 monomers, far enough apart to be non-interacting
# =============================================================================

a = 1.20577  # bond length in a monomer (Angstrom)
d = 100  # distance between monomers (Angstrom)
na = 2  # atoms per monomer
nc = 1  # number of monomers
spin = 2  # spin per monomer

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"O {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="ccpvdz", unit="A", spin=spin * nc, verbose=0)

# =============================================================================
# UHF, made stable before it is correlated; UCCSD with the chemical core frozen
# =============================================================================

mf = scf.UHF(mol)
mf.kernel()

stable = False
while not stable:
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if not stable:
        mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
print(f"UHF   energy: {mf.e_tot:.10f} Ha")

mycc = cc.CCSD(mf)  # a UHF reference makes this a UCCSD object
mycc.set_frozen()
mycc.kernel()
print(f"UCCSD energy: {mycc.e_tot:.10f} Ha")

# =============================================================================
# AFQMC/upt2CCSD
# =============================================================================
# The frozen core follows mycc.frozen. Other options worth trying:
#
#   trial="upt2ccsd"                 the chunked kernel with the full green
#   trial="upt2ccsd_sto_chol"        sample the T2-contracted two-body sum; by default each
#                                    walker uses 20% of the cholesky vectors, split
#                                    head : samples = 3 : 1. Tune it with trial_kwargs:
#       trial_kwargs={"chol_cost_ratio": 0.25}
#       trial_kwargs={"n_chol_head": 16, "n_chol_samples": 32}
#       trial_kwargs={"n_chol_head": "full"}   # no sampling; reproduces upt2ccsd_bar
#   max_memory=20                    MB per device; sizes the cholesky chunk
#   nchol_chunk=8                    fix the cholesky chunk size directly
#   mixed_precision=True             single precision for the heavy contractions
#   basis_a=c_a, basis_b=c_b         the alpha and beta bases, as in AfqmcUh; the UCCSD
#                                    amplitudes must be expressed in them

af = AfqmcMixed(
    mycc,
    trial="upt2ccsd_bar",
    dt=0.005,
    n_walkers=300,
    n_blocks=300,
    n_eql_blocks=80,
    seed=17,
    mixed_precision=False,
)
mean, err = af.kernel()

print(f"\nGuide  (AFQMC/UHF)     : {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f} Ha")
print(f"Trial  (AFQMC/upt2CCSD): {mean:.6f} +/- {err:.6f} Ha")
print(f"Reference (UCCSD)      : {mycc.e_tot:.6f} Ha")

af = AfqmcMixed(
    mycc,
    trial="upt2ccsd",
    dt=0.005,
    n_walkers=300,
    n_blocks=300,
    n_eql_blocks=80,
    seed=17,
    mixed_precision=False,
)
mean, err = af.kernel()

print(f"\nGuide  (AFQMC/UHF)     : {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f} Ha")
print(f"Trial  (AFQMC/upt2CCSD): {mean:.6f} +/- {err:.6f} Ha")
print(f"Reference (UCCSD)      : {mycc.e_tot:.6f} Ha")
