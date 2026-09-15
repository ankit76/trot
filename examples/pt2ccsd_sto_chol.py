"""
Example: AFQMC/pt2CCSD with a semistochastic cholesky sum (trial="pt2ccsd_sto_chol")
===================================================================================

trial="pt2ccsd_sto_chol" is the bar estimator (see examples/pt2ccsd_bar.py) with the
expensive half of the two-body energy sampled instead of summed over every cholesky
vector.

  exact     e2_0 (and e2_2_1 = e2_0 * gt2g), which only needs green . chol
  sampled   e2_2_2_1, e2_2_2_2, e2_2_3 -- the nocc^2 nvir^2 per vector "iajb" terms that
            dominate the cost. They are split into an exactly summed head of cholesky
            vectors and an importance sampled tail.

The proposal is pi_g ~ |e2_0_g| with a uniform floor so every pi_g > 0, and tail draws
carry weight 1 / (n_samples * pi_g), so the estimator is unbiased. The sampling noise
adds to the error bar of the walk.
"""

from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed

# =============================================================================
# Molecular system: a hydrogen chain, big enough that sampling has something to sample
# =============================================================================

mol = gto.M(
    atom="; ".join(f"H 0 0 {1.6 * i}" for i in range(10)),
    basis="631g",
    unit="b",
    verbose=0,
)

mf = scf.RHF(mol)
mf.kernel()
print(f"RHF  energy: {mf.e_tot:.10f} Ha")

mycc = cc.CCSD(mf)
mycc.kernel()
print(f"CCSD energy: {mycc.e_tot:.10f} Ha")

# =============================================================================
# AFQMC/pt2CCSD with the semistochastic cholesky sum
# =============================================================================
# The sampling is controlled through trial_kwargs. With none given, each walker uses 20% of
# the cholesky vectors (chol_cost_ratio=0.2), split head : samples = 3 : 1. The head and
# sample counts this resolves to are printed in the trial_meas_cfg dump as
# n_chol_head_used and n_chol_samples_used. The knobs to explore:
#
#   n_chol_head=16              head size outright; "full" puts every vector in the
#                               head, which turns sampling off and reproduces
#                               trial="pt2ccsd_bar" exactly
#   head_chol_ratio=0.25        head as a fraction of nchol (when n_chol_head is unset)
#   n_chol_samples=64           tail draws per walker per block
#   chol_cost_ratio=0.25        per-walker budget C = ratio * nchol (default 0.2), split
#   head_sample_ratio=3.0         head : samples = head_sample_ratio : 1 (default 3:1)
#
# For example:
#
#   trial_kwargs={"n_chol_samples": 64}
#   trial_kwargs={"chol_cost_ratio": 0.25}
#   trial_kwargs={"n_chol_head": "full"}

af = AfqmcMixed(
    mycc,
    trial="pt2ccsd_sto_chol",
    guide="rhf",
    # trial_kwargs={"n_chol_samples": 64},
    dt=0.005,
    n_prop_steps=50,
    n_walkers=200,
    n_blocks=200,
    n_eql_blocks=10,
    seed=7,
)
mean, err = af.kernel()

print(f"\nGuide  (AFQMC/RHF)    : {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f} Ha")
print(f"Trial  (AFQMC/pt2CCSD): {mean:.6f} +/- {err:.6f} Ha")
print(f"Reference (CCSD)      : {mycc.e_tot:.6f} Ha")
