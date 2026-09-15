"""
Example: sizing the AFQMC/pt2CCSD energy kernel against a memory budget
=======================================================================

The pt2CCSD energy estimator has a second kernel, energy_kernel_rw_rh_chunk, selected
with trial="pt2ccsd_chunk", that differs from the default one in two ways:

  * it walks the cholesky tensor in chunks of nchol_chunk vectors per scan step, rather
    than one vector at a time
  * it carries the heavy two-body contractions in single precision when asked, and
    accumulates every partial sum back in double

Both are memory/speed trades, and neither is a number anyone should have to guess. Pass
``max_memory`` (MB, per device, as in pyscf) and the estimator's memory model sizes the
chunking for you.

There are two knobs it can turn, and it turns them in this order:

  nchol_chunk   cholesky vectors per scan step -- gives way first
  n_chunks      walker chunks in vmap_chunked  -- rises only as a last resort

Both multiply the same dominant term, so spending the budget on one or the other buys
the same arithmetic. The difference is that walkers in flight *also* batch the work no
cholesky chunk touches -- the greens function, the determinant, the one-body amplitude
contractions -- so the walkers are the last thing to give. Only when a single cholesky
vector per step still does not fit do walkers leave flight, and the cholesky chunk is
then chosen again against the smaller walker count.

See examples/pt2ccsd_api.py for the plain run this builds on.
"""

from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.meas.pt2ccsd import plan_pt2ccsd_chunking

MB = 1024**2

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
mycc = cc.CCSD(mf)
mycc.kernel()
print(f"RHF  energy: {mf.e_tot:.10f} Ha")
print(f"CCSD energy: {mycc.e_tot:.10f} Ha")

# =============================================================================
# Set a budget and let it choose the chunking
# =============================================================================
# The trial name selects the chunked kernel; max_memory then sizes it. build_job()
# stages and plans without running, which is the cheap way to see what a budget buys.

af = AfqmcMixed(
    mycc,
    trial="pt2ccsd_chunk",
    guide="rhf",
    max_memory=10,  # MB, per device
    dt=0.005,
    n_walkers=200,
    n_blocks=200,
    n_eql_blocks=40,
    seed=17,
    mixed_precision=False,
)

job = af.build_job()
plan = job.chunk_plan
print(f"\nplan: {plan.describe()}")

# =============================================================================
# Where that number came from
# =============================================================================
# The model splits the measurement into four terms by how each one scales. With w
# walkers in flight and a chunk of k cholesky vectors:
#
#     bytes(w, k) = resident + w*per_walker + w*k*per_walker_chol
#
# resident         the cholesky tensor and its padded copy, the t2 amplitudes, the
#                  whole walker population -- no chunking touches any of it. Nothing
#                  shared scales with k: the scan slice is a view into the padded copy,
#                  and _equal_chunks keeps that copy's padding bounded whatever k is
# per_walker       walker, green, greenp, t2_green and its halves -- the per-walker
#                  work that only n_chunks can reduce
# per_walker_chol  gl_c, lt2g_c, glgp_c, lt2_1, lt2_2 and their mixed-precision casts
#                  -- the term both knobs divide, and the one that dominates at scale

model = plan.model
print("\nmemory model")
print(f"  resident         {model.resident / MB:9.3f} MB")
print(f"  per_walker       {model.per_walker / MB:9.3f} MB  x w")
print(f"  per_walker_chol  {model.per_walker_chol / MB:9.3f} MB  x w*k")
print(f"  -> at w={plan.walkers_in_flight}, k={plan.nchol_chunk}: {plan.bytes_used / MB:.3f} MB")

# =============================================================================
# What other budgets would have given
# =============================================================================
# plan_pt2ccsd_chunking is the whole policy, and it is pure arithmetic -- no staging, no
# jax. Reuse the model to see the ladder: the cholesky chunk shrinks first, and only
# once it hits 1 do the walkers start leaving flight.

# nchol = int(job.ham_data.nchol)
# n_walkers = int(job.params.n_walkers)

# print("\nthe same system at other budgets")
# for budget_mb in (100, 20, 8, 2, 1, 0.7, 0.55):
#     try:
#         p = plan_pt2ccsd_chunking(
#             model,
#             n_walkers=n_walkers,
#             nchol=nchol,
#             budget_bytes=int(budget_mb * MB),
#         )
#         print(f"  {budget_mb:>6} MB -> {p.describe()}")
#     except ValueError as exc:
#         print(f"  {budget_mb:>6} MB -> refused: {exc}")

# =============================================================================
# Run it
# =============================================================================

mean, err = af.kernel()
print(f"\nGuide  (AFQMC/RHF)    : {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f} Ha")
print(f"Trial  (AFQMC/pt2CCSD): {mean:.6f} +/- {err:.6f} Ha")


af = AfqmcMixed(
    mycc, trial="pt2ccsd", dt=0.005, n_walkers=200, n_blocks=200, n_eql_blocks=40, seed=17
)
mean, err = af.kernel()

print(f"\nGuide  (AFQMC/RHF)    : {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f} Ha")
print(f"Trial  (AFQMC/pt2CCSD): {mean:.6f} +/- {err:.6f} Ha")
print(f"Reference (CCSD)      : {mycc.e_tot:.6f} Ha")

# =============================================================================
# Overriding the plan
# =============================================================================
# Everything above is a default worth overriding when you know better:
#
#   nchol_chunk=64            fix the cholesky chunk; the budget then only decides
#                             n_chunks around it
#   n_chunks=4                a floor, never lowered -- the plan can raise it, but a
#                             value you set for reasons the model cannot see survives
#   mixed_precision           single precision for the run: the estimator's two-body
#                             contractions and the guide propagator
#   trial="pt2ccsd_bar"       the other chunking kernel, with exp(T1) on the hamiltonian
#
# trial="pt2ccsd_chunk" with no max_memory takes the default chunk size; trial="pt2ccsd"
# is the original unchunked kernel, for which a budget has nothing to size.

# af_manual = AfqmcMixed(
#     mycc,
#     trial="pt2ccsd_chunk",
#     max_memory=8,
#     nchol_chunk=8,
#     mixed_precision=True,
#     dt=0.005,
#     n_walkers=200,
#     n_blocks=50,
#     n_eql_blocks=10,
#     seed=17,
# )
# plan_manual = af_manual.build_job().chunk_plan
# print(f"\nfixed chunk, single precision: {plan_manual.describe()}")

# mean_mp, err_mp = af_manual.kernel()
# print(f"Trial  (single precision) : {mean_mp:.6f} +/- {err_mp:.6f} Ha")
# print(f"difference from double    : {abs(mean_mp - mean):.2e} Ha")
