from pyscf import cc, gto, scf
from trot.afqmc import AfqmcFp
import pickle

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

mycc = cc.UCCSD(mf)
mycc.kernel()

af = AfqmcFp(mycc)
af.save_staged("af.h5")

start = 5
n = 3

# Simulation of independent fp calculations (must use the same staged data)
for seed in range(start, start + n * af.n_traj, af.n_traj):
    af = AfqmcFp.from_staged("af.h5")
    af.n_walkers = 20
    af.ene0 = mycc.e_tot
    af.n_traj = 10
    af.n_blocks = 10
    af.walker_kind = "unrestricted"
    af.seed = seed
    af.kernel()

    # Results
    with open("qmc_result_" + str(af.seed) + ".pkl", "wb") as f:
        pickle.dump(af.qmc_result, f)

# Equivalent sequential calculation for comparison
af = AfqmcFp.from_staged("af.h5")
af.n_walkers = 20
af.ene0 = mycc.e_tot
af.n_traj = af.n_traj * n
af.n_blocks = 10
af.walker_kind = "unrestricted"
af.seed = start
e_tot, e_err = af.kernel()

# Comparison with the previous sequential results
import glob
import pickle
import jax

# Not needed here but must be added if AfqmcFp is not imported
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

files = sorted(glob.glob("qmc_result_*.pkl"))

l_e = []
l_w = []
l_s = []
print("\nReading...")
for file in files:
    print(file)
    with open(file, "rb") as f:
        qmc_result = pickle.load(f)

        l_e.append(qmc_result.block_energies)
        l_w.append(qmc_result.block_weights)
        l_s.append(qmc_result.block_observables["sign"])

e = jnp.concatenate(l_e, axis=0)
w = jnp.concatenate(l_w, axis=0)
s = jnp.concatenate(l_s, axis=0)

mean = jnp.sum(e * w, axis=0) / jnp.sum(w, axis=0)
n = e.shape[0]
err = jnp.std(e, axis=0) / jnp.sqrt(n - 1)
sign = jnp.sum(s * w, axis=0) / jnp.sum(w, axis=0)

# Must be identical to the sequential results
for e, error, s in zip(mean, err, sign):
    print(f"{e.real:14.10f} {error.real:13.7e} {s.real:6.2f}")

assert jnp.allclose(mean, e_tot, rtol=0.0)
assert jnp.allclose(err, e_err, rtol=0.0)
