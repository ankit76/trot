"""
Unrestricted LNO-AFQMC/pt2CCSD with trot.lnoafqmc
=================================================

Two well separated triplet O2 molecules (UHF, sto-6g, density fitted), one IAO fragment
per atom. A UHF mean field selects the unrestricted trial, trial="upt2ccsd": each
fragment runs on the uchol fragment hamiltonian (alpha and beta each in their own LNO
basis, one joint cholesky index from trot.cholesky.joint_df2chol), with unrestricted
walkers under the UHF guide (or guide="ucisd", the fragment UCISD), and the fragment
energy measured against the unrestricted pt2CCSD trial on the similarity transformed
hamiltonian (the exp(T1) "bar" estimator, chunked over the cholesky index).

With a tight lno_thresh each fragment's local active space is the whole active space, so
the fragment energies add up to the full-space AfqmcMixed(trial="upt2ccsd_bar") result:
exactly at tau = 0 (the UCCSD energy), within error bars after sampling. With a looser
threshold the alpha and beta active spaces generally differ in size; lno_size and
lno_nocc are then (alpha, beta) pairs per fragment.

Run from the repository root:  python examples/lnoafqmc_upt2ccsd.py
"""

# importing trot.lnoafqmc before jax initialises selects the allocator that hands device
# memory back between fragments (trot/lnoafqmc/__init__.py); it must come before trot.afqmc
import trot.lnoafqmc  # noqa: F401
from pyscf import cc, gto, mp, scf
from pyscf.data.elements import chemcore

from trot.afqmc import AfqmcMixed
from trot.lnoafqmc import LnoAfqmcMixed, iao_fragment

a = 1.20577  # O-O bond length (Angstrom)
d = 100  # centre-to-centre distance between the two molecules (Angstrom)
na, nc = 2, 2  # atoms per monomer, number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"O {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto-6g", spin=2 * nc, verbose=3)
mf = scf.UHF(mol).density_fit()  # the fragment integrals come from the DF tensor
mf.kernel()

# follow the UHF solution down to a stable one
for _ in range(5):
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        break
    mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
print(f"UHF energy: {mf.e_tot:.10f}")

nfrozen = int(chemcore(mol))

# for a UHF mean field lo_coeff is an (alpha, beta) pair and each fragment a pair of LO lists
lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="atom")

lno = LnoAfqmcMixed(
    mf,
    lo_coeff,
    frag_list,
    frag_name=frag_name,
    run_frag=None,
    lno_thresh=1e-5,
    nfrozen=nfrozen,
    trial="upt2ccsd",  # the default for a UHF mf; guide=None -> the UHF guide, or guide="ucisd"
    target_error=1e-5,
    n_walkers=300,
    n_eql_blocks=80,
    n_blocks=400,
    dt=0.005,
    seed=27,
    mixed_precision=False,
    # frag_output="./fragment.out",
    lno_output="./lno_result.out",
)
e_qmc, e_qmc_err = lno.kernel()

mymp = mp.UMP2(mf, frozen=nfrozen)
mymp.kernel()
mycc = cc.UCCSD(mf, frozen=nfrozen)
mycc.kernel()

print(f"\nLNO-MP2   E_corr = {lno.e_mp:.8f}     UMP2  E_corr = {mymp.e_corr:.8f}")
print(f"LNO-CCSD  E_corr = {lno.e_cc:.8f}     UCCSD E_corr = {mycc.e_corr:.8f}")
print(f"LNO-AFQMC E_corr = {e_qmc:.6f} +/- {e_qmc_err:.6f}")
print("per fragment:", lno.lno_eqmc, lno.lno_eqmc_err, "sizes (alpha, beta)", lno.lno_size)

# the full-space AFQMC with the same trial (bar estimator), for comparison
af = AfqmcMixed(
    mycc,
    trial="upt2ccsd_bar",
    n_walkers=300,
    n_eql_blocks=80,
    n_blocks=400,
    seed=27,
    mixed_precision=False,
)
e_ref, err_ref = af.kernel()
print(f"\nAFQMC/upt2CCSD E_corr (full space) = {e_ref - mf.e_tot:.6f} +/- {err_ref:.6f}")
