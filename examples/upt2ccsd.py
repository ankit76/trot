from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed

a = 1.20577  # O-O bond length (Angstrom)
d = 100  # centre-to-centre distance between the two molecules (Angstrom)
na, nc = 2, 1  # atoms per monomer, number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"O {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto-6g", spin=2 * nc, verbose=3)
mf= scf.UHF(mol).density_fit()  # the fragment integrals come from the DF tensor
mf.kernel()

# follow the UHF solution down to a stable one
for _ in range(5):
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        break
    mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
print(f"UHF energy: {mf.e_tot:.10f}")

# the frozen core of the hamiltonian follows cc.frozen
mycc = cc.CCSD(mf).set_frozen()
mycc.kernel()

# RHF guide, bar estimator
af = AfqmcMixed(
    mycc, 
    guide="uhf", 
    trial="upt2ccsd_bar", 
    n_walkers=300, 
    n_blocks=600, 
    tau_eql=25, 
    seed=7, 
    mixed_precision=False,
    )
e_rhf, err_rhf = af.kernel()
print(f"AFQMC/pt2CCSD (RHF guide)  energy: {e_rhf:.6f} +/- {err_rhf:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")
