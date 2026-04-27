import dataclasses
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcFp
import trot.spin_proj

mol = gto.M(
    atom="""
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """,
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
af.n_walkers = 2000
af.ene0 = mycc.e_tot
af.seed = 5
af.n_prop_steps = 20
af.n_blocks = 5
af.walker_kind = "unrestricted"
af.mixed_precision = False
af.build_job()

job = af._job

from trot.meas.ucisd import energy_kernel_gw_rh
from trot.trial.ucisd import overlap_g
from trot.core.ops import k_energy
from trot.spin_proj import make_overlap_u_s2, make_energy_kernel_uw_rh_s2

# Spin projection
## Data for the quadrature
target_spin = 0.0
betas, w_betas = trot.spin_proj.quadrature_s2(
    target_spin,
    (job.sys.nup, job.sys.ndn),
    4,
)

## Overlap and energy with spin projection
overlap_u_s2 = make_overlap_u_s2(betas, w_betas, overlap_g)
energy_kernel_uw_rh_s2 = make_energy_kernel_uw_rh_s2(betas, w_betas, overlap_g, energy_kernel_gw_rh)

job.meas_ops = dataclasses.replace(
    job.meas_ops,
    overlap=overlap_u_s2,
    kernels={
        k_energy: energy_kernel_uw_rh_s2,
    },
)

af.kernel()
