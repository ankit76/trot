import dataclasses
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcFp
import trot.trial.uccsd

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
af.n_walkers = 20
af.ene0 = mycc.e_tot
af.seed = 5
af.n_blocks = 5
af.walker_kind = "unrestricted"
af.build_job()

job = af._job

trial_coeff = (
    job.staged.trial.data["mo_a"],
    job.staged.trial.data["mo_b"],
)

# Replace the function creating the initial state
job.prop_ops = dataclasses.replace(
    job.prop_ops,
    init_prop_state=trot.trial.uccsd.make_init_prop_state(
        trial_coeff,
        mycc.t1,
        mycc.t2,
    ),
)

af.kernel()
