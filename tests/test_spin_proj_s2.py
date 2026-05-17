import pytest

# import dataclasses
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcFp
import trot.spin_proj
import trot.testing

from trot.meas.ucisd import energy_kernel_gw_rh
from trot.trial.ucisd import overlap_g

# from trot.core.ops import k_energy
from trot.spin_proj import make_overlap_u_s2, make_energy_kernel_uw_rh_s2

import jax
import jax.numpy as jnp

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      1.0000000000
    H       -0.2353791634      0.9268076728      1.0000000000
    """,
    basis="6-31g",
)
mf = scf.UHF(mol)
mf.kernel()

for i in range(2):
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)  # type: ignore
mf.stability()

mycc = cc.UCCSD(mf)
mycc.kernel()

af = AfqmcFp(mycc)
af.dt = 0.1
af.n_walkers = 10
af.ene0 = mycc.e_tot
af.seed = 5
af.n_prop_steps = 50
af.n_blocks = 1
af.walker_kind = "unrestricted"
af.n_traj = 10
af.mixed_precision = False
af.build_job()
job = af._job


# @pytest.mark.parametrize(
#    "target_spin, e_ref, err_ref",
#    [
#        (0.0, -75.9947503188, 4.9884686e-03),
#        (2.0, -75.8479635806, 7.6459366e-02),
#    ],
# )
# def test_spin_proj_s2(target_spin, e_ref, err_ref):
#
#    # Spin projection
#    ## Data for the quadrature
#    betas, w_betas = trot.spin_proj.quadrature_s2(
#        target_spin,
#        (job.sys.nup, job.sys.ndn),
#        ngrid=4,
#    )
#
#    ## Overlap and energy with spin projection
#    overlap_u_s2 = make_overlap_u_s2(betas, w_betas, overlap_g)
#    energy_kernel_uw_rh_s2 = make_energy_kernel_uw_rh_s2(
#        betas, w_betas, overlap_g, energy_kernel_gw_rh
#    )
#
#    job.meas_ops = dataclasses.replace(
#        job.meas_ops,
#        overlap=overlap_u_s2,
#        kernels={
#            k_energy: energy_kernel_uw_rh_s2,
#        },
#    )
#
#    # Important to do after changing the energy and overlap kernels if the
#    # function job._prepare_runtime() has been run before. Otherwise the initial
#    # state will not be the right one.
#    job._runtime_prop_ctx = None
#    job._runtime_meas_ctx = None
#    job._runtime_state = None
#    job._prepare_runtime()
#
#    e, err = af.kernel()
#
#    assert abs(e[-1].real - e_ref) < 1e-6, (e[-1].real, e_ref)
#    assert abs(err[-1].real - err_ref) < 1e-6, (err[-1].real, err_ref)


@pytest.mark.parametrize(
    "target_spin",
    [
        (0.0),
        (2.0),
        (4.0),
    ],
)
def test_quadrature(target_spin):
    job._prepare_runtime()
    key = jax.random.key(42)
    wa, wb = trot.testing.make_walkers(key, job.sys)
    w = (wa, wb)

    # Spin projection
    ## Data for the quadrature
    betas, w_betas = trot.spin_proj.quadrature_s2(
        target_spin,
        (job.sys.nup, job.sys.ndn),
        ngrid=10,
    )

    ## Overlap and energy with spin projection
    overlap_u_s2 = make_overlap_u_s2(betas, w_betas, overlap_g)
    energy_kernel_uw_rh_s2 = make_energy_kernel_uw_rh_s2(
        betas, w_betas, overlap_g, energy_kernel_gw_rh
    )

    trial_data = job.trial_data
    meas_ctx = job._runtime_meas_ctx
    ham_data = job.ham_data

    o1 = overlap_u_s2(w, trial_data)
    e1 = energy_kernel_uw_rh_s2(w, ham_data, meas_ctx, trial_data)

    # Spin projection
    ## Brute force
    S = target_spin / 2.0
    Sz = (job.sys.nup - job.sys.ndn) / 2.0

    ngrid = 1000
    betas = jnp.linspace(0, jnp.pi, ngrid, endpoint=False)
    wigner = lambda beta: trot.spin_proj.wigner_small_d(S, Sz, Sz, beta)
    w_betas = jax.vmap(wigner)(betas) * jnp.sin(betas) * (2 * S + 1) / 2.0 * jnp.pi / ngrid

    ## Overlap and energy with spin projection
    overlap_u_s2 = make_overlap_u_s2(betas, w_betas, overlap_g)
    energy_kernel_uw_rh_s2 = make_energy_kernel_uw_rh_s2(
        betas, w_betas, overlap_g, energy_kernel_gw_rh
    )

    o2 = overlap_u_s2(w, trial_data)
    e2 = energy_kernel_uw_rh_s2(w, ham_data, meas_ctx, trial_data)

    assert abs(o1 - o2) < 1e-6, (o1, o2)
    assert abs(e1.real - e2.real) < 1e-5, (e1.real, e2.real)


if __name__ == "__main__":
    pytest.main([__file__])
