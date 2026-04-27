from trot import config

config.configure_once()

from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from pyscf import cc, gto, scf

from trot import testing
from trot.afqmc import Afqmc
from trot.core.ops import k_energy, k_force_bias
from trot.meas.ucisd import (
    build_meas_ctx,
    energy_kernel_gw_rh,
    energy_kernel_rw_rh,
    energy_kernel_uw_rh,
    force_bias_kernel_gw_rh,
    force_bias_kernel_rw_rh,
    force_bias_kernel_uw_rh,
    make_ucisd_meas_ops,
)
from trot.prop.types import QmcParams
from trot.trial.ucisd import UcisdTrial, make_ucisd_trial_ops


def _make_ucisd_trial(
    key,
    norb: int,
    nup: int,
    ndn: int,
    *,
    memory_mode: Literal["low", "high"] = "low",
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
    scale_ci2: float = 0.02,
) -> UcisdTrial:
    """
    Random UCISD coefficients in the MO basis where the reference occupies
    ([0..nocc[0]-1], [0..nocc[1]-1]).

    We keep coefficients modest in magnitude to reduce catastrophic cancellation
    when comparing against overlap-based finite differences.
    """
    n_oa, n_ob = nup, ndn
    n_va = norb - n_oa
    n_vb = norb - n_ob
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    c1a = scale_ci1 * jax.random.normal(k1, (n_oa, n_va), dtype=dtype)
    c1b = scale_ci1 * jax.random.normal(k2, (n_ob, n_vb), dtype=dtype)
    c2aa = scale_ci2 * jax.random.normal(k3, (n_oa, n_va, n_oa, n_va), dtype=dtype)
    c2ab = scale_ci2 * jax.random.normal(k4, (n_oa, n_va, n_ob, n_vb), dtype=dtype)
    c2bb = scale_ci2 * jax.random.normal(k5, (n_ob, n_vb, n_ob, n_vb), dtype=dtype)

    # Antisymmetry
    c2aa = 0.25 * (
        c2aa
        - jnp.einsum("iajb->jaib", c2aa)
        - jnp.einsum("iajb->ibja", c2aa)
        + jnp.einsum("iajb->jbia", c2aa)
    )
    c2bb = 0.25 * (
        c2bb
        - jnp.einsum("iajb->jaib", c2bb)
        - jnp.einsum("iajb->ibja", c2bb)
        + jnp.einsum("iajb->jbia", c2bb)
    )

    # Impossible
    i, a, j, b = jnp.ogrid[:n_oa, :n_va, :n_oa, :n_va]
    c2aa = jnp.where((i == j) | (a == b), 0.0, c2aa)

    i, a, j, b = jnp.ogrid[:n_ob, :n_vb, :n_ob, :n_vb]
    c2bb = jnp.where((i == j) | (a == b), 0.0, c2bb)

    c_a = jnp.eye(norb, norb)
    c_b = testing.rand_orthonormal_cols(k6, norb, norb, dtype=jnp.float64)

    return UcisdTrial(
        mo_a=c_a,
        mo_b=c_b,
        c1a=c1a,
        c1b=c1b,
        c2aa=c2aa,
        c2ab=c2ab,
        c2bb=c2bb,
    )


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 4, 2, 2, 5),
        ("unrestricted", 4, 2, 1, 5),
        ("generalized", 4, 2, 1, 5),
    ],
)
def test_auto_force_bias_matches_manual_ucisd(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(0)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_ucisd_trial_ops,
        make_meas_ops_fn=make_ucisd_meas_ops,
    )

    meas_manual.require_kernel(k_energy)
    meas_auto.require_kernel(k_energy)
    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(1):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(v_a, v_m, atol=1e-12), (v_a, v_m)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 4, 2, 2, 5),
        ("unrestricted", 6, 3, 2, 8),
        ("generalized", 6, 3, 2, 8),
    ],
)
def test_auto_energy_matches_manual_ucisd(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(0)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_ucisd_trial_ops,
        make_meas_ops_fn=make_ucisd_meas_ops,
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(1):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        e_m = e_manual(wi, ham, ctx_manual, trial)
        e_a = e_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(e_a, e_m, rtol=5e-6, atol=5e-7), (e_a, e_m)


def test_force_bias_equal_when_wr_eq_wu():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_ucisd_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wa, wb = wi
        wi = (wa, wa)
        fbu = force_bias_kernel_uw_rh(wi, ham, ctx, trial)
        fbr = force_bias_kernel_rw_rh(wa, ham, ctx, trial)

        assert jnp.allclose(fbu, fbr, atol=1e-12), (fbu, fbr)


def test_force_bias_equal_when_wg_eq_wu():
    norb = 6
    nup, ndn = 3, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_ucisd_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(tuple, wi)
        fbu = force_bias_kernel_uw_rh(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2 * norb, nup + ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0, 0))
        wi = lax.dynamic_update_slice(wi, wb, (norb, nup))
        fbg = force_bias_kernel_gw_rh(wi, ham, ctx, trial)

        assert jnp.allclose(fbu, fbg, atol=1e-12), (fbu, fbg)


def test_energy_equal_when_wr_eq_wu():
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_ucisd_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wa, wb = wi
        wi = (wa, wa)
        eu = energy_kernel_uw_rh(wi, ham, ctx, trial)
        er = energy_kernel_rw_rh(wa, ham, ctx, trial)

        assert jnp.allclose(eu, er, atol=1e-12), (eu, er)


def test_energy_equal_when_wg_eq_wu():
    norb = 6
    nup, ndn = 3, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        ctx,
    ) = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nup=nup,
            ndn=ndn,
        ),
        make_trial_ops_fn=make_ucisd_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wi = cast(tuple, wi)
        eu = energy_kernel_uw_rh(wi, ham, ctx, trial)
        wa, wb = wi
        wi = jnp.zeros((2 * norb, nup + ndn), dtype=wa.dtype)
        wi = lax.dynamic_update_slice(wi, wa, (0, 0))
        wi = lax.dynamic_update_slice(wi, wb, (norb, nup))
        eg = energy_kernel_gw_rh(wi, ham, ctx, trial)

        assert jnp.allclose(eu, eg, atol=1e-12), (eu, eg)


def mycc():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.UHF(mol).newton()
    mf.kernel()
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    return mycc


def mycc2():
    mol = gto.M(
        atom="""
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        """,
        basis="sto-6g",
        spin=1,
    )
    mf = scf.UHF(mol).newton()
    mf.kernel()
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    return mycc


mycc = mycc()
mycc2 = mycc2()


@pytest.mark.parametrize(
    "mycc, walker_kind, e_ref, err_ref",
    [
        (mycc, "restricted", -75.72869717787768, 0.0002352900295176637),
        (mycc2, "unrestricted", -55.41533781603285, 0.0001071700818560977),
        (mycc2, "generalized", -55.41533781630213, 0.0001071700909047719),
    ],
)
def test_calc_rhf_hamiltonian(mycc, params, walker_kind, e_ref, err_ref):
    myafqmc = Afqmc(mycc)
    myafqmc.params = params
    myafqmc.walker_kind = walker_kind
    myafqmc.mixed_precision = False
    myafqmc.chol_cut = 1e-6
    mean, err = myafqmc.kernel()
    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
