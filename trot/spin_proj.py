from typing import Any, Callable

import jax
import jax.numpy as jnp


def apply_unitary(wi_a: jax.Array, wi_b: jax.Array, U: jax.Array) -> jax.Array:
    """
    Builds a GHF walker from a rotated UHF walker.

    Input:
    wi_a: alpha part of the UHF walker
    wi_b: beta  part of the UHF walker
    U   : Unitary matrix

    Output:
    wi_g: GHF walker
    """
    A, B = wi_a * U[0, 0], wi_b * U[0, 1]
    C, D = wi_a * U[1, 0], wi_b * U[1, 1]

    wi_g = jnp.block([[A, B], [C, D]])
    return wi_g


def build_unitary(betas: jax.Array) -> jax.Array:
    """
    Builds 2 by 2 unitary matrices.

    Input:
    betas: rotation angles

    Output:
    U: Unitary matrices corresponding to the different rotation angles
    """
    U = jax.vmap(
        lambda beta: jnp.array(
            [
                [jnp.cos(beta / 2), jnp.sin(beta / 2)],
                [-jnp.sin(beta / 2), jnp.cos(beta / 2)],
            ]
        )
    )(betas)
    return U


def quadrature_s2(
    target_spin: float, nelec: tuple[int, int], ngrid=4
) -> tuple[jax.Array, jax.Array]:
    """
    Returns grid points and their associated weight for the quadrature.

    Input:
    target_spin: 2S of the target state
    nelec      : number of alpha and beta electrons
    ngrid      : number of grid points for the quadrature

    Output:
    betas  : grid points
    w_betas: weights associated with each grid point
    """
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(ngrid)
    betas = jnp.arccos(x)

    S = 0.5 * target_spin
    na, nb = nelec
    assert na >= nb
    Sz = 0.5 * (na - nb)

    wigner = lambda beta: wigner_small_d(S, Sz, Sz, beta)

    w_betas = jax.vmap(wigner)(betas) * w * (2.0 * S + 1.0) * 0.5

    return betas, w_betas


def make_overlap_u_s2(
    betas: jax.Array, w_betas: jax.Array, overlap_g: Callable[[jax.Array, Any], jax.Array]
) -> Callable[[tuple[jax.Array, jax.Array], Any], jax.Array]:
    """
    Returns a function to compute the overlap using S^2 spin projection

    Input:
    betas    : quadrature points
    w_betas  : weights associated with each quadrature point
    overlap_g: function to compute the overlap using a GHF walker

    Output:
    overlap_u_s2: function to compute the overlap using S^2 spin projection
    """

    U = build_unitary(betas)

    def overlap_u_s2(walker: Any, trial_data: Any):
        wa, wb = walker

        def _wrapper(U_i):
            wg = apply_unitary(wa, wb, U_i)
            o_i = overlap_g(wg, trial_data)
            return o_i

        o = jax.vmap(_wrapper, (0,))(U)

        return jnp.sum(o * w_betas)

    return overlap_u_s2


def make_energy_kernel_uw_rh_s2(
    betas: jax.Array,
    w_betas: jax.Array,
    overlap_g: Callable[[jax.Array, Any], jax.Array],
    energy_kernel_gw_rh: Callable[[jax.Array, Any, Any, Any], jax.Array],
) -> Callable[[tuple[jax.Array, jax.Array], Any, Any, Any], jax.Array]:
    """
    Returns a function to compute the local energy using S^2 spin projection

    Input:
    betas    : quadrature points
    w_betas  : weights associated with each quadrature point
    overlap_g: function to compute the overlap using a GHF walker
    energy_kernel_gw_rh: function to compute the local energy using a GHF walker
    Output:
    energy_kernel_uw_rh_s2: function to compute the local energy using S^2 spin projection
    """

    U = build_unitary(betas)

    def energy_kernel_uw_rh_s2(walker, ham_data, meas_ctx, trial_data):
        wa, wb = walker

        def _wrapper(U_i):
            wg = apply_unitary(wa, wb, U_i)
            e_i = energy_kernel_gw_rh(wg, ham_data, meas_ctx, trial_data)
            o_i = overlap_g(wg, trial_data)
            return o_i, e_i

        o, e = jax.vmap(_wrapper, (0,))(U)

        return jnp.sum(e * o * w_betas) / jnp.sum(o * w_betas)

    return energy_kernel_uw_rh_s2


def wigner_small_d(j: float, mp: float, m: float, beta: float) -> jax.Array:
    """
    Compute small Wigner d-matrix element d_{m, mp}^j(beta).
    """
    if abs(m) > j or abs(mp) > j:
        return jnp.array(0.0)

    from jax.scipy.special import gamma

    prefactor = jnp.sqrt(
        gamma(j + m + 1).astype(int)
        * gamma(j - m + 1).astype(int)
        * gamma(j + mp + 1).astype(int)
        * gamma(j - mp + 1).astype(int)
    )

    k_min = jnp.maximum(0, m - mp).astype(int)
    k_max = jnp.minimum(j + m, j - mp).astype(int)

    def wigner_summation(carry, k):
        numerator = (-1) ** (k - m + mp)
        denominator = (
            gamma(k + 1)
            * gamma(j + m - k + 1).astype(int)
            * gamma(j - mp - k + 1).astype(int)
            * gamma(mp - m + k + 1).astype(int)
        )

        cos_term = jnp.cos(beta / 2) ** (2 * j + m - mp - 2 * k)
        sin_term = jnp.sin(beta / 2) ** (mp - m + 2 * k)

        carry += numerator / denominator * cos_term * sin_term
        return carry, None

    sum_term, _ = jax.lax.scan(wigner_summation, 0.0, jnp.arange(k_min, k_max + 1))

    d = prefactor * sum_term

    return d
