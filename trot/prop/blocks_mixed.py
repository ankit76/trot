"""
The block function of mixed guide/trial AFQMC (trot/mixed.py, trot/setup_mixed.py,
trot/driver_mixed.py): the walkers propagate under a guide wavefunction, the energy is
measured against a different trial,

    |AFQMC> = sum_i w_i |phi_i> / <G|phi_i>
    E_T     = sum_i wp_i E_loc^T(phi_i) / sum_i wp_i,   wp_i = w_i <T|phi_i> / <G|phi_i>

The trial's energy kernel returns a fixed number of components per walker (a scalar
counts as one), named by ``components``; they are averaged over the walkers with the
absorbed weight wp and returned as one block scalar "trial_components", a vector in the
recipe's component order (scanned over blocks it becomes (n_blocks, n_components), as
the branch's block_mixed_estimator returns "estimator_components"), next to
"trial_weight" = sum(wp). How the averaged components combine into an energy is the
trial recipe's business (its energy_fn), not this function's. For the pt2CCSD trials the
components are (theta, electronic_0, h_t) and the energy is
h0 + <electronic_0> + <h_t> - <theta><electronic_0>.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy
from ..core.system import System
from .blocks import BlockObs
from .types import PropOps, PropState, QmcParams

PT2_COMPONENTS: tuple[str, ...] = ("theta", "electronic_0", "h_t")


class MixedBlockFn(Protocol):
    def __call__(
        self,
        state: PropState,
        *,
        sys: System,
        params: QmcParams,
        ham_data: Any,
        guide_data: Any,
        guide_ops: TrialOps,
        guide_meas_ops: MeasOps,
        guide_meas_ctx: Any,
        guide_prop_ops: PropOps,
        guide_prop_ctx: Any,
        trial_data: Any,
        trial_meas_ops: MeasOps,
        trial_meas_ctx: Any,
        observable_names: tuple[str, ...] = (),
        sr_fn: Callable = wk.stochastic_reconfiguration,
        measure_trial: bool = True,
        components: tuple[str, ...] = PT2_COMPONENTS,
    ) -> tuple[PropState, BlockObs]: ...


def block_mixed(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_meas_ctx: Any,
    guide_prop_ops: PropOps,
    guide_prop_ctx: Any,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    trial_meas_ctx: Any,
    observable_names: tuple[str, ...] = (),
    sr_fn: Callable = wk.stochastic_reconfiguration,
    measure_trial: bool = True,
    components: tuple[str, ...] = PT2_COMPONENTS,
) -> tuple[PropState, BlockObs]:
    """
    One block: n_prop_steps of guided propagation, then the guide's local energy (for the
    population control and the outlier rejection) and, unless measure_trial=False, the
    trial's components.

    measure_trial=False skips the trial estimator entirely and returns guide scalars
    only. During equilibration the walkers are governed purely by the guide, so the
    trial energy is not used for anything and evaluating it is wasted work. The returned
    BlockObs then has no trial_* keys, so callers must branch on the same flag.
    """

    # propagation is guided with the guiding wavefunction
    step = lambda st: guide_prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=guide_data,
        trial_ops=guide_ops,
        meas_ops=guide_meas_ops,
        prop_ctx=guide_prop_ctx,
        meas_ctx=guide_meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        return step(carry), None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    guide_overlaps = wk.vmap_chunked(
        guide_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, guide_data)
    state = state._replace(walkers=walkers_new, overlaps=guide_overlaps)

    # the local energy with respect to the guide, <G|H|walker>/<G|walker>
    guide_e_kernel = guide_meas_ops.require_kernel(k_energy)
    guide_e_samples = wk.vmap_chunked(
        guide_e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, guide_meas_ctx, guide_data)
    guide_e_samples = jnp.real(guide_e_samples)

    # Outlier rejection, judged on the guide local energy. A walker whose guide energy
    # has run away is discarded by zeroing its weight, not by clamping its energy:
    # clamping would leave it contributing its full weight to the trial averages further
    # down, since trial_weights is built from guide_weights. The energy is still replaced
    # by e_ref so that a nan cannot survive as nan * 0.
    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    is_bad = ~jnp.isfinite(guide_e_samples) | (jnp.abs(guide_e_samples - e_ref) > thresh)
    guide_e_samples = jnp.where(is_bad, e_ref, guide_e_samples)

    guide_weights = jnp.where(is_bad, 0.0, state.weights)
    guide_w_block = jnp.sum(guide_weights)
    guide_e_block = jnp.sum(guide_weights * guide_e_samples) / guide_w_block
    guide_e_block = jnp.where(guide_w_block == 0, e_ref, guide_e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(guide_e_block))
    state = state._replace(
        weights=guide_weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * guide_e_block,
    )

    obs_samples: dict[str, jax.Array] = {}

    if not measure_trial:
        # equilibration: the trial estimator is not used, so do not pay for it
        key, subkey = jax.random.split(state.rng_key)
        zeta = jax.random.uniform(subkey)
        w_sr, weights_sr = sr_fn(state.walkers, state.weights, zeta, sys.walker_kind)
        overlaps_sr = wk.vmap_chunked(
            guide_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
        )(w_sr, guide_data)
        state = state._replace(walkers=w_sr, weights=weights_sr, overlaps=overlaps_sr, rng_key=key)
        return state, BlockObs(
            scalars={"guide_weight": guide_w_block, "guide_energy": guide_e_block},
            observables=obs_samples,
        )

    # measuring with respect to the trial
    trial_e_kernel = trial_meas_ops.require_kernel(k_energy)
    results = wk.vmap_chunked(
        trial_e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, trial_meas_ctx, trial_data)
    # (n_walkers, n_components); a scalar kernel is the one component case
    results = jnp.reshape(results, (wk.n_walkers(state.walkers), -1))
    if results.shape[1] != len(components):
        raise ValueError(
            f"the trial energy kernel returns {results.shape[1]} numbers per walker but the "
            f"recipe names {len(components)} components {components}"
        )
    trial_overlaps = wk.vmap_chunked(
        trial_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, trial_data)
    # wp = w_guide * <T|walker> / <G|walker>
    trial_weights = guide_weights * trial_overlaps / guide_overlaps
    trial_w_block = jnp.sum(trial_weights)
    # (n_components,): the wp-averaged components in the recipe's order
    trial_components = jnp.sum(trial_weights[:, None] * results, axis=0) / trial_w_block

    # stochastic reconfiguration on the guide weights
    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    w_sr, weights_sr = sr_fn(state.walkers, state.weights, zeta, sys.walker_kind)
    overlaps_sr = wk.vmap_chunked(
        guide_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(w_sr, guide_data)
    state = state._replace(walkers=w_sr, weights=weights_sr, overlaps=overlaps_sr, rng_key=key)

    obs = BlockObs(
        scalars={
            "guide_weight": guide_w_block,
            "guide_energy": guide_e_block,
            "trial_weight": trial_w_block,
            "trial_components": trial_components,
        },
        observables=obs_samples,
    )
    return state, obs
