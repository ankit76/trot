"""
The driver of mixed guide/trial AFQMC: equilibration and sampling blocks of
prop/blocks_mixed.block_mixed, the trial's components accumulated per block and combined
into an energy at the end.

    |AFQMC> = sum_i w_i |phi_i> / <G|phi_i>
    E_T     = energy_fn(h0, <c>),   <c> = sum_b W_b c_b / sum_b W_b

with W_b = sum_i wp_i the block's absorbed weight and c_b the wp-averaged components. The
statistics are the branch's own component analyses (stat_utils): the robust outlier mask
on per block proxy energies, and the blocking / Gamma jackknife on the ratio of component
sums, which keeps the covariance between the components in the nonlinear combination.
The final error follows params.error_method; both analyses are reported.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from . import walkers as wk
from .core.ops import MeasOps, TrialOps, k_energy
from .core.system import System
from .driver import (
    EnergyErrorAnalysis,
    _analyze_component_estimator_errors,
    _analyze_energy_errors,
    _print_energy_error_analysis,
)
from .prop.blocks_mixed import PT2_COMPONENTS, MixedBlockFn
from .prop.types import PropOps, PropState, QmcParams
from .stat_utils import (
    blocking_analysis_components,
    blocking_analysis_ratio,
    component_estimator_outlier_mask,
    reject_outliers,
)
from .walkers import stochastic_reconfiguration


class MixedQmcResult(NamedTuple):
    """
    The result of a mixed guide/trial run.

    The trial's block data is the absorbed block weight sum(wp) and the wp-averaged
    components of its energy kernel, keyed by the recipe's component names; the recipe's
    energy_fn combines them. trial_block_keep_mask records the outlier cleanup used for
    the reported trial energy; the block arrays are raw.
    """

    guide_mean_energy: float
    guide_stderr_energy: float
    guide_block_energies: jax.Array
    guide_block_weights: jax.Array
    trial_mean_energy: float
    trial_stderr_energy: float
    trial_mean_components: np.ndarray
    trial_component_names: tuple[str, ...]
    trial_block_weights: jax.Array
    trial_block_components: dict[str, jax.Array]
    trial_block_proxy_energies: np.ndarray
    trial_block_keep_mask: np.ndarray
    guide_analysis: EnergyErrorAnalysis
    trial_analysis: EnergyErrorAnalysis
    final_state: PropState


def make_run_mixed_blocks(
    *,
    mixed_block_fn: MixedBlockFn,
    sys: System,
    params: QmcParams,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_prop_ops: PropOps,
    trial_meas_ops: MeasOps,
    observable_names: tuple[str, ...] = (),
) -> Callable:
    """
    A jitted scan of mixed_block_fn over n_blocks. ham_data, the trial data and the
    contexts stay arguments, since these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks", "measure_trial"))
    def run_mixed_blocks(
        state0,
        *,
        ham_data,
        guide_data,
        guide_meas_ctx,
        guide_prop_ctx,
        trial_data,
        trial_meas_ctx,
        n_blocks: int,
        measure_trial: bool = True,
    ):
        def one_block(state, _):
            state, obs = mixed_block_fn(
                state,
                sys=sys,
                params=params,
                ham_data=ham_data,
                guide_data=guide_data,
                guide_ops=guide_ops,
                guide_meas_ops=guide_meas_ops,
                guide_meas_ctx=guide_meas_ctx,
                guide_prop_ops=guide_prop_ops,
                guide_prop_ctx=guide_prop_ctx,
                trial_data=trial_data,
                trial_meas_ops=trial_meas_ops,
                trial_meas_ctx=trial_meas_ctx,
                observable_names=observable_names,
                measure_trial=measure_trial,
            )
            obs_tuple = tuple(obs.observables[name] for name in observable_names)
            return state, (obs.scalars, obs_tuple)

        stateN, (scalars, obs) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, scalars, obs

    return run_mixed_blocks


def _init_trial_energy(
    state: PropState,
    ham_data: Any,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    trial_meas_ctx: Any,
    params: QmcParams,
    components: tuple[str, ...],
    energy_fn: Callable[[Any, Any], Any],
) -> tuple[jax.Array, jax.Array]:
    """
    The tau = 0 row: the trial energy of the initial population, sum_i wp_i c_i / sum_i wp_i
    per component and energy_fn on top, and the absorbed weight sum_i wp_i with
    wp_i = w_i <T|phi_i> / <G|phi_i>.
    """
    kernel = trial_meas_ops.require_kernel(k_energy)
    n = wk.n_walkers(state.walkers)
    out = wk.vmap_chunked(kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
        state.walkers, ham_data, trial_meas_ctx, trial_data
    )
    out = jnp.reshape(out, (n, -1))
    if out.shape[1] != len(components):
        raise ValueError(
            f"the trial energy kernel returns {out.shape[1]} numbers per walker but the "
            f"recipe names {len(components)} components {components}"
        )
    trial_overlaps = wk.vmap_chunked(
        trial_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(state.walkers, trial_data)
    wp = state.weights * trial_overlaps / state.overlaps
    w_sum = jnp.sum(wp)
    avgs = jnp.sum(wp[:, None] * out, axis=0) / w_sum
    return jnp.asarray(energy_fn(ham_data.h0, avgs)) + 0j, w_sum


def _progress_trial(h0, weights, comps, energy_fn):
    """(mean, error-or-None) of the trial energy from the blocks so far, or None."""
    if len(weights) < 2:
        return None
    stats = blocking_analysis_components(
        h0, np.asarray(weights), np.asarray(comps), energy_fn, print_q=False
    )
    return float(stats["mu"]), stats["se_star"]


def run_mixed_qmc(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_prop_ops: PropOps,
    guide_meas_ops: MeasOps,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    mix_block_fn: MixedBlockFn,
    components: tuple[str, ...] = PT2_COMPONENTS,
    energy_fn: Callable[[Any, Any], Any],
    outlier_zeta: float | None = 20.0,
    trial_name: str = "trial",
    state: PropState | None = None,
    guide_meas_ctx: Any | None = None,
    trial_meas_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
    observable_names: tuple[str, ...] = (),
) -> MixedQmcResult:
    """
    Equilibration blocks then sampling blocks. The importance sampling is governed by the
    guide and the energy is measured against the trial; the guide's energy is also
    measured, to update the population control shift and to reject runaway walkers.

    The trial is described by what its energy kernel returns per walker (components, in
    order) and how the weighted block means of those combine into an energy
    (energy_fn(h0, components), the last axis being the component axis). The trial is
    measured at tau = 0 and during sampling, not during equilibration. target_error stops
    the sampling once the trial energy's blocking error is below it.
    """
    for name in observable_names:
        guide_meas_ops.require_observable(name)

    # build the contexts
    guide_prop_ctx = guide_prop_ops.build_prop_ctx(ham_data, guide_ops.get_rdm1(guide_data), params)
    if guide_meas_ctx is None:
        guide_meas_ctx = guide_meas_ops.build_meas_ctx(ham_data, guide_data)
    if trial_meas_ctx is None:
        trial_meas_ctx = trial_meas_ops.build_meas_ctx(ham_data, trial_data)

    if state is None:
        state = guide_prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=guide_ops,
            trial_data=guide_data,
            meas_ops=guide_meas_ops,
            params=params,
            meas_ctx=guide_meas_ctx,
            mesh=mesh,
        )
    assert state is not None

    trial_energy0, trial_weights0 = _init_trial_energy(
        state, ham_data, trial_data, trial_meas_ops, trial_meas_ctx, params, components, energy_fn
    )

    # the block function is told what the trial kernel returns; the sr_fn binding for a
    # sharded population is the same mechanism
    mix_block_fn_sr = partial(mix_block_fn, components=components)
    if mesh is not None and mesh.size > 1:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        mix_block_fn_sr = partial(mix_block_fn_sr, sr_fn=sr_sharded)

    run_blocks = make_run_mixed_blocks(
        mixed_block_fn=mix_block_fn_sr,
        sys=sys,
        params=params,
        guide_ops=guide_ops,
        guide_meas_ops=guide_meas_ops,
        guide_prop_ops=guide_prop_ops,
        trial_meas_ops=trial_meas_ops,
        observable_names=observable_names,
    )

    h0 = ham_data.h0
    t0 = time.perf_counter()
    t_mark = t0
    block_time = params.dt * params.n_prop_steps

    # ---------------------------------------------------------------- equilibration
    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    guide_block_e_eq = [state.e_estimate]
    guide_block_w_eq = [jnp.sum(state.weights)]
    print("\nEquilibration:")
    print("E_Trial is not measured till the sampling phase\n")
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'tau':>6s}  {'Guide_E_blk':>14s}  {'Guide_W_blk':>12s}   "
            f"{'Trial_E_blk':>14s}  {'Trial_W_blk':>12s}   {'nodes':>10s}  {'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{0.0:6.2f}  "
        f"{float(jnp.real(guide_block_e_eq[0])):14.10f}  "
        f"{float(jnp.real(guide_block_w_eq[0])):12.6e}  "
        f"{float(trial_energy0.real):14.10f}  "
        f"{float(trial_weights0.real):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, scalars_chunk, _ = run_blocks(
            state,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            trial_data=trial_data,
            trial_meas_ctx=trial_meas_ctx,
            n_blocks=n,
            measure_trial=False,
        )
        guide_block_e_eq.extend(scalars_chunk["guide_energy"].tolist())
        guide_block_w_eq.extend(scalars_chunk["guide_weight"].tolist())
        guide_w_chunk_avg = jnp.mean(scalars_chunk["guide_weight"])
        guide_e_chunk_avg = (
            jnp.mean(scalars_chunk["guide_energy"] * scalars_chunk["guide_weight"])
            / guide_w_chunk_avg
        )
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{(start + n) * block_time:6.2f}  "
            f"{float(guide_e_chunk_avg):14.10f}  "
            f"{float(guide_w_chunk_avg):12.6e}  "
            f"{'-':>14s}  {'-':>12s}  "
            f"{int(state.node_encounters):10d}  "
            f"{time.perf_counter() - t0:8.1f}"
        )

    guide_block_w_eq_arr = jnp.asarray(guide_block_w_eq)
    guide_block_e_eq_arr = jnp.asarray(guide_block_e_eq)

    # ---------------------------------------------------------------- sampling
    print("\nSampling:\n")
    target = 0.0 if target_error is None else float(target_error)
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0

    guide_block_w_sp: list = []
    guide_block_e_sp: list = []
    trial_block_w_sp: list = []
    trial_comp_sp: list = []  # rows of (n_components,)

    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'Guide_E_avg':>14s}  {'Guide_E_err':>10s}  {'Guide_W':>12s}  "
            f"{'Trial_E_avg':>14s}  {'Trial_E_err':>10s}  {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, scalars_chunk, _ = run_blocks(
            state,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            trial_data=trial_data,
            trial_meas_ctx=trial_meas_ctx,
            n_blocks=n,
            measure_trial=True,
        )
        guide_block_w_sp.extend(scalars_chunk["guide_weight"].tolist())
        guide_block_e_sp.extend(scalars_chunk["guide_energy"].tolist())
        trial_block_w_sp.extend(scalars_chunk["trial_weight"].tolist())
        trial_comp_sp.extend(np.asarray(scalars_chunk["trial_components"]))

        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()

        guide_stats = blocking_analysis_ratio(
            np.asarray(guide_block_e_sp), np.asarray(guide_block_w_sp), print_q=False
        )
        guide_mu, guide_se = guide_stats["mu"], guide_stats["se_star"]
        trial_stats = _progress_trial(h0, trial_block_w_sp, np.asarray(trial_comp_sp), energy_fn)
        trial_e_avg, trial_error = (None, None) if trial_stats is None else trial_stats

        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{guide_mu:14.10f}  "
            f"{(f'{guide_se:10.3e}' if guide_se is not None else ' ' * 10)}  "
            f"{float(np.mean(guide_block_w_sp)):12.6e}  "
            f"{(f'{trial_e_avg:14.10f}' if trial_e_avg is not None else ' ' * 14)}  "
            f"{(f'{float(trial_error):10.3e}' if trial_error is not None else ' ' * 10)}  "
            f"{int(state.node_encounters):10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
        # the energy this run reports is the trial's, so that is the error to stop on
        if target > 0.0 and trial_error is not None and float(trial_error) <= target:
            print(f"\nTarget error {target:.3e} reached at block {start + n}.")
            break

    # ---------------------------------------------------------------- guide statistics
    guide_e_sp = np.asarray(guide_block_e_sp)
    guide_w_sp = np.asarray(guide_block_w_sp)
    guide_block_e_all = jnp.concatenate([guide_block_e_eq_arr, jnp.asarray(guide_e_sp)])
    guide_block_w_all = jnp.concatenate([guide_block_w_eq_arr, jnp.asarray(guide_w_sp)])
    guide_clean, _ = reject_outliers(np.column_stack((guide_e_sp, guide_w_sp)), obs=0)
    print(f"\nRejected {guide_e_sp.shape[0] - guide_clean.shape[0]} guide outlier blocks.")
    guide_analysis = _analyze_energy_errors(
        guide_clean[:, 0], guide_clean[:, 1], error_method=params.error_method
    )

    # ---------------------------------------------------------------- trial statistics
    trial_w = np.asarray(trial_block_w_sp)
    trial_comps = np.asarray(trial_comp_sp).reshape(len(trial_w), len(components))
    proxy, keep = component_estimator_outlier_mask(
        h0, trial_w, trial_comps, energy_fn, zeta=outlier_zeta
    )
    print(f"Rejected {int((~keep).sum())} AFQMC/{trial_name} outlier blocks.")
    trial_analysis = _analyze_component_estimator_errors(
        h0, trial_w[keep], trial_comps[keep], energy_fn, error_method=params.error_method
    )

    print("\nFinal analysis of the guide energy:")
    _print_energy_error_analysis(guide_analysis)
    print(f"\nFinal analysis of the AFQMC/{trial_name} energy:")
    _print_energy_error_analysis(trial_analysis)
    print(
        f"\nAFQMC/{trial_name} energy = {trial_analysis.mean:.6f} +/- "
        f"{trial_analysis.stderr:.6f} (1-sigma, {params.error_method})"
    )

    return MixedQmcResult(
        guide_mean_energy=guide_analysis.mean,
        guide_stderr_energy=guide_analysis.stderr,
        guide_block_energies=guide_block_e_all,
        guide_block_weights=guide_block_w_all,
        trial_mean_energy=trial_analysis.mean,
        trial_stderr_energy=trial_analysis.stderr,
        trial_mean_components=np.asarray(trial_analysis.blocking["mean_components"]),
        trial_component_names=tuple(components),
        trial_block_weights=jnp.asarray(trial_w),
        trial_block_components={
            name: jnp.asarray(trial_comps[:, i]) for i, name in enumerate(components)
        },
        trial_block_proxy_energies=np.asarray(proxy),
        trial_block_keep_mask=np.asarray(keep),
        guide_analysis=guide_analysis,
        trial_analysis=trial_analysis,
        # the jitted block runner returns an untyped state; it is the PropState it was given
        final_state=cast(PropState, state),
    )
