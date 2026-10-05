"""
The run loop of one fragment's AFQMC: trot/driver_mixed.py:run_mixed_qmc with the flow
of afqmc's script/run_lno_afqmc_pt2ccsd.py,

  tau = 0      the fragment energy of the initial walkers ("Initial Orbital energy")
  eql          n_eql_blocks blocks, guide only
  sample       n_blocks blocks, printing every chunk, stopping early once at least
               min_blocks blocks are in and the fragment error is below
               stop_ratio * max_error
  post         outlier blocks dropped, guide and fragment errors from the branch's
               analyses (params.error_method), the final numbers as host floats

The scan itself is the mixed driver's make_run_mixed_blocks over
prop/blocks_mixed.block_mixed: per block the absorbed weight sum(wp) and the wp-averaged
components (t2frg, e0frg, e1frg, e0), combined by the recipe's energy_fn into
E_F = <e0frg> + <e1frg> - <t2frg><e0>. The statistics are the branch's component
analyses, which keep the covariance between the components in that nonlinear
combination. Everything returned is on the host, so nothing of the fragment stays on the
device.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable, NamedTuple, cast

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ..core.ops import MeasOps, TrialOps
from ..core.system import System
from ..driver import (
    EnergyErrorAnalysis,
    _analyze_component_estimator_errors,
    _analyze_energy_errors,
    _print_energy_error_analysis,
)
from ..driver_mixed import _init_trial_energy, _progress_trial, make_run_mixed_blocks
from ..prop.blocks_mixed import MixedBlockFn
from ..prop.types import PropOps, PropState, QmcParams
from ..stat_utils import blocking_analysis_ratio, component_estimator_outlier_mask, reject_outliers
from ..walkers import stochastic_reconfiguration
from .meas.pt2ccsd_bar import TRIAL_COMPONENTS, frag_pt2ccsd_energy_fn

print = partial(print, flush=True)


class FragQmcResult(NamedTuple):
    """
    The result of one fragment's run. The fragment block data is the absorbed block
    weight sum(wp) and the wp-averaged components of the kernel, keyed by the recipe's
    component names; frag_block_keep_mask records the outlier cleanup used for the
    reported energy, the block arrays are raw.
    """

    guide_mean_energy: float
    guide_stderr_energy: float
    guide_block_energies: np.ndarray  # eql rows (incl. the tau=0 row) + sampling rows
    guide_block_weights: np.ndarray
    frag_mean_energy: float
    frag_stderr_energy: float
    frag_init_energy: float  # the tau = 0 fragment energy
    frag_mean_components: np.ndarray
    frag_component_names: tuple[str, ...]
    frag_block_weights: np.ndarray  # sum(wp) per sampling block
    frag_block_components: dict[str, np.ndarray]
    frag_block_proxy_energies: np.ndarray
    frag_block_keep_mask: np.ndarray
    guide_analysis: EnergyErrorAnalysis
    frag_analysis: EnergyErrorAnalysis
    n_blocks_run: int
    n_outliers: int
    weightp_over_weight: float
    final_state: PropState


def run_frag_qmc(
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
    components: tuple[str, ...] = TRIAL_COMPONENTS,
    energy_fn: Callable[[Any, Any], Any] = frag_pt2ccsd_energy_fn,
    max_error: float | None = None,
    stop_ratio: float = 0.7,
    min_blocks: int = 120,
    outlier_zeta: float | None = 20.0,
    state: PropState | None = None,
    guide_meas_ctx: Any | None = None,
    trial_meas_ctx: Any | None = None,
    mesh: Mesh | None = None,
    label: str = "",
) -> FragQmcResult:
    """
    Equilibration then sampling of one fragment. The importance sampling is governed by
    the guide; the fragment energy is measured against the trial at tau = 0 and from the
    sampling phase on. Returns the fragment correlation energy with its error, and the
    guide energy.
    """
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

    e_init_c, w_init_c = _init_trial_energy(
        state, ham_data, trial_data, trial_meas_ops, trial_meas_ctx, params, components, energy_fn
    )
    e_init = float(np.real(e_init_c))
    w_init = float(np.real(w_init_c))

    mix_block_fn_sr = partial(mix_block_fn, components=components)
    if mesh is not None and mesh.size > 1:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        mix_block_fn_sr = partial(mix_block_fn_sr, sr_fn=sr_sharded)

    run_blocks = make_run_mixed_blocks(
        mixed_block_fn=cast(Any, mix_block_fn_sr),
        sys=sys,
        params=params,
        guide_ops=guide_ops,
        guide_meas_ops=guide_meas_ops,
        guide_prop_ops=guide_prop_ops,
        trial_meas_ops=trial_meas_ops,
        observable_names=(),
    )

    h0 = ham_data.h0
    t0 = time.perf_counter()
    t_mark = t0
    block_time = params.dt * params.n_prop_steps
    tag = f"[{label}] " if label else ""

    # ------------------------------------------------------------------ equilibration
    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    guide_block_e_eq = [float(np.real(state.e_estimate))]
    guide_block_w_eq = [float(np.real(jnp.sum(state.weights)))]

    # the same layout as run_mixed_qmc: the tau = 0 row carries the initial walkers
    # measured against both the guide and the trial (here the fragment correlation
    # energy); afterwards the fragment is not measured until sampling
    print(f"\n{tag}Equilibration:")
    print("The fragment energy is not measured until the sampling phase\n")
    print(
        f"{'':4s}{'block':>9s}  {'tau':>6s}  {'Guide_E_blk':>14s}  {'Guide_W_blk':>12s}  "
        f"{'Frag_E_blk':>14s}  {'Frag_W_blk':>12s}  {'nodes':>10s}  {'t[s]':>8s}"
    )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  {0.0:6.2f}  "
        f"{guide_block_e_eq[0]:14.10f}  {guide_block_w_eq[0]:12.6e}  "
        f"{e_init:14.10f}  {w_init:12.6e}  {int(state.node_encounters):10d}  {0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, scalars, _ = run_blocks(
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
        e_chunk = np.asarray(scalars["guide_energy"])
        w_chunk = np.asarray(scalars["guide_weight"])
        guide_block_e_eq.extend(e_chunk.tolist())
        guide_block_w_eq.extend(w_chunk.tolist())
        w_avg = float(np.mean(w_chunk))
        e_avg = float(np.mean(e_chunk * w_chunk) / w_avg)
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  {(start + n) * block_time:6.2f}  "
            f"{e_avg:14.10f}  {w_avg:12.6e}  {'-':>14s}  {'-':>12s}  "
            f"{int(state.node_encounters):10d}  {time.perf_counter() - t0:8.1f}"
        )

    # ------------------------------------------------------------------ sampling
    print(f"\n{tag}Sampling:")
    if max_error is not None and max_error > 0:
        print(
            f"Early stop once the fragment error < {stop_ratio:.2f} x {max_error:.3e} "
            f"= {stop_ratio * max_error:.3e}, after at least {min_blocks} blocks"
        )
    print("")
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    chunk = print_every if print_every > 0 else 1

    guide_w_sp: list[float] = []
    guide_e_sp: list[float] = []
    frag_w_sp: list[complex] = []
    frag_comp_sp: list[np.ndarray] = []  # rows of (n_components,)

    print(
        f"{'':4s}{'block':>9s}  {'Guide_E_avg':>14s}  {'Guide_E_err':>10s}  {'Guide_W':>12s}  "
        f"{'Frag_E_avg':>14s}  {'Frag_E_err':>10s}  {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>8s}"
    )

    n_done = 0
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, scalars, _ = run_blocks(
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
        n_done = start + n
        guide_w_sp.extend(np.asarray(scalars["guide_weight"]).tolist())
        guide_e_sp.extend(np.asarray(scalars["guide_energy"]).tolist())
        frag_w_sp.extend(np.asarray(scalars["trial_weight"]).tolist())
        frag_comp_sp.extend(np.asarray(scalars["trial_components"]))

        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()

        guide_stats = blocking_analysis_ratio(
            np.asarray(guide_e_sp), np.asarray(guide_w_sp), print_q=False
        )
        guide_mu, guide_se = guide_stats["mu"], guide_stats["se_star"]
        guide_w_avg = float(np.mean(guide_w_sp))

        frag_stats = _progress_trial(h0, frag_w_sp, np.asarray(frag_comp_sp), energy_fn)
        frag_e, frag_err = (None, None) if frag_stats is None else frag_stats
        guide_se_s = f"{guide_se:10.3e}" if guide_se is not None else f"{'-':>10s}"
        frag_e_s = f"{float(np.real(frag_e)):14.10f}" if frag_e is not None else f"{'-':>14s}"
        frag_err_s = f"{float(frag_err):10.3e}" if frag_err is not None else f"{'-':>10s}"
        print(
            f"[blk {n_done:4d}/{params.n_blocks}]  {guide_mu:14.10f}  {guide_se_s}  {guide_w_avg:12.6e}  "
            f"{frag_e_s}  {frag_err_s}  {int(state.node_encounters):10d}  {dt_per_block:10.3f}  "
            f"{elapsed:8.1f}"
        )
        if (
            max_error is not None
            and max_error > 0
            and n_done >= min_blocks
            and frag_err is not None
            and float(frag_err) < stop_ratio * max_error
        ):
            print(
                f"\n{tag}Fragment error {float(frag_err):.3e} < {stop_ratio * max_error:.3e} "
                f"at block {n_done}: stopping."
            )
            break

    # ------------------------------------------------------------------ guide statistics
    guide_e = np.asarray(guide_e_sp)
    guide_w = np.asarray(guide_w_sp)
    guide_clean, _ = reject_outliers(np.column_stack((guide_e, guide_w)), obs=0)
    print(f"\n{tag}Rejected {guide_e.shape[0] - guide_clean.shape[0]} guide outlier blocks.")
    guide_analysis = _analyze_energy_errors(
        guide_clean[:, 0], guide_clean[:, 1], error_method=params.error_method
    )

    # ------------------------------------------------------------------ fragment statistics
    frag_w = np.asarray(frag_w_sp)
    frag_comps = np.asarray(frag_comp_sp).reshape(len(frag_w), len(components))
    proxy, keep = component_estimator_outlier_mask(
        h0, frag_w, frag_comps, energy_fn, zeta=outlier_zeta
    )
    n_out = int((~keep).sum())
    print(f"{tag}Rejected {n_out} fragment outlier blocks.")
    frag_analysis = _analyze_component_estimator_errors(
        h0, frag_w[keep], frag_comps[keep], energy_fn, error_method=params.error_method
    )

    print(f"\n{tag}Final analysis of the guide energy:")
    _print_energy_error_analysis(guide_analysis)
    print(f"\n{tag}Final analysis of the AFQMC/pt2CCSD fragment energy:")
    _print_energy_error_analysis(frag_analysis)

    weightp_over_weight = float(np.real(np.mean(frag_w) / np.mean(guide_w)))
    print(
        f"\n{tag}Final AFQMC/guide energy:            "
        f"{guide_analysis.mean:.6f} +/- {guide_analysis.stderr:.6f}"
    )
    print(
        f"{tag}Final AFQMC/pt2CCSD fragment energy: "
        f"{frag_analysis.mean:.6f} +/- {frag_analysis.stderr:.6f} "
        f"(1-sigma, {params.error_method})"
    )
    print(f"{tag}<t1> = weightp/weight = {weightp_over_weight:.5f}")
    print(f"{tag}Sampling blocks: {n_done}, wall time {time.perf_counter() - t0:.1f} s")

    return FragQmcResult(
        guide_mean_energy=float(guide_analysis.mean),
        guide_stderr_energy=float(guide_analysis.stderr),
        guide_block_energies=np.concatenate([np.asarray(guide_block_e_eq), guide_e]),
        guide_block_weights=np.concatenate([np.asarray(guide_block_w_eq), guide_w]),
        frag_mean_energy=float(frag_analysis.mean),
        frag_stderr_energy=float(frag_analysis.stderr),
        frag_init_energy=e_init,
        frag_mean_components=np.asarray(frag_analysis.blocking["mean_components"]),
        frag_component_names=tuple(components),
        frag_block_weights=frag_w,
        frag_block_components={name: frag_comps[:, i] for i, name in enumerate(components)},
        frag_block_proxy_energies=np.asarray(proxy),
        frag_block_keep_mask=np.asarray(keep),
        guide_analysis=guide_analysis,
        frag_analysis=frag_analysis,
        n_blocks_run=n_done,
        n_outliers=n_out,
        weightp_over_weight=weightp_over_weight,
        # the jitted block runner returns an untyped state; it is the PropState it was given
        final_state=cast(PropState, state),
    )
