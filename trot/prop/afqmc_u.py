"""
Propagation ops for the unrestricted (uchol) hamiltonian.

The companion of prop/afqmc.py for HamCholU. afqmc_step and init_prop_state are reused
unchanged; what differs is the propagation context (chol_afqmc_ops_u) and the initial
walkers, which are built per spin so norb_a may differ from norb_b.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from ..core.ops import MeasOps, TrialOps
from ..walkers import init_walkers_uh
from .afqmc import afqmc_step, init_prop_state
from .chol_afqmc_ops_u import CholAfqmcCtxU, _build_prop_ctx_u, make_trotter_ops_u
from .types import PropOps, PropState, QmcParamsBase


def init_prop_state_uh(
    *,
    sys: Any,
    ham_data: Any,
    trial_ops: TrialOps,
    trial_data: Any,
    meas_ops: MeasOps,
    params: QmcParamsBase,
    meas_ctx: Any | None = None,
    initial_walkers: Any | None = None,
    initial_e_estimate: jax.Array | None = None,
    rdm1: Any | None = None,
    mesh: Mesh | None = None,
) -> PropState:
    """
    init_prop_state with walkers from init_walkers_uh: the trial rdm1 is a pair of spin
    blocks of possibly different sizes, which init_walkers cannot take. Everything else
    (overlaps, the initial energy through the trial's energy kernel) is shared.
    """
    if initial_walkers is None:
        if rdm1 is None:
            rdm1 = trial_ops.get_rdm1(trial_data)
        initial_walkers = init_walkers_uh(sys, rdm1, params.n_walkers)

    return init_prop_state(
        sys=sys,
        ham_data=ham_data,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        params=params,
        meas_ctx=meas_ctx,
        initial_walkers=initial_walkers,
        initial_e_estimate=initial_e_estimate,
        rdm1=rdm1,
        mesh=mesh,
    )


def make_prop_ops_u(
    ham_basis: str,
    walker_kind: str,
    mixed_precision: bool = False,
    packed_cholesky: bool = False,
) -> PropOps:
    """
    PropOps for a HamCholU: make_prop_ops with the unrestricted trotter ops, propagation
    context and initial walkers. The step itself is afqmc_step.
    """
    if packed_cholesky:
        raise NotImplementedError(
            "packed cholesky storage is not implemented for the unrestricted hamiltonian."
        )
    trotter_ops = make_trotter_ops_u(ham_basis, walker_kind, mixed_precision=mixed_precision)

    def step(
        state: PropState,
        *,
        params: QmcParamsBase,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> PropState:
        return afqmc_step(
            state,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            trotter_ops=trotter_ops,  # type: ignore[arg-type]
        )

    def build_prop_ctx(ham_data: Any, rdm1: Any, params: QmcParamsBase) -> CholAfqmcCtxU:
        return _build_prop_ctx_u(
            ham_data,
            rdm1,
            params.dt,
            chol_flat_precision=jnp.float32 if mixed_precision else jnp.float64,
        )

    return PropOps(init_prop_state=init_prop_state_uh, build_prop_ctx=build_prop_ctx, step=step)
