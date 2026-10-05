from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Protocol, cast

import jax

from .typing import ham_data, trial_data

# Using Protocols for public APIs, Callables for internal helper APIs.

# trial


class OverlapFn(Protocol):
    def __call__(self, walker: Any, trial_data: Any) -> jax.Array: ...


class Rdm1Fn(Protocol):
    def __call__(self, trial_data: Any) -> jax.Array: ...


class GreensFn(Protocol):
    # returns the CPMC cache for one walker
    #   - single det: Array (n,n)
    #   - multi det:  dict like {"G": (nd,n,n), "w": (nd,)}

    def __call__(self, walker: Any, trial_data: Any) -> Any: ...


class OverlapRatioFn(Protocol):
    def __call__(
        self,
        greens: Any,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array: ...


class UpdateGreenFn(Protocol):
    def __call__(
        self,
        greens: Any,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> Any: ...


class TrialOps(NamedTuple):
    """
    Trial operations.
      - overlap: overlap for a single walker
      - get_rdm1: trial rdm1
      Optional fast update functions (mainly for CPMC):
      - calc_green: compute the greens function
      - calc_overlap_ratio: compute overlap ratio for updates
      - update_green: update greens function after walker update
    """

    overlap: OverlapFn  # (walker, trial_data) -> overlap
    get_rdm1: Rdm1Fn  # (trial_data) -> rdm1
    calc_green: GreensFn | None = None  # (walker, trial_data) -> greens
    calc_overlap_ratio: OverlapRatioFn | None = (
        None  # (greens, update_indices, update_constants) -> ratio
    )
    update_green: UpdateGreenFn | None = (
        None  # (greens, update_indices, update_constants) -> new_greens
    )


@dataclass(frozen=True)
class CpmcTrialFns:
    calc_green: GreensFn
    calc_overlap_ratio: OverlapRatioFn
    update_green: UpdateGreenFn


def require_cpmc_trial_ops(trial_ops: TrialOps) -> CpmcTrialFns:
    if trial_ops.calc_green is None:
        raise ValueError("CPMC requires trial_ops.calc_green")
    if trial_ops.calc_overlap_ratio is None:
        raise ValueError("CPMC requires trial_ops.calc_overlap_ratio")
    if trial_ops.update_green is None:
        raise ValueError("CPMC requires trial_ops.update_green")

    # cast narrows for Pylance
    return CpmcTrialFns(
        calc_green=cast(GreensFn, trial_ops.calc_green),
        calc_overlap_ratio=cast(OverlapRatioFn, trial_ops.calc_overlap_ratio),
        update_green=cast(UpdateGreenFn, trial_ops.update_green),
    )


# hamiltonian


class HamOps(NamedTuple):
    """
    Hamiltonian (would probably be helpful when adding different Hamiltonians).
    """

    n_fields: Callable[[ham_data], int]


# measurements


class MeasKernel(Protocol):
    """
    Measurement kernel protocol.
    """

    def __call__(self, walker: Any, ham_data: Any, meas_ctx: Any, trial_data: Any) -> jax.Array: ...


class CombineEnergyComponentsFn(Protocol):
    """Combine final component ratios into an energy.

    The last axis of ``components`` is the component axis; leading axes are
    preserved so the same callable can evaluate a single estimate or a batch
    of leave-one-out estimates.
    """

    def __call__(self, h0: Any, components: Any) -> Any: ...


class BlockEnergyFn(Protocol):
    """Population-level block-energy estimator.

    Unlike a :class:`MeasKernel`, this estimator acts jointly on the full
    walker population.  This supports estimators that sample walkers or
    Hamiltonian terms while keeping their random stream independent of
    propagation and stochastic reconfiguration.
    """

    def __call__(
        self,
        walkers: Any,
        weights: jax.Array,
        overlaps: jax.Array,
        rng_key: jax.Array,
        n_chunks: int,
        ham_data: Any,
        meas_ctx: Any,
        trial_data: Any,
        e_ref: jax.Array,
        energy_clip_threshold: jax.Array,
    ) -> jax.Array | BlockEnergyEstimate: ...


class BlockEnergyEstimate(NamedTuple):
    """Block energy with optional zero-mean estimator diagnostics."""

    energy: jax.Array
    diagnostics: Mapping[str, jax.Array]


class BlockComponentEstimate(NamedTuple):
    """Exact block weight and unnormalized component numerator.

    Keeping the denominator separate from the numerator makes population-level
    component sampling explicit and avoids introducing a random-ratio bias.
    """

    weight: jax.Array
    numerator: jax.Array
    diagnostics: Mapping[str, jax.Array]


class BlockComponentsFn(Protocol):
    """Population-level sufficient-statistic estimator."""

    def __call__(
        self,
        walkers: Any,
        candidate_weights: jax.Array,
        rng_key: jax.Array,
        n_chunks: int,
        ham_data: Any,
        estimator_ctx: Any,
        trial_data: Any,
    ) -> BlockComponentEstimate: ...


class BlockComponentsAdvanceFn(Protocol):
    """Advance calibration by a bounded block count."""

    def __call__(
        self,
        state: Any,
        *,
        n_blocks: int,
    ) -> tuple[Any, Mapping[str, jax.Array], Any]: ...


class BlockComponentRetuneResult(NamedTuple):
    """Replacement state and estimator context after equilibration."""

    state: Any
    estimator_ctx: Any
    initial_n_chunks: int = 1
    settling_blocks: int = 0


class BlockComponentsRetuneFn(Protocol):
    """Host-side hook for adapting a population component estimator."""

    def __call__(
        self,
        state: Any,
        equilibration_components: jax.Array,
        equilibration_weights: jax.Array,
        params: Any,
        ham_data: Any,
        estimator_ctx: Any,
        trial_data: Any,
        *,
        guide_data: Any,
        guide_meas_ops: Any,
        guide_meas_ctx: Any,
        advance_blocks: BlockComponentsAdvanceFn,
        target_error: float | None = None,
    ) -> BlockComponentRetuneResult: ...


class BlockEnergyRetuneResult(NamedTuple):
    """Replacement state and measurement context after equilibration."""

    state: Any
    meas_ctx: Any
    initial_n_chunks: int = 1
    settling_blocks: int = 0


class BlockEnergyAdvanceFn(Protocol):
    """Advance the current equilibration estimator by a bounded block count."""

    def __call__(
        self,
        state: Any,
        *,
        n_blocks: int,
    ) -> tuple[Any, Mapping[str, jax.Array], Any]: ...


class BlockEnergyRetuneFn(Protocol):
    """Host-side hook for adapting a block-energy estimator after equilibration."""

    def __call__(
        self,
        state: Any,
        equilibration_energies: jax.Array,
        equilibration_weights: jax.Array,
        params: Any,
        ham_data: Any,
        meas_ctx: Any,
        trial_data: Any,
        *,
        advance_blocks: BlockEnergyAdvanceFn,
        target_error: float | None = None,
    ) -> BlockEnergyRetuneResult: ...


# usual kernel names
k_energy = "energy"
# Optional equivalent energy implementation used only by AFQMC initialization.
k_energy_init = "energy_init"
k_force_bias = "force_bias"
d_energy_sampling_noise = "energy_sampling_noise"
d_energy_head_guard_count = "energy_head_guard_count"
d_energy_head_guard_weight = "energy_head_guard_weight"
d_energy_walker_guide_ess = "energy_walker_guide_ess"
d_energy_walker_guide_max_correction = "energy_walker_guide_max_correction"
d_pt_component_sampling_noise_real = "pt_component_sampling_noise_real"
d_pt_component_sampling_noise_imag = "pt_component_sampling_noise_imag"
d_pt_estimator_phase_coherence = "pt_estimator_phase_coherence"
d_pt_walker_proposal_ess = "pt_walker_proposal_ess"
o_rdm1 = "rdm1"
o_density_corr = "density_corr"
o_orb_corr = "orb_corr"


@dataclass(frozen=True)
class MeasOps:
    """
    Measurement ops: trial + ham estimators + optional observables.
    """

    # same as TrialOps.overlap
    overlap: OverlapFn  # (walker, trial_data) -> overlap

    # intermediates for measurements
    build_meas_ctx: Callable[[ham_data, trial_data], Any] = lambda ham_data, trial_data: None

    # algorithm kernels (e.g. "energy", "force_bias")
    kernels: Mapping[str, MeasKernel] = field(default_factory=dict)

    # optional observables (e.g. "rdm1", "density_corr", ...)
    observables: Mapping[str, MeasKernel] = field(default_factory=dict)

    # optional population-level energy estimator. When absent, the standard
    # block vmaps kernels["energy"] over walkers and forms a weighted mean.
    block_energy: BlockEnergyFn | None = None

    # optional host-side transition between equilibration and production.
    # Existing measurements leave this unset and retain one context/executable.
    retune_block_energy: BlockEnergyRetuneFn | None = None

    def has_kernel(self, name: str) -> bool:
        return name in self.kernels

    def has_observable(self, name: str) -> bool:
        return name in self.observables

    def require_kernel(self, name: str) -> MeasKernel:
        try:
            return self.kernels[name]
        except KeyError as e:
            avail = ", ".join(sorted(self.kernels.keys()))
            raise KeyError(f"missing required kernel '{name}'. available: [{avail}]") from e

    def require_observable(self, name: str) -> MeasKernel:
        try:
            return self.observables[name]
        except KeyError as e:
            avail = ", ".join(sorted(self.observables.keys()))
            raise KeyError(f"missing requested observable '{name}'. available: [{avail}]") from e

    def available_kernels(self) -> tuple[str, ...]:
        return tuple(sorted(self.kernels.keys()))

    def available_observables(self) -> tuple[str, ...]:
        return tuple(sorted(self.observables.keys()))


@dataclass(frozen=True)
class EstimatorOps:
    """Projected-energy estimator independent of the propagation guide.

    ``reference_overlap`` defines the bra relative to which the per-walker
    sufficient statistics are normalized. Mixed propagation reweights from
    the guide overlap to this reference overlap before averaging components.
    """

    reference_overlap: OverlapFn
    components: MeasKernel
    combine_energy: CombineEnergyComponentsFn
    component_names: tuple[str, ...]
    build_estimator_ctx: Callable[[ham_data, trial_data], Any] = (
        lambda ham_data, trial_data: None
    )
    # Optional population-level component estimator. The caller supplies the
    # exact complex guide-to-reference reweighting coefficients after overlap
    # validation. The hook returns their exact valid sum and an unnormalized
    # component numerator.
    block_components: BlockComponentsFn | None = None

    # Optional host-side transition between equilibration and production for
    # population-level component sampling.
    retune_block_components: BlockComponentsRetuneFn | None = None

    # Use the combined population-level estimator energy for the AFQMC
    # population-control shift.  This is useful when the guide overlap is only
    # a propagation importance function and its deterministic local energy
    # would duplicate (or dominate) the cost of a sampled block estimator.
    use_for_population_control: bool = False

    # Optional host-side builder that can reuse compatible guide intermediates.
    # The ordinary two-argument builder remains available for standalone use.
    build_estimator_ctx_from_guide: Callable[[ham_data, trial_data, trial_data, Any], Any] | None = None

    # Optional deterministic, bounded-workspace kernel for block-zero reporting.
    initial_components: MeasKernel | None = None

    def __post_init__(self) -> None:
        if not self.component_names:
            raise ValueError("EstimatorOps.component_names must be nonempty.")
        if len(set(self.component_names)) != len(self.component_names):
            raise ValueError("EstimatorOps.component_names must be unique.")
