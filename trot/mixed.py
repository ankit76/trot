from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from .core.ops import MeasOps
from .core.system import System, WalkerKind
from .meas.pt2ccsd import make_pt2ccsd_meas_ops, plan_chunking_for_run
from .meas.upt2ccsd import make_upt2ccsd_meas_ops, plan_chunking_for_run_u
from .prop.blocks import MixedBlockFn, block_mixed
from .staging import TrialInput, stage_pt2ccsd_trial, stage_upt2ccsd_trial
from .stat_utils import pt2ccsd_blocking
from .trial.pt2ccsd import make_pt2ccsd_trial_data
from .trial.upt2ccsd import make_upt2ccsd_trial_data

# Recipes for mixed guide/trial AFQMC, where the walkers propagate under one wavefunction
# (the guide) and the energy is measured against another (the trial).
#
# The class and job layers are generic; everything that differs between combinations is
# collected here. Picking a trial picks its whole pipeline -- staging, measurement ops,
# block function and blocking analysis -- so a trial can never be paired with the wrong
# estimator.


@dataclass(frozen=True)
class MixedRecipe:
    """
    One (guide, trial) combination, which is what the registry is keyed by: the same
    trial measured against a different guide is a different pipeline, not a variant of
    the same one.

    guide:                wavefunction that propagates the walkers, e.g. "rhf"
    trial:                wavefunction the energy is measured against, and with it the
                          energy kernel, e.g. "pt2ccsd_bar"
    walker_kind:          walker representation the pair supports

    stage_trial:          pyscf cc object -> TrialInput
    make_trial_data:      TrialInput.data -> trial pytree
    make_trial_meas_ops:  (sys, ...) -> MeasOps for the trial estimator
    plan_chunking:        (sys, ham_data, trial_data, ...) -> ChunkPlan, or None if the
                          estimator has no memory model and so cannot honour max_memory
    ham_basis:            the hamiltonian both halves use: "restricted" (HamChol, one
                          orbital basis) or "uchol" (HamCholU, alpha and beta each in
                          their own basis, as in AfqmcUh)

    mixed_block_fn:       per block propagate(guide) + measure(trial)
    blocking_fn:          combines the block components into (energy, stderr)

    mixed_block_fn and blocking_fn belong together: the block function decides which
    components come out of each block, and the blocking function is the only thing that
    knows how to recombine them. They are chosen as a pair, never independently.
    """

    guide: str
    trial: str
    walker_kind: WalkerKind

    stage_trial: Callable[..., TrialInput]
    make_trial_data: Callable[[dict, System], Any]
    make_trial_meas_ops: Callable[..., MeasOps]

    mixed_block_fn: MixedBlockFn
    blocking_fn: Callable[..., Any]

    plan_chunking: Callable[..., Any] | None = None
    ham_basis: str = "restricted"

    @property
    def key(self) -> tuple[str, str]:
        return (self.guide, self.trial)


def _pt2ccsd_recipe(guide: str, trial: str, measure_type: str | None) -> MixedRecipe:
    """
    The pt2CCSD trials differ only in which energy kernel they measure with, so the trial
    name is the kernel choice: "pt2ccsd" the plain estimator, "pt2ccsd_chunk" the chunked
    one, "pt2ccsd_bar" the chunked one with exp(T1) on the hamiltonian, and
    "pt2ccsd_sto_chol" that one again with the T2-contracted two-body sum sampled
    semistochastically. Staging, trial data, block function and blocking analysis are
    shared.

    Only the chunking kernels get a plan_chunking hook: max_memory has nothing to size
    for the unchunked one, and setup_mixed says so rather than ignoring the budget.
    """
    return MixedRecipe(
        guide=guide,
        trial=trial,
        walker_kind="restricted",
        stage_trial=stage_pt2ccsd_trial,
        make_trial_data=make_pt2ccsd_trial_data,
        make_trial_meas_ops=partial(make_pt2ccsd_meas_ops, measure_type=measure_type),
        plan_chunking=(
            partial(plan_chunking_for_run, measure_type=measure_type)
            if measure_type is not None
            else None
        ),
        # the pt2CCSD energy kernel returns (t2, e0, e1) per walker and the energy is
        # h0 + e0 + e1 - t2 * e0, which is nonlinear in the block averages, so these two
        # must stay matched
        mixed_block_fn=block_mixed,
        blocking_fn=pt2ccsd_blocking,
    )


def _upt2ccsd_recipe(guide: str, trial: str, measure_type: str) -> MixedRecipe:
    """
    The unrestricted pt2CCSD trials, run on the unrestricted (uchol) hamiltonian AfqmcUh
    uses: each spin in its own MO basis, with unrestricted walkers under a UHF guide.

    As for the restricted trials the name is the kernel choice: "upt2ccsd" the chunked
    estimator, "upt2ccsd_bar" with exp(T1) on the hamiltonian, and "upt2ccsd_sto_chol"
    that one with the T2-contracted two-body sum sampled. Every kernel returns the same
    (t2, e0, e1) as the restricted ones, so the block function and the blocking analysis
    are shared with them. All three are chunked, so all three can honour max_memory.
    """
    return MixedRecipe(
        guide=guide,
        trial=trial,
        walker_kind="unrestricted",
        stage_trial=stage_upt2ccsd_trial,
        make_trial_data=make_upt2ccsd_trial_data,
        make_trial_meas_ops=partial(make_upt2ccsd_meas_ops, measure_type=measure_type),
        plan_chunking=partial(plan_chunking_for_run_u, measure_type=measure_type),
        mixed_block_fn=block_mixed,
        blocking_fn=pt2ccsd_blocking,
        ham_basis="uchol",
    )


MIXED_RECIPES: dict[tuple[str, str], MixedRecipe] = {
    **{
        (guide, trial): _pt2ccsd_recipe(guide, trial, measure_type)
        for guide, trial, measure_type in (
            ("rhf", "pt2ccsd", None),
            ("rhf", "pt2ccsd_chunk", "chunk"),
            ("rhf", "pt2ccsd_bar", "bar"),
            ("rhf", "pt2ccsd_sto_chol", "sto_chol"),
        )
    },
    **{
        (guide, trial): _upt2ccsd_recipe(guide, trial, measure_type)
        for guide, trial, measure_type in (
            ("uhf", "upt2ccsd", "chunk"),
            ("uhf", "upt2ccsd_bar", "bar"),
            ("uhf", "upt2ccsd_sto_chol", "sto_chol"),
        )
    },
}


def _format_pairs() -> str:
    return ", ".join(f"{guide}+{trial}" for guide, trial in sorted(MIXED_RECIPES))


def get_mixed_recipe(trial: str, guide: str | None = None) -> MixedRecipe:
    """
    Look up the (guide, trial) pipeline.

    guide may be left out when the trial is registered against only one guide, which is
    the common case; once a trial pairs with several, it has to be said, since the guide
    changes what propagates the walkers and cannot be guessed.
    """
    if guide is not None:
        try:
            return MIXED_RECIPES[(guide.lower(), trial)]
        except KeyError:
            raise ValueError(
                f"no mixed recipe pairs guide={guide!r} with trial={trial!r}; "
                f"available: {_format_pairs()}"
            ) from None

    matches = [rec for (_, t), rec in MIXED_RECIPES.items() if t == trial]
    if not matches:
        raise ValueError(
            f"unknown mixed recipe trial={trial!r}; available: {_format_pairs()}"
        )
    if len(matches) > 1:
        guides = ", ".join(sorted(rec.guide for rec in matches))
        raise ValueError(
            f"trial={trial!r} is registered against more than one guide ({guides}); "
            "say which with guide=..."
        )
    return matches[0]


def available_mixed_recipes() -> tuple[tuple[str, str], ...]:
    """The registered (guide, trial) pairs."""
    return tuple(sorted(MIXED_RECIPES))
