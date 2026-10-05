"""
Recipes of the LNO fragment AFQMC, mirroring trot/mixed.py.

The guides are the branch's own (trot.mixed.GUIDES): the RHF one is staged by the
branch from the mean field in the fragment basis (staging.frag_mf), the UHF one is the
identity reference per spin of the uchol fragment hamiltonian (staging.stage_uhf_guide),
the CISD ones come from the fragment's CCSD amplitudes (staging.stage_cisd_guide /
stage_ucisd_guide). The trials are the fragment estimators, in a table of their own
because the names coincide with trot's: "pt2ccsd" there is the full-space trial, here it
is the fragment one (the bar estimator on the similarity transformed hamiltonian,
meas/pt2ccsd_bar.py), "pt2ccsd_fast" the same estimator with the fragment projector
kept factored so the local index is contracted last (meas/pt2ccsd_fast.py), and
"upt2ccsd" the unrestricted counterpart on the uchol fragment hamiltonian
(meas/upt2ccsd_bar.py).
get_mixed_recipe(trial, guide) pairs the two through trot's pair_recipe, so the same
compatibility rules (hamiltonian basis, walker kind) apply, and the block function and
the statistics are the branch's (prop/blocks_mixed.block_mixed, the component
analyses of stat_utils), with the fragment energy function on top.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

from ..mixed import GUIDES, GuideSpec, MixedRecipe, TrialSpec, pair_recipe
from .integral import build_ham_lno_df, build_ham_ulno_df
from .meas.pt2ccsd_bar import (
    TRIAL_COMPONENTS,
    frag_pt2ccsd_energy_fn,
    get_pt2ccsd_meas_cfg,
    make_pt2ccsd_meas_ops,
    plan_chunking_for_run,
)
from .meas.pt2ccsd_fast import (
    get_pt2ccsd_fast_meas_cfg,
    make_pt2ccsd_fast_meas_ops,
    plan_chunking_for_run_fast,
)
from .meas.upt2ccsd_bar import (
    get_upt2ccsd_meas_cfg,
    make_upt2ccsd_meas_ops,
    plan_chunking_for_run_u,
)
from .meas.upt2ccsd_fast import (
    get_upt2ccsd_fast_meas_cfg,
    make_upt2ccsd_fast_meas_ops,
    plan_chunking_for_run_u_fast,
)
from .staging import (
    LnoFragData,
    stage_cisd_guide,
    stage_pt2ccsd_fast_trial,
    stage_pt2ccsd_trial,
    stage_ucisd_guide,
    stage_uhf_guide,
    stage_upt2ccsd_fast_trial,
    stage_upt2ccsd_trial,
)
from .trial.pt2ccsd import make_pt2ccsd_trial_data
from .trial.pt2ccsd_fast import make_pt2ccsd_fast_trial_data
from .trial.upt2ccsd import make_upt2ccsd_trial_data
from .trial.upt2ccsd_fast import make_upt2ccsd_fast_trial_data


@dataclass(frozen=True)
class LnoMixedRecipe(MixedRecipe):
    """
    A MixedRecipe plus what the fragment loop needs:

    stage_trial:       LnoFragData -> TrialInput (narrowed from the CC-object version)
    stage_guide:       LnoFragData -> TrialInput for a guide built from the fragment data
                       (the amplitudes, or the per spin identity of the uchol layout),
                       None for one staged from the mean field
    build_ham:         (mf, lno_coeff, lno_frozen, chol_cut=) -> HamInput / HamInputU
    energy_fn:         (h0, averaged components) -> fragment energy; h0 is ignored
    components:        the names of the kernel's return values, in order
    needs_amplitudes:  whether step 3 (LNO-CCSD) has to run for this pair, for the
                       trial or for the guide
    """

    stage_guide: Callable[..., Any] | None = None
    build_ham: Callable[..., Any] = build_ham_lno_df
    needs_amplitudes: bool = True


# guides built from the fragment data rather than staged from the mean field: the CISD
# ones from the amplitudes, the UHF one because the uchol layout keeps each spin in its
# own basis, which the branch's mean-field staging does not
GUIDE_STAGERS: dict[str, Callable[[LnoFragData], Any]] = {
    "uhf": stage_uhf_guide,
    "cisd": stage_cisd_guide,
    "ucisd": stage_ucisd_guide,
}
AMPLITUDE_GUIDES: frozenset[str] = frozenset({"cisd", "ucisd"})


TRIALS: dict[str, TrialSpec] = {
    "pt2ccsd": TrialSpec(
        name="pt2ccsd",
        kind="pt2ccsd",
        stage=stage_pt2ccsd_trial,
        make_data=make_pt2ccsd_trial_data,
        make_meas_ops=make_pt2ccsd_meas_ops,
        components=TRIAL_COMPONENTS,
        energy_fn=frag_pt2ccsd_energy_fn,
        ham_basis="restricted",
        walker_kinds=frozenset({"restricted"}),
        cc_kind="ccsd",
        default_guide="rhf",
        plan_chunking=plan_chunking_for_run,
        cfg_getter=get_pt2ccsd_meas_cfg,
    ),
    # the same estimator with the fragment projector kept factored, U U^H, and the local
    # index contracted last: the same numbers at nlo/nocc of the cost (meas/pt2ccsd_fast.py)
    "pt2ccsd_fast": TrialSpec(
        name="pt2ccsd_fast",
        kind="pt2ccsd_fast",
        stage=stage_pt2ccsd_fast_trial,
        make_data=make_pt2ccsd_fast_trial_data,
        make_meas_ops=make_pt2ccsd_fast_meas_ops,
        components=TRIAL_COMPONENTS,
        energy_fn=frag_pt2ccsd_energy_fn,
        ham_basis="restricted",
        walker_kinds=frozenset({"restricted"}),
        cc_kind="ccsd",
        default_guide="rhf",
        plan_chunking=plan_chunking_for_run_fast,
        cfg_getter=get_pt2ccsd_fast_meas_cfg,
    ),
    # the unrestricted fragment trial, on the uchol fragment hamiltonian (each spin in its
    # own LNO basis, one shared cholesky index) with unrestricted walkers
    "upt2ccsd": TrialSpec(
        name="upt2ccsd",
        kind="upt2ccsd",
        stage=stage_upt2ccsd_trial,
        make_data=make_upt2ccsd_trial_data,
        make_meas_ops=make_upt2ccsd_meas_ops,
        components=TRIAL_COMPONENTS,
        energy_fn=frag_pt2ccsd_energy_fn,
        ham_basis="uchol",
        walker_kinds=frozenset({"unrestricted"}),
        cc_kind="uccsd",
        default_guide="uhf",
        plan_chunking=plan_chunking_for_run_u,
        cfg_getter=get_upt2ccsd_meas_cfg,
    ),
    # the unrestricted estimator with the projectors kept factored (meas/upt2ccsd_fast.py)
    "upt2ccsd_fast": TrialSpec(
        name="upt2ccsd_fast",
        kind="upt2ccsd_fast",
        stage=stage_upt2ccsd_fast_trial,
        make_data=make_upt2ccsd_fast_trial_data,
        make_meas_ops=make_upt2ccsd_fast_meas_ops,
        components=TRIAL_COMPONENTS,
        energy_fn=frag_pt2ccsd_energy_fn,
        ham_basis="uchol",
        walker_kinds=frozenset({"unrestricted"}),
        cc_kind="uccsd",
        default_guide="uhf",
        plan_chunking=plan_chunking_for_run_u_fast,
        cfg_getter=get_upt2ccsd_fast_meas_cfg,
    ),
}

_BUILD_HAM: dict[str, Callable[..., Any]] = {
    "restricted": build_ham_lno_df,
    "uchol": build_ham_ulno_df,
}

# the guide a trial pairs with when none is asked for
DEFAULT_GUIDE: dict[str, str] = {name: spec.default_guide for name, spec in TRIALS.items()}


def _lno_recipe(guide: GuideSpec, trial: TrialSpec) -> LnoMixedRecipe:
    base = pair_recipe(guide, trial)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(MixedRecipe)}
    return LnoMixedRecipe(
        **fields,
        stage_guide=GUIDE_STAGERS.get(guide.name),
        build_ham=_BUILD_HAM[trial.ham_basis],
        needs_amplitudes=trial.cc_kind in ("ccsd", "uccsd") or guide.name in AMPLITUDE_GUIDES,
    )


def _compatible(guide: GuideSpec, trial: TrialSpec) -> bool:
    return trial.ham_basis in guide.ham_bases and bool(guide.walker_kinds & trial.walker_kinds)


MIXED_RECIPES: dict[tuple[str, str], LnoMixedRecipe] = {
    (g.name, t.name): _lno_recipe(g, t)
    for g in GUIDES.values()
    for t in TRIALS.values()
    if _compatible(g, t)
}


def _format_pairs() -> str:
    return ", ".join(f"{guide}+{trial}" for guide, trial in sorted(MIXED_RECIPES))


def get_mixed_recipe(trial: str, guide: str | None = None) -> LnoMixedRecipe:
    """
    The (guide, trial) pipeline of the fragment AFQMC. guide=None takes the trial's
    default guide (the corresponding HF); an unregistered or incompatible pair lists what
    is available.
    """
    trials = sorted(TRIALS)
    if trial not in trials:
        raise ValueError(f"unknown LNO trial {trial!r}; available: {', '.join(trials)}")
    if guide is None:
        guide = DEFAULT_GUIDE[trial]
    guide = guide.lower()
    if guide not in GUIDES:
        raise ValueError(f"unknown mixed guide {guide!r}; available: {', '.join(sorted(GUIDES))}")
    try:
        return MIXED_RECIPES[(guide, trial)]
    except KeyError:
        guides = ", ".join(sorted(g for g, t in MIXED_RECIPES if t == trial))
        raise ValueError(
            f"no LNO recipe pairs guide={guide!r} with trial={trial!r}; guides for this trial: "
            f"{guides} (all pairs: {_format_pairs()})"
        ) from None


def available_mixed_recipes() -> tuple[tuple[str, str], ...]:
    """The registered (guide, trial) pairs."""
    return tuple(sorted(MIXED_RECIPES))


def needs_amplitudes(recipe: LnoMixedRecipe) -> bool:
    return bool(recipe.needs_amplitudes)


__all__ = [
    "LnoMixedRecipe",
    "MIXED_RECIPES",
    "DEFAULT_GUIDE",
    "GUIDE_STAGERS",
    "AMPLITUDE_GUIDES",
    "TRIALS",
    "get_mixed_recipe",
    "available_mixed_recipes",
    "needs_amplitudes",
    "LnoFragData",
]
