"""
Recipes for mixed guide/trial AFQMC, where the walkers propagate under one wavefunction
(the guide) and the energy is measured against another (the trial):

    |AFQMC> = sum_i w_i |phi_i> / <G|phi_i>
    E_T     = energy_fn(h0, <c>),   <c>_k = sum_i wp_i c_ik / sum_i wp_i,
    wp_i = w_i <T|phi_i> / <G|phi_i>

The guide owns everything that touches propagation (overlap, force bias, the local energy
used for population control, the rdm1 the initial walkers come from); all of that already
exists for every trial kind the branch stages, and setup._make_trial_bundle (or
setup_u._make_trial_bundle_uh on the unrestricted hamiltonian) builds it. The trial owns
everything that touches the measurement: its overlap, an energy kernel that returns named
components per walker, and how the wp-averaged components combine into an energy.

So guides and trials are registered separately (GUIDES, TRIALS) and paired on demand:
get_mixed_recipe(trial, guide) checks that the two agree on the hamiltonian basis and on a
walker kind, and returns the MixedRecipe that setup_mixed and JobMixed consume.

The trials registered here are the plain pt2CCSD estimators,

    "pt2ccsd"       restricted hamiltonian, meas/pt2ccsd.py (one cholesky vector per step)
    "pt2ccsd_bar"   restricted hamiltonian, exp(T1) applied to the right, chunked
    "upt2ccsd"      unrestricted (uchol) hamiltonian, chunked
    "upt2ccsd_bar"  unrestricted hamiltonian, exp(T1) applied to the right, chunked

all with the components (theta, electronic_0, h_t) and the energy
h0 + <electronic_0> + <h_t> - <theta><electronic_0> (meas.pt2ccsd.combine_first_order_energy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .core.ops import MeasOps
from .core.system import WalkerKind
from .meas.pt2ccsd import combine_first_order_energy, make_pt2ccsd_meas_ops
from .meas.pt2ccsd_bar import (
    get_pt2ccsd_bar_meas_cfg,
    make_pt2ccsd_bar_meas_ops,
    plan_chunking_for_run,
)
from .meas.upt2ccsd_bar_uh import (
    get_upt2ccsd_bar_meas_cfg,
    make_upt2ccsd_bar_meas_ops,
    plan_chunking_for_run_u_bar,
)
from .meas.upt2ccsd_uh import (
    get_upt2ccsd_meas_cfg,
    make_upt2ccsd_meas_ops,
    plan_chunking_for_run_u,
)
from .prop.blocks_mixed import PT2_COMPONENTS, MixedBlockFn, block_mixed
from .staging import StagedMfOrCc, TrialInput, _is_cc_like, _stage_pt2ccsd_input
from .staging_u import stage_upt2ccsd_trial_uh
from .trial.pt2ccsd import make_pt2ccsd_trial_data
from .trial.upt2ccsd_uh import make_upt2ccsd_trial_data

WALKER_KIND_ORDER: tuple[str, ...] = ("restricted", "unrestricted", "generalized")


def _scf_of(obj: Any) -> Any:
    return obj._scf if _is_cc_like(obj) else obj


def stage_pt2ccsd_trial(cc: Any, *, frozen: int | None = None) -> TrialInput:
    """The restricted pt2CCSD trial of a pyscf CCSD object (staging._stage_pt2ccsd_input)."""
    return _stage_pt2ccsd_input(StagedMfOrCc(cc, frozen))


@dataclass(frozen=True)
class GuideSpec:
    """
    A wavefunction that can propagate the walkers.

    name:          "rhf", "uhf", "cisd", "ucisd"
    kinds:         the TrialInput.kind values it stages as
    source:        "mf": staged from the mean field (the object itself, or the one under
                   a CC object); "cc": needs the CC object itself
    ham_bases:     hamiltonians it can propagate on: "restricted" and/or "uchol"
    walker_kinds:  walker representations its ops accept
    """

    name: str
    kinds: frozenset[str]
    source: str
    ham_bases: frozenset[str]
    walker_kinds: frozenset[str]

    def source_obj(self, obj: Any) -> Any:
        """The pyscf object the guide is staged from."""
        if self.source == "cc":
            if not _is_cc_like(obj):
                raise ValueError(
                    f"guide={self.name!r} is built from CC amplitudes and needs a pyscf CC "
                    f"object, got {type(obj).__name__}."
                )
            return obj
        return _scf_of(obj)


@dataclass(frozen=True)
class TrialSpec:
    """
    A wavefunction the energy can be measured against.

    name:           "pt2ccsd", "pt2ccsd_bar", "upt2ccsd", "upt2ccsd_bar"
    kind:           the TrialInput.kind stage() must produce, checked after staging
    stage:          (pyscf cc, *, frozen) -> TrialInput
    make_data:      (TrialInput.data, sys) -> trial pytree
    make_meas_ops:  (sys, *, mixed_precision, nchol_chunk) -> MeasOps with an overlap and
                    an "energy" kernel returning the components per walker
    components:     names of the energy kernel's outputs per walker, in order
    energy_fn:      (h0, components) -> energy, the last axis being the component axis
    ham_basis:      the hamiltonian it measures on
    walker_kinds:   walker representations its kernel accepts
    cc_kind:        "ccsd" / "uccsd": which CC object it needs
    default_guide:  the guide used when none is asked for (the corresponding HF)
    plan_chunking:  optional memory model, so a byte budget can size nchol_chunk
    cfg_getter:     optional (MeasOps) -> config dataclass, for dump_flags
    """

    name: str
    kind: str
    stage: Callable[..., TrialInput]
    make_data: Callable[[dict, Any], Any]
    make_meas_ops: Callable[..., MeasOps]
    components: tuple[str, ...]
    energy_fn: Callable[[Any, Any], Any]
    ham_basis: str
    walker_kinds: frozenset[str]
    cc_kind: str
    default_guide: str
    plan_chunking: Callable[..., Any] | None = None
    cfg_getter: Callable[[MeasOps], Any] | None = None


@dataclass(frozen=True)
class MixedRecipe:
    """
    One (guide, trial) pairing, as setup_mixed / JobMixed / driver_mixed consume it.

    guide, trial:         the names
    walker_kind:          the walker representation both sides accept
    ham_basis:            the hamiltonian both halves use: "restricted" or "uchol"
    stage_trial:          pyscf cc -> TrialInput
    make_trial_data:      TrialInput.data -> trial pytree
    make_trial_meas_ops:  (sys, ...) -> MeasOps for the trial estimator
    mixed_block_fn:       per block propagate(guide) + measure(trial)
    components:           names of the trial kernel's outputs, in order
    energy_fn:            (h0, averaged components) -> energy
    plan_chunking:        the trial's memory model, or None
    guide_spec, trial_spec: the specs this pairing was built from
    """

    guide: str
    trial: str
    walker_kind: WalkerKind
    ham_basis: str
    stage_trial: Callable[..., TrialInput]
    make_trial_data: Callable[[dict, Any], Any]
    make_trial_meas_ops: Callable[..., MeasOps]
    mixed_block_fn: MixedBlockFn
    components: tuple[str, ...]
    energy_fn: Callable[[Any, Any], Any]
    plan_chunking: Callable[..., Any] | None
    guide_spec: GuideSpec
    trial_spec: TrialSpec

    @property
    def key(self) -> tuple[str, str]:
        return (self.guide, self.trial)


# ------------------------------------------------------------------------------------
# guides
# ------------------------------------------------------------------------------------

GUIDES: dict[str, GuideSpec] = {
    "rhf": GuideSpec(
        name="rhf",
        kinds=frozenset({"rhf"}),
        source="mf",
        ham_bases=frozenset({"restricted"}),
        walker_kinds=frozenset({"restricted", "unrestricted", "generalized"}),
    ),
    # on the restricted hamiltonian as Afqmc(uhf) runs it, and on the unrestricted one as
    # AfqmcUh does
    "uhf": GuideSpec(
        name="uhf",
        kinds=frozenset({"uhf"}),
        source="mf",
        ham_bases=frozenset({"restricted", "uchol"}),
        walker_kinds=frozenset({"unrestricted", "generalized"}),
    ),
    "cisd": GuideSpec(
        name="cisd",
        kinds=frozenset({"cisd"}),
        source="cc",
        ham_bases=frozenset({"restricted"}),
        walker_kinds=frozenset({"restricted"}),
    ),
    # on the restricted hamiltonian as Afqmc(ucc) runs it; on the unrestricted one with
    # the meas.ucisd_uh kernels, each spin's CI coefficients in that spin's own basis
    "ucisd": GuideSpec(
        name="ucisd",
        kinds=frozenset({"ucisd"}),
        source="cc",
        ham_bases=frozenset({"restricted", "uchol"}),
        walker_kinds=frozenset({"unrestricted", "generalized"}),
    ),
}


# ------------------------------------------------------------------------------------
# trials
# ------------------------------------------------------------------------------------


def _make_pt2ccsd_meas_ops(
    sys: Any, *, mixed_precision: bool = True, nchol_chunk: int | None = None
) -> MeasOps:
    """meas.pt2ccsd's factory in the mixed signature; it scans one vector at a time."""
    return make_pt2ccsd_meas_ops(sys, mixed_precision=mixed_precision)


TRIALS: dict[str, TrialSpec] = {
    "pt2ccsd": TrialSpec(
        name="pt2ccsd",
        kind="pt2ccsd",
        stage=stage_pt2ccsd_trial,
        make_data=make_pt2ccsd_trial_data,
        make_meas_ops=_make_pt2ccsd_meas_ops,
        components=PT2_COMPONENTS,
        energy_fn=combine_first_order_energy,
        ham_basis="restricted",
        walker_kinds=frozenset({"restricted"}),
        cc_kind="ccsd",
        default_guide="rhf",
    ),
    "pt2ccsd_bar": TrialSpec(
        name="pt2ccsd_bar",
        kind="pt2ccsd",
        stage=stage_pt2ccsd_trial,
        make_data=make_pt2ccsd_trial_data,
        make_meas_ops=make_pt2ccsd_bar_meas_ops,
        components=PT2_COMPONENTS,
        energy_fn=combine_first_order_energy,
        ham_basis="restricted",
        walker_kinds=frozenset({"restricted"}),
        cc_kind="ccsd",
        default_guide="rhf",
        plan_chunking=plan_chunking_for_run,
        cfg_getter=get_pt2ccsd_bar_meas_cfg,
    ),
    "upt2ccsd": TrialSpec(
        name="upt2ccsd",
        kind="upt2ccsd",
        stage=stage_upt2ccsd_trial_uh,
        make_data=make_upt2ccsd_trial_data,
        make_meas_ops=make_upt2ccsd_meas_ops,
        components=PT2_COMPONENTS,
        energy_fn=combine_first_order_energy,
        ham_basis="uchol",
        walker_kinds=frozenset({"unrestricted"}),
        cc_kind="uccsd",
        default_guide="uhf",
        plan_chunking=plan_chunking_for_run_u,
        cfg_getter=get_upt2ccsd_meas_cfg,
    ),
    "upt2ccsd_bar": TrialSpec(
        name="upt2ccsd_bar",
        kind="upt2ccsd",
        stage=stage_upt2ccsd_trial_uh,
        make_data=make_upt2ccsd_trial_data,
        make_meas_ops=make_upt2ccsd_bar_meas_ops,
        components=PT2_COMPONENTS,
        energy_fn=combine_first_order_energy,
        ham_basis="uchol",
        walker_kinds=frozenset({"unrestricted"}),
        cc_kind="uccsd",
        default_guide="uhf",
        plan_chunking=plan_chunking_for_run_u_bar,
        cfg_getter=get_upt2ccsd_bar_meas_cfg,
    ),
}


# ------------------------------------------------------------------------------------
# pairing
# ------------------------------------------------------------------------------------


def _compatible(guide: GuideSpec, trial: TrialSpec) -> bool:
    return trial.ham_basis in guide.ham_bases and bool(guide.walker_kinds & trial.walker_kinds)


def pair_recipe(guide: GuideSpec, trial: TrialSpec) -> MixedRecipe:
    """
    The MixedRecipe of one (guide, trial) pair, or a ValueError saying why the two cannot
    be paired: they must share the hamiltonian basis and at least one walker kind. The
    walker kind is the first common one in WALKER_KIND_ORDER.
    """
    if trial.ham_basis not in guide.ham_bases:
        raise ValueError(
            f"guide={guide.name!r} runs on {sorted(guide.ham_bases)} hamiltonians but "
            f"trial={trial.name!r} needs {trial.ham_basis!r}."
        )
    common = [k for k in WALKER_KIND_ORDER if k in guide.walker_kinds and k in trial.walker_kinds]
    if not common:
        raise ValueError(
            f"guide={guide.name!r} accepts walkers {sorted(guide.walker_kinds)} and "
            f"trial={trial.name!r} accepts {sorted(trial.walker_kinds)}; no walker kind in common."
        )
    return MixedRecipe(
        guide=guide.name,
        trial=trial.name,
        walker_kind=common[0],  # type: ignore[arg-type]
        ham_basis=trial.ham_basis,
        stage_trial=trial.stage,
        make_trial_data=trial.make_data,
        make_trial_meas_ops=trial.make_meas_ops,
        mixed_block_fn=block_mixed,
        components=trial.components,
        energy_fn=trial.energy_fn,
        plan_chunking=trial.plan_chunking,
        guide_spec=guide,
        trial_spec=trial,
    )


# every compatible pair, for listing; get_mixed_recipe builds the same objects on demand
MIXED_RECIPES: dict[tuple[str, str], MixedRecipe] = {
    (g.name, t.name): pair_recipe(g, t)
    for g in GUIDES.values()
    for t in TRIALS.values()
    if _compatible(g, t)
}


def _format_pairs() -> str:
    return ", ".join(f"{guide}+{trial}" for guide, trial in sorted(MIXED_RECIPES))


def get_mixed_recipe(trial: str, guide: str | None = None) -> MixedRecipe:
    """
    The (guide, trial) pipeline. guide=None takes the trial's default guide, the
    corresponding HF. An unknown name or an incompatible pair is a ValueError that lists
    what is available.
    """
    try:
        trial_spec = TRIALS[trial]
    except KeyError:
        raise ValueError(
            f"unknown mixed trial {trial!r}; available: {', '.join(sorted(TRIALS))}"
        ) from None
    guide_name = trial_spec.default_guide if guide is None else guide.lower()
    try:
        guide_spec = GUIDES[guide_name]
    except KeyError:
        raise ValueError(
            f"unknown mixed guide {guide_name!r}; available: {', '.join(sorted(GUIDES))}"
        ) from None
    try:
        return pair_recipe(guide_spec, trial_spec)
    except ValueError as e:
        raise ValueError(f"{e} Available pairs: {_format_pairs()}") from None


def available_mixed_recipes() -> tuple[tuple[str, str], ...]:
    """The compatible (guide, trial) pairs."""
    return tuple(sorted(MIXED_RECIPES))


def available_guides() -> tuple[str, ...]:
    return tuple(sorted(GUIDES))


def available_trials() -> tuple[str, ...]:
    return tuple(sorted(TRIALS))


__all__ = [
    "GUIDES",
    "TRIALS",
    "MIXED_RECIPES",
    "GuideSpec",
    "TrialSpec",
    "MixedRecipe",
    "get_mixed_recipe",
    "pair_recipe",
    "available_mixed_recipes",
    "available_guides",
    "available_trials",
    "stage_pt2ccsd_trial",
]
