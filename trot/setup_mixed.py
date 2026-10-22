"""
Assembly for mixed guide/trial AFQMC.

The guide side of a mixed run is an ordinary single-bundle job: the walkers propagate
under the guide wavefunction with the guide's own hamiltonian, ops and propagator. So
setup._assemble_job (restricted hamiltonian) or setup_u.setup_uh (unrestricted one)
builds it unchanged, and JobMixed only adds the trial that the energy is measured
against, plus the cholesky chunk plan of a chunked trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ClassVar, Union, cast

from jax.sharding import Mesh

from . import driver_mixed
from .core.ops import MeasOps
from .core.system import WalkerKind
from .driver_mixed import MixedQmcResult
from .meas.pt2ccsd_chunking import device_memory_budget_bytes
from .mixed import MixedRecipe, get_mixed_recipe
from .prop.blocks import block as default_block
from .prop.types import QmcParams, QmcParamsBase
from .setup import Job, _assemble_job, _make_params, _make_prop, _resolve_default_walker_kind
from .setup_u import setup_uh
from .staging import StagedInputs, TrialInput


def _walker_devices(mesh: Mesh | None) -> int:
    """
    How many devices the walker axis is spread over. Walkers are sharded on the "data"
    axis, so that is what divides the population; a mesh without one is treated as whole.
    """
    if mesh is None:
        return 1
    return int(mesh.shape.get("data", mesh.size))


@dataclass
class JobMixed(Job):
    """
    A fully assembled mixed guide/trial AFQMC run bundle.

    The inherited fields carry the GUIDE: trial_data, trial_ops, meas_ops and prop_ops
    are the guide wavefunction's. The trial that the energy is measured against lives in
    mix_trial_data / mix_trial_meas_ops.
    """

    recipe: MixedRecipe = None  # type: ignore[assignment]
    mix_trial_data: Any = None
    mix_trial_meas_ops: MeasOps = None  # type: ignore[assignment]
    # how the memory budget was split between the two chunking knobs, None without a plan
    chunk_plan: Any = None
    _runtime_mix_meas_ctx: object | None = field(default=None, init=False, repr=False)

    params_cls: ClassVar[type[QmcParamsBase]] = QmcParams
    driver_fn: ClassVar[Callable[..., Any]] = staticmethod(driver_mixed.run_mixed_qmc)

    def mix_meas_ctx(self) -> Any:
        if self._runtime_mix_meas_ctx is None:
            self._runtime_mix_meas_ctx = self.mix_trial_meas_ops.build_meas_ctx(
                self.ham_data, self.mix_trial_data
            )
        return self._runtime_mix_meas_ctx

    def kernel(self, **driver_kwargs: Any) -> MixedQmcResult:
        """
        Run mixed AFQMC: propagate with the guide, measure with the trial.

        Job.prepare_runtime is deliberately not used here: the runtime layouts may compact
        the hamiltonian's cholesky tensor to a zero sized placeholder once the guide's
        contexts are built, but the trial kernels read it. The mixed driver builds all
        three contexts itself from the uncompacted hamiltonian.
        """
        assert isinstance(self.params, self.params_cls)
        driver_kwargs.setdefault("mesh", self.mesh)
        driver_kwargs.setdefault("trial_meas_ctx", self.mix_meas_ctx())
        return self.driver_fn(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            # guide: propagation
            guide_data=self.trial_data,
            guide_ops=self.trial_ops,
            guide_meas_ops=self.meas_ops,
            guide_prop_ops=self.prop_ops,
            # trial: measurement
            trial_data=self.mix_trial_data,
            trial_meas_ops=self.mix_trial_meas_ops,
            mix_block_fn=self.recipe.mixed_block_fn,
            components=self.recipe.components,
            energy_fn=self.recipe.energy_fn,
            trial_name=self.recipe.trial,
            **driver_kwargs,
        )


def setup_mixed(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # the trial half, staged separately from the guide
    recipe: MixedRecipe | str = "pt2ccsd",
    trial_input: TrialInput | None = None,
    nchol_chunk: int | None = None,
    max_memory: float | None = None,
    # staging options (used only if we need to stage)
    norb_frozen_core: Any = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    # system/prop options
    walker_kind: WalkerKind | None = None,
    mesh: Mesh | None = None,
    mixed_precision: bool = True,
    trial_mixed_precision: bool | None = None,
    # params options
    params: QmcParams | None = None,
    # overrides for customized runs
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Callable[..., Any] | None = None,
    # extra kwargs
    params_kwargs: dict[str, Any] | None = None,
    prop_kwargs: dict[str, Any] | None = None,
) -> JobMixed:
    """
    Assemble a runnable mixed AFQMC Job.

    obj_or_staged supplies the GUIDE (a mean field or a StagedInputs with the guide's
    hamiltonian and trial input). trial_input supplies the measurement trial and must be
    staged already (the recipe's stage_trial), since the trial comes from the CC object
    while the guide may come from the mean field under it.

    mixed_precision applies to the guide (its propagator and kernels) and, unless
    trial_mixed_precision is given, to the trial estimator; trial_mixed_precision sets the
    trial's precision on its own, so the two sides can be studied separately.

    The cholesky chunk of a chunked trial (pt2ccsd_bar, upt2ccsd, upt2ccsd_bar) is sized by
    the recipe's memory model against a byte budget: max_memory (MB) when given, else a
    share of the device allocator limit, else DEFAULT_NCHOL_CHUNK when the backend reports
    no limit. The plan can raise params.n_chunks (the walker chunk count), never lower it,
    and the driver's own automatic walker chunking may raise it further after compiling.
    nchol_chunk, when given, is taken as fixed and only n_chunks is derived.

    Basic usage is through AfqmcMixed rather than this function directly.
    """
    rec = get_mixed_recipe(recipe) if isinstance(recipe, str) else recipe
    trial_mp = mixed_precision if trial_mixed_precision is None else bool(trial_mixed_precision)
    if trial_input is None:
        raise ValueError(
            "setup_mixed needs a staged trial_input; the measurement trial comes from the "
            "CC object, so it cannot be inferred from the guide's staged inputs."
        )
    if trial_input.kind != rec.trial_spec.kind:
        raise ValueError(
            f"trial={rec.trial!r} needs a TrialInput of kind {rec.trial_spec.kind!r}, got "
            f"{trial_input.kind!r}."
        )

    wk = walker_kind or cast(WalkerKind, rec.walker_kind)
    if rec.ham_basis == "uchol":
        job = setup_uh(
            obj_or_staged,
            norb_frozen_core=norb_frozen_core,
            chol_cut=chol_cut,
            cache=cache,
            overwrite=overwrite,
            verbose=verbose,
            walker_kind=wk,
            mesh=mesh,
            mixed_precision=mixed_precision,
            params=params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            params_kwargs=params_kwargs,
            prop_kwargs=prop_kwargs,
            job_cls=JobMixed,
        )
    else:
        job = _assemble_job(
            obj_or_staged,
            norb_frozen_core=norb_frozen_core,
            chol_cut=chol_cut,
            cache=cache,
            overwrite=overwrite,
            verbose=verbose,
            walker_kind=wk,
            mesh=mesh,
            mixed_precision=mixed_precision,
            params=params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            params_kwargs=params_kwargs,
            prop_kwargs=prop_kwargs,
            params_builder=_make_params,
            prop_builder=_make_prop,
            default_block_fn=default_block,
            job_cls=JobMixed,
            walker_kind_resolver=_resolve_default_walker_kind,
        )
    job = cast(JobMixed, job)

    # attach the measurement trial
    job.recipe = rec
    job.mix_trial_data = rec.make_trial_data(trial_input.data, job.sys)

    # size the cholesky chunk of a chunked trial against the memory budget
    if rec.plan_chunking is not None:
        assert job.params is not None
        budget = (
            int(float(max_memory) * 1024**2)
            if max_memory is not None
            else device_memory_budget_bytes()
        )
        if budget is not None:
            plan = rec.plan_chunking(
                job.sys,
                job.ham_data,
                job.mix_trial_data,
                n_walkers=job.params.n_walkers,
                budget_bytes=budget,
                n_chunks=job.params.n_chunks,
                nchol_chunk=nchol_chunk,
                mixed_precision=trial_mp,
                n_devices=_walker_devices(mesh),
            )
            nchol_chunk = plan.nchol_chunk
            job.chunk_plan = plan
            # the plan only ever raises n_chunks, so this cannot undercut a caller's setting
            if plan.n_chunks != job.params.n_chunks:
                job.params = replace(job.params, n_chunks=plan.n_chunks)
    elif max_memory is not None:
        raise ValueError(
            f"trial {rec.trial!r} scans one cholesky vector at a time and has no memory "
            "model, so there is nothing for max_memory to size; use the chunked "
            "'pt2ccsd_bar' trial, or drop max_memory."
        )

    job.mix_trial_meas_ops = rec.make_trial_meas_ops(
        job.sys, mixed_precision=trial_mp, nchol_chunk=nchol_chunk
    )
    return job
