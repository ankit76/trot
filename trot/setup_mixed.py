from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ClassVar, Union, cast

from jax.sharding import Mesh

from . import driver
from .core.ops import MeasOps
from .core.system import System, WalkerKind
from .driver import MixedQmcResult
from .mixed import MixedRecipe, get_mixed_recipe
from .prop.afqmc import make_prop_ops, make_prop_ops_u
from .prop.blocks import block as default_block
from .prop.types import QmcParams, QmcParamsBase
from .setup import Job, _assemble_job, _make_params, _resolve_default_walker_kind
from .staging import StagedInputs, TrialInput

# Assembly for mixed guide/trial AFQMC.
#
# The guide side of a mixed run is an ordinary single-bundle job: the walkers propagate
# under the guide wavefunction with the guide's own hamiltonian, ops and propagator. So
# _assemble_job builds it unchanged, and JobMixed only adds the trial that the energy is
# measured against.


def _walker_devices(mesh: Mesh | None) -> int:
    """
    How many devices the walker axis is spread over. Walkers are sharded on the "data"
    axis, so that is what divides the population; a mesh without one is treated as whole.
    """
    if mesh is None:
        return 1
    return int(mesh.shape.get("data", mesh.size))


def _make_prop_mixed(
    ham_data: Any,
    walker_kind: str,
    sys: System | None = None,
    *,
    mixed_precision: bool,
) -> Any:
    # the unrestricted (uchol) hamiltonian has its own propagator, as in setup._make_prop
    if ham_data.basis == "uchol":
        return make_prop_ops_u(ham_data.basis, walker_kind, mixed_precision=mixed_precision)
    return make_prop_ops(ham_data.basis, walker_kind, mixed_precision=mixed_precision)


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
    # how max_memory was split between the two chunking knobs, None if it was not given
    chunk_plan: Any = None
    _runtime_mix_meas_ctx: object | None = field(default=None, init=False, repr=False)

    params_cls: ClassVar[type[QmcParamsBase]] = QmcParams
    driver_fn: ClassVar[Callable[..., Any]] = staticmethod(driver.run_mixed_qmc)

    def mix_meas_ctx(self) -> Any:
        if self._runtime_mix_meas_ctx is None:
            self._runtime_mix_meas_ctx = self.mix_trial_meas_ops.build_meas_ctx(
                self.ham_data, self.mix_trial_data
            )
        return self._runtime_mix_meas_ctx

    def kernel(self, **driver_kwargs: Any) -> MixedQmcResult:
        """
        Run mixed AFQMC: propagate with the guide, measure with the trial.

        Job._prepare_runtime is deliberately not used here. It compacts the cholesky
        tensor to a zero sized placeholder once the propagation context has been built,
        but run_mixed_qmc builds the guide propagation context itself and takes no
        prop_ctx argument, so it would be handed an empty chol. Letting the driver
        construct all three contexts also keeps this path identical to the manual setup
        in examples/pt2ccsd.py.
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
            # both come from the recipe, so they can never be mismatched
            mix_block_fn=self.recipe.mixed_block_fn,
            blocking_fn=self.recipe.blocking_fn,
            **driver_kwargs,
        )


def setup_mixed(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # the trial half, staged separately from the guide
    recipe: MixedRecipe | str = "pt2ccsd",
    trial_input: TrialInput | None = None,
    trial_kwargs: dict[str, Any] | None = None,
    # staging options (used only if we need to stage)
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    # system/prop options
    walker_kind: WalkerKind | None = None,
    mesh: Mesh | None = None,
    mixed_precision: bool = False,
    max_memory: float | None = None,
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

    obj_or_staged supplies the GUIDE (a mean-field object, StagedInputs, or a staged .h5
    path). trial_input supplies the measurement trial; if omitted it must have been
    staged already and passed in, since the trial generally comes from a different pyscf
    object than the guide.

    mixed_precision is single precision for the whole run: the guide propagator and the
    trial estimator both take it. The guide's measurement ops do not honour it yet, so
    today it reaches the guide only through the propagator.

    max_memory (MB) hands the recipe's memory model a budget for the measurement, which
    it splits between the cholesky chunk and the walker chunk. It can raise params
    .n_chunks, never lower it.

    Basic usage is through AfqmcMixed rather than this function directly.
    """
    rec = get_mixed_recipe(recipe) if isinstance(recipe, str) else recipe

    if trial_input is None:
        raise ValueError(
            "setup_mixed needs a staged trial_input; the measurement trial comes from a "
            "different object than the guide, so it cannot be inferred here."
        )

    job = _assemble_job(
        obj_or_staged,
        norb_frozen_core=norb_frozen_core,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
        walker_kind=walker_kind or cast(WalkerKind, rec.walker_kind),
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
        prop_builder=_make_prop_mixed,
        default_block_fn=default_block,
        job_cls=JobMixed,
        walker_kind_resolver=_resolve_default_walker_kind,
    )
    job = cast(JobMixed, job)

    # attach the measurement trial
    job.recipe = rec
    job.mix_trial_data = rec.make_trial_data(trial_input.data, job.sys)

    meas_kwargs = dict(trial_kwargs or {})

    if max_memory is not None:
        if rec.plan_chunking is None:
            raise ValueError(
                f"trial {rec.trial!r} measures with the unchunked energy kernel, which "
                "has nothing for max_memory to size. Use trial='pt2ccsd_chunk' or "
                "'pt2ccsd_bar', or drop max_memory."
            )
        assert job.params is not None
        plan = rec.plan_chunking(
            job.sys,
            job.ham_data,
            job.mix_trial_data,
            n_walkers=job.params.n_walkers,
            max_memory_mb=max_memory,
            n_chunks=job.params.n_chunks,
            nchol_chunk=meas_kwargs.get("nchol_chunk"),
            mixed_precision=mixed_precision,
            n_devices=_walker_devices(mesh),
        )
        meas_kwargs["nchol_chunk"] = plan.nchol_chunk
        job.chunk_plan = plan
        # the plan only ever raises n_chunks, so this cannot undercut a caller's setting
        if plan.n_chunks != job.params.n_chunks:
            job.params = replace(job.params, n_chunks=plan.n_chunks)

    job.mix_trial_meas_ops = rec.make_trial_meas_ops(
        job.sys, **meas_kwargs, mixed_precision=mixed_precision
    )
    return job
