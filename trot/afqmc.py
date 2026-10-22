from __future__ import annotations

from .config import configure_once

configure_once()

import copy
import dataclasses
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

print = partial(print, flush=True)

from jax.sharding import Mesh

from . import staging
from .cisd_workflow import (
    CisdWorkflowConfig,
    PreparedCisdModes,
    cached_representation_key,
    prepare_cisd_modes,
)
from .core.system import WalkerKind
from .driver import QmcResult
from .prop.types import QmcParams, QmcParamsBase, QmcParamsFp, QmcParamsLno
from .runtime_provenance import print_runtime_provenance
from .setup import Job
from .setup import setup as setup_job
from .setup_fp import JobFp
from .setup_fp import setup_fp as setup_job_fp

# from .setup_lno import setup_lno as setup_job_lno
# from . import setup_lno
from .staging import StagedInputs, _is_cc_like
from .staging import dump as dump_staged
from .staging import load as load_staged
from .staging import stage as stage_inputs


def banner_afqmc() -> str:
    return r"""
    ████████╗██████╗  ██████╗ ████████╗
    ╚══██╔══╝██╔══██╗██╔═══██╗╚══██╔══╝
       ██║   ██████╔╝██║   ██║   ██║
       ██║   ██╔══██╗██║   ██║   ██║
       ██║   ██║  ██║╚██████╔╝   ██║
       ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝
  Trotter-propagated Random Orbital Trajectories
differentiable auxiliary-field quantum Monte Carlo
"""


def _frozen_cache_key(frozen: int | ArrayLike | None) -> int | tuple[int, ...] | None:
    if isinstance(frozen, np.ndarray):
        arr = np.asarray(frozen, dtype=np.int64).reshape(-1)
        return tuple(int(x) for x in arr)
    if isinstance(frozen, (list, tuple)):
        arr = np.asarray(frozen, dtype=np.int64).reshape(-1)
        return tuple(int(x) for x in arr)
    if frozen is None:
        return None
    if isinstance(frozen, (int, np.integer)):
        return int(frozen)
    raise TypeError(f"Unsupported frozen type for cache key: {type(frozen)}")


class Afqmc:
    """
    AFQMC driver object.

    Parameters
    ----------
    mf_or_cc : Any
        Mean-field or coupled-cluster object from which to build Hamiltonian and trial wavefunction.
    norb_frozen_core : int, optional
        Preferred name for the number of lowest occupied core orbitals removed from the AFQMC
        Hamiltonian.
    norb_frozen : int, optional
        Backward-compatible alias for ``norb_frozen_core``. For CC objects with integer
        ``cc.frozen``, this is inferred from ``cc.frozen``. For restricted CCSD objects with
        list-valued ``cc.frozen``, the trial-space frozen occupied/virtual blocks are inferred
        from ``cc.frozen`` while ``norb_frozen_core``/``norb_frozen`` control the occupied core
        orbitals removed from the AFQMC Hamiltonian.
    chol_cut : float, optional
        Cholesky decomposition cutoff, by default 1e-5
    cache : Union[str, Path], optional
        Path to cache file for staged inputs, by default None
    n_eql_blocks : int, optional
        Number of equilibration blocks if params is not provided, by default 20
    n_blocks : int, optional
        Number of production blocks if params is not provided, by default 200
    seed : int | None, optional
        Random seed if params is not provided, by default None
    dt : float | None, optional
        Time step if params is not provided, by default None
    n_walkers : int | None, optional
        Number of walkers if params is not provided, by default None
    n_chunk : int | None, optional
        Number of chunks if params is not provided, by default 1
    error_method : {"gamma", "blocking"} | None, optional
        Primary energy-error estimator. Both analyses are always evaluated;
        "gamma" is reported by default.
    cisd_workflow : CisdWorkflowConfig | None, optional
        Opt-in retained-mode and pair-sampled energy workflow for staged CISD
        or UCISD trials. Mode construction can be cached on a CPU node with
        :meth:`prepare_cisd_trial_cache`; pair-sampling is tuned after AFQMC
        equilibration.
    """

    params_cls = QmcParams
    job_cls = Job
    setup_fn = staticmethod(setup_job)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen_core: int | None = None,
        norb_frozen: int | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
        error_method: Literal["gamma", "blocking"] | None = None,
        cisd_workflow: CisdWorkflowConfig | None = None,
    ):
        self._obj = mf_or_cc
        self._cc: Any = None
        if _is_cc_like(mf_or_cc):
            self._cc = mf_or_cc
            self._scf = mf_or_cc._scf
            self.source_kind = "cc"
        else:
            self._scf = mf_or_cc
            self.source_kind = "mf"

        resolved_norb_frozen = staging._resolve_stage_frozen_arg(
            norb_frozen_core, norb_frozen, None
        )
        assert resolved_norb_frozen is None or isinstance(resolved_norb_frozen, int)
        self.norb_frozen_core = resolved_norb_frozen
        self.norb_frozen = resolved_norb_frozen
        self.chol_cut = float(chol_cut)
        self.cache = Path(cache).expanduser().resolve() if cache is not None else None
        self.overwrite_cache = False
        self.verbose = False

        self.walker_kind: WalkerKind | None = None  # resolved in kernel
        self.mixed_precision = True
        self.cisd_workflow = cisd_workflow

        self.params: QmcParamsBase | None = None  # resolved in kernel
        defaults = self.params_cls()
        self.dt = defaults.dt if dt is None else dt
        self.n_walkers = defaults.n_walkers if n_walkers is None else n_walkers
        self.n_blocks = defaults.n_blocks if n_blocks is None else n_blocks
        self.seed = defaults.seed if seed is None else seed
        self.n_chunks = defaults.n_chunks if n_chunks is None else n_chunks
        if hasattr(defaults, "n_eql_blocks"):
            self.n_eql_blocks = defaults.n_eql_blocks if n_eql_blocks is None else n_eql_blocks
        if hasattr(defaults, "error_method"):
            self.error_method = defaults.error_method if error_method is None else error_method

        self._staged: StagedInputs | None = None
        self._job: Job | None = None
        self._job_cisd_workflow: CisdWorkflowConfig | None = None
        self._cache_key: tuple | None = None

        self.e_tot: Any = None
        self.e_err: Any = None
        self.block_energies: Any = None
        self.block_weights: Any = None

    @property
    def staged(self) -> StagedInputs | None:
        return self._staged

    @property
    def job(self) -> Job | None:
        return self._job

    def _dump_params(self, params: QmcParamsBase) -> None:
        fields = dataclasses.fields(params)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        print(f" {type(params).__name__}:")
        for field in fields:
            print(f"  {field.name:<{width}} = {getattr(params, field.name)}")
        print("")

    def _resolve_meas_cfg(self, job: Job) -> object | None:
        from .meas.cisd import get_cisd_meas_cfg
        from .meas.rhf import get_rhf_meas_cfg

        for getter in (get_cisd_meas_cfg, get_rhf_meas_cfg):
            cfg = getter(job.meas_ops)
            if cfg is not None:
                return cfg
        return None

    def _dump_cfg(self, name: str, cfg: object) -> None:
        if not dataclasses.is_dataclass(cfg):
            print(f" {name:<15} = {cfg}")
            return

        print(f" {name:<15} = {type(cfg).__name__}")
        fields = dataclasses.fields(cfg)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        for field in fields:
            value = getattr(cfg, field.name)
            if isinstance(value, type):
                value_str = value.__name__
            else:
                value_str = str(value)
            print(f"  {field.name:<{width}} = {value_str}")

    def dump_flags(self, job: Job) -> None:
        self._dump_flags_helper(job)

    def _dump_flags_helper(self, job: Job) -> None:
        meta = job.staged.meta
        src = meta["source_kind"]
        chol_cut = meta["chol_cut"]
        sys = job.sys
        nchol = job.ham_data.nchol
        params = job.params
        trial = job.staged.trial
        print("\n******** AFQMC ********")
        print(f" norb            = {sys.norb}")
        print(f" nelec_up        = {sys.nelec[0]}")
        print(f" nelec_dn        = {sys.nelec[1]}")
        print(f" nchol           = {nchol}")
        print(f" source_kind     = {src}")
        print(f" trial_kind      = {trial.kind}")
        print(f" chol_cut        = {chol_cut:g}")
        print(f" cache           = {str(self.cache) if self.cache else None}")
        print(f" walker_kind     = {sys.walker_kind}")
        print(f" mixed_precision = {self.mixed_precision}\n")
        meas_cfg = self._resolve_meas_cfg(job)
        if meas_cfg is not None:
            self._dump_cfg("meas_cfg", meas_cfg)
            print("")
        if self.cisd_workflow is not None:
            self._dump_cfg("cisd_workflow", self.cisd_workflow)
            representation = job.staged.meta.get("trial_representation")
            if representation is not None:
                self._dump_cfg("trial_repr", representation)
            print("")
        self._dump_params(params)

    def _key(self) -> tuple:
        """Key for determining whether staged/job caches are still valid."""
        cache_mtime = None
        if self.cache is not None and self.cache.exists():
            cache_mtime = self.cache.stat().st_mtime
        return (
            self.source_kind,
            _frozen_cache_key(self.norb_frozen_core),
            float(self.chol_cut),
            str(self.cache) if self.cache is not None else None,
            bool(self.overwrite_cache),
            cache_mtime,
            repr(self.cisd_workflow) if self.cisd_workflow is not None else None,
        )

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Compute or load HamInput/TrialInput.
        If cache is set and exists, loads unless overwrite_cache=True.
        """
        if isinstance(self._obj, StagedInputs):
            if self._staged is None or force:
                self._staged = self._obj
                self._cache_key = self._key()
                self._job = None
            return self._staged

        if isinstance(self.norb_frozen_core, (list, tuple, np.ndarray)):
            raise TypeError(
                "Array-valued frozen is reserved for LNO orbital-list staging; "
                "use AfqmcLnoFrag(..., frozen_orbitals=...)."
            )

        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        derived_trial_key = (
            cached_representation_key(self.cache, self.cisd_workflow)
            if self.cache is not None and self.cache.exists()
            else None
        )
        staged = stage_inputs(
            self._obj,
            norb_frozen_core=(
                int(self.norb_frozen_core) if self.norb_frozen_core is not None else None
            ),
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self.verbose,
            derived_trial_key=derived_trial_key,
        )
        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def save_staged(self, path: Union[str, Path]) -> None:
        """Write current staged inputs to a single file cache."""
        staged = self.stage()
        destination = Path(path).expanduser().resolve()
        if self.cache is not None and self.cache.exists():
            if destination != self.cache:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.cache, destination)
            return
        dump_staged(staged, destination)

    def prepare_cisd_trial_cache(
        self,
        path: Union[str, Path] | None = None,
        *,
        overwrite: bool = False,
    ) -> PreparedCisdModes:
        """Build and cache the configured CISD modes on the current host.

        The raw staged amplitudes remain canonical in ``trial/data`` and the
        derived modes are added below ``trial/derived`` in the same HDF5 file.
        Calling this immediately after the PySCF CC calculation keeps the
        factorization on the CPU preparation node. A later GPU-side
        :meth:`from_staged` call loads only the selected derived representation.
        """

        if self.cisd_workflow is None:
            raise ValueError(
                "prepare_cisd_trial_cache requires cisd_workflow to be configured."
            )
        cache_path = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.cache
        )
        if cache_path is None:
            raise ValueError(
                "prepare_cisd_trial_cache requires a path argument or Afqmc(cache=...)."
            )

        if cache_path.exists():
            prepared = prepare_cisd_modes(
                cache_path,
                self.cisd_workflow.modes,
                overwrite=overwrite,
                verbose=self.verbose,
            )
        else:
            staged = self.stage()
            prepared = prepare_cisd_modes(
                staged,
                self.cisd_workflow.modes,
                cache=cache_path,
                overwrite=overwrite,
                verbose=self.verbose,
            )

        if self.cache is not None and cache_path == self.cache:
            self._staged = load_staged(
                cache_path,
                derived_trial_key=prepared.cache_key,
            )
            self._cache_key = self._key()
            self._job = None
        return prepared

    # def load_staged(self, path: Union[str, Path]): -> StagedInputs:
    #    """Load staged inputs from a cache file and attach them to this object."""
    #    staged = load_staged(path)
    #    self._staged = staged
    #    self._cache_key = None
    #    self._job = None
    #    return staged

    def _validate_params(self, params: QmcParamsBase) -> QmcParamsBase:
        if isinstance(params, QmcParams) and params.error_method not in ("gamma", "blocking"):
            raise ValueError(
                "error_method must be either 'gamma' or 'blocking'; "
                f"received {params.error_method!r}"
            )
        return params

    def _make_params(self) -> QmcParamsBase:
        """
        Create QmcParams if user didn't provide one.
        """
        params_cls = self.params_cls

        if self.params is not None and isinstance(self.params, params_cls):
            params = self.params
        elif self.params is not None and not isinstance(self.params, params_cls):
            raise TypeError(
                f"Expected type {params_cls.__name__} for self.params, but received '{type(self.params)}'"
            )
        else:
            kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(params_cls):
                if hasattr(self, field.name):
                    val = getattr(self, field.name)
                    if val is not None:
                        kwargs[field.name] = val

            params = params_cls(**kwargs)

        return self._validate_params(params)

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Callable[..., Any] | None = None,
        prop_kwargs: dict[str, Any] | None = None,
        mesh: Mesh | None = None,
    ) -> Job:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        if (
            self._job is not None
            and not force
            and self._job_cisd_workflow == self.cisd_workflow
            and (mesh is None or self._job.mesh is mesh)
        ):
            return self._job

        staged = self.stage()
        qmc_params = self._make_params()
        self.params = qmc_params

        setup_kwargs: dict[str, Any] = {}
        if self.cisd_workflow is not None:
            if self.setup_fn is not setup_job:
                raise ValueError("cisd_workflow is currently supported only by standard AFQMC.")
            setup_kwargs["cisd_workflow"] = self.cisd_workflow

        job = self.setup_fn(
            staged,
            walker_kind=self.walker_kind,
            mesh=mesh,
            mixed_precision=self.mixed_precision,
            params=cast(Any, qmc_params),
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
            **setup_kwargs,
        )
        self._job = job
        self._job_cisd_workflow = self.cisd_workflow
        return job

    def _coerce_result(self, value: Any) -> Any:
        return float(value)

    def kernel(self, **driver_kwargs: Any) -> tuple[Any, Any]:
        """Run AFQMC and return ``(e_tot, e_err)``.

        For standard importance-sampled AFQMC, ``e_err`` uses ``error_method``
        (the Gamma method by default).  Both Gamma and blocking estimates and
        their diagnostics are retained on ``self.qmc_result``.
        """
        print(banner_afqmc())
        print_runtime_provenance()
        mesh = driver_kwargs.get("mesh")
        job = self.build_job(mesh=mesh)
        self.dump_flags(job)

        qmc_result = job.kernel(**driver_kwargs)

        e_tot = float(qmc_result.mean_energy)
        e_err = float(qmc_result.stderr_energy)

        self.qmc_result = qmc_result

        return e_tot, e_err

    run = kernel

    @classmethod
    def _from_staged_common(cls, path: Union[str, Path], **kwargs: Any):
        cache_path = Path(path).expanduser().resolve()
        workflow = kwargs.get("cisd_workflow")
        derived_trial_key = cached_representation_key(cache_path, workflow)
        staged = load_staged(cache_path, derived_trial_key=derived_trial_key)
        meta = staged.meta

        af = cls(
            None,
            norb_frozen_core=meta["frozen"],
            chol_cut=meta["chol_cut"],
            cache=cache_path,
            **kwargs,
        )
        af._staged = staged
        af.source_kind = meta["source_kind"]
        af._cache_key = af._key()
        return af

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
        error_method: Literal["gamma", "blocking"] | None = None,
        cisd_workflow: CisdWorkflowConfig | None = None,
    ) -> Afqmc:
        """
        Returns a new AFQMC object from a previously staged calculations
        (using save_staged method). The number of frozen core orbitals, norb_frozen_core
        (legacy alias ``norb_frozen``),
        and the cholesky decomposition threshold, chol_cut, cannot be changed.
        Parameters
        ----------
        path: str, pathlib.Path
        The other parameters are identical to the ones in the AFQMC class.
        """
        return cls._from_staged_common(
            path,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
            error_method=error_method,
            cisd_workflow=cisd_workflow,
        )


class AfqmcFp(Afqmc):
    params_cls = QmcParamsFp
    job_cls = JobFp
    setup_fn = staticmethod(setup_job_fp)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen_core: int | None = None,
        norb_frozen: int | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
        ene0: float | None = None,
        n_traj: int | None = None,
    ):
        super().__init__(
            mf_or_cc,
            norb_frozen_core=norb_frozen_core,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=None,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )
        defaults = self.params_cls()
        self.n_prop_steps = defaults.n_prop_steps if n_prop_steps is None else n_prop_steps
        self.n_traj = defaults.n_traj if n_traj is None else n_traj
        self.ene0 = ene0

    def _validate_params(self, params: QmcParamsBase) -> QmcParamsBase:
        assert isinstance(params, QmcParamsFp)
        if params.ene0 is None:
            raise ValueError(
                "The value of the parameter 'ene0' must be set, typically with SCF or CC energy."
            )
        return params

    def _coerce_result(self, value: Any) -> Any:
        return value

    def kernel(self, **driver_kwargs: Any) -> tuple[Any, Any]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        print(banner_afqmc())
        print_runtime_provenance()
        mesh = driver_kwargs.get("mesh")
        job = self.build_job(mesh=mesh)
        self.dump_flags(job)

        qmc_result = job.kernel(**driver_kwargs)

        e_tot = qmc_result.mean_energy
        e_err = qmc_result.stderr_energy

        self.qmc_result = qmc_result

        return e_tot, e_err

    run_fp = kernel

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
    ) -> AfqmcFp:
        """
        Returns a new AFQMC object from a previously staged calculations
        (using save_staged method). The number of frozen core orbitals, norb_frozen_core
        (legacy alias ``norb_frozen``),
        and the choliesky decomposition threshold, chol_cut, cannot be changed.
        Parameters
        ----------
        path: str, pathlib.Path
        The other parameters are identical to the ones in the AFQMC class.
        """
        return cls._from_staged_common(
            path,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_prop_steps=n_prop_steps,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )


class AfqmcLnoFrag(Afqmc):
    params_cls = QmcParamsLno

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        frozen_orbitals: ArrayLike | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
        prjlo: NDArray | None = None,
        error_method: Literal["gamma", "blocking"] | None = None,
    ):
        super().__init__(
            mf_or_cc,
            norb_frozen_core=0,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
            error_method=error_method,
        )

        self.mixed_precision = False
        self.prjlo = prjlo
        self.frozen_orbitals = frozen_orbitals

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Compute or load HamInput/TrialInput.
        If cache is set and exists, loads unless overwrite_cache=True.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        frozen_orbitals = self.frozen_orbitals
        if frozen_orbitals is None:
            frozen_orbitals = np.zeros((0,), dtype=np.int64)

        ham = staging.build_ham_lno(
            self._obj,
            frozen_orbitals=frozen_orbitals,
            chol_cut=self.chol_cut,
        )

        staged = stage_inputs(
            self._obj,
            frozen_orbitals=frozen_orbitals,
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self.verbose,
            ham=ham,
            trial=None,
        )

        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def _key(self) -> tuple:
        base = super()._key()
        return base + (_frozen_cache_key(self.frozen_orbitals),)

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
        prjlo: NDArray | None = None,
        error_method: Literal["gamma", "blocking"] | None = None,
    ) -> "AfqmcLnoFrag":
        staged = load_staged(path)
        meta = staged.meta
        frozen_orbitals = meta["frozen"]
        if frozen_orbitals is not None and not isinstance(frozen_orbitals, np.ndarray):
            frozen_orbitals = np.asarray(frozen_orbitals, dtype=np.int64)

        af = cls(
            None,
            frozen_orbitals=frozen_orbitals,
            chol_cut=meta["chol_cut"],
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
            prjlo=prjlo,
            error_method=error_method,
        )
        af._staged = staged
        af.source_kind = meta["source_kind"]
        af._cache_key = af._key()
        return af

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Callable[..., Any] | None = None,
        prop_kwargs: dict[str, Any] | None = None,
    ) -> Job:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        from .core.system import System
        from .meas.rhf import make_lno_rhf_meas_ops

        if self._job is not None and not force:
            return self._job

        if meas_ops is not None:
            raise ValueError("meas_ops must be None as we overwrite it.")

        staged = self.stage()
        params = self._make_params()
        assert isinstance(params, QmcParamsLno)
        self.params = params

        ham = staged.ham
        walker_kind = ham.basis
        sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind=walker_kind)
        meas_ops = make_lno_rhf_meas_ops(sys=sys, params=params)

        job = setup_job(
            staged,
            walker_kind=self.walker_kind,
            mixed_precision=self.mixed_precision,
            params=params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
        )
        self._job = job
        return job

    def kernel(self, **driver_kwargs: Any) -> tuple[NDArray, NDArray]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        print(banner_afqmc())
        job = self.build_job()
        self.dump_flags(job)

        obs = driver_kwargs.get("observable_names", ())
        if "orb_corr" not in obs:
            driver_kwargs["observable_names"] = obs + ("orb_corr",)

        qmc_result = job.kernel(**driver_kwargs)

        if not isinstance(qmc_result, QmcResult):
            raise TypeError(
                f"Unexpected return from Job.kernel(), expected QmcResult but received {type(qmc_result)}."
            )

        orb_corr = np.array(qmc_result.observable_means["orb_corr"].real)
        orb_corr_stderr = np.array(qmc_result.observable_stderrs["orb_corr"])

        self.qmc_result = qmc_result

        return orb_corr, orb_corr_stderr


def run_afqmc_lno_helper(
    mf: Any,
    norb_act=None,
    nelec_act=None,
    mo_coeff=None,
    frozen_orbitals: ArrayLike | None = None,
    chol_cut: float = 1e-5,
    seed: int | None = None,
    dt: float = 0.005,
    n_walkers: int = 5,
    nblocks: int = 1000,
    target_error: float = 1e-4,
    prjlo: NDArray | None = None,
    n_eql: int = 2,
):
    from pyscf import scf

    # choose the orbital basis
    if mo_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            mo_coeff = mf.mo_coeff[0]
        elif isinstance(mf, scf.rhf.RHF):
            mo_coeff = mf.mo_coeff
        else:
            raise Exception("# Invalid mean field object!")

    mf2 = copy.deepcopy(mf)
    mf2.mo_coeff = mo_coeff

    myafqmc = AfqmcLnoFrag(
        mf2,
        frozen_orbitals=frozen_orbitals,
        chol_cut=chol_cut,
        n_eql_blocks=n_eql,
        n_blocks=nblocks,
        seed=seed,
        dt=dt,
        n_walkers=n_walkers,
        prjlo=prjlo,
    )
    mean_ecorr, err_ecorr = myafqmc.kernel(target_error=target_error)

    return mean_ecorr, err_ecorr


# Backward-compatible aliases
AFQMC = Afqmc
AFQMCFp = AfqmcFp


# ======================================================================================
# unrestricted (uchol) hamiltonian
# ======================================================================================

from .setup_u import setup_uh as setup_job_uh  # noqa: E402
from .staging_u import dump_uh as dump_staged_uh  # noqa: E402
from .staging_u import load_uh as load_staged_uh  # noqa: E402
from .staging_u import stage_uh as stage_inputs_uh  # noqa: E402


class AfqmcUh(Afqmc):
    """
    AFQMC with an unrestricted (uchol) hamiltonian.

    Alpha and beta each keep their own orbital basis, so h1 and the cholesky vectors are
    carried per spin and norb_a may differ from norb_b. The auxiliary field index stays
    shared between the spins, so the opposite spin interaction is recovered from
    L^a_g and L^b_g. Contrast with ``Afqmc(mf); af.walker_kind = "unrestricted"``, which
    uses unrestricted *walkers* against a hamiltonian built in the alpha MO basis alone.

        af = AfqmcUh(mf)                       # mf a pyscf UHF, or rhf.to_uhf(): UHF trial
        af = AfqmcUh(mycc)                     # mycc a pyscf UCCSD: CC-derived UCISD trial
        mean, err = af.kernel()

    Two independently chosen orbital sets, which may differ in size (their leading
    columns must be the occupied orbitals of that spin):

        af = AfqmcUh(mf, basis_a=c_a, basis_b=c_b)

    Parameters
    ----------
    mf_or_cc : Any
        pyscf UHF mean field (trial: the UHF determinant) or UCCSD object (trial: the
        UCISD built from the CC amplitudes, each spin in its own MO basis).
    basis_a, basis_b : NDArray, optional
        Orbital bases for the two spins. Default to the object's alpha and beta MOs.
    norb_frozen_core : int | (int, int), optional
        Frozen core orbitals, the same number for both spins or a pair (n_a, n_b). For a
        UCCSD object it is cc.frozen and need not be given.
    The remaining parameters are those of ``Afqmc``. Only unrestricted walkers are
    supported; ``walker_kind`` is fixed to "unrestricted".
    """

    params_cls = QmcParams
    job_cls = Job
    setup_fn = staticmethod(setup_job_uh)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        basis_a: NDArray | None = None,
        basis_b: NDArray | None = None,
        norb_frozen_core: int | tuple[int, int] | None = None,
        norb_frozen: int | tuple[int, int] | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
        error_method: Literal["gamma", "blocking"] | None = None,
        cisd_workflow: CisdWorkflowConfig | None = None,
    ):
        from .cholesky import normalize_frozen_core_uh

        if cisd_workflow is not None:
            raise ValueError("cisd_workflow is not supported on the unrestricted hamiltonian.")
        if norb_frozen_core is not None and norb_frozen is not None:
            if normalize_frozen_core_uh(norb_frozen_core) != normalize_frozen_core_uh(norb_frozen):
                raise ValueError(
                    "norb_frozen_core and norb_frozen must match when both are passed."
                )
        frozen = norb_frozen_core if norb_frozen_core is not None else norb_frozen
        # None stays None: for a CC object the core is cc.frozen, resolved at staging
        frozen_uh = None if frozen is None else normalize_frozen_core_uh(frozen)

        # the base class resolves an integer core only; the per spin pair is kept here
        super().__init__(
            mf_or_cc,
            norb_frozen_core=None,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
            error_method=error_method,
        )
        self.norb_frozen_core = frozen_uh
        self.norb_frozen = frozen_uh
        self.basis_a = None if basis_a is None else np.asarray(basis_a)
        self.basis_b = None if basis_b is None else np.asarray(basis_b)
        # alpha and beta live in different orbital spaces, so no other kind applies
        self.walker_kind = "unrestricted"

    def _key(self) -> tuple:
        return super()._key() + (
            None if self.basis_a is None else id(self.basis_a),
            None if self.basis_b is None else id(self.basis_b),
        )

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Build the unrestricted hamiltonian and the UHF trial (staging_u.stage_uh). The
        staged inputs carry a HamInputU in the ham slot.
        """
        if isinstance(self._obj, StagedInputs):
            if self._staged is None or force:
                self._staged = self._obj
                self._cache_key = self._key()
                self._job = None
            return self._staged

        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        staged = stage_inputs_uh(
            self._obj,
            norb_frozen_core=cast(Any, self.norb_frozen_core),
            chol_cut=self.chol_cut,
            basis_a=self.basis_a,
            basis_b=self.basis_b,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self.verbose,
        )
        self._staged = staged
        # the resolved per spin core (from cc.frozen for a CC object)
        frozen_pair: Any = staged.ham.frozen  # StagedInputs.ham is typed as the restricted HamInput
        self.norb_frozen_core = (int(frozen_pair[0]), int(frozen_pair[1]))
        self.norb_frozen = self.norb_frozen_core
        self._cache_key = self._key()
        self._job = None
        return staged

    def save_staged(self, path: Union[str, Path]) -> None:
        """Write the staged uchol inputs to a single file (staging_u.dump_uh)."""
        staged = self.stage()
        destination = Path(path).expanduser().resolve()
        if self.cache is not None and self.cache.exists():
            if destination != self.cache:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.cache, destination)
            return
        dump_staged_uh(staged, destination)

    @classmethod
    def _from_staged_common(cls, path: Union[str, Path], **kwargs: Any):
        cache_path = Path(path).expanduser().resolve()
        staged = load_staged_uh(cache_path)
        meta = staged.meta

        kwargs.pop("cisd_workflow", None)
        af = cls(
            None,
            norb_frozen_core=(int(meta["frozen"][0]), int(meta["frozen"][1])),
            chol_cut=meta["chol_cut"],
            cache=cache_path,
            **kwargs,
        )
        af._staged = staged
        af.source_kind = meta["source_kind"]
        af._cache_key = af._key()
        return af


# ======================================================================================
# mixed guide/trial AFQMC (pt2CCSD estimators)
# ======================================================================================

import math  # noqa: E402

from .mixed import MixedRecipe, get_mixed_recipe  # noqa: E402
from .setup_mixed import JobMixed, setup_mixed  # noqa: E402
from .staging import StagedMfOrCc  # noqa: E402
from .staging_u import stage_ham_input_df  # noqa: E402


def _kernel_location(fn: Any) -> str:
    """
    Where a kernel is defined: the path inside the trot package, plus the function name.
    Partials are unwrapped to the function they call.
    """
    while isinstance(fn, partial):
        fn = fn.func
    name = getattr(fn, "__name__", type(fn).__name__)
    module = getattr(fn, "__module__", "")
    parts = module.split(".") if module else []
    if parts and parts[0] == __name__.split(".")[0]:
        parts = parts[1:]
    return f"{'/'.join(parts)}.py:{name}" if parts else name


class AfqmcMixed(Afqmc):
    """
    Mixed guide/trial AFQMC.

    The walkers propagate under a guide wavefunction while the energy is measured against
    a different trial:

        |AFQMC> = sum_i w_i |phi_i> / <G|phi_i>
        E_T     = energy_fn(h0, <c>),   wp_i = w_i <T|phi_i> / <G|phi_i>

    Both wavefunctions come from the one pyscf CC object given. The guide is named with
    ``guide`` and the trial with ``trial``; any registered pair that agrees on the
    hamiltonian and on a walker kind can be run (``trot.mixed``).

        mycc = cc.CCSD(mf); mycc.kernel()
        af = AfqmcMixed(mycc)                                    # RHF guide, pt2ccsd trial
        af = AfqmcMixed(mycc, guide="cisd", trial="pt2ccsd_bar") # CISD guide, bar estimator

        mycc = cc.UCCSD(mf); mycc.kernel()
        af = AfqmcMixed(mycc)                                    # UHF guide, upt2ccsd trial
        af = AfqmcMixed(mycc, guide="ucisd", trial="upt2ccsd_bar")
        mean, err = af.kernel()        # returns the TRIAL energy

    kernel() returns the trial energy; the guide result is kept alongside:

        af.e_tot,       af.e_err        # trial
        af.guide_e_tot, af.guide_e_err  # guide
        af.qmc_result                   # the full MixedQmcResult

    Parameters
    ----------
    mf_or_cc : Any
        pyscf CCSD or UCCSD object. The guide and the trial are both built from it: HF
        guides from the mean field under it, CISD guides and the pt2CCSD trials from the
        amplitudes. The frozen core follows ``cc.frozen``.
    trial : str, optional
        Which estimator the energy is measured against; ``trot.mixed.available_trials()``
        lists them. By default ``"pt2ccsd"`` for a CCSD object and ``"upt2ccsd"`` for a
        UCCSD one (the unrestricted hamiltonian of ``AfqmcUh``, alpha and beta each in
        their own basis). The ``_bar`` variants apply exp(T1) to the right, onto the
        hamiltonian and the walker, so the bra is the bare reference determinant; they
        give the same energy faster and are chunked over the cholesky index.
    guide : str, optional
        Which wavefunction propagates the walkers, by default the trial's corresponding
        HF (``"rhf"`` / ``"uhf"``); ``"cisd"`` / ``"ucisd"`` use the CC-derived CISD.
    nchol_chunk : int, optional
        Cholesky vectors per scan step of a chunked trial, set directly; by default it
        is sized by the trial's memory model against the memory budget.
    max_memory : float, optional
        Memory budget of the trial measurement in MB, per device. By default a share of
        the device allocator limit is used (meas.pt2ccsd_chunking.DEVICE_MEMORY_FRACTION);
        on a backend that reports none the default chunk size is used.
    mixed_precision : bool, optional
        Single precision for the T2 contractions of the trial estimator and the
        propagator's cholesky products, by default True; partial sums are always
        accumulated in double. It sets both sides unless one of the two knobs below
        overrides it.
    guide_mixed_precision, trial_mixed_precision : bool, optional
        The precision of the guide (its propagator and kernels) and of the trial
        estimator separately; each defaults to ``mixed_precision``. Setting them apart
        shows where a change of speed or accuracy comes from.
    basis_a, basis_b : NDArray, optional
        Unrestricted hamiltonian only: the alpha and beta orbital bases, as in AfqmcUh.
        They default to the UCCSD object's own MOs, which is where its amplitudes live.
    error_method : {"blocking", "gamma"}, optional
        The reported error of the trial energy, by default "blocking": the jackknife
        blocking analysis of the component ratios, which does not depend on a
        linearization of the energy expression. Both analyses are printed.
    tau_eql : float, optional
        Equilibration length in imaginary time. When given, the number of equilibration
        blocks is derived from it, n_eql_blocks = ceil(tau_eql / (dt * n_prop_steps)), so
        the run equilibrates to (at least) tau_eql whatever the time step and the steps
        per block are. It cannot be combined with n_eql_blocks.
    The remaining parameters are those of ``Afqmc``.
    """

    params_cls = QmcParams
    job_cls = JobMixed
    setup_fn = staticmethod(setup_mixed)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        trial: str | None = None,
        guide: str | None = None,
        nchol_chunk: int | None = None,
        max_memory: float | None = None,
        mixed_precision: bool = True,
        guide_mixed_precision: bool | None = None,
        trial_mixed_precision: bool | None = None,
        basis_a: NDArray | None = None,
        basis_b: NDArray | None = None,
        norb_frozen_core: int | None = None,
        norb_frozen: int | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
        error_method: Literal["gamma", "blocking"] | None = "blocking",
        tau_eql: float | None = None,
    ):
        from pyscf.cc.uccsd import UCCSD

        if tau_eql is not None and n_eql_blocks is not None:
            raise ValueError("pass either tau_eql or n_eql_blocks, not both.")
        if tau_eql is not None and float(tau_eql) < 0.0:
            raise ValueError(f"tau_eql must be non-negative, got {tau_eql}.")
        if not _is_cc_like(mf_or_cc):
            raise ValueError(
                "AfqmcMixed measures against a pt2CCSD trial, which needs a pyscf CC object; "
                f"got {type(mf_or_cc).__name__}."
            )
        unrestricted_cc = isinstance(mf_or_cc, UCCSD)
        if trial is None:
            trial = "upt2ccsd" if unrestricted_cc else "pt2ccsd"

        # guide=None takes the trial's default guide, the corresponding HF
        self.recipe: MixedRecipe = get_mixed_recipe(trial, guide)
        self.trial: str = self.recipe.trial
        self.guide: str = self.recipe.guide
        trial_spec = self.recipe.trial_spec
        if trial_spec.cc_kind == "uccsd" and not unrestricted_cc:
            raise ValueError(
                f"trial={self.trial!r} needs a UCCSD object, got {type(mf_or_cc).__name__}."
            )
        if trial_spec.cc_kind == "ccsd" and unrestricted_cc:
            raise ValueError(
                f"trial={self.trial!r} needs a restricted CCSD object, got a UCCSD one; "
                "use the u-prefixed trial."
            )
        if (basis_a is not None or basis_b is not None) and self.recipe.ham_basis != "uchol":
            raise ValueError(
                "basis_a / basis_b set the alpha and beta orbital bases of the unrestricted "
                f"hamiltonian, so they apply only to the unrestricted trials, not {self.trial!r}."
            )

        # the frozen core is the CC object's; an explicit count may repeat it only
        cc_frozen = getattr(mf_or_cc, "frozen", None)
        if cc_frozen is not None and not isinstance(cc_frozen, (int, np.integer)):
            raise NotImplementedError("list-valued cc.frozen is not supported by AfqmcMixed.")
        cc_core = int(cc_frozen or 0)
        given = norb_frozen_core if norb_frozen_core is not None else norb_frozen
        if given is not None and int(given) != cc_core:
            raise ValueError(f"norb_frozen_core={given} contradicts cc.frozen={cc_core}.")

        super().__init__(
            mf_or_cc,
            norb_frozen_core=cc_core,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
            error_method=error_method,
        )
        defaults = self.params_cls()
        self.n_prop_steps = defaults.n_prop_steps if n_prop_steps is None else n_prop_steps

        self.basis_a = None if basis_a is None else np.asarray(basis_a)
        self.basis_b = None if basis_b is None else np.asarray(basis_b)
        self.walker_kind = cast(WalkerKind, self.recipe.walker_kind)
        self.mixed_precision = mixed_precision
        self.guide_mixed_precision = (
            bool(mixed_precision) if guide_mixed_precision is None else bool(guide_mixed_precision)
        )
        self.trial_mixed_precision = (
            bool(mixed_precision) if trial_mixed_precision is None else bool(trial_mixed_precision)
        )
        self.nchol_chunk = nchol_chunk
        self.max_memory = max_memory
        self.tau_eql = None if tau_eql is None else float(tau_eql)

        self._trial_input: staging.TrialInput | None = None
        self.guide_e_tot: Any = None
        self.guide_e_err: Any = None

    @property
    def trial_input(self) -> staging.TrialInput | None:
        return self._trial_input

    def n_eql_blocks_for_tau(self) -> int | None:
        """
        The equilibration block count that reaches tau_eql with the current dt and
        n_prop_steps, ceil(tau_eql / (dt * n_prop_steps)); None when tau_eql is not set.
        """
        if self.tau_eql is None:
            return None
        block_time = float(self.dt) * int(self.n_prop_steps)
        if block_time <= 0.0:
            raise ValueError("dt and n_prop_steps must be positive to derive n_eql_blocks.")
        return int(math.ceil(self.tau_eql / block_time - 1e-12))

    def _make_params(self) -> QmcParamsBase:
        # tau_eql fixes the equilibration length; the block count follows the dt and
        # n_prop_steps of the params actually used, whether built from the attributes
        # or given as self.params
        params = super()._make_params()
        if self.tau_eql is not None:
            block_time = float(params.dt) * int(params.n_prop_steps)
            if block_time <= 0.0:
                raise ValueError("dt and n_prop_steps must be positive to derive n_eql_blocks.")
            n_eql = int(math.ceil(self.tau_eql / block_time - 1e-12))
            params = dataclasses.replace(params, n_eql_blocks=n_eql)
            self.n_eql_blocks = n_eql
        return params

    def _key(self) -> tuple:
        return super()._key() + (
            self.trial,
            self.guide,
            None if self.basis_a is None else id(self.basis_a),
            None if self.basis_b is None else id(self.basis_b),
        )

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Stage the guide and the trial from the one CC object.

        The guide is staged by the branch's ordinary staging from the object the guide
        spec picks (the mean field for the HF guides, the CC object for the CISD ones), so
        its data, ops and propagator are exactly those of a plain run with that
        wavefunction; on the unrestricted hamiltonian by staging_u.stage_uh. The cholesky
        vectors of the hamiltonian come from the density fitting tensor when the mean
        field is density fitted (cholesky.ao_cholesky), else from the exact AO ERIs as
        staging does. The trial is staged by its own spec. Both get the CC object's
        frozen core.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        guide_spec = self.recipe.guide_spec
        trial_spec = self.recipe.trial_spec
        norb_frozen = int(self.norb_frozen_core or 0)

        if self.recipe.ham_basis == "uchol":
            staged = stage_inputs_uh(
                self._cc,
                chol_cut=self.chol_cut,
                basis_a=self.basis_a,
                basis_b=self.basis_b,
                cache=self.cache,
                overwrite=self.overwrite_cache if self.cache is not None else False,
                verbose=self.verbose,
                trial_kind=self.guide,
            )
        else:
            guide_src = guide_spec.source_obj(self._obj)
            # the cholesky vectors come from the DF tensor when the mean field is density
            # fitted (staging_u.stage_ham_input_df); otherwise this is staging's own builder
            cache_hit = self.cache is not None and self.cache.exists() and not self.overwrite_cache
            ham = (
                None
                if cache_hit
                else stage_ham_input_df(
                    StagedMfOrCc(guide_src, norb_frozen),
                    chol_cut=self.chol_cut,
                    verbose=self.verbose,
                )
            )
            staged = stage_inputs(
                guide_src,
                norb_frozen_core=norb_frozen,
                chol_cut=self.chol_cut,
                cache=self.cache,
                overwrite=self.overwrite_cache if self.cache is not None else False,
                verbose=self.verbose,
                ham=ham,
            )
        if staged.trial.kind not in guide_spec.kinds:
            raise ValueError(
                f"guide={self.guide!r} was requested but the object stages as "
                f"{staged.trial.kind!r}; the {self.guide}+{self.trial} recipe needs "
                f"one of {sorted(guide_spec.kinds)}."
            )

        trial_input = self.recipe.stage_trial(self._cc, frozen=norb_frozen)
        if trial_input.kind != trial_spec.kind:
            raise ValueError(
                f"trial={self.trial!r} staged as {trial_input.kind!r}, expected "
                f"{trial_spec.kind!r}."
            )
        self._trial_input = trial_input

        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def build_job(  # type: ignore[override]
        self, *, force: bool = False, mesh: Mesh | None = None, **kwargs: Any
    ) -> JobMixed:
        if self._job is not None and not force and (mesh is None or self._job.mesh is mesh):
            return cast(JobMixed, self._job)

        staged = self.stage()
        qmc_params = self._make_params()
        self.params = qmc_params

        job = cast(
            JobMixed,
            self.setup_fn(
                staged,
                recipe=self.recipe,
                trial_input=self._trial_input,
                nchol_chunk=self.nchol_chunk,
                max_memory=self.max_memory,
                walker_kind=self.walker_kind,
                mesh=mesh,
                mixed_precision=self.guide_mixed_precision,
                trial_mixed_precision=self.trial_mixed_precision,
                params=cast(Any, qmc_params),
                **kwargs,
            ),
        )
        # the memory plan may have raised n_chunks, so adopt what setup_mixed settled on
        self.params = job.params
        self.n_chunks = int(job.params.n_chunks)

        self._job = job
        return job

    def _frozen_core_label(self, meta: dict) -> Any:
        """
        What dump_flags prints as nfrozen: the frozen core count, an int for the
        restricted hamiltonian and a per spin pair for the unrestricted one. Subclasses
        whose staged meta carries something else (LnoFragMixed: the frozen LNO indices)
        override this.
        """
        frozen = meta.get("frozen")
        return tuple(int(n) for n in frozen) if isinstance(frozen, (list, tuple)) else frozen

    def dump_flags(self, job: Job) -> None:  # type: ignore[override]
        """Both wavefunctions are listed, each with the kernels it measures with."""
        from .core.ops import k_energy, k_force_bias

        job = cast(JobMixed, job)
        meta = job.staged.meta
        sys = job.sys

        nfrozen = self._frozen_core_label(meta)

        print("\n******** AFQMC ********")
        print(f" nfrozen         = {nfrozen}")
        print(f" nelec           = {sys.nelec}")
        print(f" norb            = {sys.norb}")
        print(f" nchol           = {job.ham_data.nchol}")
        print(f" walker_kind     = {sys.walker_kind}")
        print(f" source_kind     = {meta['source_kind']}")
        print(f" chol_cut        = {meta['chol_cut']:g}")
        print(f" cache           = {str(self.cache) if self.cache else None}")
        print(f" mixed_precision = {self.mixed_precision}")
        print(f"   guide           = {self.guide_mixed_precision}")
        print(f"   trial           = {self.trial_mixed_precision}\n")

        # the guide propagates the walkers, so it carries a force bias; the trial only
        # measures, so it has none
        for label, name, meas_ops in (
            ("guide", self.guide, job.meas_ops),
            ("trial", self.trial, job.mix_trial_meas_ops),
        ):
            print(f" {label:<15} = {name}")
            print(f"   overlap_kernel    = {_kernel_location(meas_ops.overlap)}")
            for key, shown in ((k_force_bias, "force_bias_kernel"), (k_energy, "energy_kernel")):
                if meas_ops.has_kernel(key):
                    print(f"   {shown:<17} = {_kernel_location(meas_ops.kernels[key])}")
            if label == "trial":
                print(f"   components        = {job.recipe.components}")
                print(f"   energy_fn         = {_kernel_location(job.recipe.energy_fn)}")
        print("")

        guide_cfg = self._resolve_meas_cfg(job)
        if guide_cfg is not None:
            self._dump_cfg("meas_cfg", guide_cfg)
            print("")

        trial_spec = job.recipe.trial_spec
        trial_cfg = trial_spec.cfg_getter(job.mix_trial_meas_ops) if trial_spec.cfg_getter else None
        if trial_cfg is not None:
            self._dump_cfg("trial_meas_cfg", trial_cfg)
            width = len(max(dataclasses.fields(trial_cfg), key=lambda f: len(f.name)).name)
            print(f"  {'nchol_chunk_used':<{width}} = {job.mix_meas_ctx().nchol_chunk}")
            print("")
        if job.chunk_plan is not None:
            self._dump_chunk_plan(job.chunk_plan, int(job.ham_data.nchol or 0))

        if self.tau_eql is not None:
            params = cast(QmcParams, job.params)
            block_time = params.dt * params.n_prop_steps
            print(
                f" tau_eql         = {self.tau_eql:g} "
                f"(n_eql_blocks = {params.n_eql_blocks}, "
                f"tau reached = {params.n_eql_blocks * block_time:g})\n"
            )
        self._dump_params(job.params)

    @staticmethod
    def _dump_chunk_plan(plan: Any, nchol: int) -> None:
        """
        The trial's chunk plan: the cholesky vectors scanned per step and how many zero
        vectors pad the tensor to equal chunks, the walkers in flight per device, and
        the memory model's split of the cost into what stays resident for the whole pass,
        what every walker in flight adds (a fixed part and a part per cholesky vector in
        the chunk), the total, and the budget it was fitted to.
        """
        from .cholesky import equal_chunks

        mb = 1024**2
        model = plan.model
        n_steps, chunk, n_pad = equal_chunks(nchol, plan.nchol_chunk)
        per_walker = model.per_walker + chunk * model.per_walker_chol
        w = plan.walkers_in_flight
        print(" chunk_plan      = ChunkPlan")
        print(f"  nchol_chunk       = {chunk}  ({n_steps} steps over {nchol} + {n_pad} padded)")
        print(f"  n_chunks          = {plan.n_chunks}")
        print(f"  walkers_in_flight = {w}")
        print(f"  memory_resident   = {model.resident / mb:.1f} MB")
        print(
            f"  memory_per_walker = {per_walker / mb:.2f} MB  "
            f"({model.per_walker / mb:.2f} MB + {chunk} x {model.per_walker_chol / 1024:.1f} KB)"
        )
        print(f"  memory_walkers    = {w * per_walker / mb:.1f} MB  ({w} walkers in flight)")
        print(f"  memory_used       = {plan.bytes_used / mb:.1f} MB")
        print(f"  memory_budget     = {plan.budget_bytes / mb:.1f} MB")
        print(f"  note              = {plan.note}")
        print("")

    def kernel(self, **driver_kwargs: Any) -> tuple[Any, Any]:  # type: ignore[override]
        """Run mixed AFQMC. Returns the trial (e_tot, e_err) and stores the guide result."""
        print(banner_afqmc())
        print_runtime_provenance()
        mesh = driver_kwargs.get("mesh")
        job = self.build_job(mesh=mesh)
        self.dump_flags(job)

        qmc_result = job.kernel(**driver_kwargs)
        self.qmc_result = qmc_result

        self.guide_e_tot = float(qmc_result.guide_mean_energy)
        self.guide_e_err = float(qmc_result.guide_stderr_energy)
        self.e_tot = float(qmc_result.trial_mean_energy)
        self.e_err = float(qmc_result.trial_stderr_energy)
        return self.e_tot, self.e_err

    run = kernel

    @classmethod
    def from_staged(cls, path: Union[str, Path], **kwargs: Any):  # type: ignore[override]
        raise NotImplementedError(
            "AfqmcMixed stages the trial from the CC object; pass the CC object and use "
            "cache= for the guide's staged inputs."
        )
