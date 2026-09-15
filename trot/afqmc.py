from __future__ import annotations

from .config import configure_once

configure_once()

import copy
import dataclasses
from functools import partial
from pathlib import Path
from typing import Any, Callable, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

print = partial(print, flush=True)

from jax.sharding import Mesh

from . import staging
from .core.system import WalkerKind
from .driver import QmcResult
from .prop.types import QmcParams, QmcParamsBase, QmcParamsFp, QmcParamsLno
from .runtime_provenance import print_runtime_provenance
from .setup import Job
from .setup import setup as setup_job
from .setup_fp import JobFp
from .setup_fp import setup_fp as setup_job_fp
from .mixed import MixedRecipe, get_mixed_recipe
from .setup_mixed import JobMixed, setup_mixed

# from .setup_lno import setup_lno as setup_job_lno
# from . import setup_lno
from .staging import StagedInputs, TrialInput, _is_cc_like
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

        self.params: QmcParamsBase | None = None  # resolved in kernel
        defaults = self.params_cls()
        self.dt = defaults.dt if dt is None else dt
        self.n_walkers = defaults.n_walkers if n_walkers is None else n_walkers
        self.n_blocks = defaults.n_blocks if n_blocks is None else n_blocks
        self.seed = defaults.seed if seed is None else seed
        self.n_chunks = defaults.n_chunks if n_chunks is None else n_chunks
        if hasattr(defaults, "n_eql_blocks"):
            self.n_eql_blocks = defaults.n_eql_blocks if n_eql_blocks is None else n_eql_blocks

        self._staged: StagedInputs | None = None
        self._job: Job | None = None
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

        staged = stage_inputs(
            self._obj,
            norb_frozen_core=(
                int(self.norb_frozen_core) if self.norb_frozen_core is not None else None
            ),
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self.verbose,
        )
        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def save_staged(self, path: Union[str, Path]) -> None:
        """Write current staged inputs to a single file cache."""
        staged = self.stage()
        dump_staged(staged, path)

    # def load_staged(self, path: Union[str, Path]): -> StagedInputs:
    #    """Load staged inputs from a cache file and attach them to this object."""
    #    staged = load_staged(path)
    #    self._staged = staged
    #    self._cache_key = None
    #    self._job = None
    #    return staged

    def _validate_params(self, params: QmcParamsBase) -> QmcParamsBase:
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
        if self._job is not None and not force and (mesh is None or self._job.mesh is mesh):
            return self._job

        staged = self.stage()
        qmc_params = self._make_params()
        self.params = qmc_params

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
        )
        self._job = job
        return job

    def _coerce_result(self, value: Any) -> Any:
        return float(value)

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

        e_tot = float(qmc_result.mean_energy)
        e_err = float(qmc_result.stderr_energy)

        self.qmc_result = qmc_result

        return e_tot, e_err

    run = kernel

    @classmethod
    def _from_staged_common(cls, path: Union[str, Path], **kwargs: Any):
        staged = load_staged(path)
        meta = staged.meta

        af = cls(
            None,
            norb_frozen_core=meta["frozen"],
            chol_cut=meta["chol_cut"],
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


class AfqmcUh(Afqmc):
    """
    AFQMC with an unrestricted (uchol) hamiltonian.

    Alpha and beta each keep their own orbital basis, so h1 and the cholesky vectors are
    carried per spin and norb_a may differ from norb_b. The auxiliary field index stays
    shared between the spins.

    Contrast with ``Afqmc(mf); af.walker_kind = "unrestricted"``, which uses unrestricted
    *walkers* against a hamiltonian built in the alpha MO basis alone.

        af = AfqmcUh(mf)
        mean, err = af.kernel()

    Pass two independently chosen active spaces explicitly (what unrestricted LNO does,
    and where norb_a != norb_b comes from):

        af = AfqmcUh(mf, basis_a=c_a, basis_b=c_b)

    The cholesky vectors come from the density fitting tensor when ``mf`` carries one,
    otherwise from the modified cholesky decomposition of the AO ERIs, and are then
    projected into each spin's basis.

    Parameters
    ----------
    basis_a, basis_b : NDArray, optional
        Orbital bases for the two spins. Default to the UHF alpha and beta coefficients.
    chol_cut : float, optional
        Cholesky decomposition cutoff, by default 1e-8.
    """

    params_cls = QmcParams
    job_cls = Job
    setup_fn = staticmethod(setup_job)

    def __init__(
        self,
        mf_or_cc: Any,
        *,
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
        n_walkers: int | None = None,
        n_chunks: int | None = None,
    ):
        super().__init__(
            mf_or_cc,
            norb_frozen_core=norb_frozen_core,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )

        self.basis_a = basis_a
        self.basis_b = basis_b
        # alpha and beta live in different orbital spaces, so no other kind applies
        self.walker_kind = "unrestricted"

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Build the unrestricted hamiltonian and attach it to the staged inputs.

        Follows AfqmcLnoFrag: staging.stage() accepts a prebuilt ham, so the default
        single basis _stage_ham_input is bypassed rather than modified.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        norb_frozen = self.norb_frozen_core
        if isinstance(norb_frozen, (list, tuple, np.ndarray)):
            raise NotImplementedError(
                "AfqmcUh supports an integer frozen core only; list-valued frozen "
                "orbitals are not implemented for the unrestricted hamiltonian yet."
            )

        ham = staging.build_ham_uchol(
            self._obj,
            chol_cut=self.chol_cut,
            basis_a=self.basis_a,
            basis_b=self.basis_b,
            norb_frozen_core=int(norb_frozen or 0),
            verbose=self.verbose,
        )

        staged = stage_inputs(
            self._obj,
            norb_frozen_core=int(norb_frozen or 0),
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self.verbose,
            # StagedInputs.ham is typed as the restricted HamInput; the unrestricted
            # path carries a HamInputU through the same slot
            ham=cast(Any, ham),
        )
        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged


def _kernel_location(fn: Any) -> str:
    """
    Where a kernel is defined: the path inside the trot package, plus the function name.
    Partials are unwrapped to the function they call; anything defined outside trot keeps
    its full module path, since there is no relative path to give.
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
    a different trial. Which combination to run is chosen with ``trial``; everything that
    follows from it -- staging, measurement ops, block function and blocking analysis --
    comes from that recipe, so a trial can never be paired with the wrong estimator.

        mf = scf.RHF(mol); mf.kernel()
        mycc = cc.CCSD(mf); mycc.kernel()

        af = AfqmcMixed(mycc)                                 # RHF guide, pt2CCSD trial
        af = AfqmcMixed(mycc, trial="pt2ccsd_bar", guide="rhf")  # the bar energy kernel
        mean, err = af.kernel()        # returns the TRIAL (pt2CCSD) energy

    kernel() returns the trial energy; the guide result is kept alongside:

        af.e_tot,       af.e_err        # trial  (pt2CCSD)
        af.guide_e_tot, af.guide_e_err  # guide  (AFQMC/RHF)
        af.qmc_result                   # the full MixedQmcResult

    Parameters
    ----------
    cc : Any
        pyscf CC object. The guide is taken from ``cc._scf`` and the trial amplitudes
        from ``cc`` itself, so no second argument is needed.
    trial : str, optional
        Which mixed recipe to run. The name picks the energy kernel the trial is
        measured with. For a restricted ``CCSD`` object, with an RHF guide:

        - ``"pt2ccsd"``          the plain estimator, one cholesky vector per scan step
        - ``"pt2ccsd_chunk"``    chunked over the cholesky index
        - ``"pt2ccsd_bar"``      chunked, with exp(T1) moved onto the hamiltonian and the
          walker rather than the trial
        - ``"pt2ccsd_sto_chol"`` the bar estimator with a semistochastic cholesky sum

        For a ``UCCSD`` object, with a UHF guide on the unrestricted hamiltonian of
        ``AfqmcUh`` (alpha and beta each in their own orbital basis):

        - ``"upt2ccsd"``          chunked over the cholesky index
        - ``"upt2ccsd_bar"``      with exp(T1) moved onto the hamiltonian and the walker
        - ``"upt2ccsd_sto_chol"`` the bar estimator with a semistochastic cholesky sum

        By default ``"pt2ccsd"`` or ``"upt2ccsd"``, whichever matches the CC object; a
        trial that does not match it is an error. Within a family all the kernels compute
        the same energy. ``trot.mixed.available_mixed_recipes()`` lists what is
        registered. ``max_memory`` and ``nchol_chunk`` apply to the chunking kernels,
        which is every one but ``"pt2ccsd"``.
    guide : str, optional
        Which wavefunction propagates the walkers, by default the one the trial is
        registered against ("rhf" for the pt2CCSD trials, "uhf" for the upt2CCSD ones).
        Recipes are keyed by
        the (guide, trial) pair, so once a trial is registered against several guides
        this has to be given. It is also checked against the staged guide, so a mismatch
        is an error rather than a silently different calculation.
    memory_mode : str, optional
        The trial estimator's memory layout, by default "low". None of the pt2CCSD
        kernels branch on it today; the kernel choice lives in ``trial``.
    max_memory : float, optional
        Memory budget for the trial measurement, in MB as in pyscf, per device. The
        estimator's memory model splits it between the two chunking knobs: the cholesky
        chunk gives way first, and only when a single cholesky vector per step still does
        not fit does ``n_chunks`` rise to take walkers out of flight -- at which point the
        cholesky chunk is chosen again against the smaller walker count. This is the knob
        to turn: it stays meaningful across system sizes and walker counts, where raw
        chunk counts do not.
    nchol_chunk : int, optional
        Cholesky vectors per scan step, set directly. With ``max_memory`` it is taken as
        fixed and only ``n_chunks`` is derived.
    trial_kwargs : dict, optional
        Extra options for the trial's measurement ops, for knobs that belong to one trial
        rather than to every mixed run. ``"pt2ccsd_sto_chol"`` takes its sampling
        controls this way -- ``n_chol_head``, ``n_chol_samples``, ``chol_cost_ratio`` and
        the rest of the fields documented on ``Pt2ccsdMeasCfg``. By default each walker
        uses 20% of the cholesky vectors (``chol_cost_ratio=0.2``), split head : samples
        = 3 : 1; the resolved sizes are printed with the flags::

            AfqmcMixed(mycc, trial="pt2ccsd_sto_chol",
                       trial_kwargs={"chol_cost_ratio": 0.25})
    basis_a, basis_b : NDArray, optional
        Unrestricted trials only: the alpha and beta orbital bases of the hamiltonian, as
        in ``AfqmcUh``. They default to the CC object's own alpha and beta MOs, which is
        where its amplitudes are expressed; bases given here must be the ones the
        amplitudes were computed in (an unrestricted LNO fragment's active spaces, say).
        The frozen core follows ``cc.frozen``.
    mixed_precision : bool, optional
        Single precision for the run, by default False. It reaches the guide propagator
        and the trial estimator's heavy two-body contractions; partial sums are always
        accumulated back in double. Only the chunked energy kernel honours it on the
        trial side, and the guide's measurement ops do not honour it at all yet.

    Examples
    --------
    Measure with the chunked kernel in single precision, keeping the measurement under
    4 GB per device::

        af = AfqmcMixed(
            mycc, trial="pt2ccsd_chunk", max_memory=4000, n_walkers=50, mixed_precision=True
        )

    The chunking it settled on is on the job, and is printed with the flags::

        print(af.build_job().chunk_plan.describe())
    """

    params_cls = QmcParams
    job_cls = JobMixed
    setup_fn = staticmethod(setup_mixed)

    def __init__(
        self,
        cc: Any,
        *,
        trial: str | None = None,
        guide: str | None = None,
        memory_mode: str = "low",
        max_memory: float | None = None,
        nchol_chunk: int | None = None,
        trial_kwargs: dict[str, Any] | None = None,
        mixed_precision: bool = False,
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
    ):
        super().__init__(
            cc,
            norb_frozen_core=norb_frozen_core,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )

        defaults = self.params_cls()
        self.n_prop_steps = defaults.n_prop_steps if n_prop_steps is None else n_prop_steps

        if self._cc is None:
            raise ValueError(
                "AfqmcMixed needs a pyscf CC object: the trial is built from its "
                "amplitudes. Got a mean-field object, which supplies only the guide."
            )

        from pyscf.cc.uccsd import UCCSD

        unrestricted = isinstance(self._cc, UCCSD)
        suggested = "upt2ccsd" if unrestricted else "pt2ccsd"
        if trial is None:
            # the CC object's spin treatment picks between the two pt2CCSD families
            trial = suggested

        self.recipe: MixedRecipe = get_mixed_recipe(trial, guide)
        self.trial: str = self.recipe.trial
        self.guide: str = self.recipe.guide

        if (self.recipe.ham_basis == "uchol") != unrestricted:
            needs = "a UCCSD" if self.recipe.ham_basis == "uchol" else "a restricted CCSD"
            raise ValueError(
                f"trial={self.trial!r} needs {needs} object, got {type(self._cc).__name__}; "
                f"use trial={suggested!r} or one of its _bar / _sto_chol variants."
            )
        if (basis_a is not None or basis_b is not None) and self.recipe.ham_basis != "uchol":
            raise ValueError(
                "basis_a / basis_b set the alpha and beta orbital bases of the unrestricted "
                f"hamiltonian, so they apply only to the unrestricted trials, not {self.trial!r}."
            )
        # orbital bases of the unrestricted hamiltonian, as in AfqmcUh; None means the
        # CC object's own alpha and beta MOs, which is where its amplitudes live
        self.basis_a = basis_a
        self.basis_b = basis_b

        self.walker_kind = cast(WalkerKind, self.recipe.walker_kind)
        self.mixed_precision = mixed_precision
        self.memory_mode = memory_mode
        self.max_memory = max_memory
        self.nchol_chunk = nchol_chunk
        self.extra_trial_kwargs: dict[str, Any] = dict(trial_kwargs or {})

        self._trial_input: TrialInput | None = None
        self.guide_e_tot: Any = None
        self.guide_e_err: Any = None

    @property
    def trial_input(self) -> TrialInput | None:
        return self._trial_input

    def _trial_kwargs(self) -> dict[str, Any]:
        """
        Options for the trial measurement ops. Only the ones that were actually set are
        passed on, so a recipe is never handed a knob it does not have. Sizing against
        max_memory happens in setup_mixed, which has the hamiltonian the model needs.
        """
        kwargs: dict[str, Any] = {"memory_mode": self.memory_mode}
        if self.nchol_chunk is not None:
            kwargs["nchol_chunk"] = self.nchol_chunk
        # trial specific knobs last, so they can override the generic ones
        kwargs.update(self.extra_trial_kwargs)
        return kwargs

    def dump_flags(self, job: JobMixed) -> None:
        """
        A mixed run prints its own flags rather than the inherited ones.

        Afqmc's dump names a single "trial_kind", taken from job.staged.trial -- which in
        a mixed run is the GUIDE, and would contradict the trial named below. Both
        wavefunctions are listed here instead, each with the kernels it measures with.
        """
        from .core.ops import k_energy, k_force_bias
        from .meas.pt2ccsd import get_pt2ccsd_meas_cfg, resolve_chol_budget

        meta = job.staged.meta
        sys = job.sys

        print("\n******** AFQMC ********")
        print(f" nelec           = {sys.nelec}")
        print(f" norb            = {sys.norb}")
        print(f" nchol           = {job.ham_data.nchol}")
        print(f" walker_kind     = {sys.walker_kind}")
        print(f" source_kind     = {meta['source_kind']}")
        print(f" chol_cut        = {meta['chol_cut']:g}")
        print(f" cache           = {str(self.cache) if self.cache else None}")
        print(f" mixed_precision = {self.mixed_precision}\n")

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
        print("")

        guide_cfg = self._resolve_meas_cfg(job)
        if guide_cfg is not None:
            self._dump_cfg("meas_cfg", guide_cfg)
            print("")

        trial_cfg = get_pt2ccsd_meas_cfg(job.mix_trial_meas_ops)
        if trial_cfg is not None:
            self._dump_cfg("trial_meas_cfg", trial_cfg)
            if trial_cfg.measure_type is not None:
                # the chunk size the config actually resolves to against this hamiltonian
                width = len(max(dataclasses.fields(trial_cfg), key=lambda f: len(f.name)).name)
                print(f"  {'nchol_chunk_used':<{width}} = {job.mix_meas_ctx().nchol_chunk}")
            if trial_cfg.measure_type == "sto_chol":
                # the head and tail sizes the sampling knobs resolve to, defaults included
                nchol = int(job.ham_data.nchol)
                n_head, n_samples = resolve_chol_budget(
                    nchol,
                    trial_cfg.n_chol_head,
                    trial_cfg.head_chol_ratio,
                    trial_cfg.n_chol_samples,
                    trial_cfg.chol_cost_ratio,
                    trial_cfg.head_sample_ratio,
                )
                if n_head >= nchol:
                    n_samples = 0  # a full head leaves no tail to sample
                print(f"  {'n_chol_head_used':<{width}} = {n_head}")
                print(f"  {'n_chol_samples_used':<{width}} = {n_samples}")
            print("")

        if job.chunk_plan is not None:
            print(f" chunk_plan      = {job.chunk_plan.describe()}\n")

        self._dump_params(job.params)

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Stage the guide and the measurement trial.

        The guide comes from the SCF object underneath the CC one, which is what makes
        this a mixed calculation: the hamiltonian and propagator stay at the mean-field
        level while the trial carries the correlation.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        if self.recipe.ham_basis == "uchol":
            # as AfqmcUh.stage: build the unrestricted hamiltonian and hand it to stage(),
            # which then stages only the guide. It is built from the CC object, whose alpha
            # and beta MOs are the bases its amplitudes live in, and frozen with the core
            # the CC object froze, so the hamiltonian and the amplitudes always agree
            norb_frozen = int(staging.StagedMfOrCc(self._cc, self.norb_frozen_core).afqmc_frozen)
            ham = staging.build_ham_uchol(
                self._cc,
                chol_cut=self.chol_cut,
                basis_a=self.basis_a,
                basis_b=self.basis_b,
                norb_frozen_core=norb_frozen,
                verbose=self.verbose,
            )
            staged = stage_inputs(
                self._scf,
                norb_frozen_core=norb_frozen,
                chol_cut=self.chol_cut,
                cache=self.cache,
                overwrite=self.overwrite_cache if self.cache is not None else False,
                verbose=self.verbose,
                # StagedInputs.ham is typed as the restricted HamInput; the unrestricted
                # path carries a HamInputU through the same slot
                ham=cast(Any, ham),
            )
        else:
            staged = stage_inputs(
                self._scf,
                norb_frozen_core=(
                    int(self.norb_frozen_core) if self.norb_frozen_core is not None else None
                ),
                chol_cut=self.chol_cut,
                cache=self.cache,
                overwrite=self.overwrite_cache if self.cache is not None else False,
                verbose=self.verbose,
            )
        if staged.trial.kind != self.guide:
            raise ValueError(
                f"guide={self.guide!r} was requested but the object underneath the CC one "
                f"stages as {staged.trial.kind!r}; the {self.guide}+{self.trial} recipe "
                f"needs a {self.guide} guide."
            )
        self._trial_input = self.recipe.stage_trial(self._cc, frozen=self.norb_frozen_core)

        if self.recipe.ham_basis == "uchol":
            norb_ham = tuple(int(n) for n in staged.ham.norb)
            norb_trial = tuple(
                int(self._trial_input.data[k].shape[0]) for k in ("mo_t_a", "mo_t_b")
            )
            if norb_ham != norb_trial:
                raise ValueError(
                    f"the unrestricted hamiltonian has norb={norb_ham} but the UCCSD "
                    f"amplitudes span {norb_trial} orbitals; basis_a / basis_b must be the "
                    "bases the amplitudes are expressed in."
                )

        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def build_job(
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
                trial_kwargs=self._trial_kwargs(),
                walker_kind=self.walker_kind,
                mesh=mesh,
                mixed_precision=self.mixed_precision,
                max_memory=self.max_memory,
                params=cast(Any, qmc_params),
                **kwargs,
            ),
        )
        # the memory plan may have raised n_chunks, so adopt what setup_mixed settled on
        self.params = job.params
        self.n_chunks = int(job.params.n_chunks)

        self._job = job
        return job

    def kernel(self, **driver_kwargs: Any) -> tuple[Any, Any]:
        """
        Run mixed AFQMC. Returns the trial (e_tot, e_err) and stores the guide result.
        """
        print(banner_afqmc())
        print_runtime_provenance()
        mesh = driver_kwargs.get("mesh")
        job = self.build_job(mesh=mesh)
        self.dump_flags(job)

        qmc_result = job.kernel(**driver_kwargs)

        self.qmc_result = qmc_result
        self.guide_e_tot = float(qmc_result.guide_mean_energy.real)
        self.guide_e_err = float(qmc_result.guide_stderr_energy.real)
        self.e_tot = float(qmc_result.trial_mean_energy.real)
        self.e_err = float(qmc_result.trial_stderr_energy.real)

        return self.e_tot, self.e_err
