from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Union

import h5py
import numpy as np

from .core.system import System
from .staging import StagedInputs, TrialInput, dump, load


ModeSolver = Literal["auto", "dense", "lanczos"]


@dataclass(frozen=True)
class CisdModeConfig:
    """Host-side CISD/UCISD mode selection.

    A zero threshold or zero discarded-norm target requests the exact,
    full-rank mode representation.  A nonzero selection changes the trial
    wave function and is therefore always explicit.
    """

    threshold: float | None = None
    discarded_norm_target: float | None = 0.0
    minimum_rank: int = 0
    solver: ModeSolver = "auto"
    dense_max_dim: int | None = None
    lanczos_initial_rank: int = 256
    lanczos_tol: float = 1.0e-9
    lanczos_maxiter: int | None = None

    def __post_init__(self) -> None:
        if self.threshold is None and self.discarded_norm_target is None:
            raise ValueError("CISD modes require a threshold or discarded_norm_target.")
        if self.threshold is not None and self.threshold < 0.0:
            raise ValueError("CISD mode threshold must be nonnegative.")
        if self.discarded_norm_target is not None and not (
            0.0 <= self.discarded_norm_target < 1.0
        ):
            raise ValueError("discarded_norm_target must lie in [0, 1).")
        if self.minimum_rank < 0:
            raise ValueError("minimum_rank must be nonnegative.")
        if self.solver not in ("auto", "dense", "lanczos"):
            raise ValueError("solver must be 'auto', 'dense', or 'lanczos'.")
        if self.dense_max_dim is not None and self.dense_max_dim <= 0:
            raise ValueError("dense_max_dim must be positive when provided.")
        if self.lanczos_initial_rank <= 0:
            raise ValueError("lanczos_initial_rank must be positive.")
        if self.lanczos_tol <= 0.0:
            raise ValueError("lanczos_tol must be positive.")
        if self.lanczos_maxiter is not None and self.lanczos_maxiter <= 0:
            raise ValueError("lanczos_maxiter must be positive when provided.")

    def cache_key(self, trial_kind: str) -> str:
        kind = trial_kind.lower()
        if kind not in {"cisd", "ucisd"}:
            raise ValueError(f"CISD mode caching does not support trial kind {trial_kind!r}.")
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{kind}:{payload}".encode()).hexdigest()[:16]
        return f"{kind}_modes_{digest}"


@dataclass(frozen=True)
class CisdPairSamplingConfig:
    """Automatic walker--Cholesky pair-sampling policy.

    The detailed proposal and batching choices intentionally live here rather
    than as flat :class:`Afqmc` flags. After equilibration, the tail-noise limit
    is ``target_tail_std_fraction`` times the standard deviation of the exact
    calibration-population energies. ``kernel(target_error=...)`` controls
    stopping independently. ``final_error_sampling_fraction`` is retained for
    input compatibility but no longer affects tuning.
    """

    initial_head_fraction: float = 0.125
    initial_pair_sample_size: int = 4096
    final_error_sampling_fraction: float = 0.2
    target_tail_std_fraction: float = 0.35
    safety_factor: float = 1.0
    cross_validation_quantile: float = 1.0
    guide_chol_batch_size: int = 16
    head_chol_batch_size: int = 16
    tuning_n_chunks: int = 10
    tuning_chol_batch_size: int = 16
    tuning_population_count: int = 5
    tuning_population_spacing_blocks: int = 2
    production_initial_n_chunks: int = 1
    tail_probability_uniform_mix: float = 0.01
    walker_guide_weight_mix: float = 0.1
    settling_blocks: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial_head_fraction < 1.0:
            raise ValueError("initial_head_fraction must lie in [0, 1).")
        if self.initial_pair_sample_size < 2:
            raise ValueError(
                "initial_pair_sample_size must be at least two for half-sample diagnostics."
            )
        if not 0.0 < self.final_error_sampling_fraction <= 1.0:
            raise ValueError("final_error_sampling_fraction must lie in (0, 1].")
        if self.target_tail_std_fraction <= 0.0:
            raise ValueError("target_tail_std_fraction must be positive.")
        if self.safety_factor <= 0.0:
            raise ValueError("safety_factor must be positive.")
        if not 0.0 < self.cross_validation_quantile <= 1.0:
            raise ValueError("cross_validation_quantile must lie in (0, 1].")
        for name in (
            "guide_chol_batch_size",
            "head_chol_batch_size",
            "tuning_n_chunks",
            "tuning_chol_batch_size",
            "tuning_population_count",
            "production_initial_n_chunks",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.tuning_population_spacing_blocks <= 0:
            raise ValueError("tuning_population_spacing_blocks must be positive.")
        if not 0.0 <= self.tail_probability_uniform_mix <= 1.0:
            raise ValueError("tail_probability_uniform_mix must lie in [0, 1].")
        if not 0.0 < self.walker_guide_weight_mix <= 1.0:
            raise ValueError("walker_guide_weight_mix must lie in (0, 1].")
        if self.settling_blocks < 0:
            raise ValueError("settling_blocks must be nonnegative.")


@dataclass(frozen=True)
class CisdWorkflowConfig:
    """Opt-in retained-mode and energy-sampling workflow for CISD trials."""

    modes: CisdModeConfig = CisdModeConfig()
    pair_sampling: CisdPairSamplingConfig | None = None
    n_mode_chunks: int = 1
    memory_mode: Literal["high"] = "high"

    def __post_init__(self) -> None:
        if self.n_mode_chunks <= 0:
            raise ValueError("n_mode_chunks must be positive.")
        if self.memory_mode != "high":
            raise ValueError("retained-mode CISD workflows currently require memory_mode='high'.")

    @classmethod
    def pair_sampled(
        cls,
        *,
        discarded_norm_target: float = 0.0,
        solver: ModeSolver = "auto",
        n_mode_chunks: int = 1,
    ) -> CisdWorkflowConfig:
        return cls(
            modes=CisdModeConfig(
                threshold=None,
                discarded_norm_target=discarded_norm_target,
                solver=solver,
            ),
            pair_sampling=CisdPairSamplingConfig(),
            n_mode_chunks=n_mode_chunks,
        )


@dataclass(frozen=True)
class PreparedCisdModes:
    trial_kind: Literal["cisd", "ucisd"]
    cache_key: str
    data: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CisdRuntimeBundle:
    staged: StagedInputs
    trial_data: Any
    trial_ops: Any
    meas_ops: Any


def _jsonable_metadata(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable_metadata(item) for item in value]
    return value


def _prepare_from_trial(
    trial: TrialInput,
    config: CisdModeConfig,
    *,
    verbose: bool,
) -> PreparedCisdModes:
    kind = trial.kind.lower()
    started = time.perf_counter()
    if kind == "cisd":
        from .trial.cisd_modes import factorize_cisd_k_modes

        required = ("ci1", "ci2")
        missing = tuple(name for name in required if name not in trial.data)
        if missing:
            raise KeyError(f"staged CISD trial is missing raw datasets: {missing}")
        factorization = factorize_cisd_k_modes(
            trial.data["ci2"],
            threshold=config.threshold,
            discarded_norm_target=config.discarded_norm_target,
            minimum_rank=config.minimum_rank,
            solver=config.solver,
            dense_max_dim=config.dense_max_dim,
            lanczos_initial_rank=config.lanczos_initial_rank,
            lanczos_tol=config.lanczos_tol,
            lanczos_maxiter=config.lanczos_maxiter,
            verbose=verbose,
        )
        data = {
            "ci1": np.asarray(trial.data["ci1"]),
            "eigenvalues": np.asarray(factorization.eigenvalues),
            "modes": np.asarray(factorization.modes),
            "nocc_t_core": np.asarray(trial.data.get("nocc_t_core", 0), dtype=np.int64),
            "nvir_t_outer": np.asarray(trial.data.get("nvir_t_outer", 0), dtype=np.int64),
        }
        pair_dimension = int(np.prod(data["ci1"].shape))
        metadata = {
            "trial_kind": kind,
            "representation": "cisd_modes",
            "pair_dimension": pair_dimension,
            "mode_rank": factorization.rank,
            "natural_mode_rank": factorization.natural_rank,
            "solver": factorization.solver,
            "dense_driver": factorization.dense_driver,
            "threshold": factorization.threshold,
            "discarded_norm_target": factorization.discarded_norm_target,
            "discarded_norm_fraction": factorization.discarded_norm_fraction,
        }
    elif kind == "ucisd":
        from .trial.ucisd_k_modes import factorize_ucisd_k_blocks

        required = (
            "mo_coeff_a",
            "mo_coeff_b",
            "ci1a",
            "ci1b",
            "ci2aa",
            "ci2ab",
            "ci2bb",
        )
        missing = tuple(name for name in required if name not in trial.data)
        if missing:
            raise KeyError(f"staged UCISD trial is missing raw datasets: {missing}")
        factorization = factorize_ucisd_k_blocks(
            trial.data["ci2aa"],
            trial.data["ci2ab"],
            trial.data["ci2bb"],
            threshold=config.threshold,
            discarded_norm_target=config.discarded_norm_target,
            minimum_rank=config.minimum_rank,
            solver=config.solver,
            dense_max_dim=config.dense_max_dim,
            lanczos_initial_rank=config.lanczos_initial_rank,
            lanczos_tol=config.lanczos_tol,
            lanczos_maxiter=config.lanczos_maxiter,
            verbose=verbose,
        )
        data = {
            "mo_coeff_a": np.asarray(trial.data["mo_coeff_a"]),
            "mo_coeff_b": np.asarray(trial.data["mo_coeff_b"]),
            "c1a": np.asarray(trial.data["ci1a"]),
            "c1b": np.asarray(trial.data["ci1b"]),
            "eigenvalues": np.asarray(factorization.eigenvalues),
            "modes": np.asarray(factorization.modes),
        }
        metadata = {
            "trial_kind": kind,
            "representation": "ucisd_k_modes",
            "pair_dimensions": list(factorization.pair_dims),
            "combined_pair_dimension": factorization.combined_dim,
            "mode_rank": factorization.rank,
            "natural_mode_rank": factorization.natural_rank,
            "solver": factorization.solver,
            "dense_driver": factorization.dense_driver,
            "threshold": factorization.threshold,
            "discarded_norm_target": factorization.discarded_norm_target,
            "discarded_norm_fraction": factorization.discarded_norm_fraction,
        }
    else:
        raise ValueError(
            "CISD mode preparation requires staged trial kind 'cisd' or 'ucisd'; "
            f"received {trial.kind!r}."
        )

    cache_key = config.cache_key(kind)
    metadata.update(
        {
            "cache_key": cache_key,
            "configuration": asdict(config),
            "factorization_seconds": time.perf_counter() - started,
            "storage_bytes": int(sum(array.nbytes for array in data.values())),
            "representation_format_version": 1,
        }
    )
    return PreparedCisdModes(
        trial_kind=kind,
        cache_key=cache_key,
        data=data,
        metadata=_jsonable_metadata(metadata),
    )


def _derived_group_path(cache_key: str) -> str:
    return f"trial/derived/{cache_key}"


def _h5_string(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def has_cached_cisd_modes(path: Union[str, Path], config: CisdModeConfig) -> bool:
    cache_path = Path(path).expanduser().resolve()
    if not cache_path.exists():
        return False
    with h5py.File(cache_path, "r") as handle:
        if "trial" not in handle:
            return False
        kind = _h5_string(handle["trial"].attrs["kind"])
        return _derived_group_path(config.cache_key(kind)) in handle


def load_cached_cisd_modes(
    path: Union[str, Path],
    config: CisdModeConfig,
) -> PreparedCisdModes | None:
    cache_path = Path(path).expanduser().resolve()
    if not cache_path.exists():
        return None
    with h5py.File(cache_path, "r") as handle:
        if "trial" not in handle:
            return None
        kind = _h5_string(handle["trial"].attrs["kind"]).lower()
        key = config.cache_key(kind)
        group_path = _derived_group_path(key)
        if group_path not in handle:
            return None
        group = handle[group_path]
        metadata = json.loads(_h5_string(group.attrs["metadata_json"]))
        data_group = group["data"]
        data = {name: np.asarray(data_group[name]) for name in data_group.keys()}
    return PreparedCisdModes(
        trial_kind=kind,  # type: ignore[arg-type]
        cache_key=key,
        data=data,
        metadata=metadata,
    )


def _store_prepared_modes(
    path: Path,
    prepared: PreparedCisdModes,
    *,
    overwrite: bool,
) -> None:
    temporary_name = f"__tmp_{prepared.cache_key}_{uuid.uuid4().hex}"
    with h5py.File(path, "r+") as handle:
        trial_group = handle["trial"]
        stored_kind = _h5_string(trial_group.attrs["kind"]).lower()
        if stored_kind != prepared.trial_kind:
            raise ValueError(
                f"cannot store {prepared.trial_kind} modes in a {stored_kind} trial cache."
            )
        derived = trial_group.require_group("derived")
        if prepared.cache_key in derived:
            if not overwrite:
                return
        temporary = derived.create_group(temporary_name)
        temporary.attrs["metadata_json"] = json.dumps(
            prepared.metadata, sort_keys=True, separators=(",", ":")
        )
        data_group = temporary.create_group("data")
        for name, value in prepared.data.items():
            data_group.create_dataset(name, data=np.asarray(value))
        handle.flush()
        if prepared.cache_key in derived:
            del derived[prepared.cache_key]
        derived.move(temporary_name, prepared.cache_key)
        handle.flush()


def prepare_cisd_modes(
    staged_or_path: StagedInputs | str | Path,
    config: CisdModeConfig,
    *,
    cache: str | Path | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> PreparedCisdModes:
    """Build CISD modes on the host and optionally append them to a stage cache.

    Raw amplitudes remain in ``trial/data``.  Derived mode data are written to
    a sibling ``trial/derived/<configuration-key>`` group, making the cache
    backward compatible with readers that know only the raw representation.
    """

    if isinstance(staged_or_path, (str, Path)):
        source_path = Path(staged_or_path).expanduser().resolve()
        if cache is not None and Path(cache).expanduser().resolve() != source_path:
            raise ValueError("cache must match staged_or_path when preparing from a path.")
        cached = load_cached_cisd_modes(source_path, config)
        if cached is not None and not overwrite:
            if verbose:
                print(f"[modes] using cached derived trial {cached.cache_key} from {source_path}")
            return cached
        staged = load(source_path)
        cache_path: Path | None = source_path
    else:
        staged = staged_or_path
        cache_path = Path(cache).expanduser().resolve() if cache is not None else None
        if cache_path is not None and cache_path.exists() and not overwrite:
            cached = load_cached_cisd_modes(cache_path, config)
            if cached is not None:
                if verbose:
                    print(
                        f"[modes] using cached derived trial {cached.cache_key} "
                        f"from {cache_path}"
                    )
                return cached

    prepared = _prepare_from_trial(staged.trial, config, verbose=verbose)
    if cache_path is not None:
        if not cache_path.exists():
            dump(staged, cache_path)
        _store_prepared_modes(cache_path, prepared, overwrite=overwrite)
        if verbose:
            gib = prepared.metadata["storage_bytes"] / 1024**3
            print(
                f"[modes] stored derived trial {prepared.cache_key} in {cache_path} "
                f"({gib:.3f} GiB)"
            )
    return prepared


def cached_representation_key(
    path: Union[str, Path],
    workflow: CisdWorkflowConfig | None,
) -> str | None:
    if workflow is None or not has_cached_cisd_modes(path, workflow.modes):
        return None
    cache_path = Path(path).expanduser().resolve()
    with h5py.File(cache_path, "r") as handle:
        kind = _h5_string(handle["trial"].attrs["kind"])
    return workflow.modes.cache_key(kind)


def _make_pair_sampling_configs(
    *,
    trial_kind: str,
    n_chol: int,
    config: CisdPairSamplingConfig | None,
) -> tuple[Any | None, Any | None]:
    if config is None or n_chol <= 1:
        if config is not None and n_chol <= 1:
            print("[sampling] one Cholesky vector: using deterministic CISD energy.")
        return None, None

    head_size = min(
        n_chol - 1,
        max(0, int(round(config.initial_head_fraction * n_chol))),
    )
    common_sampling = dict(
        chol_head_size=head_size,
        pair_sample_size=config.initial_pair_sample_size,
        rank_head_by_guide=True,
        guide_chol_batch_size=config.guide_chol_batch_size,
        head_chol_batch_size=config.head_chol_batch_size,
        tail_probability_uniform_mix=config.tail_probability_uniform_mix,
        track_half_sample_diagnostic=True,
        guard_head_deviations=True,
        walker_guide_policy="head_rms",
        walker_guide_weight_mix=config.walker_guide_weight_mix,
    )
    common_tuning = dict(
        guide_policy="population_rms",
        final_error_target_ha=None,
        final_error_sampling_fraction=config.final_error_sampling_fraction,
        target_tail_std_fraction=config.target_tail_std_fraction,
        safety_factor=config.safety_factor,
        cross_validation_quantile=config.cross_validation_quantile,
        tuning_n_chunks=config.tuning_n_chunks,
        tuning_chol_batch_size=config.tuning_chol_batch_size,
        tuning_population_count=config.tuning_population_count,
        tuning_population_spacing_blocks=config.tuning_population_spacing_blocks,
        production_initial_n_chunks=config.production_initial_n_chunks,
        production_head_chol_batch_size=config.head_chol_batch_size,
        tail_probability_uniform_mix=config.tail_probability_uniform_mix,
        track_half_sample_diagnostic=True,
        guard_head_deviations=True,
        walker_guide_policy="head_rms",
        walker_guide_weight_mix=config.walker_guide_weight_mix,
        settling_blocks=config.settling_blocks,
    )
    if trial_kind == "cisd":
        from .meas.cisd_modes import CisdModePairSamplingCfg, CisdModePairTuningCfg

        return CisdModePairSamplingCfg(**common_sampling), CisdModePairTuningCfg(
            **common_tuning
        )
    if trial_kind == "ucisd":
        from .meas.ucisd_k_modes import (
            UcisdKModePairSamplingCfg,
            UcisdKModePairTuningCfg,
        )

        return UcisdKModePairSamplingCfg(**common_sampling), UcisdKModePairTuningCfg(
            **common_tuning
        )
    raise ValueError(f"unsupported CISD trial kind {trial_kind!r}.")


def make_cisd_runtime_bundle(
    sys: System,
    staged: StagedInputs,
    *,
    mixed_precision: bool,
    workflow: CisdWorkflowConfig,
) -> CisdRuntimeBundle:
    """Resolve retained modes and measurement policy during job assembly."""

    kind = staged.trial.kind.lower()
    data = staged.trial.data
    if "eigenvalues" in data and "modes" in data:
        prepared_data = {name: np.asarray(value) for name, value in data.items()}
        metadata = dict(staged.meta.get("trial_representation", {}))
    else:
        prepared = _prepare_from_trial(staged.trial, workflow.modes, verbose=True)
        prepared_data = prepared.data
        metadata = prepared.metadata

    energy_sampling, energy_tuning = _make_pair_sampling_configs(
        trial_kind=kind,
        n_chol=int(staged.ham.chol.shape[0]),
        config=workflow.pair_sampling,
    )
    if kind == "cisd":
        from .meas.cisd_modes import make_cisd_mode_meas_ops
        from .trial.cisd_modes import make_cisd_mode_trial_data, make_cisd_mode_trial_ops

        trial_data = make_cisd_mode_trial_data(
            prepared_data,
            sys,
            mixed_precision=mixed_precision,
        )
        trial_ops = make_cisd_mode_trial_ops(sys)
        meas_ops = make_cisd_mode_meas_ops(
            sys,
            mixed_precision=mixed_precision,
            n_mode_chunks=workflow.n_mode_chunks,
            energy_sampling=energy_sampling,
            energy_tuning=energy_tuning,
        )
    elif kind == "ucisd":
        from .meas.ucisd_k_modes import make_ucisd_k_mode_meas_ops
        from .trial.ucisd_k_modes import (
            make_ucisd_k_mode_trial_data,
            make_ucisd_k_mode_trial_ops,
        )

        trial_data = make_ucisd_k_mode_trial_data(
            prepared_data,
            sys,
            mixed_precision=mixed_precision,
        )
        trial_ops = make_ucisd_k_mode_trial_ops(sys)
        meas_ops = make_ucisd_k_mode_meas_ops(
            sys,
            memory_mode=workflow.memory_mode,
            mixed_precision=mixed_precision,
            n_mode_chunks=workflow.n_mode_chunks,
            energy_sampling=energy_sampling,
            energy_tuning=energy_tuning,
        )
    else:
        raise ValueError(
            "CisdWorkflowConfig requires staged trial kind 'cisd' or 'ucisd'; "
            f"received {staged.trial.kind!r}."
        )

    runtime_meta = dict(staged.meta)
    runtime_meta["trial_representation"] = metadata
    runtime_meta["cisd_workflow"] = _jsonable_metadata(asdict(workflow))
    runtime_staged = replace(
        staged,
        trial=replace(staged.trial, data=prepared_data),
        meta=runtime_meta,
    )
    return CisdRuntimeBundle(
        staged=runtime_staged,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
    )


__all__ = [
    "CisdModeConfig",
    "CisdPairSamplingConfig",
    "CisdRuntimeBundle",
    "CisdWorkflowConfig",
    "PreparedCisdModes",
    "cached_representation_key",
    "has_cached_cisd_modes",
    "load_cached_cisd_modes",
    "make_cisd_runtime_bundle",
    "prepare_cisd_modes",
]
