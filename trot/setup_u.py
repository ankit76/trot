"""
Assembly of an AFQMC Job on the unrestricted (uchol) hamiltonian.

The companion of setup.py for HamCholU. setup._assemble_job is not reused because it
builds a System with an integer norb and a HamChol runtime hamiltonian; setup_uh
mirrors it with System_uh, HamCholU, the uchol trial bundle and the uchol propagator,
and returns an ordinary Job so the driver, block function, statistics and the Afqmc API
are all shared.

    UcholRuntimeLayout   RuntimeLayout for HamCholU: single device, no compaction
    setup_uh             Job from a pyscf mean field, a StagedInputs or a staged file
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union, cast

import jax.numpy as jnp
from jax.sharding import Mesh

from .core.system import System_uh, WalkerKind
from .ham.chol_u import HamCholU
from .prop.afqmc_u import make_prop_ops_u
from .prop.blocks import block as default_block
from .prop.types import QmcParams
from .runtime_layout import DefaultRuntimeLayout, PreparedRuntime, RuntimeJob
from .setup import Job, _make_params, _setup_begin, _setup_end
from .sharding import has_model_axis
from .staging import StagedInputs
from .staging_u import load_uh, stage_uh


def make_ham_data_uh(ham: Any, mesh: Mesh | None) -> HamCholU:
    """
    Runtime HamCholU from a HamInputU (a HamCholU passes through).

    chol_a and chol_b share the auxiliary field axis, so if they were ever sharded on a
    model axis they would have to be padded and permuted identically; that is not wired
    yet, so a mesh with a model axis is refused rather than silently mis-sharded. Data
    only meshes (walker parallelism) are fine, the hamiltonian is replicated.
    """
    if isinstance(ham, HamCholU):
        return ham
    if mesh is not None and has_model_axis(mesh):
        raise NotImplementedError(
            "model-axis sharding of the cholesky vectors is not implemented for the "
            "unrestricted (uchol) hamiltonian yet; use mesh=None or a data-only mesh."
        )
    return HamCholU(
        h0=jnp.asarray(ham.h0),
        h1_a=jnp.asarray(ham.h1_a),
        h1_b=jnp.asarray(ham.h1_b),
        chol_a=jnp.asarray(ham.chol_a),
        chol_b=jnp.asarray(ham.chol_b),
        basis="uchol",
    )


@dataclass(frozen=True)
class UcholRuntimeLayout:
    """
    RuntimeLayout for the unrestricted hamiltonian. Contexts and the initial state are
    prepared exactly as DefaultRuntimeLayout does (the Job's own prop/trial/meas ops are
    already the uchol ones); only the runtime hamiltonian differs.
    """

    def make_initial_ham_data(self, ham: Any, mesh: Mesh | None) -> HamCholU:  # type: ignore[override]
        return make_ham_data_uh(ham, mesh)

    def prepare(
        self,
        job: RuntimeJob,
        *,
        state: Any = None,
        meas_ctx: object | None = None,
        prop_ctx: object | None = None,
    ) -> PreparedRuntime:
        return DefaultRuntimeLayout().prepare(
            job, state=state, meas_ctx=meas_ctx, prop_ctx=prop_ctx
        )


def _make_trial_bundle_uh(
    sys: System_uh, staged: StagedInputs, mixed_precision: bool
) -> tuple[Any, Any, Any]:
    """
    (trial_data, trial_ops, meas_ops) of the staged trial on the uchol hamiltonian.
    """
    tr = staged.trial
    kind = tr.kind.lower()
    t_bundle = _setup_begin(f"building trial bundle ({kind}, uchol)")

    if kind == "uhf":
        from .meas.uhf_uh import make_uhf_meas_ops_uh
        from .trial.uhf_uh import make_uhf_trial_data_uh, make_uhf_trial_ops_uh

        trial_data = make_uhf_trial_data_uh(tr.data, sys)
        trial_ops = make_uhf_trial_ops_uh(sys)
        meas_ops = make_uhf_meas_ops_uh(sys)
        _setup_end(t_bundle, "trial bundle ready", details=f"kind={kind} (uchol)")
        return trial_data, trial_ops, meas_ops

    if kind == "ucisd":
        from .meas.ucisd_uh import make_ucisd_meas_ops_uh
        from .trial.ucisd_uh import make_ucisd_trial_data_uh, make_ucisd_trial_ops_uh

        trial_data = make_ucisd_trial_data_uh(tr.data, sys)
        trial_ops = make_ucisd_trial_ops_uh(sys)
        meas_ops = make_ucisd_meas_ops_uh(sys, mixed_precision=mixed_precision)
        _setup_end(t_bundle, "trial bundle ready", details=f"kind={kind} (uchol)")
        return trial_data, trial_ops, meas_ops

    raise ValueError(
        f"Unsupported TrialInput.kind for the unrestricted hamiltonian: {tr.kind!r} "
        "(wired so far: 'uhf', 'ucisd')."
    )


def _make_prop_uh(
    ham_data: HamCholU,
    walker_kind: str,
    sys: Any = None,
    *,
    mixed_precision: bool,
    **prop_kwargs: Any,
) -> Any:
    return make_prop_ops_u(
        ham_data.basis, walker_kind, mixed_precision=mixed_precision, **prop_kwargs
    )


def _resolve_staged_uh(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    norb_frozen_core: Any,
    chol_cut: float,
    basis_a: Any,
    basis_b: Any,
    cache: Union[str, Path] | None,
    overwrite: bool,
    verbose: bool,
) -> StagedInputs:
    if isinstance(obj_or_staged, StagedInputs):
        return obj_or_staged
    if isinstance(obj_or_staged, (str, Path)):
        p = Path(obj_or_staged).expanduser().resolve()
        if p.exists():
            return load_uh(p)
        raise FileNotFoundError(f"staged file {p} does not exist")
    return stage_uh(
        obj_or_staged,
        norb_frozen_core=norb_frozen_core,
        chol_cut=chol_cut,
        basis_a=basis_a,
        basis_b=basis_b,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
    )


def setup_uh(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # staging options (used only if we need to stage)
    norb_frozen_core: int | tuple[int, int] | None = None,
    chol_cut: float = 1e-5,
    basis_a: Any = None,
    basis_b: Any = None,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    # system/prop options
    walker_kind: WalkerKind | None = None,
    mesh: Mesh | None = None,
    mixed_precision: bool = True,
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
    job_cls: type[Job] = Job,
) -> Job:
    """
    Assemble a runnable AFQMC Job on the unrestricted hamiltonian from a pyscf UHF (or
    RHF converted with to_uhf) object with the UHF trial, a UCCSD object with the
    CC-derived UCISD trial, a StagedInputs carrying a HamInputU, or a path to a file
    written by staging_u.dump_uh.

        job = setup_uh(mf)
        job.kernel()

    walker_kind must be "unrestricted" (the default); anything else is refused since the
    two spins live in different orbital spaces.
    """
    staged = _resolve_staged_uh(
        obj_or_staged,
        norb_frozen_core=norb_frozen_core,
        chol_cut=chol_cut,
        basis_a=basis_a,
        basis_b=basis_b,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
    )
    ham: Any = staged.ham
    if getattr(ham, "basis", None) != "uchol":
        raise ValueError(
            "setup_uh needs a StagedInputs carrying a HamInputU (basis 'uchol'); "
            f"got basis={getattr(ham, 'basis', None)!r}. Use setup() for the restricted one."
        )

    wk = "unrestricted" if walker_kind is None else str(walker_kind).lower()
    # System_uh refuses any other walker kind with a clear message
    sys = System_uh(norb=tuple(ham.norb), nelec=tuple(ham.nelec), walker_kind=wk)  # type: ignore[arg-type]

    qmc_params = _make_params(params=params, **(params_kwargs or {}))

    if trial_data is None or trial_ops is None or meas_ops is None:
        td, to, mo = _make_trial_bundle_uh(sys, staged, mixed_precision)
        trial_data = td if trial_data is None else trial_data
        trial_ops = to if trial_ops is None else trial_ops
        meas_ops = mo if meas_ops is None else meas_ops

    runtime_layout = UcholRuntimeLayout()
    t_ham_runtime = _setup_begin("preparing runtime Hamiltonian (uchol)")
    ham_data = runtime_layout.make_initial_ham_data(ham, mesh)
    _setup_end(t_ham_runtime, "runtime Hamiltonian ready", details=f"norb={ham_data.norb}")

    if prop_ops is None:
        prop_ops = _make_prop_uh(
            ham_data,
            sys.walker_kind,
            sys=sys,
            mixed_precision=mixed_precision,
            **(prop_kwargs or {}),
        )

    if block_fn is None:
        block_fn = default_block

    return job_cls(
        staged=staged,
        sys=sys,  # type: ignore[arg-type]
        params=qmc_params,
        ham_data=ham_data,  # type: ignore[arg-type]
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        # RuntimeLayout is typed for HamChol; the uchol layout returns a HamCholU
        runtime_layout=cast(Any, runtime_layout),
        mesh=mesh,
    )
