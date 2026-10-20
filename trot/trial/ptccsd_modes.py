from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System
from .cisd_modes import factorize_cisd_k_modes, mode_quadratic


def decompose_t2_modes(
    t2: np.ndarray,
    *,
    mode_threshold: float | None = 0.0,
    discarded_norm_target: float | None = None,
    minimum_rank: int = 0,
    solver: Literal["auto", "dense", "lanczos"] = "auto",
    dense_max_dim: int | None = None,
    lanczos_initial_rank: int = 256,
    lanczos_tol: float = 1.0e-9,
    lanczos_maxiter: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the restricted spin-adapted kernel formed from raw ``T2``.

    The returned modes represent

    ``K_(ia,jb) = 2 t_(ia,jb) - t_(ib,ja)``.

    Unlike the CISD-mode construction, ``t2`` must not contain the disconnected
    ``T1*T1`` contribution. Selection may use an absolute eigenvalue threshold,
    a relative Frobenius discarded-norm target, or both.
    """

    factorization = factorize_cisd_k_modes(
        t2,
        threshold=mode_threshold,
        discarded_norm_target=discarded_norm_target,
        minimum_rank=minimum_rank,
        solver=solver,
        dense_max_dim=dense_max_dim,
        lanczos_initial_rank=lanczos_initial_rank,
        lanczos_tol=lanczos_tol,
        lanczos_maxiter=lanczos_maxiter,
        verbose=verbose,
    )
    return factorization.eigenvalues, factorization.modes


def _validate_modes(
    eigenvalues: jax.Array,
    modes: jax.Array,
    *,
    nocc: int,
    nvir: int,
) -> None:
    if not all(hasattr(value, "ndim") for value in (eigenvalues, modes)):
        return
    if eigenvalues.ndim != 1:
        raise ValueError(f"eigenvalues must have rank 1, got shape {eigenvalues.shape}.")
    if modes.ndim != 3:
        raise ValueError(f"modes must have rank 3, got shape {modes.shape}.")
    rank = int(eigenvalues.shape[0])
    pair_dim = nocc * nvir
    if not 0 < rank <= pair_dim:
        raise ValueError(f"mode rank must lie in [1, {pair_dim}], got {rank}.")
    if modes.shape != (rank, nocc, nvir):
        raise ValueError(
            f"modes must have shape {(rank, nocc, nvir)}, got {modes.shape}."
        )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdModeTrial:
    """First-order restricted PT-CCSD importance guide using raw-T2 modes."""

    t1: jax.Array
    eigenvalues: jax.Array
    modes: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.t1, "ndim"):
            return
        if self.t1.ndim != 2:
            raise ValueError(f"t1 must have rank 2, got shape {self.t1.shape}.")
        _validate_modes(
            self.eigenvalues,
            self.modes,
            nocc=self.nocc,
            nvir=self.nvir,
        )

    @property
    def nocc(self) -> int:
        return int(self.t1.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.t1.shape[1])

    @property
    def norb(self) -> int:
        return self.nocc + self.nvir

    @property
    def mode_rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    def tree_flatten(self):
        return (self.t1, self.eigenvalues, self.modes), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        t1, eigenvalues, modes = children
        return cls(t1=t1, eigenvalues=eigenvalues, modes=modes)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtccsdThoulessModeTrial:
    """Restricted Thouless reference plus compact raw-``T2`` modes.

    ``nvir_t_outer`` counts trailing virtuals excluded from the doubles trial.
    The reference, Hamiltonian, and walkers still use the full orbital space.
    """

    mo_t: jax.Array
    eigenvalues: jax.Array
    modes: jax.Array
    nvir_t_outer: int = 0

    def __post_init__(self) -> None:
        if not hasattr(self.mo_t, "ndim") or not hasattr(self.modes, "ndim"):
            return
        if self.mo_t.ndim != 2:
            raise ValueError(f"mo_t must have rank 2, got shape {self.mo_t.shape}.")
        if self.modes.ndim != 3:
            raise ValueError(f"modes must have rank 3, got shape {self.modes.shape}.")
        if self.nvir_t_outer < 0:
            raise ValueError("nvir_t_outer must be nonnegative.")
        nocc = int(self.mo_t.shape[1])
        nvir = int(self.modes.shape[2])
        norb = nocc + nvir + self.nvir_t_outer
        if self.mo_t.shape[0] != norb:
            raise ValueError(
                f"mo_t must have shape {(norb, nocc)}, got {self.mo_t.shape}."
            )
        _validate_modes(self.eigenvalues, self.modes, nocc=nocc, nvir=nvir)

    @property
    def nocc(self) -> int:
        return int(self.mo_t.shape[1])

    @property
    def nvir(self) -> int:
        return int(self.modes.shape[2])

    @property
    def norb(self) -> int:
        return int(self.mo_t.shape[0])

    @property
    def mode_rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    def tree_flatten(self):
        return (self.mo_t, self.eigenvalues, self.modes), self.nvir_t_outer

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t, eigenvalues, modes = children
        return cls(mo_t=mo_t, eigenvalues=eigenvalues, modes=modes, nvir_t_outer=aux)


def make_ptccsd_mode_trial_data(
    data: dict,
    *,
    mixed_precision: bool = True,
    mode_threshold: float | None = 0.0,
    discarded_norm_target: float | None = None,
    minimum_rank: int = 0,
) -> PtccsdModeTrial:
    t1 = jnp.asarray(data["t1"], dtype=jnp.float64)
    if "eigenvalues" in data and "modes" in data:
        eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float64)
        modes = np.asarray(data["modes"])
    else:
        eigenvalues, modes = decompose_t2_modes(
            np.asarray(data["t2"]),
            mode_threshold=mode_threshold,
            discarded_norm_target=discarded_norm_target,
            minimum_rank=minimum_rank,
        )
    mode_dtype = jnp.float32 if mixed_precision else jnp.float64
    return PtccsdModeTrial(
        t1=t1,
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        modes=jnp.asarray(modes, dtype=mode_dtype),
    )


def make_ptccsd_thouless_mode_trial_data(
    data: dict,
    *,
    mixed_precision: bool = True,
    mode_threshold: float | None = 0.0,
    discarded_norm_target: float | None = None,
    minimum_rank: int = 0,
) -> PtccsdThoulessModeTrial:
    """Build a restricted PT trial without padding its FNO doubles modes.

    Supply compact ``t1`` and ``t2``/``modes`` with ``nvir_t_outer`` to retain
    the full Hamiltonian space. Alternatively, a full ``mo_t`` or full-width
    ``t1`` determines the outer count from the compact mode dimensions.
    Existing full-space inputs retain ``nvir_t_outer=0``.
    """

    if "eigenvalues" in data and "modes" in data:
        eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float64)
        modes = np.asarray(data["modes"])
    else:
        eigenvalues, modes = decompose_t2_modes(
            np.asarray(data["t2"]),
            mode_threshold=mode_threshold,
            discarded_norm_target=discarded_norm_target,
            minimum_rank=minimum_rank,
        )
    if modes.ndim != 3:
        raise ValueError(f"modes must have rank 3, got shape {modes.shape}.")
    _, nocc, nvir = modes.shape
    if "mo_t" in data:
        mo_t = jnp.asarray(data["mo_t"])
        nvir_t_outer = int(data.get("nvir_t_outer", mo_t.shape[0] - nocc - nvir))
    else:
        t1 = jnp.asarray(data["t1"])
        if t1.ndim != 2 or t1.shape[0] != nocc:
            raise ValueError(f"t1 must have {nocc} occupied rows, got shape {t1.shape}.")
        nvir_t_outer = int(data.get("nvir_t_outer", t1.shape[1] - nvir))
        if nvir_t_outer < 0:
            raise ValueError("nvir_t_outer must be nonnegative.")
        if t1.shape[1] == nvir:
            t1 = jnp.pad(t1, ((0, 0), (0, nvir_t_outer)))
        elif t1.shape[1] != nvir + nvir_t_outer:
            raise ValueError("t1 must span either the retained or full virtual space.")
        mo_t = jnp.vstack([jnp.eye(nocc, dtype=t1.dtype), t1.T])
    mode_dtype = jnp.float32 if mixed_precision else jnp.float64
    return PtccsdThoulessModeTrial(
        mo_t=mo_t,
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        modes=jnp.asarray(modes, dtype=mode_dtype),
        nvir_t_outer=nvir_t_outer,
    )


def get_rdm1_pt(trial_data: PtccsdModeTrial) -> jax.Array:
    occ = jnp.arange(trial_data.norb) < trial_data.nocc
    dm = jnp.diag(occ).astype(float)
    return jnp.stack([dm, dm], axis=0)


def get_rdm1_thouless(trial_data: PtccsdThoulessModeTrial) -> jax.Array:
    c = trial_data.mo_t
    dm = c @ jnp.linalg.solve(c.conj().T @ c, c.conj().T)
    return jnp.stack([dm, dm], axis=0)


def greens_pt_r(walker: jax.Array, trial_data: PtccsdModeTrial) -> jax.Array:
    wocc = walker[: trial_data.nocc, :]
    return jnp.linalg.solve(wocc.T, walker.T)


def hf_overlap_r(walker: jax.Array, trial_data: PtccsdModeTrial) -> jax.Array:
    det0 = jnp.linalg.det(walker[: trial_data.nocc, :])
    return det0 * det0


def theta_pt_r(walker: jax.Array, trial_data: PtccsdModeTrial) -> jax.Array:
    green_occ = greens_pt_r(walker, trial_data)[:, trial_data.nocc :]
    theta1 = 2.0 * jnp.einsum("ia,ia->", trial_data.t1, green_occ, optimize="optimal")
    return theta1 + mode_quadratic(trial_data, green_occ)


def overlap_pt_r(walker: jax.Array, trial_data: PtccsdModeTrial) -> jax.Array:
    return hf_overlap_r(walker, trial_data) * jnp.exp(theta_pt_r(walker, trial_data))


def half_green_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    overlap_mat = trial_data.mo_t.conj().T @ walker
    return jnp.linalg.solve(overlap_mat.T, walker.T)


def _det_overlap_and_green_occ_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> tuple[jax.Array, jax.Array]:
    """Return determinant overlap and the needed Green excitation block."""

    overlap_mat = trial_data.mo_t.conj().T @ walker
    half_green = jnp.linalg.solve(overlap_mat.T, walker.T)
    nocc = trial_data.nocc
    green_occ = (
        trial_data.mo_t.conj()[:nocc, :]
        @ half_green[:, nocc : nocc + trial_data.nvir]
    )
    return jnp.linalg.det(overlap_mat) ** 2, green_occ


def greens_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    return trial_data.mo_t.conj() @ half_green_thouless_r(walker, trial_data)


def greenp_thouless(
    green: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    start = trial_data.nocc
    stop = start + trial_data.nvir
    return (green - jnp.eye(trial_data.norb, dtype=green.dtype))[:, start:stop]


def det_overlap_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    return jnp.linalg.det(trial_data.mo_t.conj().T @ walker) ** 2


def theta_t2_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    _, green_occ = _det_overlap_and_green_occ_thouless_r(walker, trial_data)
    return mode_quadratic(trial_data, green_occ)


def overlap_ptccsd_thouless_r(
    walker: jax.Array,
    trial_data: PtccsdThoulessModeTrial,
) -> jax.Array:
    det_overlap, green_occ = _det_overlap_and_green_occ_thouless_r(walker, trial_data)
    return det_overlap * jnp.exp(mode_quadratic(trial_data, green_occ))


def make_ptccsd_mode_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD mode guide requires a closed-shell restricted walker.")
    return TrialOps(overlap=overlap_pt_r, get_rdm1=get_rdm1_pt)


def make_ptccsd_thouless_mode_trial_ops(
    sys: System,
    *,
    exponentiate_t2: bool = True,
) -> TrialOps:
    if sys.nup != sys.ndn or sys.walker_kind.lower() != "restricted":
        raise ValueError("PT-CCSD Thouless mode guide requires a closed-shell restricted walker.")
    overlap = overlap_ptccsd_thouless_r if exponentiate_t2 else det_overlap_thouless_r
    return TrialOps(overlap=overlap, get_rdm1=get_rdm1_thouless)
