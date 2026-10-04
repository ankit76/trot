from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System
from .mode_factorization import (
    DENSE_EIGH_DRIVER,
    dense_symmetric_eigh,
    format_dense_memory_selection,
)


@dataclass(frozen=True)
class UcisdKModeFactorization:
    """Host-side spectral factorization of the combined UCISD doubles kernel."""

    eigenvalues: np.ndarray
    modes: np.ndarray
    pair_dims: tuple[int, int]
    solver: Literal["dense", "lanczos"]
    threshold: float | None
    discarded_norm_target: float | None
    discarded_norm_fraction: float
    natural_rank: int
    dense_driver: str | None = None

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    @property
    def combined_dim(self) -> int:
        return int(sum(self.pair_dims))


def _validate_ucisd_blocks(
    ci2aa: np.ndarray,
    ci2ab: np.ndarray,
    ci2bb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    ci2aa = np.asarray(ci2aa, dtype=np.float64)
    ci2ab = np.asarray(ci2ab, dtype=np.float64)
    ci2bb = np.asarray(ci2bb, dtype=np.float64)
    if ci2aa.ndim != 4 or ci2ab.ndim != 4 or ci2bb.ndim != 4:
        raise ValueError("UCISD doubles blocks must all have rank 4.")

    noa, nva, noa_2, nva_2 = ci2aa.shape
    nob, nvb, nob_2, nvb_2 = ci2bb.shape
    if (noa_2, nva_2) != (noa, nva):
        raise ValueError(
            f"ci2aa must have shape (nocc_a, nvir_a, nocc_a, nvir_a), got {ci2aa.shape}."
        )
    if (nob_2, nvb_2) != (nob, nvb):
        raise ValueError(
            f"ci2bb must have shape (nocc_b, nvir_b, nocc_b, nvir_b), got {ci2bb.shape}."
        )
    if ci2ab.shape != (noa, nva, nob, nvb):
        raise ValueError(f"ci2ab must have shape {(noa, nva, nob, nvb)}, got {ci2ab.shape}.")

    da = int(noa * nva)
    db = int(nob * nvb)
    aa = ci2aa.reshape(da, da)
    ab = ci2ab.reshape(da, db)
    bb = ci2bb.reshape(db, db)
    for label, block in (("alpha-alpha", aa), ("beta-beta", bb)):
        norm_sq = float(np.vdot(block, block).real)
        difference_norm_sq = 0.0
        row_batch_size = 256
        for start in range(0, block.shape[0], row_batch_size):
            stop = min(start + row_batch_size, block.shape[0])
            difference = block[start:stop] - block[:, start:stop].T
            difference_norm_sq += float(np.vdot(difference, difference).real)
        relative_error = float(np.sqrt(difference_norm_sq / norm_sq)) if norm_sq else 0.0
        if relative_error > 1.0e-10:
            raise ValueError(
                f"{label} pair matrix is not symmetric: relative error={relative_error:.3e}."
            )
    return aa, ab, bb, (da, db)


def _dense_kernel(
    aa: np.ndarray,
    ab: np.ndarray,
    bb: np.ndarray,
) -> np.ndarray:
    da, db = ab.shape
    kernel = np.empty((da + db, da + db), dtype=np.float64, order="F")
    kernel[:da, :da] = aa
    kernel[:da, da:] = ab
    kernel[da:, :da] = ab.T
    kernel[da:, da:] = bb
    return kernel


def factorize_ucisd_k_blocks(
    ci2aa: np.ndarray,
    ci2ab: np.ndarray,
    ci2bb: np.ndarray,
    *,
    threshold: float | None = None,
    discarded_norm_target: float | None = None,
    minimum_rank: int = 0,
    solver: Literal["auto", "dense", "lanczos"] = "auto",
    dense_max_dim: int | None = None,
    lanczos_initial_rank: int = 256,
    lanczos_tol: float = 1.0e-9,
    lanczos_maxiter: int | None = None,
    verbose: bool = False,
) -> UcisdKModeFactorization:
    """Diagonalize and truncate the combined UCISD kernel on the host.

    ``solver="dense"`` constructs the full combined matrix and uses LAPACK
    ``DSYEVR`` through :func:`scipy.linalg.eigh`. ``solver="lanczos"`` keeps the staged spin
    blocks separate and adaptively requests the largest-magnitude eigenpairs
    through a matrix-free ARPACK solve. ``"auto"`` selects the dense solver
    whenever its estimated peak fits the currently available host/cgroup
    memory and LAPACK workspace-index limits, and uses Lanczos otherwise. If
    the dense solve still raises :class:`MemoryError`, an inexact automatic
    selection retries with Lanczos. ``dense_max_dim`` is accepted for
    compatibility but no longer limits the automatic dense path.

    Selection may be controlled by an absolute eigenvalue ``threshold``, a
    relative Frobenius ``discarded_norm_target``, or both.  When both are
    supplied, enough modes are retained to satisfy both criteria.  The
    ``minimum_rank`` option may retain additional modes, for example to give a
    guide and estimator a common rank.

    The Lanczos solve grows its requested rank until the supplied selection
    criteria can be certified from the largest-magnitude eigenpairs.  The full
    Frobenius norm is evaluated directly from the dense spin blocks, so a
    discarded-norm target does not require computing the discarded modes.
    """
    if threshold is None and discarded_norm_target is None:
        raise ValueError("supply threshold, discarded_norm_target, or both.")
    if threshold is not None and threshold < 0.0:
        raise ValueError("mode threshold must be nonnegative.")
    if discarded_norm_target is not None and not 0.0 <= discarded_norm_target < 1.0:
        raise ValueError("discarded_norm_target must lie in [0, 1).")
    if solver not in ("auto", "dense", "lanczos"):
        raise ValueError("solver must be 'auto', 'dense', or 'lanczos'.")
    if dense_max_dim is not None and dense_max_dim <= 0:
        raise ValueError("dense_max_dim must be positive when provided.")
    if lanczos_initial_rank <= 0:
        raise ValueError("lanczos_initial_rank must be positive.")
    if lanczos_tol <= 0.0:
        raise ValueError("lanczos_tol must be positive.")
    if lanczos_maxiter is not None and lanczos_maxiter <= 0:
        raise ValueError("lanczos_maxiter must be positive when provided.")

    aa, ab, bb, pair_dims = _validate_ucisd_blocks(ci2aa, ci2ab, ci2bb)
    combined_dim = int(sum(pair_dims))
    if not 0 <= minimum_rank <= combined_dim:
        raise ValueError(
            f"minimum_rank must lie in [0, {combined_dim}], got {minimum_rank}."
        )
    full_norm_sq = (
        float(np.vdot(aa, aa).real)
        + 2.0 * float(np.vdot(ab, ab).real)
        + float(np.vdot(bb, bb).real)
    )
    full_norm = float(np.sqrt(full_norm_sq))

    selected_solver: Literal["dense", "lanczos"]
    dense_resource_safe = True
    dense_memory_message = ""
    if solver == "auto":
        dense_resource_safe, dense_memory_message = format_dense_memory_selection(combined_dim)
        exact_selection = threshold == 0.0 or discarded_norm_target == 0.0
        if (exact_selection or minimum_rank == combined_dim) and not dense_resource_safe:
            raise MemoryError(
                "exact mode selection requires dense diagonalization, but it is not "
                f"safe under the automatic resource checks: {dense_memory_message}. "
                "Request more host memory."
            )
        selected_solver = (
            "dense"
            if exact_selection
            or minimum_rank == combined_dim
            or dense_resource_safe
            else "lanczos"
        )
    else:
        selected_solver = solver
    if verbose:
        print(
            "[modes] combined UCISD K factorization: "
            f"pair_dims={pair_dims}, combined_dim={combined_dim}, "
            f"solver={selected_solver}, "
            f"threshold={threshold if threshold is not None else 'none'}, "
            "discarded_norm_target="
            f"{discarded_norm_target if discarded_norm_target is not None else 'none'}, "
            f"minimum_rank={minimum_rank}"
        )
        if solver == "auto":
            print(
                "[modes] dense auto-selection resource estimate: "
                f"{dense_memory_message}"
            )
            if dense_max_dim is not None:
                print("[modes] dense_max_dim is deprecated and ignored by solver='auto'.")

    if selected_solver == "lanczos" and (
        threshold == 0.0
        or discarded_norm_target == 0.0
        or minimum_rank == combined_dim
    ):
        raise ValueError(
            "an exact or full-rank selection requires the dense solver because a "
            "complete basis cannot be obtained from scipy.sparse.linalg.eigsh."
        )

    required_retained_norm_sq = (
        (1.0 - discarded_norm_target**2) * full_norm_sq
        if discarded_norm_target is not None
        else None
    )

    if selected_solver == "dense" or combined_dim <= 2:
        selected_solver = "dense"
        if verbose:
            gib = combined_dim * combined_dim * np.dtype(np.float64).itemsize / 1024**3
            print(
                f"[modes] constructing dense K ({gib:.3f} GiB) and "
                f"diagonalizing with DSY{DENSE_EIGH_DRIVER.upper()}..."
            )
        try:
            eigenvalues, eigenvectors = dense_symmetric_eigh(
                _dense_kernel(aa, ab, bb)
            )
        except MemoryError:
            can_retry_lanczos = (
                solver == "auto"
                and combined_dim > 2
                and threshold != 0.0
                and discarded_norm_target != 0.0
                and minimum_rank != combined_dim
            )
            if not can_retry_lanczos:
                raise
            selected_solver = "lanczos"
            if verbose:
                print(
                    "[modes] dense diagonalization raised MemoryError; "
                    "retrying with Lanczos"
                )
    if selected_solver == "lanczos":
        from scipy.sparse.linalg import LinearOperator, eigsh

        da, db = pair_dims

        def matvec(vector: np.ndarray) -> np.ndarray:
            vector_a = vector[:da]
            vector_b = vector[da:]
            return np.concatenate(
                (
                    aa @ vector_a + ab @ vector_b,
                    ab.T @ vector_a + bb @ vector_b,
                )
            )

        def matmat(vectors: np.ndarray) -> np.ndarray:
            vectors_a = vectors[:da]
            vectors_b = vectors[da:]
            return np.concatenate(
                (
                    aa @ vectors_a + ab @ vectors_b,
                    ab.T @ vectors_a + bb @ vectors_b,
                ),
                axis=0,
            )

        operator = LinearOperator(
            shape=(combined_dim, combined_dim),
            matvec=matvec,
            matmat=matmat,
            dtype=np.dtype(np.float64),
        )
        requested_rank = min(max(lanczos_initial_rank, minimum_rank), combined_dim - 1)
        rng = np.random.default_rng(91_733)
        v0 = rng.standard_normal(combined_dim)
        while True:
            if verbose:
                print(f"[modes] Lanczos largest-magnitude solve: requested_rank={requested_rank}")
            eigenvalues, eigenvectors = eigsh(
                operator,
                k=requested_rank,
                which="LM",
                tol=lanczos_tol,
                maxiter=lanczos_maxiter,
                v0=v0,
            )
            threshold_satisfied = threshold is None or (
                np.count_nonzero(np.abs(eigenvalues) > threshold) < requested_rank
            )
            retained_norm_satisfied = required_retained_norm_sq is None or (
                float(np.vdot(eigenvalues, eigenvalues).real)
                >= required_retained_norm_sq
            )
            if threshold_satisfied and retained_norm_satisfied:
                break
            if requested_rank == combined_dim - 1:
                raise RuntimeError(
                    "Lanczos could not satisfy the requested mode selection before "
                    "reaching combined_dim - 1 modes; use solver='dense' or relax "
                    "the selection criteria."
                )
            requested_rank = min(combined_dim - 1, 2 * requested_rank)

    eigenvalues_all = np.asarray(eigenvalues, dtype=np.float64)
    eigenvectors_all = np.asarray(eigenvectors, dtype=np.float64)
    order = np.argsort(np.abs(eigenvalues_all))[::-1]
    ordered_eigenvalues = eigenvalues_all[order]

    threshold_rank = (
        int(np.count_nonzero(np.abs(ordered_eigenvalues) > threshold))
        if threshold is not None
        else 0
    )
    norm_rank = 0
    if required_retained_norm_sq is not None and full_norm_sq > 0.0:
        cumulative_norm_sq = np.cumsum(ordered_eigenvalues**2, dtype=np.float64)
        norm_rank = int(
            np.searchsorted(
                cumulative_norm_sq,
                min(required_retained_norm_sq, cumulative_norm_sq[-1]),
                side="left",
            )
            + 1
        )
    natural_rank = max(threshold_rank, norm_rank)
    if natural_rank == 0 and discarded_norm_target is not None:
        natural_rank = 1
    retained_rank = max(natural_rank, minimum_rank)
    if retained_rank == 0:
        raise ValueError("mode selection removed every UCISD mode.")
    if retained_rank > ordered_eigenvalues.size:
        raise RuntimeError(
            f"mode selection requires {retained_rank} eigenpairs but only "
            f"{ordered_eigenvalues.size} were computed."
        )
    retained_indices = order[:retained_rank]
    eigenvalues = np.asarray(eigenvalues_all[retained_indices], dtype=np.float64)
    modes = np.empty((retained_indices.size, combined_dim), dtype=np.float64)
    for output_index, input_index in enumerate(retained_indices):
        modes[output_index] = eigenvectors_all[:, input_index]

    retained_norm_sq = float(np.vdot(eigenvalues, eigenvalues).real)
    discarded_norm_fraction = (
        float(np.sqrt(max(0.0, full_norm_sq - retained_norm_sq)) / full_norm) if full_norm else 0.0
    )
    if verbose:
        mode_gib = modes.nbytes / 1024**3
        print(
            "[modes] retained combined UCISD K modes: "
            f"rank={eigenvalues.size}/{combined_dim}, storage={mode_gib:.3f} GiB, "
            f"discarded_norm_fraction={discarded_norm_fraction:.3e}, "
            f"natural_rank={natural_rank}"
        )
    return UcisdKModeFactorization(
        eigenvalues=eigenvalues,
        modes=modes,
        pair_dims=pair_dims,
        solver=selected_solver,
        threshold=float(threshold) if threshold is not None else None,
        discarded_norm_target=discarded_norm_target,
        discarded_norm_fraction=discarded_norm_fraction,
        natural_rank=natural_rank,
        dense_driver=DENSE_EIGH_DRIVER if selected_solver == "dense" else None,
    )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdKModeTrial:
    """UCISD trial stored as modes of the combined doubles kernel.

    For ``z = concatenate((g_alpha, g_beta))``,

    ``K ~= modes.T @ diag(eigenvalues) @ modes``

    and the doubles overlap is ``0.5 * z.T @ K @ z``. The mode rows can mix
    alpha and beta pair spaces, but the UHF orbital bases remain distinct and
    walkers remain restricted.
    """

    mo_coeff_a: jax.Array
    mo_coeff_b: jax.Array
    c1a: jax.Array
    c1b: jax.Array
    eigenvalues: jax.Array
    modes: jax.Array

    def __post_init__(self) -> None:
        arrays = (
            self.mo_coeff_a,
            self.mo_coeff_b,
            self.c1a,
            self.c1b,
            self.eigenvalues,
            self.modes,
        )
        if not all(hasattr(value, "ndim") for value in arrays):
            return
        if self.mo_coeff_a.ndim != 2 or self.mo_coeff_b.ndim != 2:
            raise ValueError("mo_coeff_a and mo_coeff_b must have rank 2.")
        if self.mo_coeff_a.shape != self.mo_coeff_b.shape:
            raise ValueError(
                "mo_coeff_a and mo_coeff_b must have identical shapes, got "
                f"{self.mo_coeff_a.shape} and {self.mo_coeff_b.shape}."
            )
        if self.c1a.ndim != 2 or self.c1b.ndim != 2:
            raise ValueError("c1a and c1b must have rank 2.")
        if self.mo_coeff_a.shape != (self.norb, self.norb):
            raise ValueError(
                f"MO coefficient matrices must have shape {(self.norb, self.norb)}, "
                f"got {self.mo_coeff_a.shape}."
            )
        if self.eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues must have rank 1, got shape {self.eigenvalues.shape}.")
        if self.modes.ndim != 2:
            raise ValueError(f"modes must have rank 2, got shape {self.modes.shape}.")
        expected = (self.mode_rank, sum(self.pair_dim))
        if self.modes.shape != expected:
            raise ValueError(f"modes must have shape {expected}, got {self.modes.shape}.")
        if jnp.issubdtype(self.modes.dtype, jnp.complexfloating):
            raise ValueError("UcisdKModeTrial currently requires real modes.")

    @property
    def norb(self) -> int:
        return int(self.mo_coeff_b.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.c1a.shape[0]), int(self.c1b.shape[0]))

    @property
    def nvir(self) -> tuple[int, int]:
        return (int(self.c1a.shape[1]), int(self.c1b.shape[1]))

    @property
    def pair_dim(self) -> tuple[int, int]:
        return (
            int(self.nocc[0] * self.nvir[0]),
            int(self.nocc[1] * self.nvir[1]),
        )

    @property
    def mode_rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    def tree_flatten(self):
        return (
            self.mo_coeff_a,
            self.mo_coeff_b,
            self.c1a,
            self.c1b,
            self.eigenvalues,
            self.modes,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


def get_rdm1(trial_data: UcisdKModeTrial) -> jax.Array:
    """Return the UHF reference density matrices in the alpha orbital basis."""
    norb = trial_data.norb
    noa, nob = trial_data.nocc
    occ_a = jnp.arange(norb) < noa
    c_b = trial_data.mo_coeff_b
    dm_a = jnp.diag(occ_a)
    dm_b = c_b[:, :nob] @ c_b[:, :nob].conj().T
    return jnp.stack((dm_a, dm_b), axis=0)


def combined_pair_vector(
    trial_data: UcisdKModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
) -> jax.Array:
    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrix_a.shape != expected_a or matrix_b.shape != expected_b:
        raise ValueError(
            f"matrix pair must have shapes {expected_a} and {expected_b}, got "
            f"{matrix_a.shape} and {matrix_b.shape}."
        )
    return jnp.concatenate((matrix_a.reshape(-1), matrix_b.reshape(-1)))


def mode_projections(
    trial_data: UcisdKModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
) -> jax.Array:
    """Project the alpha and beta pair spaces separately before promotion."""
    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrix_a.shape != expected_a or matrix_b.shape != expected_b:
        raise ValueError(
            f"matrix pair must have shapes {expected_a} and {expected_b}, got "
            f"{matrix_a.shape} and {matrix_b.shape}."
        )

    da, _ = trial_data.pair_dim
    modes_a = trial_data.modes[:, :da]
    modes_b = trial_data.modes[:, da:]
    vector_a_r = jnp.real(matrix_a).reshape(-1).astype(trial_data.modes.dtype)
    vector_b_r = jnp.real(matrix_b).reshape(-1).astype(trial_data.modes.dtype)
    projection_a_r = jnp.einsum("rp,p->r", modes_a, vector_a_r, optimize="optimal")
    projection_b_r = jnp.einsum("rp,p->r", modes_b, vector_b_r, optimize="optimal")
    projection_r = projection_a_r.astype(jnp.float64) + projection_b_r.astype(jnp.float64)
    is_complex = jnp.issubdtype(matrix_a.dtype, jnp.complexfloating) or jnp.issubdtype(
        matrix_b.dtype, jnp.complexfloating
    )
    if not is_complex:
        return projection_r

    vector_a_i = jnp.imag(matrix_a).reshape(-1).astype(trial_data.modes.dtype)
    vector_b_i = jnp.imag(matrix_b).reshape(-1).astype(trial_data.modes.dtype)
    projection_a_i = jnp.einsum("rp,p->r", modes_a, vector_a_i, optimize="optimal")
    projection_b_i = jnp.einsum("rp,p->r", modes_b, vector_b_i, optimize="optimal")
    projection_i = projection_a_i.astype(jnp.float64) + projection_b_i.astype(jnp.float64)
    return projection_r.astype(jnp.complex128) + 1.0j * projection_i.astype(jnp.complex128)


def k_mode_apply(
    trial_data: UcisdKModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    projections: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Apply the retained combined modes and split the result by spin."""
    if projections is None:
        projections = mode_projections(trial_data, matrix_a, matrix_b)
    weighted = trial_data.eigenvalues.astype(jnp.float64) * projections
    modes = trial_data.modes
    if not jnp.issubdtype(projections.dtype, jnp.complexfloating):
        applied = jnp.einsum("r,rp->p", weighted, modes, optimize="optimal")
    else:
        applied_r = jnp.einsum("r,rp->p", jnp.real(weighted), modes, optimize="optimal")
        applied_i = jnp.einsum("r,rp->p", jnp.imag(weighted), modes, optimize="optimal")
        applied = applied_r.astype(jnp.complex128)
        applied += 1.0j * applied_i.astype(jnp.complex128)

    da, _ = trial_data.pair_dim
    shape_a = (trial_data.nocc[0], trial_data.nvir[0])
    shape_b = (trial_data.nocc[1], trial_data.nvir[1])
    return applied[:da].reshape(shape_a), applied[da:].reshape(shape_b)


def k_mode_quadratic(
    trial_data: UcisdKModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    projections: jax.Array | None = None,
) -> jax.Array:
    """Return ``0.5 * z.T @ K_R @ z`` as a bilinear mode contraction."""
    if projections is None:
        projections = mode_projections(trial_data, matrix_a, matrix_b)
    dtype = (
        jnp.complex128 if jnp.issubdtype(projections.dtype, jnp.complexfloating) else jnp.float64
    )
    return 0.5 * jnp.sum(
        trial_data.eigenvalues.astype(jnp.float64) * projections * projections,
        dtype=dtype,
    )


def _greens_restricted(
    walker: jax.Array,
    trial_data: UcisdKModeTrial,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    noa, nob = trial_data.nocc
    wa = walker[:, :noa]
    wb = trial_data.mo_coeff_b.T @ walker[:, :nob]
    woa = wa[:noa]
    wob = wb[:nob]
    green_a = jnp.linalg.solve(woa.T, wa.T)
    green_b = jnp.linalg.solve(wob.T, wb.T)
    det0 = jnp.linalg.det(woa) * jnp.linalg.det(wob)
    return green_a, green_b, det0


def overlap_r(walker: jax.Array, trial_data: UcisdKModeTrial) -> jax.Array:
    """Overlap for a restricted walker and retained combined-K modes."""
    green_a, green_b, det0 = _greens_restricted(walker, trial_data)
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_b[:, nob : nob + trial_data.nvir[1]]
    singles = jnp.einsum("ia,ia->", trial_data.c1a, green_occ_a, optimize="optimal")
    singles += jnp.einsum("ia,ia->", trial_data.c1b, green_occ_b, optimize="optimal")
    doubles = k_mode_quadratic(trial_data, green_occ_a, green_occ_b)
    return det0 * (1.0 + singles + doubles)


def make_ucisd_k_mode_trial_data(
    data: dict,
    sys: System,
    *,
    mixed_precision: bool = True,
) -> UcisdKModeTrial:
    """Build a full-rank or truncated combined-K mode trial."""
    del sys
    c1a = jnp.asarray(data["c1a"], dtype=jnp.float64)
    c1b = jnp.asarray(data["c1b"], dtype=jnp.float64)
    eigenvalues = jnp.asarray(data["eigenvalues"], dtype=jnp.float64)
    if eigenvalues.ndim != 1:
        raise ValueError(f"eigenvalues must have rank 1, got shape {eigenvalues.shape}.")
    combined_dim = int(c1a.size + c1b.size)
    rank = int(eigenvalues.shape[0])
    if "modes" in data:
        modes = jnp.asarray(data["modes"])
    elif "eigenvectors" in data:
        eigenvectors = jnp.asarray(data["eigenvectors"])
        if eigenvectors.shape != (combined_dim, rank):
            raise ValueError(
                f"eigenvectors must have shape {(combined_dim, rank)}, got {eigenvectors.shape}."
            )
        modes = eigenvectors.T
    else:
        raise KeyError("combined-K mode data requires 'modes' or 'eigenvectors'.")

    mode_dtype = jnp.float32 if mixed_precision else jnp.float64
    return UcisdKModeTrial(
        mo_coeff_a=jnp.asarray(data["mo_coeff_a"]),
        mo_coeff_b=jnp.asarray(data["mo_coeff_b"]),
        c1a=c1a,
        c1b=c1b,
        eigenvalues=eigenvalues,
        modes=jnp.asarray(modes, dtype=mode_dtype),
    )


def make_ucisd_k_mode_trial_ops(sys: System) -> TrialOps:
    """Build combined-K mode trial operations for restricted walkers."""
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"UCISD K-mode trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
