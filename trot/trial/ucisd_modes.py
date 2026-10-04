from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


def _validate_factor(
    name: str,
    values: jax.Array,
    modes: jax.Array,
    pair_shape: tuple[int, int],
) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} values must have rank 1, got shape {values.shape}.")
    if modes.ndim != 3:
        raise ValueError(f"{name} modes must have rank 3, got shape {modes.shape}.")
    rank = int(values.shape[0])
    pair_dim = int(pair_shape[0] * pair_shape[1])
    if not 0 <= rank <= pair_dim:
        raise ValueError(f"{name} rank must lie in [0, {pair_dim}], got {rank}.")
    if modes.shape != (rank,) + pair_shape:
        raise ValueError(
            f"{name} modes must have shape {(rank,) + pair_shape}, got {modes.shape}."
        )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdModeTrial:
    """UCISD trial stored as same-spin EVDs and an opposite-spin SVD.

    With occupied--virtual pair dimensions ``Da`` and ``Db``, the doubles
    blocks are represented as

        Caa = Uaa diag(eigenvalues_aa) Uaa.T,
        Cab = Uab diag(singular_values_ab) Vab.T,
        Cbb = Ubb diag(eigenvalues_bb) Ubb.T.

    Rows of each ``*_modes`` array are the reshaped columns of the corresponding
    factor matrix. Retaining complete factors reproduces dense UCISD; retaining
    prefixes defines a consistent truncated UCISD trial.
    """

    mo_coeff_a: jax.Array
    mo_coeff_b: jax.Array
    c1a: jax.Array
    c1b: jax.Array
    eigenvalues_aa: jax.Array
    modes_aa: jax.Array
    singular_values_ab: jax.Array
    left_modes_ab: jax.Array
    right_modes_ab: jax.Array
    eigenvalues_bb: jax.Array
    modes_bb: jax.Array

    def __post_init__(self) -> None:
        arrays = (
            self.mo_coeff_a,
            self.mo_coeff_b,
            self.c1a,
            self.c1b,
            self.eigenvalues_aa,
            self.modes_aa,
            self.singular_values_ab,
            self.left_modes_ab,
            self.right_modes_ab,
            self.eigenvalues_bb,
            self.modes_bb,
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

        pair_a = (self.nocc[0], self.nvir[0])
        pair_b = (self.nocc[1], self.nvir[1])
        _validate_factor("alpha-alpha", self.eigenvalues_aa, self.modes_aa, pair_a)
        _validate_factor("beta-beta", self.eigenvalues_bb, self.modes_bb, pair_b)

        if self.singular_values_ab.ndim != 1:
            raise ValueError(
                "alpha-beta singular values must have rank 1, got shape "
                f"{self.singular_values_ab.shape}."
            )
        rank_ab = int(self.singular_values_ab.shape[0])
        maximum_rank = min(self.pair_dim[0], self.pair_dim[1])
        if not 0 <= rank_ab <= maximum_rank:
            raise ValueError(
                f"alpha-beta rank must lie in [0, {maximum_rank}], got {rank_ab}."
            )
        if self.left_modes_ab.shape != (rank_ab,) + pair_a:
            raise ValueError(
                f"left alpha-beta modes must have shape {(rank_ab,) + pair_a}, "
                f"got {self.left_modes_ab.shape}."
            )
        if self.right_modes_ab.shape != (rank_ab,) + pair_b:
            raise ValueError(
                f"right alpha-beta modes must have shape {(rank_ab,) + pair_b}, "
                f"got {self.right_modes_ab.shape}."
            )

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
    def mode_rank(self) -> tuple[int, int, int]:
        return (
            int(self.eigenvalues_aa.shape[0]),
            int(self.singular_values_ab.shape[0]),
            int(self.eigenvalues_bb.shape[0]),
        )

    def tree_flatten(self):
        return (
            self.mo_coeff_a,
            self.mo_coeff_b,
            self.c1a,
            self.c1b,
            self.eigenvalues_aa,
            self.modes_aa,
            self.singular_values_ab,
            self.left_modes_ab,
            self.right_modes_ab,
            self.eigenvalues_bb,
            self.modes_bb,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


class UcisdModeProjections(NamedTuple):
    """Four reusable projections of alpha/beta matrices onto UCISD modes."""

    aa: jax.Array
    ab_left: jax.Array
    ab_right: jax.Array
    bb: jax.Array


def get_rdm1(trial_data: UcisdModeTrial) -> jax.Array:
    """Return the UHF reference density matrices in the alpha orbital basis."""
    norb = trial_data.norb
    noa, nob = trial_data.nocc
    occ_a = jnp.arange(norb) < noa
    c_b = trial_data.mo_coeff_b
    dm_a = jnp.diag(occ_a)
    dm_b = c_b[:, :nob] @ c_b[:, :nob].conj().T
    return jnp.stack((dm_a, dm_b), axis=0)


def _mode_projections(modes: jax.Array, matrix: jax.Array) -> jax.Array:
    matrix_r = jnp.real(matrix).astype(modes.dtype)
    projections_r = jnp.einsum("rpt,pt->r", modes, matrix_r, optimize="optimal")
    if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        return projections_r.astype(jnp.float64)
    matrix_i = jnp.imag(matrix).astype(modes.dtype)
    projections_i = jnp.einsum("rpt,pt->r", modes, matrix_i, optimize="optimal")
    return projections_r.astype(jnp.complex128) + 1.0j * projections_i.astype(jnp.complex128)


def _mode_apply(
    values: jax.Array,
    modes: jax.Array,
    matrix: jax.Array,
    projections: jax.Array | None = None,
) -> jax.Array:
    if projections is None:
        projections = _mode_projections(modes, matrix)
    weighted = values.astype(jnp.float64) * projections
    if not jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        return jnp.einsum("r,rpt->pt", weighted, modes, optimize="optimal").astype(jnp.float64)
    applied_r = jnp.einsum("r,rpt->pt", jnp.real(weighted), modes, optimize="optimal")
    applied_i = jnp.einsum("r,rpt->pt", jnp.imag(weighted), modes, optimize="optimal")
    return applied_r.astype(jnp.complex128) + 1.0j * applied_i.astype(jnp.complex128)


def _mode_quadratic(
    values: jax.Array,
    modes: jax.Array,
    matrix: jax.Array,
    projections: jax.Array | None = None,
) -> jax.Array:
    if projections is None:
        projections = _mode_projections(modes, matrix)
    dtype = jnp.complex128 if jnp.issubdtype(matrix.dtype, jnp.complexfloating) else jnp.float64
    return jnp.sum(values.astype(jnp.float64) * projections * projections, dtype=dtype)


def doubles_projections(
    trial_data: UcisdModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
) -> UcisdModeProjections:
    """Project alpha/beta pair matrices once for apply and quadratic contractions."""
    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrix_a.shape != expected_a or matrix_b.shape != expected_b:
        raise ValueError(
            f"matrix pair must have shapes {expected_a} and {expected_b}, got "
            f"{matrix_a.shape} and {matrix_b.shape}."
        )
    return UcisdModeProjections(
        aa=_mode_projections(trial_data.modes_aa, matrix_a),
        ab_left=_mode_projections(trial_data.left_modes_ab, matrix_a),
        ab_right=_mode_projections(trial_data.right_modes_ab, matrix_b),
        bb=_mode_projections(trial_data.modes_bb, matrix_b),
    )


def doubles_apply(
    trial_data: UcisdModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    *,
    projections: UcisdModeProjections | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(Caa @ a + Cab @ b, Cbb @ b + Cab.T @ a)``."""
    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrix_a.shape != expected_a or matrix_b.shape != expected_b:
        raise ValueError(
            f"matrix pair must have shapes {expected_a} and {expected_b}, got "
            f"{matrix_a.shape} and {matrix_b.shape}."
        )

    if projections is None:
        projections = doubles_projections(trial_data, matrix_a, matrix_b)
    applied_a = _mode_apply(
        trial_data.eigenvalues_aa,
        trial_data.modes_aa,
        matrix_a,
        projections.aa,
    )
    applied_b = _mode_apply(
        trial_data.eigenvalues_bb,
        trial_data.modes_bb,
        matrix_b,
        projections.bb,
    )
    weighted_a = trial_data.singular_values_ab.astype(jnp.float64) * projections.ab_right
    weighted_b = trial_data.singular_values_ab.astype(jnp.float64) * projections.ab_left

    if jnp.issubdtype(matrix_a.dtype, jnp.complexfloating) or jnp.issubdtype(
        matrix_b.dtype, jnp.complexfloating
    ):
        cross_a = jnp.einsum(
            "r,rpt->pt", jnp.real(weighted_a), trial_data.left_modes_ab, optimize="optimal"
        ).astype(jnp.complex128)
        cross_a += 1.0j * jnp.einsum(
            "r,rpt->pt", jnp.imag(weighted_a), trial_data.left_modes_ab, optimize="optimal"
        ).astype(jnp.complex128)
        cross_b = jnp.einsum(
            "r,rpt->pt", jnp.real(weighted_b), trial_data.right_modes_ab, optimize="optimal"
        ).astype(jnp.complex128)
        cross_b += 1.0j * jnp.einsum(
            "r,rpt->pt", jnp.imag(weighted_b), trial_data.right_modes_ab, optimize="optimal"
        ).astype(jnp.complex128)
    else:
        cross_a = jnp.einsum(
            "r,rpt->pt", weighted_a, trial_data.left_modes_ab, optimize="optimal"
        ).astype(jnp.float64)
        cross_b = jnp.einsum(
            "r,rpt->pt", weighted_b, trial_data.right_modes_ab, optimize="optimal"
        ).astype(jnp.float64)
    return applied_a + cross_a, applied_b + cross_b


def doubles_quadratic(
    trial_data: UcisdModeTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    *,
    projections: UcisdModeProjections | None = None,
) -> jax.Array:
    """Return the UCISD doubles overlap bilinear for a pair of matrices."""
    expected_a = (trial_data.nocc[0], trial_data.nvir[0])
    expected_b = (trial_data.nocc[1], trial_data.nvir[1])
    if matrix_a.shape != expected_a or matrix_b.shape != expected_b:
        raise ValueError(
            f"matrix pair must have shapes {expected_a} and {expected_b}, got "
            f"{matrix_a.shape} and {matrix_b.shape}."
        )
    if projections is None:
        projections = doubles_projections(trial_data, matrix_a, matrix_b)
    dtype = (
        jnp.complex128
        if jnp.issubdtype(matrix_a.dtype, jnp.complexfloating)
        or jnp.issubdtype(matrix_b.dtype, jnp.complexfloating)
        else jnp.float64
    )
    cross = jnp.sum(
        trial_data.singular_values_ab.astype(jnp.float64)
        * projections.ab_left
        * projections.ab_right,
        dtype=dtype,
    )
    return (
        0.5
        * _mode_quadratic(
            trial_data.eigenvalues_aa,
            trial_data.modes_aa,
            matrix_a,
            projections.aa,
        )
        + cross
        + 0.5
        * _mode_quadratic(
            trial_data.eigenvalues_bb,
            trial_data.modes_bb,
            matrix_b,
            projections.bb,
        )
    )


def _greens_restricted(
    walker: jax.Array,
    trial_data: UcisdModeTrial,
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


def overlap_r(walker: jax.Array, trial_data: UcisdModeTrial) -> jax.Array:
    """Overlap of a restricted walker with a mode-native UCISD trial."""
    green_a, green_b, det0 = _greens_restricted(walker, trial_data)
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_b[:, nob : nob + trial_data.nvir[1]]
    singles = jnp.einsum("ia,ia->", trial_data.c1a, green_occ_a, optimize="optimal")
    singles += jnp.einsum("ia,ia->", trial_data.c1b, green_occ_b, optimize="optimal")
    doubles = doubles_quadratic(trial_data, green_occ_a, green_occ_b)
    return det0 * (1.0 + singles + doubles)


def _load_modes(
    data: dict,
    *,
    modes_key: str,
    vectors_key: str,
    pair_shape: tuple[int, int],
    rank: int,
) -> jax.Array:
    if modes_key in data:
        modes = jnp.asarray(data[modes_key])
    elif vectors_key in data:
        vectors = jnp.asarray(data[vectors_key])
        pair_dim = int(pair_shape[0] * pair_shape[1])
        if vectors.shape != (pair_dim, rank):
            raise ValueError(
                f"{vectors_key} must have shape {(pair_dim, rank)}, got {vectors.shape}."
            )
        modes = vectors.T.reshape((rank,) + pair_shape)
    else:
        raise KeyError(f"mode-native UCISD data requires {modes_key!r} or {vectors_key!r}.")
    return modes


def make_ucisd_mode_trial_data(
    data: dict,
    sys: System,
    *,
    mixed_precision: bool = True,
) -> UcisdModeTrial:
    """Build a full-rank or truncated mode-native UCISD trial."""
    del sys
    c1a = jnp.asarray(data["c1a"], dtype=jnp.float64)
    c1b = jnp.asarray(data["c1b"], dtype=jnp.float64)
    pair_a = (int(c1a.shape[0]), int(c1a.shape[1]))
    pair_b = (int(c1b.shape[0]), int(c1b.shape[1]))
    values_aa = jnp.asarray(data["eigenvalues_aa"], dtype=jnp.float64)
    values_ab = jnp.asarray(data["singular_values_ab"], dtype=jnp.float64)
    values_bb = jnp.asarray(data["eigenvalues_bb"], dtype=jnp.float64)
    modes_aa = _load_modes(
        data,
        modes_key="modes_aa",
        vectors_key="eigenvectors_aa",
        pair_shape=pair_a,
        rank=int(values_aa.shape[0]),
    )
    left_modes_ab = _load_modes(
        data,
        modes_key="left_modes_ab",
        vectors_key="left_singular_vectors_ab",
        pair_shape=pair_a,
        rank=int(values_ab.shape[0]),
    )
    right_modes_ab = _load_modes(
        data,
        modes_key="right_modes_ab",
        vectors_key="right_singular_vectors_ab",
        pair_shape=pair_b,
        rank=int(values_ab.shape[0]),
    )
    modes_bb = _load_modes(
        data,
        modes_key="modes_bb",
        vectors_key="eigenvectors_bb",
        pair_shape=pair_b,
        rank=int(values_bb.shape[0]),
    )
    mode_dtype = jnp.float32 if mixed_precision else jnp.float64
    return UcisdModeTrial(
        mo_coeff_a=jnp.asarray(data["mo_coeff_a"]),
        mo_coeff_b=jnp.asarray(data["mo_coeff_b"]),
        c1a=c1a,
        c1b=c1b,
        eigenvalues_aa=values_aa,
        modes_aa=jnp.asarray(modes_aa, dtype=mode_dtype),
        singular_values_ab=values_ab,
        left_modes_ab=jnp.asarray(left_modes_ab, dtype=mode_dtype),
        right_modes_ab=jnp.asarray(right_modes_ab, dtype=mode_dtype),
        eigenvalues_bb=values_bb,
        modes_bb=jnp.asarray(modes_bb, dtype=mode_dtype),
    )


def make_ucisd_mode_trial_ops(sys: System) -> TrialOps:
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "UCISD mode trial currently supports only restricted walkers, got: "
            f"{sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
