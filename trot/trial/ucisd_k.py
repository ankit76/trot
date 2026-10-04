from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


def build_ucisd_kernel(
    ci2aa: jax.Array,
    ci2ab: jax.Array,
    ci2bb: jax.Array,
) -> jax.Array:
    """Return the symmetric UCISD doubles kernel in combined pair space.

    With flattened occupied--virtual pair dimensions ``Da`` and ``Db``,

    ``K = [[Caa, Cab], [Cab.T, Cbb]]``.

    The doubles overlap is ``0.5 * z.T @ K @ z`` for
    ``z = concatenate((g_alpha, g_beta))``. Contractions are bilinear; no
    complex conjugation is implied.
    """
    ci2aa = jnp.asarray(ci2aa)
    ci2ab = jnp.asarray(ci2ab)
    ci2bb = jnp.asarray(ci2bb)
    if ci2aa.ndim != 4 or ci2ab.ndim != 4 or ci2bb.ndim != 4:
        raise ValueError("UCISD doubles blocks must all have rank 4.")

    noa, nva, noa_2, nva_2 = ci2aa.shape
    nob, nvb, nob_2, nvb_2 = ci2bb.shape
    if (noa_2, nva_2) != (noa, nva):
        raise ValueError(
            "ci2aa must have shape (nocc_a, nvir_a, nocc_a, nvir_a), "
            f"got {ci2aa.shape}."
        )
    if (nob_2, nvb_2) != (nob, nvb):
        raise ValueError(
            "ci2bb must have shape (nocc_b, nvir_b, nocc_b, nvir_b), "
            f"got {ci2bb.shape}."
        )
    if ci2ab.shape != (noa, nva, nob, nvb):
        raise ValueError(
            f"ci2ab must have shape {(noa, nva, nob, nvb)}, got {ci2ab.shape}."
        )

    da = int(noa * nva)
    db = int(nob * nvb)
    aa = ci2aa.reshape(da, da)
    ab = ci2ab.reshape(da, db)
    bb = ci2bb.reshape(db, db)
    return jnp.block([[aa, ab], [ab.T, bb]])


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdKTrial:
    """Exact UCISD trial stored as one symmetric doubles kernel.

    Alpha and beta amplitudes retain their own UHF orbital bases. The kernel
    acts only in the concatenated occupied--virtual pair space; walkers remain
    restricted so spin projection is preserved.
    """

    mo_coeff_a: jax.Array
    mo_coeff_b: jax.Array
    c1a: jax.Array
    c1b: jax.Array
    k: jax.Array

    def __post_init__(self) -> None:
        arrays = (self.mo_coeff_a, self.mo_coeff_b, self.c1a, self.c1b, self.k)
        # JAX may reconstruct registered pytrees once with opaque placeholders.
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
        if self.k.ndim != 2:
            raise ValueError(f"k must have rank 2, got shape {self.k.shape}.")
        combined_dim = sum(self.pair_dim)
        if self.k.shape != (combined_dim, combined_dim):
            raise ValueError(
                f"k must have shape {(combined_dim, combined_dim)}, got {self.k.shape}."
            )
        if jnp.issubdtype(self.k.dtype, jnp.complexfloating):
            raise ValueError("UcisdKTrial currently requires a real K matrix.")

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

    def tree_flatten(self):
        return (self.mo_coeff_a, self.mo_coeff_b, self.c1a, self.c1b, self.k), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


def get_rdm1(trial_data: UcisdKTrial) -> jax.Array:
    """Return the UHF reference density matrices in the alpha orbital basis."""
    norb = trial_data.norb
    noa, nob = trial_data.nocc
    occ_a = jnp.arange(norb) < noa
    c_b = trial_data.mo_coeff_b
    dm_a = jnp.diag(occ_a)
    dm_b = c_b[:, :nob] @ c_b[:, :nob].conj().T
    return jnp.stack((dm_a, dm_b), axis=0)


def _combined_pair_vector(
    trial_data: UcisdKTrial,
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


def k_apply(
    trial_data: UcisdKTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply the exact combined UCISD kernel to alpha/beta pair matrices."""
    vector = _combined_pair_vector(trial_data, matrix_a, matrix_b)
    vector_r = jnp.real(vector).astype(trial_data.k.dtype)
    applied_r = jnp.einsum("pq,q->p", trial_data.k, vector_r, optimize="optimal")
    if not jnp.issubdtype(vector.dtype, jnp.complexfloating):
        applied = applied_r
    else:
        vector_i = jnp.imag(vector).astype(trial_data.k.dtype)
        applied_i = jnp.einsum("pq,q->p", trial_data.k, vector_i, optimize="optimal")
        complex_dtype = jnp.complex64 if trial_data.k.dtype == jnp.float32 else jnp.complex128
        applied = applied_r.astype(complex_dtype)
        applied += jnp.asarray(1.0j, dtype=complex_dtype) * applied_i.astype(complex_dtype)

    da, _ = trial_data.pair_dim
    shape_a = (trial_data.nocc[0], trial_data.nvir[0])
    shape_b = (trial_data.nocc[1], trial_data.nvir[1])
    return applied[:da].reshape(shape_a), applied[da:].reshape(shape_b)


def k_quadratic(
    trial_data: UcisdKTrial,
    matrix_a: jax.Array,
    matrix_b: jax.Array,
    applied: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Return ``0.5 * z.T @ K @ z`` using a bilinear contraction."""
    if applied is None:
        applied = k_apply(trial_data, matrix_a, matrix_b)
    applied_a, applied_b = applied
    dtype = (
        jnp.complex128
        if jnp.issubdtype(matrix_a.dtype, jnp.complexfloating)
        or jnp.issubdtype(matrix_b.dtype, jnp.complexfloating)
        else jnp.float64
    )
    quadratic = jnp.sum(matrix_a * applied_a, dtype=dtype)
    quadratic += jnp.sum(matrix_b * applied_b, dtype=dtype)
    return 0.5 * quadratic


def _greens_restricted(
    walker: jax.Array,
    trial_data: UcisdKTrial,
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


def overlap_r(walker: jax.Array, trial_data: UcisdKTrial) -> jax.Array:
    """Exact overlap for a restricted walker and combined-K UCISD trial."""
    green_a, green_b, det0 = _greens_restricted(walker, trial_data)
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_b[:, nob : nob + trial_data.nvir[1]]
    singles = jnp.einsum("ia,ia->", trial_data.c1a, green_occ_a, optimize="optimal")
    singles += jnp.einsum("ia,ia->", trial_data.c1b, green_occ_b, optimize="optimal")
    doubles = k_quadratic(trial_data, green_occ_a, green_occ_b)
    return det0 * (1.0 + singles + doubles)


def make_ucisd_k_trial_data(data: dict, sys: System) -> UcisdKTrial:
    """Build an exact combined-K UCISD trial from K or dense spin blocks."""
    del sys
    if "k" in data:
        kernel = jnp.asarray(data["k"], dtype=jnp.float64)
    elif all(key in data for key in ("ci2aa", "ci2ab", "ci2bb")):
        kernel = build_ucisd_kernel(
            jnp.asarray(data["ci2aa"], dtype=jnp.float64),
            jnp.asarray(data["ci2ab"], dtype=jnp.float64),
            jnp.asarray(data["ci2bb"], dtype=jnp.float64),
        )
    else:
        raise KeyError("K-native UCISD trial data requires 'k' or all three dense ci2 blocks.")

    return UcisdKTrial(
        mo_coeff_a=jnp.asarray(data["mo_coeff_a"]),
        mo_coeff_b=jnp.asarray(data["mo_coeff_b"]),
        c1a=jnp.asarray(data["c1a"], dtype=jnp.float64),
        c1b=jnp.asarray(data["c1b"], dtype=jnp.float64),
        k=kernel,
    )


def make_ucisd_k_trial_ops(sys: System) -> TrialOps:
    """Build combined-K UCISD trial operations for restricted walkers."""
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            "UCISD K trial currently supports only restricted walkers, got: "
            f"{sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
