from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PtuccsdThoulessTrial:
    """Unrestricted first-order PT-UCCSD trial with T1 in the reference.

    ``mo_t_a`` is the alpha occupied Thouless determinant in the alpha MO
    basis. ``mo_t_b`` is the corresponding beta determinant in the beta MO
    basis, and ``mo_coeff_b`` rotates alpha-basis walkers into that beta basis.
    The three doubles blocks contain raw UCCSD amplitudes in ``(i,a,j,b)``
    layout, without disconnected ``T1*T1`` terms. Each spin basis is ordered
    [occupied | retained virtual | discarded virtual]; T2 uses only retained
    virtuals while the Thouless reference matrices have all orbital rows.
    """

    mo_t_a: jax.Array
    mo_t_b: jax.Array
    mo_coeff_b: jax.Array
    t2aa: jax.Array
    t2ab: jax.Array
    t2bb: jax.Array

    def __post_init__(self) -> None:
        if not hasattr(self.mo_t_a, "ndim"):
            return
        if self.mo_t_a.ndim != 2 or self.mo_t_b.ndim != 2:
            raise ValueError("mo_t_a and mo_t_b must be rank 2.")
        if self.t2aa.ndim != 4 or self.t2ab.ndim != 4 or self.t2bb.ndim != 4:
            raise ValueError("t2aa, t2ab, and t2bb must be rank 4.")

        noa, nva = self.t2aa.shape[:2]
        nob, nvb = self.t2bb.shape[:2]
        norb = int(self.mo_t_a.shape[0])
        if self.t2aa.shape != (noa, nva, noa, nva):
            raise ValueError(
                "t2aa must have shape (nocc_a,nvir_a,nocc_a,nvir_a); "
                f"got {self.t2aa.shape}."
            )
        if self.t2ab.shape != (noa, nva, nob, nvb):
            raise ValueError(
                f"t2ab must have shape {(noa, nva, nob, nvb)}; got {self.t2ab.shape}."
            )
        if self.t2bb.shape != (nob, nvb, nob, nvb):
            raise ValueError(
                "t2bb must have shape (nocc_b,nvir_b,nocc_b,nvir_b); "
                f"got {self.t2bb.shape}."
            )
        if noa + nva > norb or nob + nvb > norb:
            raise ValueError("Retained trial orbitals exceed the full orbital dimension.")
        if self.mo_t_a.shape != (norb, noa):
            raise ValueError(
                f"mo_t_a must have shape {(norb, noa)}; got {self.mo_t_a.shape}."
            )
        if self.mo_t_b.shape != (norb, nob):
            raise ValueError(
                f"mo_t_b must have shape {(norb, nob)}; got {self.mo_t_b.shape}."
            )
        if self.mo_coeff_b.shape != (norb, norb):
            raise ValueError(
                "mo_coeff_b must be the full beta orbital rotation with shape "
                f"{(norb, norb)}; got {self.mo_coeff_b.shape}."
            )

    @property
    def norb(self) -> int:
        return int(self.mo_t_a.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.mo_t_a.shape[1]), int(self.mo_t_b.shape[1]))

    @property
    def nvir(self) -> tuple[int, int]:
        return (int(self.t2aa.shape[1]), int(self.t2bb.shape[1]))

    def tree_flatten(self):
        return (
            self.mo_t_a,
            self.mo_t_b,
            self.mo_coeff_b,
            self.t2aa,
            self.t2ab,
            self.t2bb,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        mo_t_a, mo_t_b, mo_coeff_b, t2aa, t2ab, t2bb = children
        return cls(
            mo_t_a=mo_t_a,
            mo_t_b=mo_t_b,
            mo_coeff_b=mo_coeff_b,
            t2aa=t2aa,
            t2ab=t2ab,
            t2bb=t2bb,
        )


def thouless_mo_from_t1(t1: jax.Array, norb: int | None = None) -> jax.Array:
    """Return exp(T1)|HF> orbitals, with zero rows for discarded trial virtuals."""
    nocc, nvir = t1.shape
    occupied = jnp.vstack([jnp.eye(nocc, dtype=t1.dtype), t1.T])
    if norb is None:
        return occupied
    if norb < nocc + nvir:
        raise ValueError("Retained trial orbitals exceed the full orbital dimension.")
    return jnp.pad(occupied, ((0, norb - nocc - nvir), (0, 0)))


def _t2_from_pyscf_layout(t2: jax.Array) -> jax.Array:
    return jnp.asarray(t2).transpose(0, 2, 1, 3)


def _projector(coeff: jax.Array) -> jax.Array:
    metric = coeff.conj().T @ coeff
    return coeff @ jnp.linalg.solve(metric, coeff.conj().T)


def get_rdm1(trial_data: PtuccsdThoulessTrial) -> jax.Array:
    dm_a = _projector(trial_data.mo_t_a)
    beta_occ_alpha_basis = trial_data.mo_coeff_b @ trial_data.mo_t_b
    dm_b = _projector(beta_occ_alpha_basis)
    return jnp.stack([dm_a, dm_b], axis=0)


def _half_green_from_overlap_matrix(walker: jax.Array, overlap: jax.Array) -> jax.Array:
    return jnp.linalg.solve(overlap.T, walker.T)


def _spin_green(walker: jax.Array, mo_t: jax.Array) -> jax.Array:
    overlap = mo_t.conj().T @ walker
    return mo_t.conj() @ _half_green_from_overlap_matrix(walker, overlap)


def greens_unrestricted(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessTrial,
) -> tuple[jax.Array, jax.Array]:
    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    return (
        _spin_green(walker_a, trial_data.mo_t_a),
        _spin_green(walker_b_beta, trial_data.mo_t_b),
    )


def greenp_from_green(green: jax.Array, nocc: int, nvir: int | None = None) -> jax.Array:
    stop = green.shape[1] if nvir is None else nocc + nvir
    return (green - jnp.eye(green.shape[0], dtype=green.dtype))[:, nocc:stop]


def theta_t2_from_greens(
    green_a: jax.Array,
    green_b: jax.Array,
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    noa, nob = trial_data.nocc
    green_occ_a = green_a[:noa, noa : noa + trial_data.nvir[0]]
    green_occ_b = green_b[:nob, nob : nob + trial_data.nvir[1]]
    theta_aa = 0.5 * jnp.einsum(
        "iajb,ia,jb->", trial_data.t2aa, green_occ_a, green_occ_a, optimize="optimal"
    )
    theta_ab = jnp.einsum(
        "iajb,ia,jb->", trial_data.t2ab, green_occ_a, green_occ_b, optimize="optimal"
    )
    theta_bb = 0.5 * jnp.einsum(
        "iajb,ia,jb->", trial_data.t2bb, green_occ_b, green_occ_b, optimize="optimal"
    )
    return theta_aa + theta_ab + theta_bb


def theta_t2_u(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    green_a, green_b = greens_unrestricted(walker, trial_data)
    return theta_t2_from_greens(green_a, green_b, trial_data)


def reference_overlap_u(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    walker_a, walker_b = walker
    walker_b_beta = trial_data.mo_coeff_b.conj().T @ walker_b
    overlap_a = trial_data.mo_t_a.conj().T @ walker_a
    overlap_b = trial_data.mo_t_b.conj().T @ walker_b_beta
    return jnp.linalg.det(overlap_a) * jnp.linalg.det(overlap_b)


def overlap_u(
    walker: tuple[jax.Array, jax.Array],
    trial_data: PtuccsdThoulessTrial,
) -> jax.Array:
    return reference_overlap_u(walker, trial_data) * jnp.exp(theta_t2_u(walker, trial_data))


def _split_restricted_walker(
    walker: jax.Array,
    trial_data: PtuccsdThoulessTrial,
) -> tuple[jax.Array, jax.Array]:
    noa, nob = trial_data.nocc
    if walker.shape[1] < max(noa, nob):
        raise ValueError(
            "restricted walker has too few occupied columns for the PT-UCCSD trial: "
            f"walker shape={walker.shape}, nocc={trial_data.nocc}."
        )
    return walker[:, :noa], walker[:, :nob]


def reference_overlap_r(walker: jax.Array, trial_data: PtuccsdThoulessTrial) -> jax.Array:
    return reference_overlap_u(_split_restricted_walker(walker, trial_data), trial_data)


def overlap_r(walker: jax.Array, trial_data: PtuccsdThoulessTrial) -> jax.Array:
    return overlap_u(_split_restricted_walker(walker, trial_data), trial_data)


def make_ptuccsd_thouless_trial_ops(sys: System) -> TrialOps:
    walker_kind = sys.walker_kind.lower()
    if walker_kind == "restricted":
        if sys.nup < sys.ndn:
            raise ValueError("Restricted PT2-UCCSD trials require nup >= ndn.")
        overlap_fn = overlap_r
    elif walker_kind == "unrestricted":
        overlap_fn = overlap_u
    else:
        raise ValueError(
            "PT-UCCSD Thouless trial supports restricted/unrestricted walkers, "
            f"got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_fn, get_rdm1=get_rdm1)


def make_ptuccsd_thouless_trial_data(
    data: dict,
    sys: System | None = None,
) -> PtuccsdThoulessTrial:
    layout = str(data.get("t2_layout", "pyscf")).lower()
    if layout in {"pyscf", "ijab"}:
        t2aa = _t2_from_pyscf_layout(data["t2aa"])
        t2ab = _t2_from_pyscf_layout(data["t2ab"])
        t2bb = _t2_from_pyscf_layout(data["t2bb"])
    elif layout == "iajb":
        t2aa = jnp.asarray(data["t2aa"])
        t2ab = jnp.asarray(data["t2ab"])
        t2bb = jnp.asarray(data["t2bb"])
    else:
        raise ValueError(f"Unknown PT-UCCSD t2_layout: {layout!r}")

    mo_coeff_b_raw = data.get("mo_coeff_b", data.get("mo_b"))
    if mo_coeff_b_raw is None:
        raise KeyError("PT-UCCSD trial data requires 'mo_coeff_b' or 'mo_b'.")
    mo_coeff_b_raw = jnp.asarray(mo_coeff_b_raw)
    norb = int(mo_coeff_b_raw.shape[0])

    mo_t_a = (
        jnp.asarray(data["mo_t_a"])
        if "mo_t_a" in data
        else thouless_mo_from_t1(jnp.asarray(data["t1a"]), norb)
    )
    mo_t_b = (
        jnp.asarray(data["mo_t_b"])
        if "mo_t_b" in data
        else thouless_mo_from_t1(jnp.asarray(data["t1b"]), norb)
    )

    trial_data = PtuccsdThoulessTrial(
        mo_t_a=mo_t_a,
        mo_t_b=mo_t_b,
        mo_coeff_b=jnp.asarray(mo_coeff_b_raw),
        t2aa=t2aa,
        t2ab=t2ab,
        t2bb=t2bb,
    )
    if sys is not None:
        if trial_data.norb != sys.norb:
            raise ValueError(
                "PT2-UCCSD trial/system orbital mismatch: "
                f"trial norb={trial_data.norb}, system norb={sys.norb}."
            )
        if trial_data.nocc != sys.nelec:
            raise ValueError(
                "PT2-UCCSD trial/system electron mismatch: "
                f"trial nocc={trial_data.nocc}, system nelec={sys.nelec}."
            )
    return trial_data


__all__ = [
    "PtuccsdThoulessTrial",
    "get_rdm1",
    "greenp_from_green",
    "greens_unrestricted",
    "make_ptuccsd_thouless_trial_data",
    "make_ptuccsd_thouless_trial_ops",
    "overlap_r",
    "overlap_u",
    "reference_overlap_r",
    "reference_overlap_u",
    "theta_t2_from_greens",
    "theta_t2_u",
    "thouless_mo_from_t1",
]
