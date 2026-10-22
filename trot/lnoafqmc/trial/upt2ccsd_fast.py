from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

from ...trial.upt2ccsd_uh import overlap_u as _overlap_u

__all__ = ["Upt2ccsdFastTrial", "overlap_u", "make_upt2ccsd_fast_trial_data"]


def overlap_u(walker: Any, trial_data: Any) -> jax.Array:
    """<exp(T1)HF|walker>: the branch's overlap, which reads only mo_t_a / mo_t_b."""
    return _overlap_u(walker, trial_data)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdFastTrial:
    """
    Unrestricted pt2CCSD trial of one LNO fragment with the fragment projectors kept in
    factored form (the "upt2ccsd_fast" trial, meas/upt2ccsd_fast.py): trial/upt2ccsd.py
    with prjlo_s = U_s U_s^H, U_s = <act_occ_s|lo_s> of shape (nocc_s, nlo), and the
    doubles carrying the local index instead of the projected occupied one,

        t2aa_u_{Iajb} = sum_i t2aa_{iajb} U_a_{iI}       (nlo, nvir_a, nocc_a, nvir_a)
        t2ab_u_{Iajb} = sum_i t2ab_{iajb} U_a_{iI}       (nlo, nvir_a, nocc_b, nvir_b)
        t2ba_u_{Iajb} = sum_i t2ba_{iajb} U_b_{iI}       (nlo, nvir_b, nocc_a, nvir_a)
        t2bb_u_{Iajb} = sum_i t2bb_{iajb} U_b_{iI}       (nlo, nvir_b, nocc_b, nvir_b)

    The second factor U_s^H is applied inside the kernel with the local index contracted
    last. The same-spin blocks are antisymmetric in (a, b) (the projection touches only
    the first occupied index), which is what lets one contraction per index pair serve
    both the direct and the exchange term; t2ab_u and t2ba_u are independent.

    Arrays:
      mo_t_a, mo_t_b: (norb_s, nocc_s)   exp(T1_s)|HF_s> by Thouless' theorem
      t2aa_u, t2ab_u, t2ba_u, t2bb_u     as above
      u_a, u_b: (nocc_s, nlo)            U_s = <act_occ_s|lo_s>
      t1a, t1b: (nocc_s, nvir_s)         the singles, unprojected (for e0t1orb)
    """

    mo_t_a: jax.Array
    mo_t_b: jax.Array
    t2aa_u: jax.Array
    t2ab_u: jax.Array
    t2ba_u: jax.Array
    t2bb_u: jax.Array
    u_a: jax.Array
    u_b: jax.Array
    t1a: jax.Array
    t1b: jax.Array

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.u_a.shape[0]), int(self.u_b.shape[0]))

    @property
    def nlo(self) -> int:
        return int(self.u_a.shape[1])

    @property
    def nvir(self) -> tuple[int, int]:
        return (int(self.t2ab_u.shape[1]), int(self.t2ab_u.shape[3]))

    @property
    def norb(self) -> tuple[int, int]:
        nocc, nvir = self.nocc, self.nvir
        return (nocc[0] + nvir[0], nocc[1] + nvir[1])

    @property
    def prjlo_a(self) -> jax.Array:
        return self.u_a @ self.u_a.conj().T

    @property
    def prjlo_b(self) -> jax.Array:
        return self.u_b @ self.u_b.conj().T

    def tree_flatten(self):
        children = (
            self.mo_t_a,
            self.mo_t_b,
            self.t2aa_u,
            self.t2ab_u,
            self.t2ba_u,
            self.t2bb_u,
            self.u_a,
            self.u_b,
            self.t1a,
            self.t1b,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t_a, mo_t_b, t2aa_u, t2ab_u, t2ba_u, t2bb_u, u_a, u_b, t1a, t1b = children
        return cls(
            mo_t_a=mo_t_a,
            mo_t_b=mo_t_b,
            t2aa_u=t2aa_u,
            t2ab_u=t2ab_u,
            t2ba_u=t2ba_u,
            t2bb_u=t2bb_u,
            u_a=u_a,
            u_b=u_b,
            t1a=t1a,
            t1b=t1b,
        )


def make_upt2ccsd_fast_trial_data(data: dict, sys: Any) -> Upt2ccsdFastTrial:
    """From staging.stage_upt2ccsd_fast_trial's data; sys is a System_uh."""
    td = Upt2ccsdFastTrial(
        mo_t_a=jnp.asarray(data["mo_t_a"])[:, : sys.nup],
        mo_t_b=jnp.asarray(data["mo_t_b"])[:, : sys.ndn],
        t2aa_u=jnp.asarray(data["t2aa_u"]),
        t2ab_u=jnp.asarray(data["t2ab_u"]),
        t2ba_u=jnp.asarray(data["t2ba_u"]),
        t2bb_u=jnp.asarray(data["t2bb_u"]),
        u_a=jnp.asarray(data["u_a"]),
        u_b=jnp.asarray(data["u_b"]),
        t1a=jnp.asarray(data["t1a"]),
        t1b=jnp.asarray(data["t1b"]),
    )
    norb = sys.norb if isinstance(sys.norb, tuple) else (sys.norb, sys.norb)
    if td.nocc != (sys.nup, sys.ndn) or td.norb != tuple(norb):
        raise ValueError(
            f"the fragment amplitudes span norb={td.norb} with nocc={td.nocc} but the "
            f"fragment hamiltonian has norb={tuple(norb)} and nelec={(sys.nup, sys.ndn)}."
        )
    return td
