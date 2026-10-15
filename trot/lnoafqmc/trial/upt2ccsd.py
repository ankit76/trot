from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

from ...trial.upt2ccsd_uh import overlap_u as _overlap_u

__all__ = ["Upt2ccsdTrial", "overlap_u", "make_upt2ccsd_trial_data"]


def overlap_u(walker: Any, trial_data: Any) -> jax.Array:
    """<exp(T1)HF|walker>: the branch's overlap, which reads only mo_t_a / mo_t_b."""
    return _overlap_u(walker, trial_data)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdTrial:
    """
    Unrestricted pt2CCSD trial of one LNO fragment. Each spin lives in its own LNO basis,
    the one the uchol fragment hamiltonian is built in, with that spin's reference
    occupying its first nocc orbitals. The branch's trial/upt2ccsd_uh.Upt2ccsdTrial plus
    the fragment projectors.

    Arrays:
      mo_t_a: (norb_a, nocc_a)                 exp(T1a)|HF_a> by Thouless' theorem
      mo_t_b: (norb_b, nocc_b)                 exp(T1b)|HF_b>
      t2aa:   (nocc_a, nvir_a, nocc_a, nvir_a) t2_{k a j b} = sum_i t2_{i a j b} prjlo_a_{i k}
      t2ab:   (nocc_a, nvir_a, nocc_b, nvir_b) projected on its alpha (first) index
      t2ba:   (nocc_b, nvir_b, nocc_a, nvir_a) t2ab with the spins swapped, projected on
                                               its beta (first) index
      t2bb:   (nocc_b, nvir_b, nocc_b, nvir_b) projected with prjlo_b
      prjlo_a, prjlo_b: (nocc_s, nocc_s)       U_s U_s^T, U_s = <act_occ_s|lo_s>
      t1a, t1b: (nocc_s, nvir_s)               the singles, unprojected (for e0t1orb)

    The projection acts on the first occupied index only, so the same-spin blocks stay
    antisymmetric in (a, b) but not in (i, j), and t2ab and t2ba are independent.
    """

    mo_t_a: jax.Array
    mo_t_b: jax.Array
    t2aa: jax.Array
    t2ab: jax.Array
    t2ba: jax.Array
    t2bb: jax.Array
    prjlo_a: jax.Array
    prjlo_b: jax.Array
    t1a: jax.Array
    t1b: jax.Array

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.t2ab.shape[0]), int(self.t2ab.shape[2]))

    @property
    def nvir(self) -> tuple[int, int]:
        return (int(self.t2ab.shape[1]), int(self.t2ab.shape[3]))

    @property
    def norb(self) -> tuple[int, int]:
        nocc, nvir = self.nocc, self.nvir
        return (nocc[0] + nvir[0], nocc[1] + nvir[1])

    def tree_flatten(self):
        children = (
            self.mo_t_a,
            self.mo_t_b,
            self.t2aa,
            self.t2ab,
            self.t2ba,
            self.t2bb,
            self.prjlo_a,
            self.prjlo_b,
            self.t1a,
            self.t1b,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t_a, mo_t_b, t2aa, t2ab, t2ba, t2bb, prjlo_a, prjlo_b, t1a, t1b = children
        return cls(
            mo_t_a=mo_t_a,
            mo_t_b=mo_t_b,
            t2aa=t2aa,
            t2ab=t2ab,
            t2ba=t2ba,
            t2bb=t2bb,
            prjlo_a=prjlo_a,
            prjlo_b=prjlo_b,
            t1a=t1a,
            t1b=t1b,
        )


def make_upt2ccsd_trial_data(data: dict, sys: Any) -> Upt2ccsdTrial:
    """From staging.stage_upt2ccsd_trial's data; sys is a System_uh."""
    td = Upt2ccsdTrial(
        mo_t_a=jnp.asarray(data["mo_t_a"])[:, : sys.nup],
        mo_t_b=jnp.asarray(data["mo_t_b"])[:, : sys.ndn],
        t2aa=jnp.asarray(data["t2aa"]),
        t2ab=jnp.asarray(data["t2ab"]),
        t2ba=jnp.asarray(data["t2ba"]),
        t2bb=jnp.asarray(data["t2bb"]),
        prjlo_a=jnp.asarray(data["prjlo_a"]),
        prjlo_b=jnp.asarray(data["prjlo_b"]),
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
