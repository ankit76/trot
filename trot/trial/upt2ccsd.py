from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.system import System_uh


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdTrial:
    """
    Unrestricted pt2CCSD trial. Each spin lives in its own MO basis, the one the uchol
    hamiltonian is built in, with that spin's reference occupying its first nocc orbitals.

    Arrays:
      mo_t_a: (norb_a, nocc_a)                 exp(T1a)|HF_a> by Thouless theorem, alpha MO basis
      mo_t_b: (norb_b, nocc_b)                 exp(T1b)|HF_b>, beta MO basis
      t2aa:   (nocc_a, nvir_a, nocc_a, nvir_a) antisymmetrized, t2_{i a j b}
      t2ab:   (nocc_a, nvir_a, nocc_b, nvir_b)
      t2bb:   (nocc_b, nvir_b, nocc_b, nvir_b) antisymmetrized

    The spin pairs (nocc, nvir, norb) are returned as (alpha, beta). norb_a and norb_b may
    differ, as they may in HamCholU.
    """

    mo_t_a: jax.Array
    mo_t_b: jax.Array
    t2aa: jax.Array
    t2ab: jax.Array
    t2bb: jax.Array

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
        children = (self.mo_t_a, self.mo_t_b, self.t2aa, self.t2ab, self.t2bb)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t_a, mo_t_b, t2aa, t2ab, t2bb = children
        return cls(mo_t_a=mo_t_a, mo_t_b=mo_t_b, t2aa=t2aa, t2ab=t2ab, t2bb=t2bb)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: Upt2ccsdTrial) -> jax.Array:
    # <exp(T1)HF|walker>
    wu, wd = walker
    return jnp.linalg.det(trial_data.mo_t_a.T.conj() @ wu) * jnp.linalg.det(
        trial_data.mo_t_b.T.conj() @ wd
    )


def make_upt2ccsd_trial_data(data: dict, sys: System_uh) -> Upt2ccsdTrial:
    return Upt2ccsdTrial(
        mo_t_a=jnp.asarray(data["mo_t_a"])[:, : sys.nup],
        mo_t_b=jnp.asarray(data["mo_t_b"])[:, : sys.ndn],
        t2aa=jnp.asarray(data["t2aa"]),
        t2ab=jnp.asarray(data["t2ab"]),
        t2bb=jnp.asarray(data["t2bb"]),
    )
