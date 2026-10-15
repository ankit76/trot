from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ...core.system import System
from ...trial.pt2ccsd import overlap_r

__all__ = ["Pt2ccsdTrial", "overlap_r", "make_pt2ccsd_trial_data"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdTrial:
    """
    Restricted pt2CCSD trial of one LNO fragment, in the LNO basis where the reference
    determinant occupies the first nocc orbitals. trot's trial/pt2ccsd.Pt2ccsdTrial plus
    the fragment projector.

    Arrays:
      mo_t:  (norb, nocc)              exp(T1)|HF> by Thouless' theorem
      t2:    (nocc, nvir, nocc, nvir)  t2_{k a j b} = sum_i t2_{i a j b} prjlo_{i k}: the
                                       doubles projected on the fragment in their first
                                       occupied index, so no longer symmetric in (ia)(jb)
      prjlo: (nocc, nocc)              U U^T, U = <act_occ|lo> of the fragment
      t1:    (nocc, nvir)              the singles, unprojected (for e0t1orb)
    """

    mo_t: jax.Array
    t2: jax.Array
    prjlo: jax.Array
    t1: jax.Array

    @property
    def nocc(self) -> int:
        return int(self.t2.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.t2.shape[1])

    @property
    def norb(self) -> int:
        return int(self.nocc + self.nvir)

    def tree_flatten(self):
        return (self.mo_t, self.t2, self.prjlo, self.t1), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t, t2, prjlo, t1 = children
        return cls(mo_t=mo_t, t2=t2, prjlo=prjlo, t1=t1)


def make_pt2ccsd_trial_data(data: dict, sys: System) -> Pt2ccsdTrial:
    mo_t = jnp.asarray(data["mo_t"])[:, : sys.nup]
    return Pt2ccsdTrial(
        mo_t=mo_t,
        t2=jnp.asarray(data["t2"]),
        prjlo=jnp.asarray(data["prjlo"]),
        t1=jnp.asarray(data["t1"]),
    )
