"""
The unrestricted pt2CCSD trial on the unrestricted (uchol) hamiltonian.

Each spin lives in its own orbital basis (HamCholU), so the trial carries a Thouless
reference exp(T1_s)|HF_s> per spin, each in that spin's basis, and the three doubles
blocks. Alpha and beta may have different orbital counts. The reference overlap the
estimator is normalised by is the bare determinant product

    <exp(T1)HF | phi> = det(mo_t_a^T wa) det(mo_t_b^T wb),

which is what the mixed driver's reweighting wp = w <T|phi>/<G|phi> uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdTrial:
    """
    Arrays:
      mo_t_a: (norb_a, nocc_a)   exp(T1_a)|HF_a> in the alpha basis
      mo_t_b: (norb_b, nocc_b)   exp(T1_b)|HF_b> in the beta basis
      t2aa:   (nocc_a, nvir_a, nocc_a, nvir_a)   (i, a, j, b), antisymmetrized
      t2ab:   (nocc_a, nvir_a, nocc_b, nvir_b)
      t2bb:   (nocc_b, nvir_b, nocc_b, nvir_b)   antisymmetrized
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
        return (self.mo_t_a, self.mo_t_b, self.t2aa, self.t2ab, self.t2bb), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t_a, mo_t_b, t2aa, t2ab, t2bb = children
        return cls(mo_t_a=mo_t_a, mo_t_b=mo_t_b, t2aa=t2aa, t2ab=t2ab, t2bb=t2bb)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: Upt2ccsdTrial) -> jax.Array:
    """<exp(T1)HF|phi>, the bare determinant product."""
    wa, wb = walker
    return jnp.linalg.det(trial_data.mo_t_a.T.conj() @ wa) * jnp.linalg.det(
        trial_data.mo_t_b.T.conj() @ wb
    )


def make_upt2ccsd_trial_data(data: dict, sys: Any) -> Upt2ccsdTrial:
    """From staging_u.stage_upt2ccsd_trial_uh's data; sys is a System_uh."""
    norb_a, norb_b = sys.norb if isinstance(sys.norb, tuple) else (sys.norb, sys.norb)
    td = Upt2ccsdTrial(
        mo_t_a=jnp.asarray(data["mo_t_a"])[:, : sys.nup],
        mo_t_b=jnp.asarray(data["mo_t_b"])[:, : sys.ndn],
        t2aa=jnp.asarray(data["t2aa"]),
        t2ab=jnp.asarray(data["t2ab"]),
        t2bb=jnp.asarray(data["t2bb"]),
    )
    if td.nocc != (sys.nup, sys.ndn) or td.norb != (norb_a, norb_b):
        raise ValueError(
            f"the UCCSD amplitudes span norb={td.norb} with nocc={td.nocc} but the unrestricted "
            f"hamiltonian has norb={(norb_a, norb_b)} and nelec={(sys.nup, sys.ndn)}."
        )
    return td
