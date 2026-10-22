from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ...core.system import System
from ...trial.pt2ccsd import overlap_r

__all__ = ["Pt2ccsdFastTrial", "overlap_r", "make_pt2ccsd_fast_trial_data"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdFastTrial:
    """
    Restricted pt2CCSD trial of one LNO fragment with the fragment projector kept in
    factored form (the "pt2ccsd_fast" trial, meas/pt2ccsd_fast.py).

    The projector of trial/pt2ccsd.py is prjlo = U U^H with U = <act_occ|lo> the
    (nocc, nlo) overlap of the active occupied orbitals with the fragment's orthonormal
    local orbitals (U itself need not have orthonormal columns: the active occupied
    space of a truncated fragment does not contain the local orbitals entirely), and the
    doubles are stored there projected on their first occupied index,
    t2_{k a j b} = sum_i t2_{i a j b} prjlo_{i k}. Here the doubles carry only the
    local index, t2u_{I a j b} = sum_i t2_{i a j b} U_{i I}, and the second factor
    U^H_{I k} is applied inside the energy kernel, where the local index I is contracted
    last. nlo is the number of local orbitals of the fragment (the IAOs on one atom,
    say), typically much smaller than nocc, so the T2 contractions and the projected
    two-body terms cost nlo/nocc of the pt2ccsd ones.

    Every direct/exchange pair of T2 terms in the kernel is of the form
    2 (X_{Ia} t2u_{Iajb} Y_{jb}) - (X_{Ib} t2u_{Iajb} Y_{ja}), which is one contraction
    with the exchange folded into the amplitudes,

        t2x_{I a j b} = 2 t2u_{I a j b} - t2u_{I b j a}      (nlo, nvir, nocc, nvir)

    so that is what the trial stores (t2u = (2 t2x + t2x_{Ibja}) / 3 if ever needed).

    Arrays:
      mo_t:  (norb, nocc)              exp(T1)|HF> by Thouless' theorem
      t2x:   (nlo, nvir, nocc, nvir)   the projected doubles, exchange folded in
      u:     (nocc, nlo)               U = <act_occ|lo>
      t1:    (nocc, nvir)              the singles, unprojected (for e0t1orb)
    """

    mo_t: jax.Array
    t2x: jax.Array
    u: jax.Array
    t1: jax.Array

    @property
    def nocc(self) -> int:
        return int(self.u.shape[0])

    @property
    def nlo(self) -> int:
        return int(self.u.shape[1])

    @property
    def nvir(self) -> int:
        return int(self.t2x.shape[1])

    @property
    def norb(self) -> int:
        return int(self.nocc + self.nvir)

    @property
    def prjlo(self) -> jax.Array:
        """The (nocc, nocc) fragment projector U U^H, for checks against trial/pt2ccsd.py."""
        return self.u @ self.u.conj().T

    def tree_flatten(self):
        return (self.mo_t, self.t2x, self.u, self.t1), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t, t2x, u, t1 = children
        return cls(mo_t=mo_t, t2x=t2x, u=u, t1=t1)


def make_pt2ccsd_fast_trial_data(data: dict, sys: System) -> Pt2ccsdFastTrial:
    mo_t = jnp.asarray(data["mo_t"])[:, : sys.nup]
    td = Pt2ccsdFastTrial(
        mo_t=mo_t,
        t2x=jnp.asarray(data["t2x"]),
        u=jnp.asarray(data["u"]),
        t1=jnp.asarray(data["t1"]),
    )
    if td.nocc != sys.nup or td.norb != sys.norb:
        raise ValueError(
            f"the fragment amplitudes span norb={td.norb} with nocc={td.nocc} but the "
            f"fragment hamiltonian has norb={sys.norb} and nocc={sys.nup}."
        )
    return td
