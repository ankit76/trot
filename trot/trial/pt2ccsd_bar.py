"""
The pt2CCSD trial with exp(T1) applied to the right (pt2ccsd_bar).

The trial is the same Pt2ccsdTrial as trial/pt2ccsd.py: the Thouless reference
mo_t = exp(T1)|HF> and the doubles t2. What the bar variant adds is the similarity
transform that moves exp(T1) off the bra and onto the hamiltonian and the walker,

    <exp(T1)HF| H |phi>  =  <HF| exp(T1) H exp(-T1) exp(T1) |phi>
                         =  <HF| H_bar |exp(T1) phi>,

    H_bar:   h1_bar = exp(T1) h1 exp(-T1),   L_bar_g = exp(T1) L_g exp(-T1)
    walker:  walker_bar = exp(T1) walker

so the bra in the energy kernel is the bare reference determinant, the identity on the
leading nocc orbitals. Against it the greens function has nonzero rows only in its
first nocc rows, which is what makes the bar kernel faster and lighter per chunk (see
meas/pt2ccsd_bar.py).

The T1 generator has only an occupied-virtual block, so its matrix X squares to zero
and exp(X) = 1 + X exactly, with inverse 1 - X: no matrix exponential is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ham.chol import HamChol
from .pt2ccsd import Pt2ccsdTrial, make_pt2ccsd_trial_data, overlap_r

__all__ = [
    "Pt2ccsdTrial",
    "make_pt2ccsd_trial_data",
    "overlap_r",
    "t1_from_mo_t",
    "bar_transforms",
    "build_bar_intermediates",
]


def t1_from_mo_t(mo_t: jax.Array, nocc: int) -> jax.Array:
    """
    Recover the singles amplitudes from the Thouless reference.

    Staging builds mo_t as exp_t1[:nocc].T, so its occupied block is the identity and its
    virtual block is t1.T. Dividing the gauge out anyway keeps this correct for any
    equivalent mo_t, since the orbitals of a determinant are fixed only up to right
    multiplication by an nocc x nocc matrix.
    """
    return (mo_t[nocc:, :] @ jnp.linalg.inv(mo_t[:nocc, :])).T  # (nocc, nvir)


def bar_transforms(mo_t: jax.Array, nocc: int) -> tuple[jax.Array, jax.Array]:
    """(exp_t1, exp_mt1) = (1 + X, 1 - X) with X[:nocc, nocc:] = t1."""
    norb = int(mo_t.shape[0])
    t1 = t1_from_mo_t(mo_t, nocc)
    x = jnp.zeros((norb, norb), dtype=t1.dtype).at[:nocc, nocc:].set(t1)
    eye = jnp.eye(norb, dtype=t1.dtype)
    return eye + x, eye - x


def build_bar_intermediates(ham_data: HamChol, trial_data: Pt2ccsdTrial) -> dict:
    """
    The similarity transformed hamiltonian: exp_t1, exp_mt1, h1_bar (norb, norb) and
    chol_bar (nchol, norb, norb). The overlap det(mo_t^T walker) equals
    det((exp_t1 walker)[:nocc]), so trial.overlap_r needs no change. What it costs is
    chol_bar, a second copy of the cholesky tensor.
    """
    exp_t1, exp_mt1 = bar_transforms(trial_data.mo_t, trial_data.nocc)
    h1_bar = exp_t1 @ ham_data.h1 @ exp_mt1
    chol_bar = jnp.einsum("pr,grs,sq->gpq", exp_t1, ham_data.chol, exp_mt1, optimize="optimal")
    return {"exp_t1": exp_t1, "exp_mt1": exp_mt1, "h1_bar": h1_bar, "chol_bar": chol_bar}
