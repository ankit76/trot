"""
The unrestricted pt2CCSD trial with exp(T1) applied to the right (upt2ccsd_bar), on the
unrestricted (uchol) hamiltonian.

The trial is the Upt2ccsdTrial of trial/upt2ccsd_uh.py. As for the restricted bar
variant (trial/pt2ccsd_bar.py), each spin's exp(T1_s) moves off the bra and onto that
spin's hamiltonian and walker,

    h1_bar_s   = exp(T1_s) h1_s   exp(-T1_s)
    L_bar_s,g  = exp(T1_s) L_s,g  exp(-T1_s)      (shared index g)
    walker_bar_s = exp(T1_s) walker_s

so the bra is the bare reference determinant of each spin. The two transforms never mix,
since alpha and beta have separate orbital spaces.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..ham.chol_u import HamCholU
from .pt2ccsd_bar import bar_transforms, t1_from_mo_t
from .upt2ccsd_uh import Upt2ccsdTrial, make_upt2ccsd_trial_data, overlap_u

__all__ = [
    "Upt2ccsdTrial",
    "make_upt2ccsd_trial_data",
    "overlap_u",
    "t1_from_mo_t",
    "build_bar_intermediates_u",
]


def build_bar_intermediates_u(ham_data: HamCholU, trial_data: Upt2ccsdTrial) -> dict:
    """
    The similarity transformed hamiltonian per spin: exp_t1_s, exp_mt1_s, h1_bar_s and
    chol_bar_s (nchol, norb_s, norb_s). chol_bar_a and chol_bar_b are a second copy of the
    cholesky tensor per spin.
    """
    nocc_a, nocc_b = trial_data.nocc
    out = {}
    for s, mo_t, nocc, h1, chol in (
        ("a", trial_data.mo_t_a, nocc_a, ham_data.h1_a, ham_data.chol_a),
        ("b", trial_data.mo_t_b, nocc_b, ham_data.h1_b, ham_data.chol_b),
    ):
        exp_t1, exp_mt1 = bar_transforms(mo_t, nocc)
        out[f"exp_t1_{s}"] = exp_t1
        out[f"exp_mt1_{s}"] = exp_mt1
        out[f"h1_bar_{s}"] = exp_t1 @ h1 @ exp_mt1
        out[f"chol_bar_{s}"] = jnp.einsum(
            "pr,grs,sq->gpq", exp_t1, chol, exp_mt1, optimize="optimal"
        )
    return out
