"""
UHF trial on the unrestricted (uchol) hamiltonian.

The UhfTrial pytree of trial/uhf.py already keeps the two spins separate, so it is
reused; what changes is where the coefficients come from and how the rdm1 is returned.
In the uchol layout each spin's basis has that spin's occupied orbitals in its leading
columns, so the trial determinant is the leading nocc_s columns of the identity, and
the rdm1 must be a pair of blocks since norb_a may differ from norb_b.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp

from ..core.ops import Rdm1Fn, TrialOps
from .uhf import UhfTrial, overlap_u


def get_rdm1_uh(trial_data: UhfTrial) -> tuple[jax.Array, jax.Array]:
    """
    Trial rdm1 as a pair of spin blocks (norb_a, norb_a), (norb_b, norb_b).

    trial.uhf.get_rdm1 stacks them into (2, norb, norb), which cannot hold
    norb_a != norb_b, so the unrestricted path needs the pair form.
    """
    c_a = trial_data.mo_coeff_a
    c_b = trial_data.mo_coeff_b
    return (c_a @ c_a.conj().T, c_b @ c_b.conj().T)


def make_uhf_trial_ops_uh(sys: Any) -> TrialOps:
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError(
            "the unrestricted hamiltonian path requires walker_kind='unrestricted', "
            f"got {sys.walker_kind!r}"
        )
    # Rdm1Fn is typed as returning a single array; the unrestricted path returns the
    # two spin blocks as a pair, which cannot be stacked when norb_a != norb_b
    return TrialOps(overlap=overlap_u, get_rdm1=cast(Rdm1Fn, get_rdm1_uh))


def make_uhf_trial_data_uh(data: dict | None, sys: Any) -> UhfTrial:
    """
    Trial in each spin's own orbital basis.

    With data == None (or without mo_a/mo_b) the determinant is the leading nocc columns
    of the identity of each spin, which is what staging_u produces: the uchol
    hamiltonian is expressed in bases whose leading columns are the occupied orbitals.
    Coefficients given in data are taken as is, in those same per spin bases.
    """
    norb_a, norb_b = sys.norb if isinstance(sys.norb, tuple) else (sys.norb, sys.norb)
    if data is not None and "mo_a" in data and "mo_b" in data:
        mo_a = jnp.asarray(data["mo_a"])
        mo_b = jnp.asarray(data["mo_b"])
    else:
        mo_a = jnp.eye(norb_a)
        mo_b = jnp.eye(norb_b)
    if mo_a.shape[0] != norb_a or mo_b.shape[0] != norb_b:
        raise ValueError(
            f"trial coefficients span ({mo_a.shape[0]}, {mo_b.shape[0]}) orbitals but the "
            f"hamiltonian has norb={(norb_a, norb_b)}"
        )
    return UhfTrial(mo_a[:, : sys.nup], mo_b[:, : sys.ndn])
