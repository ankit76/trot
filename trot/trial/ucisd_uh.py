"""
UCISD trial on the unrestricted (uchol) hamiltonian.

Alpha and beta each live in their own orbital basis and the reference determinant of
each spin occupies the leading nocc orbitals of that basis (staging_u's convention).
The CI coefficients staged by staging._stage_ucisd_input are expressed exactly there:
they are the UCCSD amplitudes in the UHF MO basis of each spin. So the UcisdTrial
pytree of trial/ucisd.py is reused with mo_coeff_s the identity of that spin's space;
the rotation of the beta reference into the alpha basis that the restricted-hamiltonian
path carries in mo_coeff_b is not needed, and the kernels in meas.ucisd_uh never rotate.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp

from ..core.ops import Rdm1Fn, TrialOps
from .ucisd import UcisdTrial


def make_ucisd_trial_data_uh(data: dict, sys: Any) -> UcisdTrial:
    norb_a, norb_b = sys.norb if isinstance(sys.norb, tuple) else (sys.norb, sys.norb)
    c1a = jnp.asarray(data["ci1a"])
    c1b = jnp.asarray(data["ci1b"])
    nocc_a, nvir_a = c1a.shape
    nocc_b, nvir_b = c1b.shape
    if (nocc_a, nocc_b) != (sys.nup, sys.ndn) or (nocc_a + nvir_a, nocc_b + nvir_b) != (
        norb_a,
        norb_b,
    ):
        raise ValueError(
            f"the UCISD amplitudes span ({nocc_a}+{nvir_a}, {nocc_b}+{nvir_b}) orbitals with "
            f"({nocc_a}, {nocc_b}) electrons but the unrestricted hamiltonian has "
            f"norb={(norb_a, norb_b)} and nelec={(sys.nup, sys.ndn)}."
        )
    # square identities: UcisdTrial.norb reads mo_coeff_b.shape[0], and the restricted
    # path's overlap treats mo_coeff_b as a full rotation, which the identity satisfies
    return UcisdTrial(
        mo_coeff_a=jnp.eye(norb_a),
        mo_coeff_b=jnp.eye(norb_b),
        c1a=c1a,
        c1b=c1b,
        c2aa=jnp.asarray(data["ci2aa"]),
        c2ab=jnp.asarray(data["ci2ab"]),
        c2bb=jnp.asarray(data["ci2bb"]),
    )


def get_rdm1_uh(trial_data: UcisdTrial) -> tuple[jax.Array, jax.Array]:
    """
    The reference determinant's rdm1 as a pair of spin blocks, as the UHF path returns
    it: it seeds the mean-field shift and the initial walkers. trial.ucisd.get_rdm1
    stacks the two blocks, which cannot hold norb_a != norb_b.
    """
    n_oa, n_ob = trial_data.nocc
    c_a = trial_data.mo_coeff_a[:, :n_oa]
    c_b = trial_data.mo_coeff_b[:, :n_ob]
    return (c_a @ c_a.conj().T, c_b @ c_b.conj().T)


def make_ucisd_trial_ops_uh(sys: Any) -> TrialOps:
    from ..meas.ucisd_uh import overlap_uw_uh

    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError(
            "the unrestricted hamiltonian path requires walker_kind='unrestricted', "
            f"got {sys.walker_kind!r}"
        )
    # Rdm1Fn is typed as returning a single array; the pair form cannot be stacked when
    # norb_a != norb_b
    return TrialOps(overlap=overlap_uw_uh, get_rdm1=cast(Rdm1Fn, get_rdm1_uh))
