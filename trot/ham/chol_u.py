from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
from jax import tree_util

HamBasisU = Literal["uchol"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamCholU:
    """
    Unrestricted cholesky hamiltonian.

    basis="uchol":
      h1_a:   (norb_a, norb_a)
      h1_b:   (norb_b, norb_b)
      chol_a: (n_fields, norb_a, norb_a)
      chol_b: (n_fields, norb_b, norb_b)

    The auxiliary field index is SHARED between spins, the orbital indices are not:
    chol_a and chol_b must agree on n_fields, but norb_a and norb_b may differ. This is
    what unrestricted LNO produces, where the alpha and beta local active spaces are
    chosen independently from a common set of cholesky vectors,

        L^sigma_g = C_sigma.T @ L_g @ C_sigma,

    so the same g indexes both spins while the orbital dimensions are unrelated.

    Contrast with HamChol, whose single chol/h1 forces norb_a == norb_b.
    """

    h0: jax.Array
    h1_a: jax.Array
    h1_b: jax.Array
    chol_a: jax.Array
    chol_b: jax.Array
    basis: HamBasisU = "uchol"
    nchol: int | None = None

    def __post_init__(self):
        if self.basis != "uchol":
            raise ValueError(f"unknown basis: {self.basis}")

        shape_a = getattr(self.chol_a, "shape", None)
        shape_b = getattr(self.chol_b, "shape", None)
        if shape_a is None or shape_b is None:
            return

        if len(shape_a) != 3 or len(shape_b) != 3:
            raise ValueError(
                f"chol_a and chol_b must be 3d (n_fields, norb, norb), "
                f"got chol_a.shape={shape_a} and chol_b.shape={shape_b}"
            )

        n_chol_a = int(shape_a[0])
        n_chol_b = int(shape_b[0])
        if n_chol_a != n_chol_b:
            raise ValueError(
                f"alpha and beta must share the auxiliary field index: "
                f"chol_a.shape[0]={n_chol_a} is inconsistent with chol_b.shape[0]={n_chol_b}"
            )

        if shape_a[1] != shape_a[2] or shape_b[1] != shape_b[2]:
            raise ValueError(
                f"each cholesky vector must be square, got chol_a.shape={shape_a} "
                f"and chol_b.shape={shape_b}"
            )

        # a compacted chol (see runtime_layout) is a zero sized placeholder, in which case
        # the orbital dimensions live only in h1 and there is nothing to cross check.
        if n_chol_a > 0 and shape_a[1] > 0:
            for name, h1, shape in (("a", self.h1_a, shape_a), ("b", self.h1_b, shape_b)):
                h1_shape = getattr(h1, "shape", None)
                if h1_shape is not None and tuple(h1_shape) != (shape[1], shape[2]):
                    raise ValueError(
                        f"h1_{name}.shape={tuple(h1_shape)} is inconsistent with "
                        f"chol_{name}.shape={shape}"
                    )

        nchol = self.nchol
        if nchol is None:
            object.__setattr__(self, "nchol", n_chol_a)
        elif n_chol_a not in (0, int(nchol)):
            raise ValueError(f"nchol={nchol} is inconsistent with chol_a.shape[0]={n_chol_a}")

    @property
    def norb_a(self) -> int:
        return int(self.h1_a.shape[0])

    @property
    def norb_b(self) -> int:
        return int(self.h1_b.shape[0])

    @property
    def norb(self) -> tuple[int, int]:
        return (self.norb_a, self.norb_b)

    def tree_flatten(self):
        children = (self.h0, self.h1_a, self.h1_b, self.chol_a, self.chol_b)
        nchol = self.nchol
        assert nchol is not None
        aux = (self.basis, int(nchol))
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1_a, h1_b, chol_a, chol_b = children
        basis, nchol = aux
        return cls(
            h0=h0,
            h1_a=h1_a,
            h1_b=h1_b,
            chol_a=chol_a,
            chol_b=chol_b,
            basis=basis,
            nchol=nchol,
        )


def n_fields(ham: HamCholU) -> int:
    nchol = ham.nchol
    assert nchol is not None
    return int(nchol)


def from_ham_chol(ham, *, h1_a=None, h1_b=None) -> HamCholU:
    """
    Build a HamCholU from a spin free HamChol by duplicating chol (and h1) across spins.

    Mainly for testing: the resulting propagator must reproduce the existing
    ("restricted", "unrestricted") path exactly.
    """
    if ham.basis != "restricted":
        raise ValueError(f"expected HamChol.basis == 'restricted', got {ham.basis!r}")
    return HamCholU(
        h0=ham.h0,
        h1_a=ham.h1 if h1_a is None else h1_a,
        h1_b=ham.h1 if h1_b is None else h1_b,
        chol_a=ham.chol,
        chol_b=ham.chol,
        nchol=ham.nchol,
    )
