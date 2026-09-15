from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

WalkerKind = Literal["restricted", "unrestricted", "generalized"]


@dataclass(frozen=True)
class System:
    """
    Static system configuration
      - norb: number of spatial orbitals
      - nelec: (n_up, n_dn)
      - walker_kind: how walkers are represented
    """

    norb: int
    nelec: Tuple[int, int]
    walker_kind: WalkerKind

    def __post_init__(self):
        object.__setattr__(self, "walker_kind", self.walker_kind.lower())

    @property
    def nup(self) -> int:
        return self.nelec[0]

    @property
    def ndn(self) -> int:
        return self.nelec[1]

    @property
    def ne(self) -> int:
        return self.nelec[0] + self.nelec[1]


@dataclass(frozen=True)
class System_uh:
    """
    Static system configuration for an unrestricted (uchol) hamiltonian.

      - norb: (norb_a, norb_b), the alpha and beta orbital space sizes
      - nelec: (n_up, n_dn)
      - walker_kind: must be "unrestricted"

    The difference from System is that norb is a pair rather than a single int, since
    unrestricted LNO chooses the alpha and beta local active spaces independently and
    the two sizes need not agree. A single int is accepted and used for both spins.

    Pairs with HamCholU, whose .norb property is the same (norb_a, norb_b) tuple:

        sys = System_uh(norb=ham.norb, nelec=(nup, ndn))
    """

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    walker_kind: WalkerKind = "unrestricted"

    def __post_init__(self):
        object.__setattr__(self, "walker_kind", self.walker_kind.lower())
        if self.walker_kind != "unrestricted":
            raise ValueError(
                f"System_uh requires walker_kind='unrestricted', got {self.walker_kind!r}"
            )

        norb = self.norb
        if isinstance(norb, int):
            norb = (norb, norb)
        else:
            norb = tuple(int(n) for n in norb)
            if len(norb) != 2:
                raise ValueError(f"norb must be (norb_a, norb_b), got {self.norb!r}")
        object.__setattr__(self, "norb", norb)

        object.__setattr__(self, "nelec", tuple(int(n) for n in self.nelec))

        for name, nocc, n in (("alpha", self.nup, norb[0]), ("beta", self.ndn, norb[1])):
            if nocc < 0 or n <= 0:
                raise ValueError(f"{name}: need norb > 0 and nelec >= 0, got {n} and {nocc}")
            if nocc > n:
                raise ValueError(f"{name} has {nocc} electrons but only {n} orbitals")

    @property
    def norb_a(self) -> int:
        return self.norb[0]

    @property
    def norb_b(self) -> int:
        return self.norb[1]

    @property
    def nup(self) -> int:
        return self.nelec[0]

    @property
    def ndn(self) -> int:
        return self.nelec[1]

    @property
    def ne(self) -> int:
        return self.nelec[0] + self.nelec[1]
