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
    Static system configuration for the unrestricted (uchol) hamiltonian, where the two
    spins live in different orbital spaces.
      - norb: (norb_a, norb_b), the number of orbitals per spin (an int is broadcast)
      - nelec: (n_up, n_dn)
      - walker_kind: only "unrestricted" walkers can represent two orbital spaces

    Built from a HamCholU as System_uh(norb=ham.norb, nelec=ham.nelec). Consumers that
    only need nup/ndn/ne can treat it like System; anything indexing orbitals must use
    norb_a / norb_b.
    """

    norb: Tuple[int, int]
    nelec: Tuple[int, int]
    walker_kind: WalkerKind = "unrestricted"

    def __post_init__(self):
        wk = self.walker_kind.lower()
        if wk != "unrestricted":
            raise ValueError(
                "the unrestricted hamiltonian supports walker_kind='unrestricted' only "
                f"(alpha and beta live in different orbital spaces), got {self.walker_kind!r}"
            )
        object.__setattr__(self, "walker_kind", wk)

        norb = self.norb
        if isinstance(norb, int):
            norb = (norb, norb)
        try:
            norb_a, norb_b = (int(n) for n in norb)
        except (TypeError, ValueError):
            raise ValueError(f"norb must be (norb_a, norb_b), got {self.norb!r}") from None
        norb = (norb_a, norb_b)
        object.__setattr__(self, "norb", norb)

        nelec = (int(self.nelec[0]), int(self.nelec[1]))
        object.__setattr__(self, "nelec", nelec)
        if not (0 <= nelec[0] <= norb[0] and 0 <= nelec[1] <= norb[1]):
            raise ValueError(f"nelec={nelec} does not fit norb={norb}")

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
