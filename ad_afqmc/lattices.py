from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import numpy as np
from jax import jit, lax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


class lattice(ABC):
    """Abstract base class for lattice objects."""

    @abstractmethod
    def get_site_num(self, pos: Sequence) -> int:
        """Return the site number corresponding to a given position."""
        pass

    @abstractmethod
    def get_nearest_neighbors(self, pos: Sequence) -> Sequence:
        """Return the nearest neighbors of a site."""
        pass

    @abstractmethod
    def __hash__(self):
        pass


@dataclass
@register_pytree_node_class
class one_dimensional_chain(lattice):
    n_sites: int
    shape: Optional[tuple] = None
    sites: Optional[Sequence] = None
    bonds: Optional[Sequence] = None
    hop_signs: Sequence = (1.0, -1.0)
    coord_num: int = 2

    def __post_init__(self):
        self.shape = (self.n_sites,)
        self.sites = tuple([(i,) for i in range(self.n_sites)])
        self.bonds = (
            tuple(
                [
                    (
                        0,
                        i,
                    )
                    for i in range(self.n_sites)
                ]
            )
            if self.n_sites > 2
            else tuple([(0, 0)])
        )

    def get_bond_mode_distance(self, bond, mode):
        neigboring_sites = self.get_neighboring_sites(bond)
        dist_1 = self.get_distance(neigboring_sites[0], mode[1:])
        dist_2 = self.get_distance(neigboring_sites[1], mode[1:])
        lr = (dist_1 < dist_2) * 1.0 - (dist_1 > dist_2) * 1.0
        return jnp.min(jnp.array([dist_1, dist_2])), lr

    def get_site_num(self, pos):
        return pos[0]

    def get_marshall_sign(self, walker):
        if isinstance(walker, list):
            # TODO: this is a bit hacky
            walker = walker[0]
        walker_a = walker[::2]
        return (-1) ** jnp.sum(jnp.where(walker_a > 0, 1, 0))

    def get_symm_fac(self, pos, k):
        return (
            jnp.exp(2 * jnp.pi * 1.0j * k[0] * pos[0] / self.n_sites)
            if k is not None
            else 1.0
        )

    @partial(jit, static_argnums=(0,))
    def get_neighboring_bonds(self, pos):
        return jnp.array(
            [
                (
                    0,
                    (pos[0] - 1) % self.n_sites,
                ),
                (
                    0,
                    pos[0],
                ),
            ]
        )

    # ordering is used in the ssh model
    @partial(jit, static_argnums=(0,))
    def get_nearest_neighbors(self, pos):
        return jnp.array(
            [((pos[0] - 1) % self.n_sites,), ((pos[0] + 1) % self.n_sites,)]
        )

    def get_nearest_neighbor_modes(self, pos):
        return (
            jnp.array(
                [
                    (
                        0,
                        (pos[0] - 1) % self.n_sites,
                    ),
                    (
                        0,
                        (pos[0] + 1) % self.n_sites,
                    ),
                ]
            )
            if self.n_sites > 2
            else jnp.array(
                [
                    (
                        0,
                        1 - pos[0],
                    )
                ]
            )
        )

    @partial(jit, static_argnums=(0,))
    def get_neighboring_sites(self, bond):
        return [(bond[1] % self.n_sites,), ((bond[1] + 1) % self.n_sites,)]

    def get_neighboring_modes(self, bond):
        return [
            (
                bond[0],
                bond[1] % self.n_sites,
            ),
            (
                bond[0],
                (bond[1] + 1) % self.n_sites,
            ),
        ]

    def get_distance(self, pos_1, pos_2):
        return jnp.min(
            jnp.array(
                [
                    jnp.abs(pos_1[0] - pos_2[0]),
                    self.n_sites - jnp.abs(pos_1[0] - pos_2[0]),
                ]
            )
        )

    def get_bond_distance(self, pos_1, pos_2):
        return jnp.min(
            jnp.array(
                [
                    jnp.abs(pos_1[1] - pos_2[1]),
                    self.n_sites - jnp.abs(pos_1[1] - pos_2[1]),
                ]
            )
        )

    def create_adjacency_matrix(self):
        n_sites = self.n_sites
        h = np.zeros((n_sites, n_sites), dtype=int)

        for r in range(n_sites):
            neighbors = self.get_nearest_neighbors((r,))
            for nr in neighbors:
                h[r, nr] = 1
                h[nr, r] = 1
        return h

    def __hash__(self):
        return hash((self.n_sites, self.shape, self.sites, self.bonds))

    def tree_flatten(self):
        return (), (self.n_sites, self.shape, self.sites, self.bonds, self.coord_num)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@dataclass
@register_pytree_node_class
class two_dimensional_grid(lattice):
    l_x: int
    l_y: int
    shape: Optional[tuple] = None
    shell_distances: Optional[Sequence] = None
    bond_shell_distances: Optional[Sequence] = None
    sites: Optional[Sequence] = None
    bonds: Optional[Sequence] = None
    n_sites: Optional[int] = None
    hop_signs: Sequence = (-1.0, -1.0, 1.0, 1.0)
    coord_num: int = 4
    boundary: str = "pbc"

    def __post_init__(self):
        self.shape = (self.l_y, self.l_x)
        self.n_sites = self.l_x * self.l_y
        distances = []
        for x in range(self.l_x // 2 + 1):
            for y in range(self.l_y // 2 + 1):
                dist = x**2 + y**2
                distances.append(dist)
        distances = [*set(distances)]
        distances.sort()
        self.shell_distances = tuple(distances)
        self.sites = tuple(
            [(i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
        )

        bond_distances = []
        for x in range(self.l_x + 1):
            for y in range(self.l_y + 1):
                if x % 2 == y % 2:
                    dist = x**2 + y**2
                    bond_distances.append(dist)
        bond_distances = [*set(bond_distances)]
        bond_distances.sort()
        self.bond_shell_distances = tuple(bond_distances)
        # if (self.l_x == 2) & (self.l_y == 2):
        #     self.bonds = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0))
        # elif self.l_x == 2:
        #     self.bonds = tuple(
        #         [(0, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
        #         + [(1, i, 0) for i in range(self.l_y)]
        #     )
        # elif self.l_y == 2:
        #     self.bonds = tuple(
        #         [(0, 0, i) for i in self.l_x]
        #         + [(1, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
        #     )
        # else:
        self.bonds = tuple(
            [(0, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
            + [(1, i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y)]
        )

    def get_site_num(self, pos):
        return pos[1] + self.l_x * pos[0]

    def get_marshall_sign(self, walker):
        if isinstance(walker, list):
            # TODO: this is a bit hacky
            walker = walker[0]
        walker_a = walker[::2, ::2]
        return (-1) ** jnp.sum(jnp.where(walker_a > 0, 1, 0))

    def get_symm_fac(self, pos, k):
        return (
            jnp.exp(2 * jnp.pi * 1.0j * k[0] * pos[0] / self.l_x)
            * jnp.exp(2 * jnp.pi * 1.0j * k[1] * pos[1] / self.l_y)
            if k is not None
            else 1.0
        )

    @partial(jit, static_argnums=(0,))
    def get_distance(self, pos_1, pos_2):
        dist_y = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[0] - pos_2[0]), self.l_y - jnp.abs(pos_1[0] - pos_2[0])]
            )
        )
        dist_x = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[1] - pos_2[1]), self.l_x - jnp.abs(pos_1[1] - pos_2[1])]
            )
        )
        dist = dist_x**2 + dist_y**2
        shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
        return shell_number

    @partial(jit, static_argnums=(0,))
    def get_bond_distance(self, pos_1, pos_2):
        shifted_pos_1 = 2 * jnp.array(pos_1[1:])
        shifted_pos_1 = shifted_pos_1.at[pos_1[0]].add(1)
        shifted_pos_2 = 2 * jnp.array(pos_2[1:])
        shifted_pos_2 = shifted_pos_2.at[pos_2[0]].add(1)
        dist_y = jnp.min(
            jnp.array(
                [
                    jnp.abs(shifted_pos_1[0] - shifted_pos_2[0]),
                    2 * self.l_y - jnp.abs(shifted_pos_1[0] - shifted_pos_2[0]),
                ]
            )
        )
        dist_x = jnp.min(
            jnp.array(
                [
                    jnp.abs(shifted_pos_1[1] - shifted_pos_2[1]),
                    2 * self.l_x - jnp.abs(shifted_pos_1[1] - shifted_pos_2[1]),
                ]
            )
        )
        dist = dist_x**2 + dist_y**2
        shell_number = jnp.searchsorted(jnp.array(self.bond_shell_distances), dist)
        return shell_number

    def get_neighboring_bonds(self, pos):
        down = (0, *pos)
        right = (1, *pos)
        up = (0, (pos[0] - 1) % self.l_y, pos[1])
        left = (1, pos[0], (pos[1] - 1) % self.l_x)
        neighbors = []
        if (self.l_x == 2) & (self.l_y == 2):
            neighbors = [(0, 0, pos[1]), (1, pos[0], 0)]
        elif self.l_x == 2:
            neighbors = [right if pos[1] == 0 else left, down, up]
        elif self.l_y == 2:
            neighbors = [right, down if pos[0] == 0 else up, left]
        else:
            neighbors = [right, down, left, up]
        return jnp.array(neighbors)

    @partial(jit, static_argnums=(0,))
    def get_neighboring_sites(self, bond):
        # neighbors = [ ]
        # if bond[0] == 0:
        #  neighbors = [ (bond[1], bond[2]), ((bond[1] + 1) % self.l_y, bond[2]) ]
        # else:
        #  neighbors = [ (bond[1], bond[2]), (bond[1], (bond[2] + 1) % self.l_x) ]
        neighbors = lax.cond(
            bond[0] == 0,
            lambda x: [(bond[1], bond[2]), ((bond[1] + 1) % self.l_y, bond[2])],
            lambda x: [(bond[1], bond[2]), (bond[1], (bond[2] + 1) % self.l_x)],
            0,
        )
        return jnp.array(neighbors)

    @partial(jit, static_argnums=(0,))
    def get_neighboring_modes(self, bond):
        neighbors = lax.cond(
            bond[0] == 0,
            lambda x: [(0, bond[1], bond[2]), (0, (bond[1] + 1) % self.l_y, bond[2])],
            lambda x: [(1, bond[1], bond[2]), (1, bond[1], (bond[2] + 1) % self.l_x)],
            0,
        )
        return jnp.array(neighbors)

    @partial(jit, static_argnums=(0,))
    def get_nearest_neighbors(self, pos):
        if self.boundary == "pbc":
            right = (pos[0], (pos[1] + 1) % self.l_x)
            down = ((pos[0] + 1) % self.l_y, pos[1])
            left = (pos[0], (pos[1] - 1) % self.l_x)
            up = ((pos[0] - 1) % self.l_y, pos[1])
        else:  # obc
            right = (pos[0], pos[1] + 1)
            down = (pos[0] + 1, pos[1])
            left = (pos[0], pos[1] - 1)
            up = (pos[0] - 1, pos[1])

        return jnp.array([right, down, left, up])

    # used in the ssh model
    @partial(jit, static_argnums=(0,))
    def get_nearest_neighbor_modes(self, pos):
        right = (1, pos[0], (pos[1] + 1) % self.l_x)
        down = (0, (pos[0] + 1) % self.l_y, pos[1])
        left = (1, pos[0], (pos[1] - 1) % self.l_x)
        up = (0, (pos[0] - 1) % self.l_y, pos[1])
        neighbors = [right, down, left, up]
        return jnp.array(neighbors)

    # used in the ssh model
    @partial(jit, static_argnums=(0,))
    def get_bond_mode_distance(self, bond, mode):
        neighboring_sites = self.get_neighboring_sites(bond)
        dist_1 = self.get_distance(neighboring_sites[0], mode[1:])
        dist_2 = self.get_distance(neighboring_sites[1], mode[1:])
        # evaluate both parallel and perpendicular cases
        # parallel
        dist_bond_1 = jnp.min(
            jnp.array(
                [
                    jnp.abs(neighboring_sites[0][mode[0]] - mode[1:][mode[0]]),
                    jnp.array(self.shape)[mode[0]]
                    - jnp.abs(neighboring_sites[0][mode[0]] - mode[1:][mode[0]]),
                ]
            )
        )
        dist_bond_2 = jnp.min(
            jnp.array(
                [
                    jnp.abs(neighboring_sites[1][mode[0]] - mode[1:][mode[0]]),
                    jnp.array(self.shape)[mode[0]]
                    - jnp.abs(neighboring_sites[1][mode[0]] - mode[1:][mode[0]]),
                ]
            )
        )
        lr_bond = (dist_bond_1 < dist_bond_2) * 1.0 - (dist_bond_1 > dist_bond_2) * 1.0
        # perpendicular
        dist_site_1 = (
            neighboring_sites[1 - (dist_1 < dist_2)][mode[0]] - mode[1:][mode[0]]
        ) % jnp.array(self.shape)[mode[0]]
        dist_site_2 = (
            mode[1:][mode[0]] - neighboring_sites[1 - (dist_1 < dist_2)][mode[0]]
        ) % jnp.array(self.shape)[mode[0]]
        lr_site = (dist_site_1 < dist_site_2) * 1.0 - (dist_site_1 > dist_site_2) * 1.0
        lr = lr_bond * (bond[0] == mode[0]) + lr_site * (bond[0] != mode[0])
        return jnp.min(jnp.array([dist_1, dist_2])), lr

    def create_adjacency_matrix(self):
        width, height = self.l_x, self.l_y
        size = width * height
        h = np.zeros((size, size), dtype=int)

        for r in range(width):
            for q in range(height):
                i = q * width + r
                neighbors = self.get_nearest_neighbors((q, r))
                for nq, nr in neighbors:
                    if 0 <= nq < height and 0 <= nr < width:  # Check bounds
                        j = nq * width + nr
                        h[i, j] = 1
                        h[j, i] = 1
        return h

    def __hash__(self):
        return hash(
            (
                self.l_x,
                self.l_y,
                self.shape,
                self.shell_distances,
                self.bond_shell_distances,
                self.sites,
                self.bonds,
                self.coord_num,
            )
        )

    def tree_flatten(self):
        return (), (
            self.l_x,
            self.l_y,
            self.shape,
            self.shell_distances,
            self.bond_shell_distances,
            self.sites,
            self.bonds,
            self.coord_num,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@dataclass
@register_pytree_node_class
class triangular_grid(lattice):
    l_x: int  # height
    l_y: int  # width
    shape: tuple = ()
    sites: Sequence = ()
    n_sites: int = 0
    coord_num: int = 6
    boundary: str = "pbc"

    def __post_init__(self):
        self.shape = (self.l_x, self.l_y)
        self.n_sites = self.l_x * self.l_y
        self.sites = tuple(
            [(i // self.l_y, i % self.l_y) for i in range(self.l_x * self.l_y)]
        )

    def get_site_num(self, pos):
        return pos[1] + self.l_y * pos[0]

    def get_site_coordinate(self, pos):
        """
        Returns the real space coordinate of the site specified by `pos`,
        assuming a primitive lattice vector of unit length.
        """
        theta = np.pi / 3.0
        lattice_vecs = np.array([[np.cos(theta), 1.0], [np.sin(theta), 0.0]])

        if (self.boundary == "xc") or (self.boundary == "oxc"):
            L2, L1 = lattice_vecs.T
            L3 = L2 - L1
            Ly = [L2, L3]
            coords = np.zeros(2)

            for i in range(1, pos[0] + 1):
                coords += Ly[(i - 1) % 2]

            coords += pos[1] * L1

        elif self.boundary == "yc":
            theta = np.pi / 6.0
            lattice_vecs = np.array([[0.0, np.cos(theta)], [1.0, np.sin(theta)]])
            L1, L2 = lattice_vecs.T
            L3 = L2 - L1
            Lx = [L3, L2]
            coords = np.zeros(2)

            for i in range(1, pos[1] + 1):
                coords += Lx[(i - 1) % 2]

            coords += pos[0] * L1

        else:  # PBC and OBC.
            coords = pos @ lattice_vecs.T

        return coords

    # @partial(jit, static_argnums=(0,))
    def get_nearest_neighbors(self, pos):
        if self.boundary == "xc":
            n1 = (pos[0], pos[1] + 1)
            n3 = (pos[0], pos[1] - 1)
            if pos[0] % 2 == 1:
                n5 = ((pos[0] + 1) % self.l_x, pos[1] + 1)
                n6 = ((pos[0] - 1) % self.l_x, pos[1] + 1)
            else:
                n5 = ((pos[0] + 1) % self.l_x, pos[1] - 1)
                n6 = ((pos[0] - 1) % self.l_x, pos[1] - 1)
            n2 = ((pos[0] + 1) % self.l_x, pos[1])
            n4 = ((pos[0] - 1) % self.l_x, pos[1])

        elif self.boundary == "oxc":  # open-XC.
            n1 = (pos[0], pos[1] + 1)
            n3 = (pos[0], pos[1] - 1)
            if pos[0] % 2 == 1:
                n5 = (pos[0] + 1, pos[1] + 1)
                n6 = (pos[0] - 1, pos[1] + 1)
            else:
                n5 = (pos[0] + 1, pos[1] - 1)
                n6 = (pos[0] - 1, pos[1] - 1)
            n2 = (pos[0] + 1, pos[1])
            n4 = (pos[0] - 1, pos[1])

        elif self.boundary == "yc":
            n1 = (pos[0], pos[1] + 1)
            n3 = (pos[0], pos[1] - 1)
            if pos[1] % 2 == 1:
                n5 = ((pos[0] - 1) % self.l_x, pos[1] + 1)
                n6 = ((pos[0] - 1) % self.l_x, pos[1] - 1)
            else:
                n5 = ((pos[0] + 1) % self.l_x, pos[1] + 1)
                n6 = ((pos[0] + 1) % self.l_x, pos[1] - 1)
            n2 = ((pos[0] + 1) % self.l_x, pos[1])
            n4 = ((pos[0] - 1) % self.l_x, pos[1])

        elif self.boundary == "pbc":
            n1 = (pos[0], (pos[1] + 1) % self.l_y)
            n3 = (pos[0], (pos[1] - 1) % self.l_y)
            n5 = ((pos[0] + 1) % self.l_x, (pos[1] - 1) % self.l_y)
            n6 = ((pos[0] - 1) % self.l_x, (pos[1] + 1) % self.l_y)
            n2 = ((pos[0] + 1) % self.l_x, pos[1])
            n4 = ((pos[0] - 1) % self.l_x, pos[1])

        elif self.boundary == "obc":
            n1 = (pos[0], pos[1] + 1)
            n3 = (pos[0], pos[1] - 1)
            n5 = (pos[0] + 1, pos[1] - 1)
            n6 = (pos[0] - 1, pos[1] + 1)
            n2 = (pos[0] + 1, pos[1])
            n4 = (pos[0] - 1, pos[1])

        return jnp.array([n1, n2, n3, n4, n5, n6])

    def get_neighboring_bonds(self, adjacency_matrix):
        # k = 1 excludes diagonal.
        i, j = np.where(np.triu(adjacency_matrix, k=1) != 0)
        return np.column_stack((i, j))

    def create_adjacency_matrix(self):
        width, height = self.l_y, self.l_x
        size = width * height
        h = np.zeros((size, size), dtype=int)

        for r in range(width):
            for q in range(height):
                i = q * width + r
                neighbors = self.get_nearest_neighbors((q, r))
                for nq, nr in neighbors:
                    if 0 <= nq < height and 0 <= nr < width:  # Check bounds
                        j = nq * width + nr
                        h[i, j] = 1
                        h[j, i] = 1
        return h

    def get_neel_guess(self, angle=0.0):
        boundary = self.boundary
        sites_0 = []
        sites_1 = []
        sites_2 = []
        for site in self.sites:
            x, y = site
            site_n = self.get_site_num(site)
            if boundary == "xc" or boundary == "oxc":
                sub = (y + 2 * (x & 1)) % 3
            elif boundary == "yc":
                sub = (x + (y & 1)) % 3
            else:
                raise ValueError(
                    "Neel guess only implemented for XC or OXC or YC boundaries."
                )
            if sub == 0:
                sites_0.append(site_n)
            elif sub == 1:
                sites_1.append(site_n)
            else:
                sites_2.append(site_n)

        def spinor(theta, phi):
            up = np.cos(theta / 2.0)
            dn = np.exp(1j * phi) * np.sin(theta / 2.0)
            return np.array([up, dn])

        spinor_0 = spinor(angle, 0)
        spinor_0_dm = np.outer(spinor_0, spinor_0.conj())
        theta = 2 * np.pi / 3 + angle
        spinor_1 = spinor(theta, 0)
        spinor_1_dm = np.outer(spinor_1, spinor_1.conj())
        spinor_2 = spinor(-theta + angle, 0)
        spinor_2_dm = np.outer(spinor_2, spinor_2.conj())
        dm = np.zeros((2 * self.n_sites, 2 * self.n_sites), dtype=complex)

        def put_block(i, rho):
            dm[i, i] = rho[0, 0]
            dm[i, i + self.n_sites] = rho[0, 1]
            dm[i + self.n_sites, i] = rho[1, 0]
            dm[i + self.n_sites, i + self.n_sites] = rho[1, 1]

        for i in sites_0:
            put_block(i, spinor_0_dm)
        for i in sites_1:
            put_block(i, spinor_1_dm)
        for i in sites_2:
            put_block(i, spinor_2_dm)
        return dm

    def get_chiral_guess(self, tilt=np.pi / 3):
        boundary = self.boundary
        if boundary not in ("xc", "oxc", "yc"):
            raise ValueError(
                "Chiral guess only implemented for 'xc', 'oxc' or 'yc' boundaries."
            )
        sites_0, sites_1, sites_2 = [], [], []
        for x, y in self.sites:
            site_n = self.get_site_num((x, y))
            if boundary == "xc" or boundary == "oxc":
                sub = (y + 2 * (x & 1)) % 3
            elif boundary == "yc":
                sub = (x + (y & 1)) % 3
            if sub == 0:
                sites_0.append(site_n)
            elif sub == 1:
                sites_1.append(site_n)
            else:
                sites_2.append(site_n)

        def spinor(theta, phi):
            up = np.cos(theta / 2.0)
            dn = np.exp(1j * phi) * np.sin(theta / 2.0)
            return np.array([up, dn])

        s0 = spinor(tilt, 0.0)
        s1 = spinor(tilt, 2.0 * np.pi / 3.0)
        s2 = spinor(tilt, 4.0 * np.pi / 3.0)

        rho0 = np.outer(s0, s0.conj())
        rho1 = np.outer(s1, s1.conj())
        rho2 = np.outer(s2, s2.conj())

        dm = np.zeros((2 * self.n_sites, 2 * self.n_sites), dtype=complex)

        def put_block(i, rho):
            dm[i, i] = rho[0, 0]
            dm[i, i + self.n_sites] = rho[0, 1]
            dm[i + self.n_sites, i] = rho[1, 0]
            dm[i + self.n_sites, i + self.n_sites] = rho[1, 1]

        for i in sites_0:
            put_block(i, rho0)
        for i in sites_1:
            put_block(i, rho1)
        for i in sites_2:
            put_block(i, rho2)

        return dm

    def get_square_guess(self, angle=0.0):
        sites_up = []
        sites_dn = []

        for x, y in self.sites:
            site_n = self.get_site_num((x, y))
            if (x + y) % 2 == 0:
                sites_up.append(site_n)
            else:
                sites_dn.append(site_n)

        dm = np.zeros((2 * self.n_sites, 2 * self.n_sites))

        def spinor(theta, phi):
            up = np.cos(theta / 2.0)
            dn = np.exp(1j * phi) * np.sin(theta / 2.0)
            return np.array([up, dn])

        x_up = spinor(angle, 0.0)
        x_dn = spinor(angle, np.pi)

        rho_up = np.outer(x_up, x_up.conj())
        rho_dn = np.outer(x_dn, x_dn.conj())

        dm = np.zeros((2 * self.n_sites, 2 * self.n_sites), dtype=complex)

        def put_block(i, rho):
            dm[i, i] = rho[0, 0]
            dm[i, i + self.n_sites] = rho[0, 1]
            dm[i + self.n_sites, i] = rho[1, 0]
            dm[i + self.n_sites, i + self.n_sites] = rho[1, 1]

        for i in sites_up:
            put_block(i, rho_up)

        for i in sites_dn:
            put_block(i, rho_dn)

        return dm

    def _build_flux_h1(self, phi=np.pi / 2, t=-1.0):
        """
        Single-particle NN Hamiltonian with uniform flux `phi` per triangle.

        H_ij = t * exp(i A_ij), with A_ij depending only on bond direction:
            vertical (0,±1):  phase 0
            horizontal (±1,0): phase ±phi
            diagonal (±1,±1): phase 0

        Works for 'xc' and 'yc' (PBC in x, OBC in y). For other boundaries we
        just use whatever neighbors exist, same phase rule.
        """
        nx, ny = self.l_x, self.l_y
        N = self.n_sites
        H = np.zeros((N, N), dtype=complex)

        def idx(pos):
            return self.get_site_num(pos)

        # helper to get dx with periodicity in x, open in y
        def bond_phase(pos_i, pos_j):
            (x1, y1), (x2, y2) = pos_i, pos_j
            dx = x2 - x1
            # periodic in x for xc/yc/pbc
            if self.boundary in ("xc", "oxc", "yc", "pbc"):
                if dx > nx // 2:
                    dx -= nx
                elif dx < -nx // 2:
                    dx += nx
            dy = y2 - y1  # y is open for cylinders

            # vertical bonds
            if dx == 0 and abs(dy) == 1:
                return 0.0

            # horizontal bonds
            if dy == 0 and abs(dx) == 1:
                return phi if dx == 1 else -phi

            # diagonals (±1, ±1) -> no extra phase
            if abs(dx) == 1 and abs(dy) == 1:
                return 0.0

            raise ValueError(f"Unexpected bond direction dx={dx}, dy={dy}")

        # loop over all sites and nearest neighbors, build Hermitian H
        for x, y in self.sites:
            i = idx((x, y))
            for xn, yn in self.get_nearest_neighbors((x, y)):
                # respect open boundaries in y
                if not (0 <= xn < nx and 0 <= yn < ny):
                    continue
                j = idx((xn, yn))
                if j <= i:
                    continue  # avoid double counting

                phase = bond_phase((x, y), (xn, yn))
                hop = t * np.exp(1j * phase)
                H[i, j] = hop
                H[j, i] = hop.conjugate()

        return H

    def get_flux_band_dm(self, nelec, phi=np.pi / 2, t=-1.0):
        """
        Construct a spinful 1RDM from a flux-phase band Hamiltonian.

        Parameters
        ----------
        nelec : int or (int, int)
            Total number of electrons, or (n_up, n_down).
            If int, we assume spin-balanced: n_up = n_down = nelec // 2.
        phi : float
            Flux per elementary triangle (in radians).
            phi != 0, pi breaks time reversal.
        t : float
            NN hopping amplitude for the auxiliary flux Hamiltonian.

        Returns
        -------
        spin_rdm1 : (2N, 2N) complex array
            1RDM in basis [up(0..N-1), down(N..2N-1)].
        """
        N = self.n_sites

        # electron numbers
        if isinstance(nelec, tuple):
            n_up, n_dn = nelec
        else:
            if nelec % 2 != 0:
                raise ValueError(
                    "For scalar nelec, require even nelec for spin-balanced state."
                )
            n_up = n_dn = nelec // 2

        if n_up > N or n_dn > N:
            raise ValueError("Cannot occupy more orbitals than sites.")

        # build flux Hamiltonian and diagonalize
        H_flux = self._build_flux_h1(phi=phi, t=t)
        eps, U = np.linalg.eigh(H_flux)  # U[:,n] is eigenvector n

        # projectors for up and down spins
        P_up = U[:, :n_up] @ U[:, :n_up].conj().T
        P_dn = U[:, :n_dn] @ U[:, :n_dn].conj().T

        spin_rdm1 = np.zeros((2 * N, 2 * N), dtype=complex)
        spin_rdm1[:N, :N] = P_up
        spin_rdm1[N:, N:] = P_dn
        # no transverse spin in this seed: r_ud = r_du = 0

        return spin_rdm1

    def plot_observables(
        self,
        spin_rdm1,
        arrow_scale=1.5,
        charge_scale=None,
        annotate_sites=True,
        cmap="viridis",
        size=None,
        show_currents=False,
        current_scale=0.6,
        current_threshold=1e-3,
        edge_only=False,
        h1=None,
        t=1.0,
        current_color="magenta",
        show_triangle_currents=False,
        triangle_cmap="coolwarm",
        triangle_alpha=0.4,
        ax=None,
        show_charge_colorbar=True,
        show_current_colorbar=True,
    ):
        """
        Plot charge density (color) and spin expectation (arrows) on a triangular
        lattice, optionally with bond currents and triangle loop currents.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        n_sites = self.n_sites
        nx, ny = self.l_x, self.l_y

        boundary = self.boundary
        if boundary not in ("xc", "oxc", "yc"):
            raise ValueError(
                "plot_observables only implemented for 'xc', 'oxc', or 'yc'."
            )

        r_uu = spin_rdm1[:n_sites, :n_sites]
        r_dd = spin_rdm1[n_sites:, n_sites:]
        r_ud = spin_rdm1[:n_sites, n_sites:]
        r_du = spin_rdm1[n_sites:, :n_sites]

        sz = (np.diag(r_uu) - np.diag(r_dd)) / 2.0
        charge = np.diag(r_uu) + np.diag(r_dd)
        sx = (np.diag(r_ud) + np.diag(r_du)) / 2.0
        sy = -0.5j * (np.diag(r_ud) - np.diag(r_du))

        sz = sz.real.reshape(nx, ny)
        sx = sx.real.reshape(nx, ny)
        sy = sy.real.reshape(nx, ny)
        charge = charge.real.reshape(nx, ny)  # type: ignore

        P = r_uu + r_dd

        coords = []
        xs, ys, cs = [], [], []
        for i, (x, y) in enumerate(self.sites):
            X, Y = self.get_site_coordinate((x, y))
            coords.append((X, Y))
            xs.append(X)
            ys.append(Y)
            cs.append(charge[x, y])
        xs = np.array(xs)
        ys = np.array(ys)
        cs = np.array(cs)

        # def get_figsize(fig_height_pt, ratio):
        #     inches_per_pt = 1.0 / 72.0
        #     fig_height = fig_height_pt * inches_per_pt
        #     fig_width = fig_height / ratio
        #     return [fig_width, fig_height]

        # if size is None:
        #     size = 100 * nx
        # fig, ax = plt.subplots(figsize=get_figsize(size, ratio=nx / ny))
        # --- create / reuse axes ---
        own_fig = False
        if ax is None:
            if size is None:
                size = 100 * nx

            def get_figsize(fig_height_pt, ratio):
                inches_per_pt = 1.0 / 72.0
                fig_height = fig_height_pt * inches_per_pt
                fig_width = fig_height / ratio
                return [fig_width, fig_height]

            fig, ax = plt.subplots(figsize=get_figsize(size, ratio=nx / ny))
            own_fig = True
        else:
            fig = ax.figure
            if size is None:
                size = 100 * nx

        if charge_scale is None:
            charge_scale = 30 * (size / 100)

        def is_edge_site(x, y):
            if boundary in ("xc", "oxc"):  # PBC in x, open in y
                return (y == 0) or (y == ny - 1)
            elif boundary == "yc":  # PBC in y, open in x
                return (x == 0) or (x == nx - 1)
            else:
                return False

        need_curr = show_currents or show_triangle_currents
        has_bond = np.zeros((n_sites, n_sites), dtype=bool)
        is_pbc_bond = np.zeros((n_sites, n_sites), dtype=bool)
        Jbond = np.zeros((n_sites, n_sites), dtype=float)
        neighbors = [[] for _ in range(n_sites)]

        for i, (x, y) in enumerate(self.sites):
            Xi, Yi = coords[i]
            for nn in self.get_nearest_neighbors((x, y)):
                xn, yn = int(nn[0]), int(nn[1])
                if not (0 <= xn < nx and 0 <= yn < ny):
                    continue
                j = self.get_site_num((xn, yn))
                if j <= i:
                    continue

                Xj, Yj = coords[j]

                is_pbc = (abs(xn - x) > 1) or (abs(yn - y) > 1)

                # store adjacency
                has_bond[i, j] = has_bond[j, i] = True
                is_pbc_bond[i, j] = is_pbc_bond[j, i] = is_pbc
                neighbors[i].append(j)
                neighbors[j].append(i)

                # bond current (or 0 if not needed)
                Jij = 0.0
                if need_curr and not is_pbc:
                    tij = h1[i, j] if h1 is not None else t
                    if tij != 0:
                        Jij = 2.0 * np.imag(tij * P[i, j])
                    Jbond[i, j] = Jij
                    Jbond[j, i] = -Jij

                # plot bond line
                if is_pbc:
                    continue
                    # ax.plot([Xi, Xj], [Yi, Yj], "k--", lw=0.001, zorder=0)
                else:
                    ax.plot([Xi, Xj], [Yi, Yj], "k-", lw=0.6, zorder=1)

                # plot bond current arrow
                if show_currents and not is_pbc:
                    if abs(Jij) < current_threshold:
                        continue
                    if edge_only and not (is_edge_site(x, y) or is_edge_site(xn, yn)):
                        continue

                    xm = 0.5 * (Xi + Xj)
                    ym = 0.5 * (Yi + Yj)
                    dx = Xj - Xi
                    dy = Yj - Yi
                    L = np.hypot(dx, dy)
                    if L == 0:
                        continue
                    ux, uy = dx / L, dy / L

                    direction = np.sign(Jij) if Jij != 0 else 1.0
                    length = current_scale * abs(Jij)

                    ax.arrow(
                        xm,
                        ym,
                        direction * length * ux,
                        direction * length * uy,
                        head_width=0.1,
                        length_includes_head=True,
                        linewidth=0.4,
                        zorder=2.5,
                        color=current_color,
                    )

        if show_triangle_currents:
            tris = []
            vals = []
            for i in range(n_sites):
                ni = neighbors[i]
                for a in range(len(ni)):
                    j = ni[a]
                    if j <= i:
                        continue
                    for b in range(a + 1, len(ni)):
                        k = ni[b]
                        if k <= i:
                            continue
                        if not has_bond[j, k]:
                            continue
                        # skip triangles that use any PBC bond (wrap-around)
                        if is_pbc_bond[i, j] or is_pbc_bond[j, k] or is_pbc_bond[k, i]:
                            continue

                        xi, yi = coords[i]
                        xj, yj = coords[j]
                        xk, yk = coords[k]

                        area2 = (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
                        if np.isclose(area2, 0.0):
                            continue  # degenerate

                        if area2 > 0:
                            tri = [(xi, yi), (xj, yj), (xk, yk)]
                            J_delta = Jbond[i, j] + Jbond[j, k] + Jbond[k, i]
                        else:
                            tri = [(xi, yi), (xk, yk), (xj, yj)]
                            J_delta = Jbond[i, k] + Jbond[k, j] + Jbond[j, i]

                        tris.append(tri)
                        vals.append(J_delta)

            if tris:
                pc = PolyCollection(
                    tris,
                    array=np.array(vals),
                    cmap=triangle_cmap,
                    alpha=triangle_alpha,
                    edgecolors="none",
                    zorder=0.5,
                )
                ax.add_collection(pc)
                # cbar2 = plt.colorbar(
                #     pc, ax=ax, fraction=0.046, pad=0.10, orientation="horizontal"
                # )
                # cbar2.set_label(r"$J_\Delta$")
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                if show_current_colorbar:

                    cax2 = inset_axes(
                        ax,
                        width="50%",  # bar length (relative to main axes width)
                        height="3%",  # bar thickness (relative to main axes height)
                        loc="lower center",
                        bbox_to_anchor=(0.0, -0.05, 1, 1),  # y < 0 pushes it below
                        bbox_transform=ax.transAxes,
                        borderpad=0,
                    )
                    cbar2 = fig.colorbar(pc, cax=cax2, orientation="horizontal")
                    cbar2.set_label(r"$J_\Delta$")

        sc = ax.scatter(
            xs,
            ys,
            c=cs,
            cmap=cmap,
            s=charge_scale,
            edgecolors="k",
            linewidths=0.3,
            zorder=2,
        )

        for i, (x, y) in enumerate(self.sites):
            X, Y = coords[i]
            ax.arrow(
                X,
                Y,
                arrow_scale * sx[x, y],  # type: ignore
                arrow_scale * sz[x, y],  # type: ignore
                head_width=0.1,
                length_includes_head=True,
                linewidth=0.4,
                zorder=3,
            )
            if annotate_sites:
                ax.text(
                    X + 0.1,
                    Y + 0.1,
                    f"{i}",
                    color="red",
                    zorder=4,
                )

        # cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label("charge density")
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        if show_charge_colorbar:
            cax1 = inset_axes(
                ax,
                width="3%",  # bar thickness (relative to main axes width)
                height="60%",  # bar length (relative to main axes height)
                loc="center left",
                bbox_to_anchor=(1.02, 0.0, 1, 1),  # (x, y, w, h) in axes coords
                bbox_transform=ax.transAxes,
                borderpad=0.0,  # distance from the main axes
            )
            cbar = fig.colorbar(sc, cax=cax1)
            # cbar.set_label(r"$n$")
            cbar.ax.set_xlabel(r"$n$", labelpad=3)  # text and spacing
            cbar.ax.xaxis.set_label_position("bottom")  # put it below, not above

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(False)
        if own_fig:
            plt.tight_layout()
            plt.show()

        return fig, ax

    def build_pg_ops_oxc(self, parity=1.0):
        """
        Full space group ops for triangular XC cylinder (OBC in x, OBC in y)
        """
        nx = self.l_x
        ny = self.l_y

        if nx % 2 != 0:
            raise ValueError("nx must be even to have a 2-row unit cell along x.")

        n_sites = nx * ny
        E = np.eye(n_sites, dtype=float)
        perm_c2 = np.arange(n_sites - 1, -1, -1)
        U_c2 = E[:, perm_c2]
        pg_ops = [E, U_c2]
        pg_chars = [1.0, parity]
        return pg_ops, pg_chars

    def build_pg_ops_xc(self):
        """
        Full space-group ops for triangular XC cylinder (PBC in x, OBC in y)
        with a 2-row unit cell along x.

        Returns
        -------
        pg_ops  : list of (N, N) arrays
            All symmetry operators: {E, C2, G, C2G} * {T2^n}, n = 0..nx/2-1.
            Total number of ops = 2 * nx.
        pg_chars: list of ints
            Characters for the totally symmetric irrep (all +1).
        """
        nx = self.l_x
        ny = self.l_y

        if nx % 2 != 0:
            raise ValueError("nx must be even to have a 2-row unit cell along x.")

        n_sites = nx * ny
        E = np.eye(n_sites, dtype=float)

        def idx(x, y):
            return (x % nx) * ny + y

        # --- C2: inversion ---
        # (x, y) -> (nx-1-x, ny-1-y)  <=>  i -> N-1-i in row-major
        perm_c2 = np.arange(n_sites - 1, -1, -1)
        U_c2 = E[:, perm_c2]

        # --- Glide G: (x, y) -> (x+1, ny-1-y) ---
        perm_g = np.empty(n_sites, dtype=int)
        for x in range(nx):
            for y in range(ny):
                i = idx(x, y)
                perm_g[i] = idx((x + 1) % nx, ny - 1 - y)
        U_g = E[:, perm_g]

        # --- T2^n: translation by 2n rows: (x, y) -> (x+2n, y) ---
        def U_t2_power(n):
            perm = np.empty(n_sites, dtype=int)
            for x in range(nx):
                for y in range(ny):
                    i = idx(x, y)
                    perm[i] = idx((x + 2 * n) % nx, y)
            return E[:, perm]

        # Base (point-group) ops without translations
        base_ops = [E, U_c2, U_g, U_c2 @ U_g]

        pg_ops = []
        pg_chars = []

        # n = 0: just the base ops
        for U in base_ops:
            if not any(np.array_equal(U, V) for V in pg_ops):
                pg_ops.append(U)
                pg_chars.append(1)

        # n = 1 .. nx/2-1: T2^n and its products with base ops
        maxn = nx // 2
        for n in range(1, maxn):
            U_tn = U_t2_power(n)
            for U0 in base_ops:
                U = U0 @ U_tn
                if not any(np.array_equal(U, V) for V in pg_ops):
                    pg_ops.append(U)
                    pg_chars.append(1)

        return pg_ops, pg_chars

    def build_pg_ops_yc(self):
        """
        Full space-group ops for triangular YC cylinder (PBC in x, OBC in y)
        with a 1-row unit cell along x.

        Returns
        -------
        pg_ops  : list of (N, N) arrays
            All symmetry operators: {E, C2, G, C2G} * {T^n}, n = 0..nx-1.
            Total number of ops = 4 * nx.
        pg_chars: list of ints
            Characters for the totally symmetric irrep (all +1).
        """
        nx = self.l_x
        ny = self.l_y
        n_sites = nx * ny

        E = np.eye(n_sites, dtype=float)

        def idx(x, y):
            return (x % nx) * ny + y

        # --- C2: inversion ---
        perm_c2 = np.arange(n_sites - 1, -1, -1)
        U_c2 = E[:, perm_c2]

        # --- Glide G (YC) ---
        perm_g = np.empty(n_sites, dtype=int)
        for x in range(nx):
            for y in range(ny):
                i = idx(x, y)
                if y % 2 == 0:  # even column
                    xp = (x + 1) % nx
                else:  # odd column
                    xp = x
                yp = ny - 1 - y
                perm_g[i] = idx(xp, yp)
        U_g = E[:, perm_g]

        # --- T^n: translation by n rows: (x, y) -> (x + n, y) ---
        def U_t_power(n):
            perm = np.empty(n_sites, dtype=int)
            for x in range(nx):
                for y in range(ny):
                    i = idx(x, y)
                    perm[i] = idx(x + n, y)
            return E[:, perm]

        # Base (point-group) ops without translations
        base_ops = [E, U_c2, U_g, U_c2 @ U_g]

        pg_ops = []
        pg_chars = []

        # n = 0: just the base ops
        for U in base_ops:
            if not any(np.array_equal(U, V) for V in pg_ops):
                pg_ops.append(U)
                pg_chars.append(1)

        # n = 1 .. nx-1: T^n and its products with base ops
        for n in range(1, nx):
            U_tn = U_t_power(n)
            for U0 in base_ops:
                U = U0 @ U_tn
                if not any(np.array_equal(U, V) for V in pg_ops):
                    pg_ops.append(U)
                    pg_chars.append(1)

        return pg_ops, pg_chars

    def build_pg_ops(self):
        if self.boundary in "xc":
            return self.build_pg_ops_xc()
        elif self.boundary == "yc":
            return self.build_pg_ops_yc()
        elif self.boundary == "oxc":
            return self.build_pg_ops_oxc(parity=1.0)
        else:
            raise ValueError("build_pg_ops only implemented for 'xc' or 'yc'.")

    def classify_orbitals_D2_xc(self, tol_e=1e-8, tol_sym=1e-6):
        """
        Classify one-electron eigenstates of an XC triangular lattice according to
        the D2 subgroup (generated by G^N/2 and C2) of the dihedral D_N symmetry.

        Each orbital is labelled by its eigenvalues, which
        are mapped to string irreps:
          ( +1, +1 ) -> A1
          ( +1, -1 ) -> A2
          ( -1, +1 ) -> B1
          ( -1, -1 ) -> B2
        """
        nx = self.l_x
        ny = self.l_y
        h1 = -1.0 * self.create_adjacency_matrix()
        n_sites = nx * ny
        if h1.shape != (n_sites, n_sites):
            raise ValueError(f"h1 must be of shape {(n_sites, n_sites)}")

        if nx % 2 != 0:
            raise ValueError("nx must be even for this D_N XC lattice.")
        if ny % 2 != 0:
            raise ValueError("ny must be even for this D_N XC lattice.")

        def idx(x, y):
            return x * ny + y

        E = np.eye(n_sites, dtype=float)

        perm_c2 = np.arange(n_sites - 1, -1, -1)
        U_c2 = E[:, perm_c2]

        perm_g = np.empty(n_sites, dtype=int)
        for x in range(nx):
            for y in range(ny):
                perm_g[idx(x, y)] = idx((x + 1) % nx, ny - 1 - y)
        U_g = E[:, perm_g]

        step = nx // 2
        perm_a = np.empty(n_sites, dtype=int)
        for x in range(nx):
            for y in range(ny):
                x_new = (x + step) % nx
                if step % 2 == 0:
                    y_new = y
                else:
                    y_new = ny - 1 - y
                perm_a[idx(x, y)] = idx(x_new, y_new)
        U_a = E[:, perm_a]

        assert np.allclose(U_a @ U_a, E)
        assert np.allclose(U_c2 @ U_c2, E)
        assert np.allclose(U_a @ U_c2, U_c2 @ U_a)

        energies, vecs = np.linalg.eigh(h1)

        blocks = []
        i = 0
        while i < n_sites:
            j = i + 1
            while j < n_sites and abs(energies[j] - energies[i]) < tol_e:
                j += 1
            blocks.append(list(range(i, j)))
            i = j

        vecs_sym = vecs.copy()
        ir_labels = [""] * n_sites
        parities = [tuple()] * n_sites

        parity_to_label = {
            (+1, +1): "A1",
            (+1, -1): "A2",
            (-1, +1): "B1",
            (-1, -1): "B2",
        }

        for block in blocks:
            inds = np.array(block)
            Vb = vecs_sym[:, inds]
            d = Vb.shape[1]

            rep_a = Vb.conj().T @ (U_a @ Vb)
            rep_b = Vb.conj().T @ (U_c2 @ Vb)

            wa, Ua = np.linalg.eigh(rep_a)
            Vb = Vb @ Ua
            rep_b = Ua.conj().T @ rep_b @ Ua
            vecs_sym[:, inds] = Vb

            sub_i = 0
            while sub_i < d:
                sub_j = sub_i + 1
                while sub_j < d and abs(wa[sub_j] - wa[sub_i]) < tol_sym:
                    sub_j += 1
                sub = slice(sub_i, sub_j)
                rep_b_sub = rep_b[sub, sub]
                wb, Ub = np.linalg.eigh(rep_b_sub)
                Vb[:, sub] = Vb[:, sub] @ Ub
                rep_b[sub, :] = Ub.conj().T @ rep_b[sub, :]
                rep_b[:, sub] = rep_b[:, sub] @ Ub
                vecs_sym[:, inds] = Vb
                sub_i = sub_j

            for local_j, global_j in enumerate(inds):
                phi = vecs_sym[:, global_j]
                la = np.vdot(phi, U_a @ phi).real
                lb = np.vdot(phi, U_c2 @ phi).real
                la = +1 if la >= 0 else -1
                lb = +1 if lb >= 0 else -1
                parities[global_j] = (la, lb)
                ir_labels[global_j] = parity_to_label[(la, lb)]

        return energies, vecs_sym, ir_labels, parities

    def classify_orbitals_D2_yc(self, tol_e=1e-8, tol_sym=1e-6):
        """
        Classify one-electron eigenstates of a YC triangular lattice with
        according to a D2 subgroup of the full D_{2N} symmetry.

        Each orbital is labelled by its eigenvalues, which
        we map to string irreps:
          ( +1, +1 ) -> "A1"
          ( +1, -1 ) -> "A2"
          ( -1, +1 ) -> "B1"
          ( -1, -1 ) -> "B2"

        """
        nx = self.l_x
        ny = self.l_y
        h1 = -1.0 * self.create_adjacency_matrix()
        n_sites = nx * ny
        if h1.shape != (n_sites, n_sites):
            raise ValueError(f"h1 must be of shape {(n_sites, n_sites)}")

        if nx % 2 != 0:
            raise ValueError("nx must be even for this D_{2N} YC lattice.")
        if ny % 2 != 0:
            print(
                "Warning: ny is odd; C2 may not be an exact symmetry of this YC cluster."
            )

        def idx(x, y):
            return x * ny + y

        E = np.eye(n_sites, dtype=float)

        perm_c2 = np.empty(n_sites, dtype=int)
        for x in range(nx):
            for y in range(ny):
                perm_c2[idx(x, y)] = idx(nx - 1 - x, ny - 1 - y)
        U_c2 = E[:, perm_c2]

        step = nx // 2
        perm_a = np.empty(n_sites, dtype=int)
        for x in range(nx):
            for y in range(ny):
                x_new = (x + step) % nx
                y_new = y
                perm_a[idx(x, y)] = idx(x_new, y_new)
        U_a = E[:, perm_a]

        if not np.allclose(U_a @ U_a, E):
            raise RuntimeError("U_a^2 != I; T_half seems not to be order 2.")
        if not np.allclose(U_c2 @ U_c2, E):
            raise RuntimeError("U_c2^2 != I; C2 seems not to be order 2.")
        if not np.allclose(U_a @ U_c2, U_c2 @ U_a):
            raise RuntimeError("U_a and U_c2 do not commute; D2 construction broken.")

        energies, vecs = np.linalg.eigh(h1)

        blocks = []
        i = 0
        while i < n_sites:
            j = i + 1
            while j < n_sites and abs(energies[j] - energies[i]) < tol_e:
                j += 1
            blocks.append(list(range(i, j)))
            i = j

        vecs_sym = vecs.copy()
        ir_labels = [""] * n_sites
        parities = [tuple()] * n_sites

        parity_to_label = {
            (+1, +1): "A1",
            (+1, -1): "A2",
            (-1, +1): "B1",
            (-1, -1): "B2",
        }

        for block in blocks:
            inds = np.array(block)
            Vb = vecs_sym[:, inds]
            d = Vb.shape[1]

            rep_a = Vb.conj().T @ (U_a @ Vb)
            rep_b = Vb.conj().T @ (U_c2 @ Vb)

            wa, Ua = np.linalg.eigh(rep_a)
            Vb = Vb @ Ua
            rep_b = Ua.conj().T @ rep_b @ Ua
            vecs_sym[:, inds] = Vb

            sub_i = 0
            while sub_i < d:
                sub_j = sub_i + 1
                while sub_j < d and abs(wa[sub_j] - wa[sub_i]) < tol_sym:
                    sub_j += 1
                sub = slice(sub_i, sub_j)
                rep_b_sub = rep_b[sub, sub]
                wb, Ub = np.linalg.eigh(rep_b_sub)
                Vb[:, sub] = Vb[:, sub] @ Ub
                rep_b[sub, :] = Ub.conj().T @ rep_b[sub, :]
                rep_b[:, sub] = rep_b[:, sub] @ Ub
                vecs_sym[:, inds] = Vb
                sub_i = sub_j

            for local_j, global_j in enumerate(inds):
                phi = vecs_sym[:, global_j]
                la = np.vdot(phi, U_a @ phi).real
                lb = np.vdot(phi, U_c2 @ phi).real
                la = +1 if la >= 0 else -1
                lb = +1 if lb >= 0 else -1
                parities[global_j] = (la, lb)
                ir_labels[global_j] = parity_to_label[(la, lb)]

        return energies, vecs_sym, ir_labels, parities

    def classify_orbitals_C2(self, tol_e=1e-8, tol_sym=1e-6):
        h1 = -1.0 * self.create_adjacency_matrix()
        n_sites = self.n_sites
        if h1.shape != (n_sites, n_sites):
            raise ValueError(f"h1 must be of shape {(n_sites, n_sites)}")
        E = np.eye(n_sites, dtype=float)
        perm_c2 = np.arange(n_sites - 1, -1, -1)
        U_c2 = E[:, perm_c2]
        energies, vecs = np.linalg.eigh(h1)
        ir_labels = [""] * n_sites
        parities = [0] * n_sites
        for i in range(n_sites):
            phi = vecs[:, i]
            l = np.vdot(phi, U_c2 @ phi).real
            l = +1 if l >= 0 else -1
            parities[i] = l
            ir_labels[i] = "A" if l == +1 else "B"
        return energies, vecs, ir_labels, parities

    def classify_orbitals_D2(self, tol_e=1e-8, tol_sym=1e-6):
        if self.boundary in "xc":
            return self.classify_orbitals_D2_xc(tol_e=tol_e, tol_sym=tol_sym)
        elif self.boundary == "yc":
            return self.classify_orbitals_D2_yc(tol_e=tol_e, tol_sym=tol_sym)
        else:
            raise ValueError("classify_orbitals_D2 only implemented for 'xc' or 'yc'.")

    def __hash__(self):
        return hash(
            (
                self.l_x,
                self.l_y,
                self.shape,
                self.sites,
                self.coord_num,
            )
        )

    def tree_flatten(self):
        return (), (
            self.l_x,
            self.l_y,
            self.shape,
            self.sites,
            self.coord_num,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@dataclass
@register_pytree_node_class
class three_dimensional_grid(lattice):
    l_x: int
    l_y: int
    l_z: int
    shape: Optional[tuple] = None
    shell_distances: Optional[Sequence] = None
    sites: Optional[Sequence] = None
    bonds: Optional[Sequence] = None
    n_sites: Optional[int] = None
    coord_num: int = 6

    def __post_init__(self):
        self.shape = (self.l_z, self.l_y, self.l_x)
        self.n_sites = self.l_x * self.l_y * self.l_z
        distances = []
        for x in range(self.l_x // 2 + 1):
            for y in range(self.l_y // 2 + 1):
                for z in range(self.l_z // 2 + 1):
                    dist = x**2 + y**2 + z**2
                    distances.append(dist)
        distances = [*set(distances)]
        distances.sort()
        self.shell_distances = tuple(distances)
        self.sites = tuple(
            [
                (
                    i // (self.l_x * self.l_y),
                    (i % (self.l_x * self.l_y)) // self.l_x,
                    (i % (self.l_x * self.l_y)) % self.l_x,
                )
                for i in range(self.l_x * self.l_y * self.l_z)
            ]
        )
        # TODO: fix bonds
        # self.bonds = tuple([ (i // self.l_x, i % self.l_x) for i in range(self.l_x * self.l_y) ])

    def get_site_num(self, pos):
        return pos[2] + self.l_x * pos[1] + (self.l_x * self.l_y) * pos[0]

    # does not work
    def get_symm_fac(self, pos, k):
        return 1.0
        # return jnp.exp(2 * jnp.pi * 1.j * k[0] * pos[0] / self.n_sites) * jnp.exp(2 * jnp.pi * 1.j * k[1] * pos[1] / self.n_sites) if k is not None else 1.

    def get_distance(self, pos_1, pos_2):
        dist_z = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[0] - pos_2[0]), self.l_z - jnp.abs(pos_1[0] - pos_2[0])]
            )
        )
        dist_y = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[1] - pos_2[1]), self.l_y - jnp.abs(pos_1[1] - pos_2[1])]
            )
        )
        dist_x = jnp.min(
            jnp.array(
                [jnp.abs(pos_1[2] - pos_2[2]), self.l_x - jnp.abs(pos_1[2] - pos_2[2])]
            )
        )
        dist = dist_x**2 + dist_y**2 + dist_z**2
        shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
        return shell_number

    def get_bond_distance(self, pos_1, pos_2):
        # dist_y = jnp.min(jnp.array([jnp.abs(pos_1[0] - pos_2[0]), self.l_y - jnp.abs(pos_1[0] - pos_2[0])]))
        # dist_x = jnp.min(jnp.array([jnp.abs(pos_1[1] - pos_2[1]), self.l_x - jnp.abs(pos_1[1] - pos_2[1])]))
        # dist = dist_x**2 + dist_y**2
        # shell_number = jnp.searchsorted(jnp.array(self.shell_distances), dist)
        # return shell_number
        return 0

    # TODO: fix this
    def get_neighboring_bonds(self, pos):
        return jnp.array([pos])

    # ignoring side length 1 and 2 special cases
    def get_nearest_neighbors(self, pos):
        right = (pos[0], (pos[1] + 1) % self.l_y, pos[2])
        down = ((pos[0] + 1) % self.l_z, pos[1], pos[2])
        left = (pos[0], (pos[1] - 1) % self.l_y, pos[2])
        up = ((pos[0] - 1) % self.l_z, pos[1], pos[2])
        front = (pos[0], pos[1], (pos[2] + 1) % self.l_x)
        back = (pos[0], pos[1], (pos[2] - 1) % self.l_x)
        neighbors = [right, down, left, up, front, back]
        return jnp.array(neighbors)

    def __hash__(self):
        return hash(
            (
                self.l_x,
                self.l_y,
                self.l_z,
                self.shape,
                self.shell_distances,
                self.sites,
                self.bonds,
                self.coord_num,
            )
        )

    def tree_flatten(self):
        return (), (
            self.l_x,
            self.l_y,
            self.l_z,
            self.shape,
            self.shell_distances,
            self.sites,
            self.bonds,
            self.coord_num,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)
