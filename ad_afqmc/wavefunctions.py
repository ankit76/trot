from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial, reduce, singledispatchmethod
from itertools import product
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as la
import numpy as np
from jax import jit, jvp, lax, random, vjp, vmap
from jax._src.typing import DTypeLike

from ad_afqmc import linalg_utils, walkers
from ad_afqmc.walkers import GHFWalkers, RHFWalkers, UHFWalkers, walker_batch


class wave_function(ABC):
    """Base class for wave functions. Contains methods for wave function measurements.

    The measurement methods support three types of walker chunks:

    1) generalized / GHF : walkers is a jax.Array of shape (nwalkers, norb, nelec[0] + nelec[1]).

    2) unrestricted / UHF : walkers is a list ([up, down]). up and down are jax.Arrays of shapes
    (nwalkers, norb, nelec[sigma]). In this case the _calc_<property>_unrestricted method is mapped over.

    3) restricted / RHF (up and down dets are assumed to be the same): walkers is a jax.Array of shape
    (nwalkers, max(nelec[0], nelec[1])). In this case the _calc_<property>_restricted method is mapped over.
    By default this method is defined to call _calc_<property>. For certain trial states, one can override
    it for computational efficiency.

    A minimal implementation of a wave function should define the _calc_<property> methods for
    property = overlap, force_bias, energy.

    The wave function data is stored in a separate wave_data dictionary. Its structure depends on the
    wave function type and is described in the corresponding class. It may contain "rdm1" which is a
    one-body spin RDM (2, norb, norb). If it is not provided, wave function specific methods are called.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_chunks: Number of chunks used in scan.
        projector: Type of symmetry projector used in the trial wave function.
    """

    norb: int
    nelec: Tuple[int, int]
    n_chunks: int = 1
    projector: Optional[str] = None

    def chain_external_projectors(self, wave_data: dict):
        def _chain(mats):
            """Multiply an array of matrices: M[k] @ ... @ M[1] @ M[0].

            Args:
                wave_data : dict
                    The trial wave function data.

            Returns:
                wave_data with added keys "chained_projectors".
            """
            return reduce(np.matmul, mats)

        chained_projectors = {}
        groups = wave_data["groups"]
        chars = wave_data["characters"]
        chained_chars = []
        labels = [[f"{G}_{i}" for i in range(len(groups[G]))] for G in groups]

        for idx_tuple in product(*[range(len(G)) for G in groups.values()]):
            ops = np.array(
                [list(groups.values())[iG][i] for iG, i in enumerate(idx_tuple)]
            )
            coeffs = np.array(
                [list(chars.values())[iG][i] for iG, i in enumerate(idx_tuple)]
            )
            # ops = ops * coeffs.reshape(-1, 1, 1)
            ops_labels = tuple(labels[iG][i] for iG, i in enumerate(idx_tuple))

            # Apply groups[0] first, then groups[1], ...  => multiply reversed
            op = _chain(ops[::-1])
            chained_projectors[ops_labels] = op
            char = np.prod(coeffs)
            chained_chars.append(char)

        wave_data["ext_ops"] = jnp.array(list(chained_projectors.values()))
        wave_data["ext_chars"] = jnp.array(chained_chars)

        return wave_data

    @singledispatchmethod
    def calc_overlap(self, walkers, wave_data: dict) -> jax.Array:
        """Calculate the overlap < psi_t | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The overlap of the walkers with the trial wave function.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_overlap.register
    def _(self, walkers: UHFWalkers, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_overlap_unrestricted_handler, self.n_chunks, wave_data
        )

    def _calc_overlap_unrestricted_handler(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        if self.projector == "s2":
            return self._calc_overlap_s2(walker_up, walker_dn, wave_data)
        elif self.projector == "tr" and self.nelec[0] == self.nelec[1]:
            return self._calc_overlap_tr(walker_up, walker_dn, wave_data)
        else:
            return self._calc_overlap_unrestricted(walker_up, walker_dn, wave_data)

    def _calc_overlap_s2(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        # assume s = Sz = walkers[0].shape[1] - walkers[1].shape[1]
        _, _, w_betas, betas = wave_data["betas"]

        RotMatrix = vmap(
            lambda beta: jnp.array(
                [
                    [jnp.cos(beta / 2), jnp.sin(beta / 2)],
                    [-jnp.sin(beta / 2), jnp.cos(beta / 2)],
                ]
            )
        )(betas)

        def applyRotMat(detA, detB, mat):
            A, B = detA * mat[0, 0], detB * mat[0, 1]
            C, D = detA * mat[1, 0], detB * mat[1, 1]

            detAout = jnp.block([[A, B], [C, D]])
            return detAout

        # Shape (nbeta, 2*norb, nocc).
        S2walkers = vmap(applyRotMat, (None, None, 0))(walker_up, walker_dn, RotMatrix)
        ovlp = vmap(self._calc_overlap_generalized, (0, None))(S2walkers, wave_data)
        totalOvlp = jnp.sum(ovlp * w_betas)
        return totalOvlp

    def _calc_overlap_tr(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        overlap_1 = self._calc_overlap_unrestricted(walker_up, walker_dn, wave_data)
        overlap_2 = self._calc_overlap_unrestricted(walker_dn, walker_up, wave_data)
        return overlap_1 + overlap_2

    @calc_overlap.register
    def _(self, walkers: RHFWalkers, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_overlap_restricted, self.n_chunks, wave_data
        )

    @calc_overlap.register
    def _(self, walkers: GHFWalkers, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_overlap_generalized, self.n_chunks, wave_data
        )

    def _calc_overlap_generalized_handler(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        if self.projector is not None:
            raise NotImplementedError(
                "Symmetry projectors are not implemented for generalized walkers."
            )
        else:
            return self._calc_overlap_generalized(walker, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Overlap for a single restricted walker."""
        return self._calc_overlap_unrestricted(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Overlap for a single unrestricted walker."""
        raise NotImplementedError("Overlap not defined")

    @partial(jit, static_argnums=0)
    def _calc_overlap_generalized(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Overlap for a single generalized walker."""
        raise NotImplementedError("Overlap not defined")

    @singledispatchmethod
    def calc_force_bias(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the force bias < psi_T | chol | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The force bias.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_force_bias.register
    def _(self, walkers: UHFWalkers, ham_data: dict, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_force_bias_unrestricted, self.n_chunks, ham_data, wave_data
        )

    @calc_force_bias.register
    def _(self, walkers: RHFWalkers, ham_data: dict, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_force_bias_restricted, self.n_chunks, ham_data, wave_data
        )

    @calc_force_bias.register
    def _(self, walkers: GHFWalkers, ham_data: dict, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_force_bias_generalized, self.n_chunks, ham_data, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Force bias for a single restricted walker."""
        return self._calc_force_bias_unrestricted(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_force_bias_generalized(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Force bias for a single walker."""
        raise NotImplementedError("Force bias not defined")

    @partial(jit, static_argnums=0)
    def _calc_force_bias_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Force bias for a single walker."""
        raise NotImplementedError("Force bias not defined")

    @singledispatchmethod
    def calc_energy(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the energy < psi_T | H | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The energy.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_energy.register
    def _(self, walkers: UHFWalkers, ham_data: dict, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_energy_unrestricted_handler, self.n_chunks, ham_data, wave_data
        )

    def _calc_energy_unrestricted_handler(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        if self.projector == "s2":
            return self._calc_energy_s2(walker_up, walker_dn, ham_data, wave_data)
        elif self.projector == "tr" and self.nelec[0] == self.nelec[1]:
            return self._calc_energy_tr(walker_up, walker_dn, ham_data, wave_data)
        else:
            return self._calc_energy_unrestricted(
                walker_up, walker_dn, ham_data, wave_data
            )

    def _calc_energy_tr(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        energy_1 = self._calc_energy_unrestricted(
            walker_up, walker_dn, ham_data, wave_data
        )
        energy_2 = self._calc_energy_unrestricted(
            walker_dn, walker_up, ham_data, wave_data
        )
        overlap_1 = self._calc_overlap_unrestricted(walker_up, walker_dn, wave_data)
        overlap_2 = self._calc_overlap_unrestricted(walker_dn, walker_up, wave_data)
        return (energy_1 * overlap_1 + energy_2 * overlap_2) / (overlap_1 + overlap_2)

    def _calc_energy_s2(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        # assume s = Sz = walkers[0].shape[1] - walkers[1].shape[1]
        _, _, w_betas, betas = wave_data["betas"]

        RotMatrix = vmap(
            lambda beta: jnp.array(
                [
                    [jnp.cos(beta / 2), jnp.sin(beta / 2)],
                    [-jnp.sin(beta / 2), jnp.cos(beta / 2)],
                ]
            )
        )(betas)

        def applyRotMat(detA, detB, mat):
            A, B = detA * mat[0, 0], detB * mat[0, 1]
            C, D = detA * mat[1, 0], detB * mat[1, 1]

            detAout = jnp.block([[A, B], [C, D]])
            return detAout

        S2walkers = vmap(applyRotMat, (None, None, 0))(walker_up, walker_dn, RotMatrix)
        ovlp = vmap(self._calc_overlap_generalized, (0, None))(S2walkers, wave_data)
        Eloc = vmap(self._calc_energy_generalized, (0, None, None))(
            S2walkers, ham_data, wave_data
        )
        totalOvlp = jnp.sum(ovlp * w_betas)
        return jnp.sum(Eloc * ovlp * w_betas) / totalOvlp

    @calc_energy.register
    def _(self, walkers: RHFWalkers, ham_data: dict, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_energy_restricted, self.n_chunks, ham_data, wave_data
        )

    @calc_energy.register
    def _(self, walkers: GHFWalkers, ham_data: dict, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_energy_generalized, self.n_chunks, ham_data, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Energy for a single restricted walker."""
        return self._calc_energy_unrestricted(
            walker[:, : self.nelec[0]], walker[:, : self.nelec[1]], ham_data, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Energy for a single walker."""
        raise NotImplementedError("Energy not defined")

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self,
        walker: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Energy for a single walker."""
        raise NotImplementedError("Energy not defined")

    def get_rdm1(self, wave_data: dict) -> jax.Array:
        """Returns the one-body spin reduced density matrix of the trial.
        Used for calculating mean-field shift and as a default value in cases of large
        deviations in observable samples. If wave_data contains "rdm1" this value is used,
        calls otherwise _calc_rdm1.

        Args:
            wave_data : The trial wave function data.

        Returns:
            rdm1: The one-body spin reduced density matrix (2, norb, norb).
        """
        if "rdm1" in wave_data:
            return jnp.array(wave_data["rdm1"])
        else:
            return self._calc_rdm1(wave_data)

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        """Calculate the one-body spin reduced density matrix. Exact or approximate rdm1
        of the trial state.

        Args:
            wave_data : The trial wave function data.

        Returns:
            rdm1: The one-body spin reduced density matrix (2, norb, norb).
        """
        raise NotImplementedError(
            "One-body spin RDM not found in wave_data and not implemented for this trial."
        )

    def get_init_walkers(
        self, wave_data: dict, n_walkers: int, walker_type: str
    ) -> walker_batch:
        """Get the initial walkers. Uses the rdm1 natural orbitals.

        Args:
            wave_data: The trial wave function data.
            n_walkers: The number of walkers.
            restricted: Whether the walkers should be restricted.

        Returns:
            walkers: The initial walkers.
                If restricted, a single jax.Array of shape (nwalkers, norb, nelec[0]).
                If unrestricted, a list of two jax.Arrays each of shape (nwalkers, norb, nelec[sigma]).
                If generalized, a single jax.Array of shape (nwalkers, norb, nelec[0] + nelec[1]).
        """
        rdm1 = self.get_rdm1(wave_data)
        natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, : self.nelec[0]]
        natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, : self.nelec[1]]

        if walker_type == "restricted":
            rdm1_avg = rdm1[0] + rdm1[1]
            natorbs = jnp.linalg.eigh(rdm1_avg)[1][:, ::-1]
            return RHFWalkers(
                jnp.array([natorbs[:, : self.nelec[0]] + 0.0j] * n_walkers)
            )
        elif walker_type == "unrestricted":
            return UHFWalkers(
                [
                    jnp.array([natorbs_up + 0.0j] * n_walkers),
                    jnp.array([natorbs_dn + 0.0j] * n_walkers),
                ]
            )
        elif walker_type == "generalized":
            natorbs = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][
                :, : self.nelec[0] + self.nelec[1]
            ]
            return GHFWalkers(jnp.array([natorbs + 0.0j] * n_walkers))
        else:
            raise Exception("Unknown walker_type.")

    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Build intermediates for measurements in ham_data. This method is called by the hamiltonian class.

        Args:
            ham_data: The hamiltonian data.
            wave_data: The trial wave function data.

        Returns:
            ham_data: The updated Hamiltonian data.
        """
        return ham_data

    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
        """Optimize the wave function parameters.

        Args:
            ham_data: The hamiltonian data.
            wave_data: The trial wave function data.

        Returns:
            wave_data: The updated trial wave function data.
        """
        return wave_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


class wave_function_cpmc(wave_function):
    """This is used in CPMC. Not as well tested and supported as ab initio currently."""

    norb: int
    nelec: Tuple[int, int]
    n_chunks: int = 1
    projector: Optional[str] = None

    def _calc_overlap_unrestricted_handler(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        if self.projector == "s2_ghf":
            return self._calc_overlap_s2_ghf(walker_up, walker_dn, wave_data)
        elif self.projector == "ext":
            return self._calc_overlap_ext(walker_up, walker_dn, wave_data)
        elif self.projector == "ext_s2_ghf":
            return self._calc_overlap_ext_s2_ghf(walker_up, walker_dn, wave_data)
        else:
            return self._calc_overlap_unrestricted(walker_up, walker_dn, wave_data)

    def _calc_overlap_s2_ghf(self, walker_up, walker_dn, wave_data):
        """
        Singlet projection of a UHF determinant via full beta and alpha integration
        """
        beta_vals, w_beta = wave_data["beta"]
        alpha_vals, w_alpha = wave_data["alpha"]

        def Ry(beta):
            c, s = jnp.cos(beta / 2.0), jnp.sin(beta / 2.0)
            return jnp.array([[c, -s], [s, c]])

        def rotate_and_phase(beta, alpha):
            u = Ry(beta)
            a, b = u[0, 0] * walker_up, u[0, 1] * walker_dn
            c, d = u[1, 0] * walker_up, u[1, 1] * walker_dn
            ket = jnp.block([[a, b], [c, d]])

            norb = walker_up.shape[0]
            phase_up = jnp.exp(-0.5j * alpha)
            phase_dn = jnp.exp(+0.5j * alpha)
            phases = jnp.concatenate(
                [
                    jnp.full((norb,), phase_up),
                    jnp.full((norb,), phase_dn),
                ]
            )
            ket = ket * phases[:, None]
            return ket

        def ovlp_one(beta, alpha):
            ket = rotate_and_phase(beta, alpha)
            return self._calc_overlap_generalized(ket, wave_data)

        ovlp_alpha = vmap(lambda b: vmap(lambda a: ovlp_one(b, a))(alpha_vals))(
            beta_vals
        )
        total = jnp.sum(w_beta[:, None] * ovlp_alpha) * w_alpha
        return total.real

    def _calc_overlap_ext_s2_ghf(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        # External projectors.
        ops = wave_data["ext_ops"]
        ovlps = vmap(
            lambda op: self._calc_overlap_s2_ghf(
                op @ walker_up, op @ walker_dn, wave_data
            )
        )(ops)
        chars = wave_data["ext_chars"]
        ovlps = ovlps * chars
        return jnp.sum(ovlps)

    def _calc_overlap_ext(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        ops = wave_data["ext_ops"]
        ovlps = vmap(
            lambda op: self._calc_overlap_unrestricted(
                op @ walker_up, op @ walker_dn, wave_data
            )
        )(ops)
        chars = wave_data["ext_chars"]
        ovlps = ovlps * chars
        return jnp.sum(ovlps)

    def _calc_energy_unrestricted_handler(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        if self.projector == "s2_ghf":
            return self._calc_energy_s2_ghf(walker_up, walker_dn, ham_data, wave_data)
        elif self.projector == "ext":
            return self._calc_energy_ext(walker_up, walker_dn, ham_data, wave_data)
        elif self.projector == "ext_s2_ghf":
            return self._calc_energy_ext_s2_ghf(
                walker_up, walker_dn, ham_data, wave_data
            )
        else:
            return self._calc_energy_unrestricted(
                walker_up, walker_dn, ham_data, wave_data
            )

    def _calc_energy_ext(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        ops = wave_data["ext_ops"]
        ovlps = vmap(
            lambda op: self._calc_overlap_unrestricted(
                op @ walker_up, op @ walker_dn, wave_data
            )
        )(ops)
        chars = wave_data["ext_chars"]
        ovlps = ovlps * chars
        energies = vmap(
            lambda op: self._calc_energy_unrestricted(
                op @ walker_up, op @ walker_dn, ham_data, wave_data
            )
        )(ops)
        num = jnp.sum(energies * ovlps)
        denom = jnp.sum(ovlps)
        return num / denom

    def _calc_energy_ext_s2_ghf(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        # assume s = Sz = walkers[0].shape[1] - walkers[1].shape[1]
        ops = wave_data["ext_ops"]
        ovlps = vmap(
            lambda op: self._calc_overlap_s2_ghf(
                op @ walker_up, op @ walker_dn, wave_data
            )
        )(ops)
        chars = wave_data["ext_chars"]
        ovlps = ovlps * chars
        energies = vmap(
            lambda op: self._calc_energy_s2_ghf(
                op @ walker_up, op @ walker_dn, ham_data, wave_data
            )
        )(ops)
        num = jnp.sum(energies * ovlps)
        denom = jnp.sum(ovlps)
        return num / denom

    def _calc_energy_s2_ghf(self, walker_up, walker_dn, ham_data, wave_data):
        """
        Singlet-projected local energy of a UHF determinant via full beta and alpha integration
        """
        beta_vals, w_beta = wave_data["beta"]
        alpha_vals, w_alpha = wave_data["alpha"]

        def Ry(beta):
            c, s = jnp.cos(beta / 2.0), jnp.sin(beta / 2.0)
            return jnp.array([[c, -s], [s, c]], dtype=walker_up.dtype)

        def rotate_and_phase(beta, alpha):
            u = Ry(beta)
            a, b = u[0, 0] * walker_up, u[0, 1] * walker_dn
            c, d = u[1, 0] * walker_up, u[1, 1] * walker_dn
            ket = jnp.block([[a, b], [c, d]])

            norb = walker_up.shape[0]
            phase_up = jnp.exp(-0.5j * alpha)
            phase_dn = jnp.exp(+0.5j * alpha)
            phases = jnp.concatenate(
                [
                    jnp.full((norb,), phase_up),
                    jnp.full((norb,), phase_dn),
                ]
            )
            return ket * phases[:, None]

        def eval_one(beta, alpha):
            ket = rotate_and_phase(beta, alpha)
            ovlp = self._calc_overlap_generalized(ket, wave_data)
            Eloc = self._calc_energy_generalized(ket, ham_data, wave_data)
            return Eloc * ovlp, ovlp

        num_alpha, den_alpha = vmap(
            lambda b: vmap(lambda a: eval_one(b, a))(alpha_vals)
        )(beta_vals)
        num = jnp.sum(w_beta[:, None] * num_alpha).real * w_alpha
        den = jnp.sum(w_beta[:, None] * den_alpha).real * w_alpha
        return num / den

    @singledispatchmethod
    def calc_green_diagonal(self, walkers, wave_data: dict) -> jax.Array:
        """Calculate the diagonal elements of the Green's function.

        Args:
            walkers: The walkers. (mapped over)
            wave_data: The trial wave function data.

        Returns:
            diag_green: The diagonal elements of the Green's function.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_green_diagonal.register
    def _(self, walkers: list, wave_data: dict) -> jax.Array:
        return vmap(self._calc_green_diagonal_unrestricted, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @calc_green_diagonal.register
    def _(self, walkers: jax.Array, wave_data: dict) -> jax.Array:
        return vmap(self._calc_green_diagonal_generalized, in_axes=(0, None))(
            walkers, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_green_diagonal_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Diagonal of the Green's function for a single walker."""
        raise NotImplementedError("Green's function diagonal not defined")

    @partial(jit, static_argnums=0)
    def _calc_green_diagonal_generalized(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Diagonal of the Green's function for a single walker."""
        raise NotImplementedError("Green's function diagonal not defined")

    @singledispatchmethod
    def calc_green_full(self, walkers, wave_data: dict) -> jax.Array | dict:
        """Calculate the Green's function.

        Args:
            walkers: The walkers. (mapped over)
            wave_data: The trial wave function data.

        Returns:
            green: The Green's function.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_green_full.register
    def _(self, walkers: list, wave_data: dict) -> jax.Array | dict:
        return vmap(self._calc_green_full_unrestricted, in_axes=(0, 0, None))(
            walkers[0], walkers[1], wave_data
        )

    @calc_green_full.register
    def _(self, walkers: jax.Array, wave_data: dict) -> jax.Array | dict:
        return vmap(self._calc_green_full_generalized, in_axes=(0, None))(
            walkers, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_green_full_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array | dict:
        """Full Green's function for a single walker."""
        raise NotImplementedError("Full Green's function not defined")

    @partial(jit, static_argnums=0)
    def _calc_green_full_generalized(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array | dict:
        """Full Green's function for a single walker."""
        raise NotImplementedError("Full Green's function not defined")

    def calc_overlap_ratio(
        self,
        green: jax.Array | dict,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        """Calculate the overlap ratio.

        Args:
            green: The Green's functions. (mapped over)
            update_indices: Proposed update indices.
            constants: Proposed update constants.

        Returns:
            overlap_ratios: The overlap ratios.
        """
        return vmap(self._calc_overlap_ratio, in_axes=(0, None, None))(
            green, update_indices, update_constants
        )

    @partial(jit, static_argnums=0)
    def _calc_overlap_ratio(
        self,
        green: jax.Array | dict,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        """Overlap ratio for a single walker."""
        raise NotImplementedError("Overlap ratio not defined")

    def update_green(
        self,
        green: jax.Array | dict,
        ratios: jax.Array,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        """Update the Green's function.

        Args:
            green: The old Green's functions. (mapped over)
            ratios: The overlap ratios. (mapped over)
            indices: Where to update.
            constants: What to update with. (mapped over)

        Returns:
            green: The updated Green's functions.
        """
        return vmap(self._update_green, in_axes=(0, 0, None, 0))(
            green, ratios, update_indices, update_constants
        )

    @partial(jit, static_argnums=0)
    def _update_green(
        self,
        green: jax.Array | dict,
        ratio: jax.Array,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        """Update Green's function for each single walker."""
        raise NotImplementedError("Update Green's function  not defined")

    @singledispatchmethod
    def calc_density_corr(self, walkers, wave_data: dict) -> jax.Array:
        """Calculate < psi_T | n_i n_j | walker > / < psi_T | walker > for a batch of walkers where i and j are spin orbitals.

        Currently only used with GHF trial in CPMC. The projectors have to be applied
        to the trial unlike energy.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The density correlations.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_density_corr.register
    def _(self, walkers: UHFWalkers, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_density_corr_unrestricted_handler, self.n_chunks, wave_data
        )

    def _calc_density_corr_unrestricted_handler(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        if self.projector == "s2_ghf":
            return self._calc_density_corr_s2_ghf(walker_up, walker_dn, wave_data)
        elif self.projector == "ext":
            return self._calc_density_corr_ext(walker_up, walker_dn, wave_data)
        elif self.projector == "ext_s2_ghf":
            return self._calc_density_corr_ext_s2_ghf(walker_up, walker_dn, wave_data)
        else:
            return self._calc_density_corr_unrestricted(walker_up, walker_dn, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_density_corr_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        """Density correlation for a single walker."""
        raise NotImplementedError("Density correlation not defined")

    @partial(jit, static_argnums=0)
    def _calc_density_corr_s2_ghf(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        """Density correlation for a single walker with s2 projector."""
        raise NotImplementedError("Density correlation not defined")

    @partial(jit, static_argnums=0)
    def _calc_density_corr_ext(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        """Density correlation for a single walker with ext projector."""
        raise NotImplementedError("Density correlation not defined")

    @partial(jit, static_argnums=0)
    def _calc_density_corr_ext_s2_ghf(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        """Density correlation for a single walker with ext and s2 projector."""
        raise NotImplementedError("Density correlation not defined")


@dataclass
class sum_state(wave_function):
    """Sum of multiple states. wave_data should contain the coeffs in the expansion."""

    norb: int
    nelec: Tuple[int, int]
    states: Tuple[wave_function, ...]
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        coeffs = wave_data["coeffs"]
        return jnp.sum(
            jnp.array(
                [
                    state._calc_overlap_restricted(walker, wave_data[f"{i}"])
                    * coeffs[i]
                    for i, state in enumerate(self.states)
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        coeffs = wave_data["coeffs"]
        return jnp.sum(
            jnp.array(
                [
                    state._calc_overlap_unrestricted(
                        walker_up, walker_dn, wave_data[f"{i}"]
                    )
                    * coeffs[i]
                    for i, state in enumerate(self.states)
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        coeffs = wave_data["coeffs"]
        force_biases_i = jnp.array(
            [
                state._calc_force_bias_restricted(
                    walker, ham_data[f"{i}"], wave_data[f"{i}"]
                )
                for i, state in enumerate(self.states)
            ]
        )
        overlaps_i = jnp.array(
            [
                state._calc_overlap_restricted(walker, wave_data[f"{i}"])
                for i, state in enumerate(self.states)
            ]
        )
        return jnp.einsum("ig,i->g", force_biases_i, overlaps_i * coeffs) / jnp.sum(
            overlaps_i * coeffs
        )

    @partial(jit, static_argnums=0)
    def _calc_force_bias_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        coeffs = wave_data["coeffs"]
        force_biases_i = jnp.array(
            [
                state._calc_force_bias_unrestricted(
                    walker_up, walker_dn, ham_data[f"{i}"], wave_data[f"{i}"]
                )
                for i, state in enumerate(self.states)
            ]
        )
        overlaps_i = jnp.array(
            [
                state._calc_overlap_unrestricted(
                    walker_up, walker_dn, wave_data[f"{i}"]
                )
                for i, state in enumerate(self.states)
            ]
        )
        return jnp.einsum("ig,i->g", force_biases_i, overlaps_i * coeffs) / jnp.sum(
            overlaps_i * coeffs
        )

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(self, walker_up, walker_dn, ham_data, wave_data):
        coeffs = wave_data["coeffs"]
        energies_i = jnp.array(
            [
                state._calc_energy_unrestricted(
                    walker_up, walker_dn, ham_data[f"{i}"], wave_data[f"{i}"]
                )
                for i, state in enumerate(self.states)
            ]
        )
        overlaps_i = jnp.array(
            [
                state._calc_overlap_unrestricted(
                    walker_up, walker_dn, wave_data[f"{i}"]
                )
                for i, state in enumerate(self.states)
            ]
        )
        return jnp.sum(energies_i * overlaps_i * coeffs) / jnp.sum(overlaps_i * coeffs)

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data, wave_data):
        for i, state in enumerate(self.states):
            ham_data[f"{i}"] = ham_data.copy()
            ham_data[f"{i}"] = state._build_measurement_intermediates(
                ham_data[f"{i}"], wave_data[f"{i}"]
            )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class rhf(wave_function):
    """Class for the restricted Hartree-Fock wave function.

    The corresponding wave_data should contain "mo_coeff", a jax.Array of shape (norb, nelec).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_chunks: int = 1
    projector: Optional[str] = None

    def __post_init__(self):
        assert (
            self.nelec[0] == self.nelec[1]
        ), "RHF requires equal number of up and down electrons."

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        return jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker) ** 2

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            wave_data["mo_coeff"].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_overlap_generalized(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            jnp.vstack(
                [
                    wave_data["mo_coeff"].T.conj() @ walker[: self.norb],
                    wave_data["mo_coeff"].T.conj() @ walker[self.norb :],
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Calculates the half green's function.

        Args:
            walker: The walker.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function.
        """
        return (walker.dot(jnp.linalg.inv(wave_data["mo_coeff"].T.conj() @ walker))).T

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: Sequence, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        green_walker = self._calc_green(walker, wave_data)
        fb = 2.0 * jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_force_bias_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = green_walker_up + green_walker_dn
        fb = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker, wave_data)
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc

        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker_up = self._calc_green(walker_up, wave_data)
        green_walker_dn = self._calc_green(walker_dn, wave_data)
        green_walker = [green_walker_up, green_walker_dn]
        ene1 = jnp.sum((green_walker[0] + green_walker[1]) * rot_h1)
        f_up = jnp.einsum(
            "gij,jk->gik", rot_chol, green_walker[0].T, optimize="optimal"
        )
        f_dn = jnp.einsum(
            "gij,jk->gik", rot_chol, green_walker[1].T, optimize="optimal"
        )
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (
            jnp.sum(c_up * c_up)
            + jnp.sum(c_dn * c_dn)
            + 2.0 * jnp.sum(c_up * c_dn)
            - exc_up
            - exc_dn
        ) / 2.0
        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T] * 2)
        return rdm1

    @partial(jit, static_argnums=0)
    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[0], h1.shape[0]))
        nelec = self.nelec[0]
        h1 = (h1 + h1.T) / 2.0

        def scanned_fun(carry, x):
            dm = carry
            f = jnp.einsum("gij,ik->gjk", h2, dm)
            c = vmap(jnp.trace)(f)
            vj = jnp.einsum("g,gij->ij", c, h2)
            vk = jnp.einsum("glj,gjk->lk", f, h2)
            vhf = vj - 0.5 * vk
            fock = h1 + vhf
            mo_energy, mo_coeff = linalg_utils._eigh(fock)
            idx = jnp.argmax(abs(mo_coeff.real), axis=0)
            mo_coeff = jnp.where(
                mo_coeff[idx, jnp.arange(len(mo_energy))].real < 0, -mo_coeff, mo_coeff
            )
            e_idx = jnp.argsort(mo_energy)
            nmo = mo_energy.size
            mo_occ = jnp.zeros(nmo)
            nocc = nelec
            mo_occ = mo_occ.at[e_idx[:nocc]].set(2)
            mocc = mo_coeff[:, jnp.nonzero(mo_occ, size=nocc)[0]]
            dm = (mocc * mo_occ[jnp.nonzero(mo_occ, size=nocc)[0]]).dot(mocc.T)
            return dm, mo_coeff

        dm0 = 2 * wave_data["mo_coeff"] @ wave_data["mo_coeff"].T.conj()
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)
        wave_data["mo_coeff"] = mo_coeff[-1][:, :nelec]
        return wave_data

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = jnp.einsum(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
        )
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf(wave_function):
    """Class for the unrestricted Hartree-Fock wave function.

    The corresponding wave_data contains "mo_coeff", a list of two jax.Arrays of shape (norb, nelec[sigma]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> complex:
        return jnp.linalg.det(
            wave_data["mo_coeff"][0].T.conj() @ walker_up
        ) * jnp.linalg.det(wave_data["mo_coeff"][1].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_overlap_generalized(
        self,
        walker: jax.Array,
        wave_data: dict,
    ) -> complex:
        # Atrial, Btrial = wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]
        # bra = jnp.block([[Atrial, 0*Btrial],[0*Atrial, Btrial]])
        # return jnp.linalg.det(bra.T.conj() @ walker)

        return jnp.linalg.det(
            jnp.vstack(
                [
                    wave_data["mo_coeff"][0].T.conj() @ walker[: self.norb],
                    wave_data["mo_coeff"][1].T.conj() @ walker[self.norb :],
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_green(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> list:
        """Calculates the half green's function.

        Args:
            walker_up: The walker for spin up.
            walker_dn: The walker for spin down.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function for spin up and spin down.
        """
        green_up = (
            walker_up.dot(jnp.linalg.inv(wave_data["mo_coeff"][0].T.conj() @ walker_up))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(wave_data["mo_coeff"][1].T.conj() @ walker_dn))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_force_bias_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        fb_up = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"][0], green_walker[0], optimize="optimal"
        )
        fb_dn = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"][1], green_walker[1], optimize="optimal"
        )
        return fb_up + fb_dn

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker[0] * rot_h1[0]) + jnp.sum(
            green_walker[1] * rot_h1[1]
        )
        f_up = jnp.einsum(
            "gij,jk->gik", rot_chol[0], green_walker[0].T, optimize="optimal"
        )
        f_dn = jnp.einsum(
            "gij,jk->gik", rot_chol[1], green_walker[1].T, optimize="optimal"
        )
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (
            jnp.sum(c_up * c_up)
            + jnp.sum(c_dn * c_dn)
            + 2.0 * jnp.sum(c_up * c_dn)
            - exc_up
            - exc_dn
        ) / 2.0

        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self,
        walker: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0

        Atrial, Btrial = wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]
        bra = jnp.block([[Atrial, 0 * Btrial], [0 * Atrial, Btrial]])
        gf = (walker.dot(jnp.linalg.inv(bra.T.conj() @ walker))).T

        gfA, gfB = gf[: self.nelec[0], : self.norb], gf[self.nelec[0] :, self.norb :]
        gfAB, gfBA = gf[: self.nelec[0], self.norb :], gf[self.nelec[0] :, : self.norb]

        ene1 = jnp.sum(gfA * rot_h1[0]) + jnp.sum(gfB * rot_h1[1])

        f_up = jnp.einsum("gij,jk->gik", rot_chol[0], gfA.T, optimize="optimal")
        f_dn = jnp.einsum("gij,jk->gik", rot_chol[1], gfB.T, optimize="optimal")
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        J = jnp.sum(c_up * c_up) + jnp.sum(c_dn * c_dn) + 2.0 * jnp.sum(c_up * c_dn)

        K = jnp.sum(vmap(lambda x: x * x.T)(f_up)) + jnp.sum(
            vmap(lambda x: x * x.T)(f_dn)
        )

        f_ab = jnp.einsum("gip,pj->gij", rot_chol[0], gfBA.T, optimize="optimal")
        f_ba = jnp.einsum("gip,pj->gij", rot_chol[1], gfAB.T, optimize="optimal")
        K += 2.0 * jnp.sum(vmap(lambda x, y: x * y.T)(f_ab, f_ba))

        return ene1 + (J - K) / 2.0 + h0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        dm_up = wave_data["mo_coeff"][0] @ wave_data["mo_coeff"][0].T.conj()
        dm_dn = wave_data["mo_coeff"][1] @ wave_data["mo_coeff"][1].T.conj()
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def optimize(self, ham_data: dict, wave_data: dict) -> dict:
        h1 = ham_data["h1"]
        h1 = h1.at[0].set((h1[0] + h1[0].T) / 2.0)
        h1 = h1.at[1].set((h1[1] + h1[1].T) / 2.0)
        h2 = ham_data["chol"]
        h2 = h2.reshape((h2.shape[0], h1.shape[1], h1.shape[1]))
        nelec = self.nelec

        def scanned_fun(carry, x):
            dm = carry
            f_up = jnp.einsum("gij,ik->gjk", h2, dm[0])
            c_up = vmap(jnp.trace)(f_up)
            vj_up = jnp.einsum("g,gij->ij", c_up, h2)
            vk_up = jnp.einsum("glj,gjk->lk", f_up, h2)
            f_dn = jnp.einsum("gij,ik->gjk", h2, dm[1])
            c_dn = vmap(jnp.trace)(f_dn)
            vj_dn = jnp.einsum("g,gij->ij", c_dn, h2)
            vk_dn = jnp.einsum("glj,gjk->lk", f_dn, h2)
            fock_up = h1[0] + vj_up + vj_dn - vk_up
            fock_dn = h1[1] + vj_up + vj_dn - vk_dn
            mo_energy_up, mo_coeff_up = linalg_utils._eigh(fock_up)
            mo_energy_dn, mo_coeff_dn = linalg_utils._eigh(fock_dn)

            nmo = mo_energy_up.size

            idx_up = jnp.argmax(abs(mo_coeff_up.real), axis=0)
            mo_coeff_up = jnp.where(
                mo_coeff_up[idx_up, jnp.arange(len(mo_energy_up))].real < 0,
                -mo_coeff_up,
                mo_coeff_up,
            )
            e_idx_up = jnp.argsort(mo_energy_up)
            mo_occ_up = jnp.zeros(nmo)
            nocc_up = nelec[0]
            mo_occ_up = mo_occ_up.at[e_idx_up[:nocc_up]].set(1)
            mocc_up = mo_coeff_up[:, jnp.nonzero(mo_occ_up, size=nocc_up)[0]]
            dm_up = (mocc_up * mo_occ_up[jnp.nonzero(mo_occ_up, size=nocc_up)[0]]).dot(
                mocc_up.T
            )

            idx_dn = jnp.argmax(abs(mo_coeff_dn.real), axis=0)
            mo_coeff_dn = jnp.where(
                mo_coeff_dn[idx_dn, jnp.arange(len(mo_energy_dn))].real < 0,
                -mo_coeff_dn,
                mo_coeff_dn,
            )
            e_idx_dn = jnp.argsort(mo_energy_dn)
            mo_occ_dn = jnp.zeros(nmo)
            nocc_dn = nelec[1]
            mo_occ_dn = mo_occ_dn.at[e_idx_dn[:nocc_dn]].set(1)
            mocc_dn = mo_coeff_dn[:, jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]
            dm_dn = (mocc_dn * mo_occ_dn[jnp.nonzero(mo_occ_dn, size=nocc_dn)[0]]).dot(
                mocc_dn.T
            )

            return jnp.array([dm_up, dm_dn]), jnp.array([mo_coeff_up, mo_coeff_dn])

        dm0 = self._calc_rdm1(wave_data)
        _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=self.n_opt_iter)

        wave_data["mo_coeff"] = [
            mo_coeff[-1][0][:, : nelec[0]],
            mo_coeff[-1][1][:, : nelec[1]],
        ]
        return wave_data

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = [
            wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
            wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
        ]
        ham_data["rot_chol"] = [
            jnp.einsum(
                "pi,gij->gpj",
                wave_data["mo_coeff"][0].T.conj(),
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
            jnp.einsum(
                "pi,gij->gpj",
                wave_data["mo_coeff"][1].T.conj(),
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
        ]
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf_cpmc(uhf, wave_function_cpmc):
    """Class for the unrestricted Hartree-Fock wave function for CPMC.

    The corresponding wave_data contains "mo_coeff", a list of two jax.Arrays of shape (norb, nelec[sigma]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_green_diagonal_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        green_up = (
            walker_up
            @ jnp.linalg.inv(wave_data["mo_coeff"][0].T.dot(walker_up))
            @ wave_data["mo_coeff"][0].T
        ).diagonal()
        green_dn = (
            walker_dn
            @ jnp.linalg.inv(wave_data["mo_coeff"][1].T.dot(walker_dn))
            @ wave_data["mo_coeff"][1].T
        ).diagonal()
        return jnp.array([green_up, green_dn])

    @partial(jit, static_argnums=0)
    def _calc_green_full_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        green_up = (
            walker_up
            @ jnp.linalg.inv(wave_data["mo_coeff"][0].T.dot(walker_up))
            @ wave_data["mo_coeff"][0].T
        ).T
        green_dn = (
            walker_dn
            @ jnp.linalg.inv(wave_data["mo_coeff"][1].T.dot(walker_dn))
            @ wave_data["mo_coeff"][1].T
        ).T
        return jnp.array([green_up, green_dn])

    @partial(jit, static_argnums=0)
    def _calc_overlap_ratio(
        self, green: jax.Array, update_indices: jax.Array, update_constants: jax.Array
    ) -> float:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        ratio = (1 + update_constants[0] * green[spin_i, i, i]) * (
            1 + update_constants[1] * green[spin_j, j, j]
        ) - (spin_i == spin_j) * update_constants[0] * update_constants[1] * (
            green[spin_i, i, j] * green[spin_j, j, i]
        )
        return ratio

    @partial(jit, static_argnums=0)
    def _update_green(
        self,
        green: jax.Array,
        ratio: float,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        sg_i = green[spin_i, i].at[i].add(-1)
        sg_j = green[spin_j, j].at[j].add(-1)
        g_ii = green[spin_i, i, i]
        g_jj = green[spin_j, j, j]
        g_ij = (spin_i == spin_j) * green[spin_i, i, j]
        g_ji = (spin_i == spin_j) * green[spin_j, j, i]
        green = green.at[spin_i, :, :].add(
            (update_constants[0] / ratio)
            * jnp.outer(
                green[spin_i, :, i],
                update_constants[1] * (g_ij * sg_j - g_jj * sg_i) - sg_i,
            )
        )
        green = green.at[spin_j, :, :].add(
            (update_constants[1] / ratio)
            * jnp.outer(
                green[spin_j, :, j],
                update_constants[0] * (g_ji * sg_i - g_ii * sg_j) - sg_j,
            )
        )
        green = jnp.where(jnp.isinf(green), 0.0, green)
        green = jnp.where(jnp.isnan(green), 0.0, green)
        return green

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green = self._calc_green_full_unrestricted(walker_up, walker_dn, wave_data)
        h1 = ham_data["h1"]
        u = ham_data["u"]
        energy_1 = jnp.sum(green[0] * h1[0]) + jnp.sum(green[1] * h1[1])
        energy_2 = u * jnp.sum(green[0].diagonal() * green[1].diagonal())
        return energy_1 + energy_2

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uhf_cpmc_nn(uhf_cpmc):
    """Class for the unrestricted Hartree-Fock wave function for NN-CPMC.

    The corresponding wave_data contains "mo_coeff", a list of two jax.Arrays of shape (norb, nelec[sigma]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    neighbors: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    neighbors_ij: jnp.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self.neighbors_ij = jnp.asarray(self.neighbors, dtype=jnp.int32)

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green = self._calc_green_full_unrestricted(walker_up, walker_dn, wave_data)
        h1 = ham_data["h1"]
        u = ham_data["u"]
        u_1 = ham_data["u_1"]
        energy_1 = jnp.sum(green[0] * h1[0]) + jnp.sum(green[1] * h1[1])

        # onsite interactions
        energy_2 = u * jnp.sum(green[0].diagonal() * green[1].diagonal())

        # nearest-neighbor interactions
        i = self.neighbors_ij[:, 0] 
        j = self.neighbors_ij[:, 1] 
        green_tot = green[0].diagonal() + green[1].diagonal()
        hartree = green_tot[i] * green_tot[j]
        exchange = green[0][i, j] * green[0][j, i] + green[1][i, j] * green[1][j, i]
        energy_2 += u_1 * jnp.sum(hartree - exchange)

        return energy_1 + energy_2

    def __hash__(self) -> int:
        # return hash(tuple(self.__dict__.values()))
        items = []
        for k, v in self.__dict__.items():
            # skip JAX arrays
            if hasattr(v, "shape") and type(v).__module__.startswith("jax"):
                continue

            items.append((k, v))
        
        return hash(tuple(items))


@dataclass
class ghf_complex(wave_function):
    """Class for the complex-valued generalized Hartree-Fock wave function.

    The corresponding wave_data should contain "mo_coeff", a jax.Array of shape (norb, nelec).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        return jnp.linalg.det(
            jnp.hstack(
                [
                    wave_data["mo_coeff"][: self.norb].T.conj() @ walker,
                    wave_data["mo_coeff"][self.norb :].T.conj() @ walker,
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walkerA: jax.Array, walkerB: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            jnp.hstack(
                [
                    wave_data["mo_coeff"][: self.norb].T.conj() @ walkerA,
                    wave_data["mo_coeff"][self.norb :].T.conj() @ walkerB,
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_overlap_generalized(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        # det(mo_coeff^\dagger . walker)
        return jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker)

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Calculates the half green's function.

        Args:
            walker: The walker.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function.
        """
        return (walker.dot(jnp.linalg.inv(wave_data["mo_coeff"].T.conj() @ walker))).T

    @partial(jit, static_argnums=0)
    def _calc_force_bias_generalized(
        self, walker: Sequence, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        green_walker = self._calc_green(walker, wave_data)
        fb = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker, wave_data)

        # <ghf|H_1|w><ghf|w> = Tr(green_walker.T . rot_h1)
        ene1 = jnp.sum(green_walker * rot_h1)

        # <ghf|H_2|w><ghf|w> = 0.5 \sum_\gamma Tr(rot_chol_\gamma . green_walker.T)^2
        # - Tr(rot_chol_\gamma . green_walker.T . rot_chol_\gamma . green_walker.T)
        # f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        # c = vmap(jnp.trace)(f)
        # exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        # ene2 = (jnp.sum(c * c) - exc) / 2.0

        p1 = jnp.einsum(
            "ri,gir,sj,gjs->",
            green_walker.T,
            rot_chol,
            green_walker.T,
            rot_chol,
            optimize="optimal",
        )
        p2 = jnp.einsum(
            "sk,gkr,rl,gls->",
            green_walker.T,
            rot_chol,
            green_walker.T,
            rot_chol,
            optimize="optimal",
        )
        ene2 = 0.5 * (p1 - p2)

        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        rdm1 = jnp.array([wave_data["mo_coeff"] @ wave_data["mo_coeff"].T.conj()])
        return rdm1

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        """Builds half rotated integrals for efficient force bias and energy calculations."""
        # ham_data["h1"] = (
        #    ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        # )
        # ham_data["h1"] = (
        #    ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        # )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ (
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        )
        ham_data["rot_chol"] = jnp.einsum(
            "pi,gij->gpj",
            wave_data["mo_coeff"].T.conj(),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
        )
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ghf(wave_function):
    """Class for the generalized Hartree-Fock wave function.

    The corresponding wave_data should contain "mo_coeff", a jax.Array of shape (2 * norb, nelec[0] + nelec[1]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations
    """

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        return jnp.linalg.det(
            jnp.hstack(
                [
                    wave_data["mo_coeff"][: self.norb].T.conj() @ walker,
                    wave_data["mo_coeff"][self.norb :].T.conj() @ walker,
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(
            jnp.hstack(
                [
                    wave_data["mo_coeff"][: self.norb].T.conj() @ walker_up,
                    wave_data["mo_coeff"][self.norb :].T.conj() @ walker_dn,
                ]
            )
        )

    @partial(jit, static_argnums=0)
    def _calc_overlap_generalized(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        return jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker)

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        overlap_mat = jnp.hstack(
            [
                wave_data["mo_coeff"][: self.norb].T.conj() @ walker_up,
                wave_data["mo_coeff"][self.norb :].T.conj() @ walker_dn,
            ]
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            jnp.vstack(
                [walker_up @ inv[: self.nelec[0]], walker_dn @ inv[self.nelec[0] :]]
            )
        ).T
        return green

    @partial(jit, static_argnums=0)
    def _calc_force_bias_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        fb = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_force_bias_generalized(
        self, walker: Sequence, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        green_walker = self._calc_green(walker, wave_data)
        fb = jnp.einsum(
            "gij,ij->g", ham_data["rot_chol"], green_walker, optimize="optimal"
        )
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker_up, walker_dn, wave_data)
        ene1 = jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        coul = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = (jnp.sum(coul * coul) - exc) / 2.0
        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker, wave_data)

        # <ghf|H_1|w><ghf|w> = Tr(green_walker.T . rot_h1)
        ene1 = jnp.sum(green_walker * rot_h1)

        # <ghf|H_2|w><ghf|w> = 0.5 \sum_\gamma Tr(rot_chol_\gamma . green_walker.T)^2
        # - Tr(rot_chol_\gamma . green_walker.T . rot_chol_\gamma . green_walker.T)
        # f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        # c = vmap(jnp.trace)(f)
        # exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        # ene2 = (jnp.sum(c * c) - exc) / 2.0

        p1 = jnp.einsum(
            "ri,gir,sj,gjs->",
            green_walker.T,
            rot_chol,
            green_walker.T,
            rot_chol,
            optimize="optimal",
        )
        p2 = jnp.einsum(
            "sk,gkr,rl,gls->",
            green_walker.T,
            rot_chol,
            green_walker.T,
            rot_chol,
            optimize="optimal",
        )
        ene2 = 0.5 * (p1 - p2)

        return ene2 + ene1 + ene0

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        dm = (
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]]
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
        )
        dm_up = dm[: self.norb, : self.norb]
        dm_dn = dm[self.norb :, self.norb :]
        return jnp.array([dm_up, dm_dn])

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ jnp.block(
            [
                [ham_data["h1"][0], jnp.zeros_like(ham_data["h1"][1])],
                [jnp.zeros_like(ham_data["h1"][0]), ham_data["h1"][1]],
            ]
        )
        ham_data["rot_chol"] = vmap(
            lambda x: jnp.hstack(
                [
                    wave_data["mo_coeff"].T.conj()[:, : self.norb] @ x,
                    wave_data["mo_coeff"].T.conj()[:, self.norb :] @ x,
                ]
            ),
            in_axes=(0),
        )(ham_data["chol"].reshape(-1, self.norb, self.norb))
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ghf_cpmc(ghf, wave_function_cpmc):

    norb: int
    nelec: Tuple[int, int]
    n_opt_iter: int = 30
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_green_diagonal_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)
        overlap_mat = (
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
            @ walker_ghf
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            walker_ghf
            @ inv
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
        ).diagonal()
        return jnp.array([green[: self.norb], green[self.norb :]])

    @partial(jit, static_argnums=0)
    def _calc_green_diagonal_generalized(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        overlap_mat = (
            wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj() @ walker
        )
        inv = jnp.linalg.inv(overlap_mat)
        green = (
            walker
            @ inv
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
        ).diagonal()
        return green

    @partial(jit, static_argnums=0)
    def _calc_green_full_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)
        green = (
            walker_ghf
            @ jnp.linalg.inv(
                wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
                @ walker_ghf
            )
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
        ).T
        return green

    @partial(jit, static_argnums=0)
    def _calc_green_full_generalized(
        self, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        green = (
            walker
            @ jnp.linalg.inv(
                wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
                @ walker
            )
            @ wave_data["mo_coeff"][:, : self.nelec[0] + self.nelec[1]].T.conj()
        ).T
        return green

    @partial(jit, static_argnums=0)
    def _calc_overlap_ratio(
        self, green: jax.Array, update_indices: jax.Array, update_constants: jax.Array
    ) -> jax.Array:
        """
        Method for UHF/GHF walkers.
        """
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        i = i + (spin_i == 1) * self.norb
        j = j + (spin_j == 1) * self.norb
        ratio = (1 + update_constants[0] * green[i, i]) * (
            1 + update_constants[1] * green[j, j]
        ) - update_constants[0] * update_constants[1] * (green[i, j] * green[j, i])
        return ratio

    @partial(jit, static_argnums=0)
    def _update_green(
        self,
        green: jax.Array,
        ratio: float,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        """
        Method for UHF/GHF walkers.
        """
        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]
        i = i + (spin_i == 1) * self.norb
        j = j + (spin_j == 1) * self.norb
        sg_i = green[i].at[i].add(-1)
        sg_j = green[j].at[j].add(-1)
        green += (update_constants[0] / ratio) * jnp.outer(
            green[:, i],
            update_constants[1] * (green[i, j] * sg_j - green[j, j] * sg_i) - sg_i,
        ) + (update_constants[1] / ratio) * jnp.outer(
            green[:, j],
            update_constants[0] * (green[j, i] * sg_i - green[i, i] * sg_j) - sg_j,
        )
        green = jnp.where(jnp.isinf(green), 0.0, green)
        green = jnp.where(jnp.isnan(green), 0.0, green)
        return green

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(self, walker_up, walker_dn, ham_data, wave_data):
        green = self._calc_green_full_unrestricted(walker_up, walker_dn, wave_data)
        u = ham_data["u"]
        h1 = ham_data["h1"]
        energy_1 = jnp.sum(green[: self.norb, : self.norb] * h1[0]) + jnp.sum(
            green[self.norb :, self.norb :] * h1[1]
        )
        energy_2 = u * (
            jnp.sum(
                green[: self.norb, : self.norb].diagonal()
                * green[self.norb :, self.norb :].diagonal()
            )
            - jnp.sum(
                green[: self.norb, self.norb :].diagonal()
                * green[self.norb :, : self.norb].diagonal()
            )
        )
        return energy_1 + energy_2

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        green = self._calc_green_full_generalized(walker, wave_data)
        u = ham_data["u"]
        h1 = ham_data["h1"]
        energy_1 = jnp.sum(green[: self.norb, : self.norb] * h1[0]) + jnp.sum(
            green[self.norb :, self.norb :] * h1[1]
        )
        energy_2 = u * (
            jnp.sum(
                green[: self.norb, : self.norb].diagonal()
                * green[self.norb :, self.norb :].diagonal()
            )
            - jnp.sum(
                green[: self.norb, self.norb :].diagonal()
                * green[self.norb :, : self.norb].diagonal()
            )
        )
        return energy_1 + energy_2

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        return ham_data

    @partial(jit, static_argnums=0)
    def _calc_density_corr_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        green = self._calc_green_full_unrestricted(walker_up, walker_dn, wave_data)
        green_diag = jnp.diagonal(green)
        density_corr = (
            green_diag[:, None] * green_diag[None, :]
            - green * green.T
            + jnp.diag(green_diag)
        )
        return density_corr

    @partial(jit, static_argnums=0)
    def _calc_density_corr_s2_ghf(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        rot_bras, rot_coeffs = wave_data["rot_bras"], wave_data["rot_coeffs"]

        def _calc_density_corr(bra, ket):
            green = (ket @ jnp.linalg.inv(bra.T.conj() @ ket) @ bra.T.conj()).T
            green_diag = jnp.diagonal(green)
            density_corr = (
                green_diag[:, None] * green_diag[None, :]
                - green * green.T
                + jnp.diag(green_diag)
            )
            return density_corr

        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)
        density_corrs = vmap(_calc_density_corr, in_axes=(0, None))(
            rot_bras, walker_ghf
        )

        def calc_overlap(bra, ket):
            return jnp.linalg.det(bra.T.conj() @ ket)

        overlaps = vmap(calc_overlap, in_axes=(0, None))(rot_bras, walker_ghf)
        weights = rot_coeffs.conj() * overlaps
        density_corr = jnp.sum(
            weights[:, None, None] * density_corrs, axis=0
        ) / jnp.sum(weights)
        return density_corr.real

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ghf_cpmc_nn(ghf_cpmc):
    """Class for the generalized Hartree-Fock wave function for NN-CPMC.

    The corresponding wave_data contains "mo_coeff", a list of two jax.Arrays of shape (norb, nelec[sigma]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        n_opt_iter: Number of optimization scf iterations.
    """

    neighbors: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    neighbors_ij: jnp.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self.neighbors_ij = jnp.asarray(self.neighbors, dtype=jnp.int32)

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        green = self._calc_green_full_unrestricted(walker_up, walker_dn, wave_data)
        green_uu = green[: self.norb, : self.norb]
        green_ud = green[: self.norb, self.norb :]
        green_du = green[self.norb :, : self.norb]
        green_dd = green[self.norb :, self.norb :]

        h1 = ham_data["h1"]
        u = ham_data["u"]
        u_1 = ham_data["u_1"]
        energy_1 = jnp.sum(green_uu * h1[0]) + jnp.sum(green_dd * h1[1])

        # onsite interactions
        energy_2 = u * (
            jnp.sum(green_uu.diagonal() * green_dd.diagonal())
            - jnp.sum(green_ud.diagonal() * green_du.diagonal())
        )

        # nearest-neighbor interactions
        i = self.neighbors_ij[:, 0] 
        j = self.neighbors_ij[:, 1] 
        green_tot = green_uu.diagonal() + green_dd.diagonal()
        hartree = green_tot[i] * green_tot[j]
        exchange = (
            green_uu[i, j] * green_uu[j, i] + green_ud[i, j] * green_du[j, i]
            + green_du[i, j] * green_ud[j, i] + green_dd[i, j] * green_dd[j, i]
        )
        energy_2 += u_1 * jnp.sum(hartree - exchange)

        return energy_1 + energy_2

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        green = self._calc_green_full_generalized(walker, wave_data)
        green_uu = green[: self.norb, : self.norb]
        green_ud = green[: self.norb, self.norb :]
        green_du = green[self.norb :, : self.norb]
        green_dd = green[self.norb :, self.norb :]

        h1 = ham_data["h1"]
        u = ham_data["u"]
        u_1 = ham_data["u_1"]
        energy_1 = jnp.sum(green_uu * h1[0]) + jnp.sum(green_dd * h1[1])

        # onsite interactions
        energy_2 = u * (
            jnp.sum(green_uu.diagonal() * green_dd.diagonal())
            - jnp.sum(green_ud.diagonal() * green_du.diagonal())
        )

        # nearest-neighbor interactions
        i = self.neighbors_ij[:, 0] 
        j = self.neighbors_ij[:, 1] 
        green_tot = green_uu.diagonal() + green_dd.diagonal()
        hartree = green_tot[i] * green_tot[j]
        exchange = (
            green_uu[i, j] * green_uu[j, i] + green_ud[i, j] * green_du[j, i]
            + green_du[i, j] * green_ud[j, i] + green_dd[i, j] * green_dd[j, i]
        )
        energy_2 += u_1 * jnp.sum(hartree - exchange)

        return energy_1 + energy_2


    def __hash__(self) -> int:
        # return hash(tuple(self.__dict__.values()))
        items = []
        for k, v in self.__dict__.items():
            # skip JAX arrays
            if hasattr(v, "shape") and type(v).__module__.startswith("jax"):
                continue

            items.append((k, v))
        
        return hash(tuple(items))


@dataclass
class multi_ghf_cpmc(wave_function_cpmc):
    """Multi-GHF trial for CPMC, e.g. S^2/point-group projected GHF.

    wave_data must contain:
        "ci_coeffs": jax.Array of shape (ndets,)
        "mo_coeffs": jax.Array of shape (ndets, 2*norb, nelec_tot)
    """

    norb: int
    nelec: Tuple[int, int]
    n_chunks: int = 1
    projector: Optional[str] = None
    green_real_dtype: DTypeLike = jnp.float32
    green_complex_dtype: DTypeLike = jnp.complex64

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        ci_coeffs = wave_data["ci_coeffs"]
        mo_coeffs = wave_data["mo_coeffs"]

        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)

        def per_det(Ck, ck):
            overlap_mat = Ck.T.conj() @ walker_ghf
            Ok = jnp.linalg.det(overlap_mat)
            return ck * Ok

        contributions = vmap(per_det, in_axes=(0, 0))(mo_coeffs, ci_coeffs)
        return jnp.sum(contributions).real

    @partial(jit, static_argnums=0)
    def _calc_green_full_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: dict,
    ) -> dict:
        """Full per-determinant greens + weights for a single UHF walker."""
        ci_coeffs = wave_data["ci_coeffs"].astype(self.green_complex_dtype)
        mo_coeffs = wave_data["mo_coeffs"].astype(self.green_complex_dtype)

        nelec_tot = self.nelec[0] + self.nelec[1]
        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn).astype(
            self.green_real_dtype
        )

        def per_det(Ck, ck):
            Cocc = Ck[:, :nelec_tot]
            overlap_mat = Cocc.T.conj() @ walker_ghf
            inv = jnp.linalg.inv(overlap_mat)
            Gk = (walker_ghf @ inv @ Cocc.T.conj()).T
            Ok = jnp.linalg.det(overlap_mat)
            wk = ck * Ok
            return Gk, wk

        Gk, wk = vmap(per_det, in_axes=(0, 0))(mo_coeffs, ci_coeffs)
        greens = {"G": Gk, "w": wk}
        return greens

    @partial(jit, static_argnums=0)
    def _calc_overlap_ratio(
        self,
        greens: dict,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> jax.Array:
        """Overlap ratio R = <Psi_T | phi'> / <Psi_T | phi> for a single walker."""
        G_states = greens["G"]
        w_states = greens["w"]

        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]

        i_eff = i + (spin_i == 1) * self.norb
        j_eff = j + (spin_j == 1) * self.norb

        u0, u1 = update_constants[0].astype(self.green_real_dtype), update_constants[
            1
        ].astype(self.green_real_dtype)

        def ratio_one(G):
            return (1.0 + u0 * G[i_eff, i_eff]) * (1.0 + u1 * G[j_eff, j_eff]) - (
                u0 * u1 * G[i_eff, j_eff] * G[j_eff, i_eff]
            )

        r_k = vmap(ratio_one, in_axes=0)(G_states)

        W_old = jnp.sum(w_states).real
        W_new = jnp.sum(w_states * r_k).real

        ratio = jnp.where(
            jnp.abs(W_old) < 1.0e-16,
            0.0 + 0.0j,
            W_new / W_old,
        )

        return ratio.real

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """
        Local energy for a single UHF walker.
        """
        ci_coeffs = wave_data["ci_coeffs"]
        mo_coeffs = wave_data["mo_coeffs"]

        norb = self.norb
        nelec_tot = self.nelec[0] + self.nelec[1]

        u = ham_data["u"]
        h1 = ham_data["h1"]

        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)

        def per_det(Ck, ck):
            """
            Compute (E_k, w_k) for a single determinant k.
            """
            Cocc = Ck[:, :nelec_tot]
            overlap_mat = Cocc.T.conj() @ walker_ghf
            inv = jnp.linalg.inv(overlap_mat)

            Gk = (walker_ghf @ inv @ Cocc.T.conj()).T

            G_up = Gk[:norb, :norb]
            G_dn = Gk[norb:, norb:]

            energy_1 = jnp.sum(G_up * h1[0]) + jnp.sum(G_dn * h1[1])

            G_up_diag = jnp.diagonal(G_up)
            G_dn_diag = jnp.diagonal(G_dn)

            G_ud = Gk[:norb, norb:]
            G_du = Gk[norb:, :norb]

            energy_2 = u * (
                jnp.sum(G_up_diag * G_dn_diag)
                - jnp.sum(jnp.diagonal(G_ud) * jnp.diagonal(G_du))
            )

            E_k = energy_1 + energy_2

            Ok = jnp.linalg.det(overlap_mat)
            w_k = ck * Ok

            return E_k, w_k

        E_k, w_k = vmap(per_det, in_axes=(0, 0))(mo_coeffs, ci_coeffs)

        num = jnp.sum(w_k * E_k).real
        den = jnp.sum(w_k).real

        E_L = jnp.where(
            jnp.abs(den) < 1.0e-16,
            0.0,
            num / den,
        )

        return E_L

    @partial(jit, static_argnums=0)
    def _calc_density_corr_unrestricted(self, walker_up, walker_dn, wave_data):
        """
        Density correlation for a single UHF walker.
        """
        ci_coeffs = wave_data["ci_coeffs"]
        mo_coeffs = wave_data["mo_coeffs"]

        norb = self.norb
        nelec_tot = self.nelec[0] + self.nelec[1]

        walker_ghf = jsp.linalg.block_diag(walker_up, walker_dn)

        def per_det(Ck, ck):
            Cocc = Ck[:, :nelec_tot]
            overlap_mat = Cocc.T.conj() @ walker_ghf
            inv = jnp.linalg.inv(overlap_mat)
            Gk = (walker_ghf @ inv @ Cocc.T.conj()).T
            Gk_diag = jnp.diagonal(Gk)
            density_corr = (
                Gk_diag[:, None] * Gk_diag[None, :] - Gk * Gk.T + jnp.diag(Gk_diag)
            )
            Ok = jnp.linalg.det(overlap_mat)
            wk = ck * Ok
            return density_corr, wk

        density_corrs, w_k = vmap(per_det, in_axes=(0, 0))(mo_coeffs, ci_coeffs)

        num = jnp.sum(w_k[:, None, None] * density_corrs, axis=0).real
        den = jnp.sum(w_k).real

        density_corr = jnp.where(
            jnp.abs(den) < 1.0e-16,
            jnp.zeros((2 * norb, 2 * norb)),
            num / den,
        )

        return density_corr

    @partial(jit, static_argnums=0)
    def _update_green(
        self,
        greens: dict,
        ratio: jax.Array,
        update_indices: jax.Array,
        update_constants: jax.Array,
    ) -> dict:
        """
        Fast update of per-determinant greens and weights for a single walker.
        """
        G_states = greens["G"]  # (ndets, 2*norb, 2*norb)
        w_states = greens["w"]  # (ndets,)

        spin_i, i = update_indices[0]
        spin_j, j = update_indices[1]

        i_eff = i + (spin_i == 1) * self.norb
        j_eff = j + (spin_j == 1) * self.norb

        u0, u1 = update_constants[0].astype(self.green_real_dtype), update_constants[
            1
        ].astype(self.green_real_dtype)

        def ratio_one(G):
            return (1.0 + u0 * G[i_eff, i_eff]) * (1.0 + u1 * G[j_eff, j_eff]) - (
                u0 * u1 * G[i_eff, j_eff] * G[j_eff, i_eff]
            )

        r_k = vmap(ratio_one, in_axes=0)(G_states)  # (ndets,)

        def update_one(G, r):
            complex_one = jnp.array(1.0 + 0.0j, dtype=self.green_complex_dtype)
            complex_zero = jnp.array(0.0 + 0.0j, dtype=self.green_complex_dtype)
            real_one = jnp.array(1.0, dtype=self.green_real_dtype)

            r_safe = jnp.array(jnp.where(jnp.abs(r) < 1.0e-8, complex_one, r))

            sg_i = G[i_eff].at[i_eff].add(-real_one)
            sg_j = G[j_eff].at[j_eff].add(-real_one)

            G_new = (
                G
                + (u0 / r_safe)
                * jnp.outer(
                    G[:, i_eff],
                    u1 * (G[i_eff, j_eff] * sg_j - G[j_eff, j_eff] * sg_i) - sg_i,
                )
                + (u1 / r_safe)
                * jnp.outer(
                    G[:, j_eff],
                    u0 * (G[j_eff, i_eff] * sg_i - G[i_eff, i_eff] * sg_j) - sg_j,
                )
            )

            G_new = jnp.array(jnp.where(jnp.isinf(G_new), complex_zero, G_new))
            G_new = jnp.array(jnp.where(jnp.isnan(G_new), complex_zero, G_new))
            return G_new

        G_states_new = vmap(update_one, in_axes=(0, 0))(G_states, r_k)
        w_states_new = w_states * r_k

        return {"G": G_states_new, "w": w_states_new}

    def prepare_wave_data_symm(
        self,
        wave_data: dict,
        ham_data: dict | None = None,
        test_walker: tuple[jax.Array, jax.Array] | None = None,
        energy_tol: float = 1.0e-3,
        tol_same: float = 1.0e-5,
        auto_grid: bool = False,
        k_projection: bool = False,
    ) -> dict:
        """
        Prepare symmetry-projected multi-GHF expansion.
        """

        mo0_jax = jnp.array(wave_data["mo_coeff"])
        norb = mo0_jax.shape[0] // 2
        nelec_tot = mo0_jax.shape[1]

        def _apply_pg_to_ket(ket: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
            """U acts in orbital space on both spin blocks."""
            up = U @ ket[:norb, :]
            dn = U @ ket[norb:, :]
            return jnp.vstack([up, dn])

        if "ext_ops" in wave_data:
            ext_ops = jnp.array(wave_data["ext_ops"])
            ext_chars = jnp.array(wave_data["ext_chars"])
            print(f"Found point-group projection with {ext_ops.shape[0]} operations.")

            rotated_all = vmap(lambda U: _apply_pg_to_ket(mo0_jax, U))(ext_ops)

            rotated_np = np.asarray(rotated_all)
            ext_chars_np = np.asarray(ext_chars)
            n_pg = rotated_np.shape[0]

            unique_mos = []
            unique_coeffs = []

            for k in range(n_pg):
                Ck = rotated_np[k]
                chi_k = np.conj(ext_chars_np[k])

                if not unique_mos:
                    unique_mos.append(Ck)
                    unique_coeffs.append(chi_k)
                    continue

                matched = False
                for m, Cref in enumerate(unique_mos):
                    S = Cref.conj().T @ Ck
                    detS = np.linalg.det(S)

                    if abs(abs(detS) - 1.0) < tol_same:
                        unique_coeffs[m] += chi_k * detS
                        matched = True
                        break

                if not matched:
                    unique_mos.append(Ck)
                    unique_coeffs.append(chi_k)

            mo_pg = jnp.asarray(unique_mos)
            ci_pg = jnp.asarray(unique_coeffs) / n_pg

            mask = jnp.abs(ci_pg) > tol_same
            mo_pg = mo_pg[mask]
            ci_pg = ci_pg[mask]

            print(
                f"PG projection: {len(ci_pg)} unique determinants from {n_pg} operations."
            )

        else:
            mo_pg = mo0_jax[jnp.newaxis, ...]
            ci_pg = jnp.array([1.0 + 0.0j])

        has_alpha = "alpha" in wave_data
        has_beta = "beta" in wave_data

        if test_walker is None and "rdm1" in wave_data:
            rdm1_up, rdm1_dn = wave_data["rdm1"]

            evals_u, evecs_u = jnp.linalg.eigh(rdm1_up)
            evals_d, evecs_d = jnp.linalg.eigh(rdm1_dn)
            evecs_u = evecs_u[:, ::-1]
            evecs_d = evecs_d[:, ::-1]

            n_up, n_dn = self.nelec
            walker_up = evecs_u[:, :n_up]
            walker_dn = evecs_d[:, :n_dn]
            test_walker = (walker_up, walker_dn)

        candidate_grids = [
            (3, 4),
            (4, 4),
            (5, 4),
            (5, 6),
            (6, 6),
            (6, 8),
            (8, 8),
            (10, 10),
        ]

        def _make_alpha(n_alpha: int):
            alpha_vals = jnp.pi * (jnp.arange(n_alpha) + 0.5) / n_alpha
            w_alpha = 1.0 / n_alpha
            return alpha_vals, w_alpha

        from numpy.polynomial.legendre import leggauss

        def _make_beta(n_beta: int):
            x, w = leggauss(int(n_beta))
            beta = np.arccos(x)
            order = np.argsort(beta)
            beta_vals = jnp.asarray(beta[order])
            w_beta = jnp.asarray(w[order])
            return beta_vals, w_beta

        def _build_pg_s2_expansion(alpha_vals, w_alpha, beta_vals, w_beta):
            def rotate_trial(beta, alpha, mo_coeff):
                c, s = jnp.cos(beta / 2.0), jnp.sin(beta / 2.0)
                u_rot = jnp.array([[c, s], [-s, c]])

                phase_up = jnp.exp(+0.5j * alpha)
                phase_dn = jnp.exp(-0.5j * alpha)

                C_up = mo_coeff[:norb, :]
                C_dn = mo_coeff[norb:, :]

                C_up_p = phase_up * C_up
                C_dn_p = phase_dn * C_dn

                a = u_rot[0, 0] * C_up_p + u_rot[0, 1] * C_dn_p
                b = u_rot[1, 0] * C_up_p + u_rot[1, 1] * C_dn_p

                return jnp.concatenate([a, b], axis=0)  # (2*norb, nelec_tot)

            mo_coeffs_grid = vmap(
                lambda b: vmap(lambda a: vmap(lambda C: rotate_trial(b, a, C))(mo_pg))(
                    alpha_vals
                )
            )(beta_vals)

            mo_coeffs = mo_coeffs_grid.reshape(
                -1, mo_coeffs_grid.shape[-2], mo_coeffs_grid.shape[-1]
            )

            ci_full = jnp.einsum(
                "i,j,k->ijk",
                w_beta,
                jnp.full(alpha_vals.shape, w_alpha),
                ci_pg,
            ).reshape(-1)

            wd_local = dict(wave_data)
            wd_local["alpha"] = (alpha_vals, w_alpha)
            wd_local["beta"] = (beta_vals, w_beta)
            wd_local["mo_coeffs"] = mo_coeffs
            wd_local["ci_coeffs"] = ci_full
            return wd_local

        def _apply_k_projection_to_expansion(mo_coeffs, ci_coeffs):
            mo_conj = jnp.conj(mo_coeffs)
            ci_conj = jnp.conj(ci_coeffs)
            mo_all = jnp.concatenate([mo_coeffs, mo_conj], axis=0)
            ci_all = jnp.concatenate([ci_coeffs, ci_conj], axis=0)
            return mo_all, ci_all

        if (
            ((not (has_alpha and has_beta)) or auto_grid)
            and (ham_data is not None)
            and (test_walker is not None)
        ):
            print("Auto-selecting (alpha, beta) grid via energy convergence...")
            walker_up, walker_dn = test_walker

            Es = []
            grids = []
            for n_alpha, n_beta in candidate_grids:
                alpha_vals, w_alpha = _make_alpha(n_alpha)
                beta_vals, w_beta = _make_beta(n_beta)

                wd_tmp = _build_pg_s2_expansion(alpha_vals, w_alpha, beta_vals, w_beta)
                if k_projection:
                    mo_k, ci_k = _apply_k_projection_to_expansion(
                        wd_tmp["mo_coeffs"], wd_tmp["ci_coeffs"]
                    )
                    wd_tmp = dict(wd_tmp)
                    wd_tmp["mo_coeffs"] = mo_k
                    wd_tmp["ci_coeffs"] = ci_k
                E = self._calc_energy_unrestricted(
                    walker_up, walker_dn, ham_data, wd_tmp
                )

                Es.append(float(E))
                grids.append((alpha_vals, w_alpha, beta_vals, w_beta))
                print(f"(n_alpha={n_alpha}, n_beta={n_beta}) -> E = {float(E):.8f}")

            E_ref = Es[-1]

            best_idx = len(Es) - 1
            for k, E_k in enumerate(Es):
                if abs(E_k - E_ref) < energy_tol:
                    best_idx = k
                    break

            alpha_vals, w_alpha, beta_vals, w_beta = grids[best_idx]
            wave_data["alpha"] = (alpha_vals, w_alpha)
            wave_data["beta"] = (beta_vals, w_beta)

            print(
                f"Chosen grid: n_alpha={alpha_vals.shape[0]}, n_beta={beta_vals.shape[0]}"
            )

        if "alpha" not in wave_data or "beta" not in wave_data:
            alpha_vals, w_alpha = _make_alpha(5)
            beta_vals, w_beta = _make_beta(8)
            wave_data["alpha"] = (alpha_vals, w_alpha)
            wave_data["beta"] = (beta_vals, w_beta)
        else:
            alpha_vals, w_alpha = wave_data["alpha"]
            beta_vals, w_beta = wave_data["beta"]

        wd_final = _build_pg_s2_expansion(alpha_vals, w_alpha, beta_vals, w_beta)

        if k_projection:
            mo_k, ci_k = _apply_k_projection_to_expansion(
                wd_final["mo_coeffs"], wd_final["ci_coeffs"]
            )
            wd_final["mo_coeffs"] = mo_k
            wd_final["ci_coeffs"] = ci_k
            print(
                f"K projection: {wd_final['ci_coeffs'].shape[0]} determinants "
                f"after adding conjugates."
            )

        wave_data["mo_coeffs"] = wd_final["mo_coeffs"]
        wave_data["ci_coeffs"] = wd_final["ci_coeffs"]

        print(
            f"Final multi-GHF expansion: {wave_data['ci_coeffs'].shape[0]} determinants."
        )

        return wave_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class noci(wave_function):
    """Class for the NOCI wave function.

    The corresponding wave_data should contain "ci_coeffs_dets", a list [ci_coeffs, dets]
    where ci_coeffs is a jax.Array of shape (ndets) and dets is a list [dets_up, dets_dn]
    with each being a jax.Array of shape (ndets, norb, nelec[sigma]), both ci_coeffs and dets
    are assumed to be real. The measurement methods make use of half-rotated integrals
    which are stored in ham_data (rot_h1 and rot_chol for each det).

    Attributes:
        norb: Number of orbitals.
        nelec: Number of electrons of each spin.
        ndets: Number of determinants in the NOCI expansion.
    """

    norb: int
    nelec: Tuple[int, int]
    ndets: int
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_overlap_single_det(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> jax.Array:
        """Calculate the overlap with a single determinant in the NOCI trial."""
        return jnp.linalg.det(
            trial_up[:, : self.nelec[0]].T.conj() @ walker_up
        ) * jnp.linalg.det(trial_dn[:, : self.nelec[1]].T.conj() @ walker_dn)

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> jax.Array:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        return jnp.sum(ci_coeffs * overlaps)

    @partial(jit, static_argnums=0)
    def _calc_green_single_det(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> List:
        """Calculate the half greens function with a single determinant in the NOCI trial."""
        green_up = (
            walker_up.dot(jnp.linalg.inv(trial_up[:, : self.nelec[0]].T.dot(walker_up)))
        ).T
        green_dn = (
            walker_dn.dot(jnp.linalg.inv(trial_dn[:, : self.nelec[1]].T.dot(walker_dn)))
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> Tuple:
        """Calculate the half greens function for the full trial."""
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        up_greens, dn_greens = vmap(
            self._calc_green_single_det, in_axes=(None, None, 0, 0)
        )(walker_up, walker_dn, dets[0], dets[1])
        return up_greens, dn_greens, overlaps

    @partial(jit, static_argnums=0)
    def _calc_force_bias_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        up_greens, dn_greens, overlaps = self._calc_green(
            walker_up, walker_dn, wave_data
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        fb_up = (
            jnp.einsum(
                "ngij,nij,n->g",
                ham_data["rot_chol"][0],
                up_greens,
                ci_coeffs * overlaps,
                optimize="optimal",
            )
            / overlap
        )
        fb_dn = (
            jnp.einsum(
                "ngij,nij,n->g",
                ham_data["rot_chol"][1],
                dn_greens,
                ci_coeffs * overlaps,
                optimize="optimal",
            )
            / overlap
        )
        return fb_up + fb_dn

    @partial(jit, static_argnums=0)
    def _calc_energy_single_det(
        self,
        h0: float,
        rot_h1_up: jax.Array,
        rot_h1_dn: jax.Array,
        rot_chol_up: jax.Array,
        rot_chol_dn: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        trial_up: jax.Array,
        trial_dn: jax.Array,
    ) -> jax.Array:
        """Calculate the energy with a single determinant in the NOCI trial."""
        ene0 = h0
        green_walker = self._calc_green_single_det(
            walker_up, walker_dn, trial_up, trial_dn
        )
        ene1 = jnp.sum(green_walker[0] * rot_h1_up) + jnp.sum(
            green_walker[1] * rot_h1_dn
        )
        f_up = jnp.einsum(
            "gij,jk->gik", rot_chol_up, green_walker[0].T, optimize="optimal"
        )
        f_dn = jnp.einsum(
            "gij,jk->gik", rot_chol_dn, green_walker[1].T, optimize="optimal"
        )
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        exc_up = jnp.sum(vmap(lambda x: x * x.T)(f_up))
        exc_dn = jnp.sum(vmap(lambda x: x * x.T)(f_dn))
        ene2 = (
            jnp.sum(c_up * c_up)
            + jnp.sum(c_dn * c_dn)
            + 2.0 * jnp.sum(c_up * c_dn)
            - exc_up
            - exc_dn
        ) / 2.0

        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0))(
            walker_up, walker_dn, dets[0], dets[1]
        )
        overlap = jnp.sum(ci_coeffs * overlaps)
        energies = vmap(
            self._calc_energy_single_det, in_axes=(None, 0, 0, 0, 0, None, None, 0, 0)
        )(
            h0,
            rot_h1[0],
            rot_h1[1],
            rot_chol[0],
            rot_chol[1],
            walker_up,
            walker_dn,
            dets[0],
            dets[1],
        )
        ene = jnp.sum(ci_coeffs * overlaps * energies) / overlap
        return ene

    @partial(jit, static_argnums=0)
    def _get_trans_rdm1_single_det(self, sd_0_up, sd_0_dn, sd_1_up, sd_1_dn) -> list:
        dm_up = (
            (sd_0_up[:, : self.nelec[0]])
            .dot(
                jnp.linalg.inv(
                    sd_1_up[:, : self.nelec[0]].T.dot(sd_0_up[:, : self.nelec[0]])
                )
            )
            .dot(sd_1_up[:, : self.nelec[0]].T)
        )
        dm_dn = (
            (sd_0_dn[:, : self.nelec[1]])
            .dot(
                jnp.linalg.inv(
                    sd_1_dn[:, : self.nelec[1]].T.dot(sd_0_dn[:, : self.nelec[1]])
                )
            )
            .dot(sd_1_dn[:, : self.nelec[1]].T)
        )
        return [dm_up, dm_dn]

    @partial(jit, static_argnums=0)
    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        ci_coeffs = wave_data["ci_coeffs_dets"][0]
        dets = wave_data["ci_coeffs_dets"][1]
        overlaps = vmap(
            vmap(self._calc_overlap_single_det, in_axes=(None, None, 0, 0)),
            in_axes=(0, 0, None, None),
        )(dets[0], dets[1], dets[0], dets[1])
        overlap = jnp.sum(jnp.outer(ci_coeffs, ci_coeffs) * overlaps)
        up_rdm1s, dn_rdm1s = vmap(
            vmap(self._get_trans_rdm1_single_det, in_axes=(0, 0, None, None)),
            in_axes=(None, None, 0, 0),
        )(dets[0], dets[1], dets[0], dets[1])
        up_rdm1 = (
            jnp.einsum(
                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, up_rdm1s
            )
            / overlap
        )
        dn_rdm1 = (
            jnp.einsum(
                "hg,hgij->ij", jnp.outer(ci_coeffs, ci_coeffs) * overlaps, dn_rdm1s
            )
            / overlap
        )
        return jnp.array([up_rdm1, dn_rdm1])

    @partial(jit, static_argnums=(0,))
    def _rot_orbs_single_det(
        self, ham_data: dict, trial_up: jax.Array, trial_dn: jax.Array
    ) -> Tuple:
        rot_h1 = [
            trial_up.T.conj() @ ham_data["h1"][0],
            trial_dn.T.conj() @ ham_data["h1"][1],
        ]
        rot_chol = [
            jnp.einsum(
                "pi,gij->gpj",
                trial_up.T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
            jnp.einsum(
                "pi,gij->gpj",
                trial_dn.T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
        ]
        return rot_h1, rot_chol

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        trial = wave_data["ci_coeffs_dets"][1]
        ham_data["rot_h1"], ham_data["rot_chol"] = vmap(
            self._rot_orbs_single_det, in_axes=(None, 0, 0)
        )(ham_data, trial[0], trial[1])
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class wave_function_auto(wave_function):
    """This wave function only requires the definition of overlap functions.
    It evaluates force bias and local energy by differentiating various overlaps
    (single derivatives with AD and double with FD)."""

    def __post_init__(self):
        """eps is the finite difference step size in local energy calculations."""
        if not hasattr(self, "eps"):
            self.eps = 1.0e-4

    @partial(jit, static_argnums=0)
    def _overlap_with_rot_sd_restricted(
        self,
        x_gamma: jax.Array,
        walker: jax.Array,
        chol: jax.Array,
        wave_data: dict,
    ) -> jax.Array:
        """Helper function for calculating force bias using AD,
        evaluates < psi_T | exp(x_gamma * chol) | walker > to linear order"""
        x_chol = jnp.einsum(
            "gij,g->ij", chol.reshape(-1, self.norb, self.norb), x_gamma
        )
        walker_1 = walker + x_chol.dot(walker)
        return self._calc_overlap_restricted(walker_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker > by differentiating
        < psi_T | exp(x_gamma * chol) | walker > / < psi_T | walker >"""
        val, grad = vjp(
            self._overlap_with_rot_sd_restricted,
            jnp.zeros((ham_data["chol"].shape[0],)) + 0.0j,
            walker,
            ham_data["chol"],
            wave_data,
        )
        return grad(1.0 + 0.0j)[0] / val

    @partial(jit, static_argnums=0)
    def _overlap_with_rot_sd(
        self,
        x_gamma: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        chol: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating force bias using AD,
        evaluates < psi_T | exp(x_gamma * chol) | walker > to linear order"""
        x_chol = jnp.einsum(
            "gij,g->ij", chol.reshape(-1, self.norb, self.norb), x_gamma
        )
        walker_up_1 = walker_up + x_chol.dot(walker_up)
        walker_dn_1 = walker_dn + x_chol.dot(walker_dn)
        return self._calc_overlap_unrestricted(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_force_bias_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: Any,
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker > by differentiating
        < psi_T | exp(x_gamma * chol) | walker > / < psi_T | walker >"""
        val, grad = vjp(
            self._overlap_with_rot_sd,
            jnp.zeros((ham_data["chol"].shape[0],)) + 0.0j,
            walker_up,
            walker_dn,
            ham_data["chol"],
            wave_data,
        )
        return grad(1.0 + 0.0j)[0] / val

    @partial(jit, static_argnums=0)
    def _overlap_with_single_rot_restricted(
        self, x: float, h1: jax.Array, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walker_2 = walker + x * h1.dot(walker)
        return self._calc_overlap_restricted(walker_2, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot_restricted(
        self, x: float, chol_i: jax.Array, walker: jax.Array, wave_data: dict
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * chol_i) | walker > to quadratic order"""
        walker2 = (
            walker
            + x * chol_i.dot(walker)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker))
        )
        return self._calc_overlap_restricted(walker2, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self,
        walker: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        """Calculates local energy using AD and finite difference for the two body term"""

        h0, h1, chol, v0 = (
            ham_data["h0"],
            (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0,
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["normal_ordering_term"],
        )

        x = 0.0
        # one body
        f1 = lambda a: self._overlap_with_single_rot_restricted(
            a, h1 + v0, walker, wave_data
        )
        overlap, d_overlap = jvp(f1, [x], [1.0])

        # two body
        eps = self.eps

        # carry: [eps, walker, wave_data]
        def scanned_fun(carry, x):
            eps, walker, wave_data = carry
            return carry, self._overlap_with_double_rot_restricted(
                eps, x, walker, wave_data
            )

        _, overlap_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, overlap_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, overlap_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

        return (d_overlap + jnp.sum(d_2_overlap) / 2.0) / overlap + h0

    @partial(jit, static_argnums=0)
    def _overlap_with_single_rot_generalized(
        self,
        x: float,
        h1: jax.Array,
        walker: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walkerout = walker + x * jnp.block(
            [
                [
                    h1[0].dot(walker[: self.norb, : self.nelec[0]]),
                    h1[0].dot(walker[: self.norb, self.nelec[0] :]),
                ],
                [
                    h1[1].dot(walker[self.norb :, : self.nelec[0]]),
                    h1[1].dot(walker[self.norb :, self.nelec[0] :]),
                ],
            ]
        )
        return self._calc_overlap_generalized(walkerout, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_single_rot(
        self,
        x: float,
        h1: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * h1) | walker > to linear order"""
        walker_up_1 = walker_up + x * h1[0].dot(walker_up)
        walker_dn_1 = walker_dn + x * h1[1].dot(walker_dn)
        return self._calc_overlap_unrestricted(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot(
        self,
        x: float,
        chol_i: jax.Array,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * chol_i) | walker > to quadratic order"""
        walker_up_1 = (
            walker_up
            + x * chol_i.dot(walker_up)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker_up))
        )
        walker_dn_1 = (
            walker_dn
            + x * chol_i.dot(walker_dn)
            + x**2 / 2.0 * chol_i.dot(chol_i.dot(walker_dn))
        )
        return self._calc_overlap_unrestricted(walker_up_1, walker_dn_1, wave_data)

    @partial(jit, static_argnums=0)
    def _overlap_with_double_rot_generalized(
        self,
        x: float,
        chol_i: jax.Array,
        walker: jax.Array,
        wave_data: Any,
    ) -> jax.Array:
        """Helper function for calculating local energy using AD,
        evaluates < psi_T | exp(x * chol_i) | walker > to quadratic order"""
        walker1 = x * jnp.block(
            [
                [
                    chol_i.dot(walker[: self.norb, : self.nelec[0]]),
                    chol_i.dot(walker[: self.norb, self.nelec[0] :]),
                ],
                [
                    chol_i.dot(walker[self.norb :, : self.nelec[0]]),
                    chol_i.dot(walker[self.norb :, self.nelec[0] :]),
                ],
            ]
        )

        walker2 = x * jnp.block(
            [
                [
                    chol_i.dot(walker1[: self.norb, : self.nelec[0]]),
                    chol_i.dot(walker1[: self.norb, self.nelec[0] :]),
                ],
                [
                    chol_i.dot(walker1[self.norb :, : self.nelec[0]]),
                    chol_i.dot(walker1[self.norb :, self.nelec[0] :]),
                ],
            ]
        )

        walker_out = walker + walker1 + walker2 / 2.0
        return self._calc_overlap_generalized(walker_out, wave_data)

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: Any,
    ) -> jax.Array:
        """Calculates local energy using AD and finite difference for the two body term"""

        h0, h1, chol, v0 = (
            ham_data["h0"],
            ham_data["h1"],
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["normal_ordering_term"],
        )

        x = 0.0
        # one body
        f1 = lambda a: self._overlap_with_single_rot(
            a, h1 + v0, walker_up, walker_dn, wave_data
        )
        val1, dx1 = jvp(f1, [x], [1.0])

        # two body
        # vmap_fun = vmap(
        #     self._overlap_with_double_rot, in_axes=(None, 0, None, None, None)
        # )

        eps = self.eps

        # carry: [eps, walker, wave_data]
        def scanned_fun(carry, chol_i):
            eps, walker_up, walker_dn, wave_data = carry
            return carry, self._overlap_with_double_rot(
                eps, chol_i, walker_up, walker_dn, wave_data
            )

        _, overlap_p = lax.scan(
            scanned_fun, (eps, walker_up, walker_dn, wave_data), chol
        )
        _, overlap_0 = lax.scan(
            scanned_fun, (0.0, walker_up, walker_dn, wave_data), chol
        )
        _, overlap_m = lax.scan(
            scanned_fun, (-1.0 * eps, walker_up, walker_dn, wave_data), chol
        )
        d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

        # dx2 = (
        #     (
        #         vmap_fun(eps, chol, walker_up, walker_dn, wave_data)
        #         - 2.0 * vmap_fun(zero, chol, walker_up, walker_dn, wave_data)
        #         + vmap_fun(-1.0 * eps, chol, walker_up, walker_dn, wave_data)
        #     )
        #     / eps
        #     / eps
        # )

        return (dx1 + jnp.sum(d_2_overlap) / 2.0) / val1 + h0

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self,
        walker: jax.Array,
        ham_data: dict,
        wave_data: Any,
    ) -> jax.Array:
        """Calculates local energy using AD and finite difference for the two body term"""

        h0, h1, chol, v0 = (
            ham_data["h0"],
            ham_data["h1"],
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["normal_ordering_term"],
        )

        x = 0.0
        # one body
        f1 = lambda a: self._overlap_with_single_rot_generalized(
            a, h1 + v0, walker, wave_data
        )
        val1, dx1 = jvp(f1, [x], [1.0])

        # two body
        # vmap_fun = vmap(
        #     self._overlap_with_double_rot, in_axes=(None, 0, None, None, None)
        # )

        eps = self.eps

        # carry: [eps, walker, wave_data]
        def scanned_fun(carry, chol_i):
            eps, walker, wave_data = carry
            return carry, self._overlap_with_double_rot_generalized(
                eps, chol_i, walker, wave_data
            )

        _, overlap_p = lax.scan(scanned_fun, (eps, walker, wave_data), chol)
        _, overlap_0 = lax.scan(scanned_fun, (0.0, walker, wave_data), chol)
        _, overlap_m = lax.scan(scanned_fun, (-1.0 * eps, walker, wave_data), chol)
        d_2_overlap = (overlap_p - 2.0 * overlap_0 + overlap_m) / eps / eps

        # dx2 = (
        #     (
        #         vmap_fun(eps, chol, walker_up, walker_dn, wave_data)
        #         - 2.0 * vmap_fun(zero, chol, walker_up, walker_dn, wave_data)
        #         + vmap_fun(-1.0 * eps, chol, walker_up, walker_dn, wave_data)
        #     )
        #     / eps
        #     / eps
        # )

        return (dx1 + jnp.sum(d_2_overlap) / 2.0) / val1 + h0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        ham_data["normal_ordering_term"] = -v0
        return ham_data


@dataclass
class multislater(wave_function_auto):
    """Multislater wave function implemented using the auto class.

    We work in the orbital basis of the wave function.
    Associated wave_data consists of excitation indices and ci coefficients:
        Acre: Alpha creation indices.
        Ades: Alpha destruction indices.
        Bcre: Beta creation indices.
        Bdes: Beta destruction indices.
        coeff: Coefficients of the determinants.
        ref_det: Reference determinant.
        orbital_rotation: Rotation matrix to transform afqmc basis walkers
        to wave function CI orbital basis.
    """

    norb: int
    nelec: Tuple[int, int]
    max_excitation: int  # maximum of sum of alpha and beta excitation ranks
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _det_overlap(
        self, green: jax.Array, cre: jax.Array, des: jax.Array
    ) -> jax.Array:
        return jnp.linalg.det(green[jnp.ix_(cre, des)])

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        ref_det = wave_data["ref_det"][0]
        return (
            walker.dot(
                jnp.linalg.inv(walker[jnp.nonzero(ref_det, size=self.nelec[0])[0], :])
            )
        ).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        """Calclulates < psi_T | walker > efficiently using Wick's theorem"""
        Acre, Ades, Bcre, Bdes, coeff, ref_det, orbital_rotation = (
            wave_data["Acre"],
            wave_data["Ades"],
            wave_data["Bcre"],
            wave_data["Bdes"],
            wave_data["coeff"],
            wave_data["ref_det"],
            wave_data["orbital_rotation"],
        )
        walker = orbital_rotation @ walker
        green = self._calc_green_restricted(walker, wave_data)

        # overlap with the reference determinant
        overlap_0 = (
            jnp.linalg.det(walker[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :])
            ** 2
        )

        # overlap / overlap_0
        overlap = coeff[(0, 0)] + 0.0j

        for i in range(1, self.max_excitation + 1):
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green, Acre[(i, 0)], Ades[(i, 0)]
            ).dot(coeff[(i, 0)])
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green, Bcre[(0, i)], Bdes[(0, i)]
            ).dot(coeff[(0, i)])

            for j in range(1, self.max_excitation - i + 1):
                overlap_a = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green, Acre[(i, j)], Ades[(i, j)]
                )
                overlap_b = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green, Bcre[(i, j)], Bdes[(i, j)]
                )
                overlap += (overlap_a * overlap_b) @ coeff[(i, j)]

        return (overlap * overlap_0)[0]

    @partial(jit, static_argnums=0)
    def _calc_green(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> list:
        ref_det = wave_data["ref_det"]
        green_up = (
            walker_up.dot(
                jnp.linalg.inv(
                    walker_up[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :]
                )
            )
        ).T
        green_dn = (
            walker_dn.dot(
                jnp.linalg.inv(
                    walker_dn[jnp.nonzero(ref_det[1], size=self.nelec[1])[0], :]
                )
            )
        ).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        """Calclulates < psi_T | walker > efficiently using Wick's theorem"""
        Acre, Ades, Bcre, Bdes, coeff, ref_det, orbital_rotation = (
            wave_data["Acre"],
            wave_data["Ades"],
            wave_data["Bcre"],
            wave_data["Bdes"],
            wave_data["coeff"],
            wave_data["ref_det"],
            wave_data["orbital_rotation"],
        )
        walker_up = orbital_rotation[0] @ walker_up
        walker_dn = orbital_rotation[1] @ walker_dn
        green = self._calc_green(walker_up, walker_dn, wave_data)

        # overlap with the reference determinant
        overlap_0 = jnp.linalg.det(
            walker_up[jnp.nonzero(ref_det[0], size=self.nelec[0])[0], :]
        ) * jnp.linalg.det(walker_dn[jnp.nonzero(ref_det[1], size=self.nelec[1])[0], :])

        # overlap / overlap_0
        overlap = coeff[(0, 0)] + 0.0j

        for i in range(1, self.max_excitation + 1):
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green[0], Acre[(i, 0)], Ades[(i, 0)]
            ).dot(coeff[(i, 0)])
            overlap += vmap(self._det_overlap, in_axes=(None, 0, 0))(
                green[1], Bcre[(0, i)], Bdes[(0, i)]
            ).dot(coeff[(0, i)])

            for j in range(1, self.max_excitation - i + 1):
                overlap_a = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green[0], Acre[(i, j)], Ades[(i, j)]
                )
                overlap_b = vmap(self._det_overlap, in_axes=(None, 0, 0))(
                    green[1], Bcre[(i, j)], Bdes[(i, j)]
                )
                overlap += (overlap_a * overlap_b) @ coeff[(i, j)]

        return (overlap * overlap_0)[0]

    def _calc_rdm1(self, wave_data: dict) -> jax.Array:
        """Spatial 1RDM of the reference det"""
        ref_det = wave_data["ref_det"]
        orb_mat = np.eye(self.norb)
        orbs_up = orb_mat[:, ref_det[0] > 0]
        orbs_dn = orb_mat[:, ref_det[1] > 0]
        rdm_up = orbs_up @ orbs_up.T
        rdm_dn = orbs_dn @ orbs_dn.T
        return jnp.array([rdm_up, rdm_dn])

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class CIS(wave_function_auto):
    """This class contains functions for the excited state CIS wavefunction c(ia) |ia>.
    The wave_data need to store the coefficient C(ia).
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1 = walker.shape[1], wave_data["ci1"]
        gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = jnp.einsum("ia,ia", ci1, gf[:, nocc:])
        return 2 * o1 * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class CISD(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = self._calc_green_restricted(walker)
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
        o2 = 2 * jnp.einsum(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]
        ) - jnp.einsum("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:])

        return (1.0 + 2 * o1 + o2) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class ccsd(wave_function):
    """This is meant to be used in free projection as the initial state.
    The wave_data need to store the coefficient T1(ia) and T2(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    nocc: int
    nvirt: int
    n_chunks: int = 1
    mixed_real_dtype: DTypeLike = jnp.float64
    mixed_complex_dtype: DTypeLike = jnp.complex128
    memory_mode: Literal["high", "low"] = "low"
    _mixed_real_dtype_testing: DTypeLike = jnp.float32
    _mixed_complex_dtype_testing: DTypeLike = jnp.complex64

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = h0
        green_walker = self._calc_green(walker, wave_data)
        ene1 = 2.0 * jnp.sum(green_walker * rot_h1)
        f = jnp.einsum("gij,jk->gik", rot_chol, green_walker.T, optimize="optimal")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        ene2 = 2.0 * jnp.sum(c * c) - exc

        return ene2 + ene1 + ene0

    @partial(jit, static_argnums=0)
    def _calc_green(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        """Calculates the half green's function.

        Args:
            walker: The walker.
            wave_data: The trial wave function data.

        Returns:
            green: The half green's function.
        """
        return (walker.dot(jnp.linalg.inv(wave_data["mo_coeff"].T.conj() @ walker))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> jax.Array:
        return jnp.linalg.det(wave_data["mo_coeff"].T.conj() @ walker) ** 2

    def hs_op(self, wave_data: dict, t2):

        nO = self.nocc
        nV = self.nvirt
        nex = nO * nV

        assert t2.shape == (nO, nO, nV, nV)

        # T2 = LL^T
        t2 = jnp.einsum("ijab->aibj", t2)
        t2 = t2.reshape(nex, nex)
        e_val, e_vec = jnp.linalg.eigh(t2)
        L = e_vec @ jnp.diag(np.sqrt(e_val + 0.0j))
        assert abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

        # Summation on the left
        L = L.T.reshape(nex, nV, nO)

        wave_data["T2_L"] = L

        return wave_data

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def get_init_walkers(
        self, wave_data: dict, n_walkers: int, walker_type: str
    ) -> walker_batch:
        nO, nV = self.nocc, self.nvirt
        n = nO + nV
        nex = nO * nV

        t1 = wave_data["t1"]

        C_occ, C_vir = jnp.split(wave_data["mo_coeff"], [nO], axis=1)

        # e^T1
        e_t1 = t1.T + 0.0j
        ops = jnp.array([e_t1] * n_walkers)

        L = wave_data["T2_L"]

        wave_data["key"], subkey = random.split(wave_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                n_walkers,
                L.shape[0],
            ),
        )

        # e^{T1+T2}
        ops = ops + jnp.einsum("wg,gai->wai", fields, L)

        # Initial walkers
        rdm1 = self.get_rdm1(wave_data)
        natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, : self.nelec[0]]

        walkers = jnp.array([natorbs_up + 0.0j] * n_walkers)
        identity = jnp.array([np.identity(n) + 0.0j] * n_walkers)

        # e^{T1+T2} \ket{\phi}
        walkers = (
            identity + jnp.einsum("pa,wai,iq -> wpq", C_vir, ops, C_occ.T)
        ) @ walkers
        walkers = jnp.array(walkers)

        walkers = RHFWalkers(walkers)
        return walkers

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class uccsd(wave_function):
    """This is meant to be used in free projection as the initial state.
    The wave_data need to store the coefficient T1(ia) and T2(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    nocc: Tuple[int, int]
    nvirt: Tuple[int, int]
    n_chunks: int = 1
    mixed_real_dtype: DTypeLike = jnp.float64
    mixed_complex_dtype: DTypeLike = jnp.complex128
    memory_mode: Literal["high", "low"] = "low"
    _mixed_real_dtype_testing: DTypeLike = jnp.float32
    _mixed_complex_dtype_testing: DTypeLike = jnp.complex64

    def hs_op(self, wave_data: dict, t2aa, t2ab, t2bb) -> dict:
        nOa, nOb = self.nocc
        nVa, nVb = self.nvirt
        n = self.norb

        # Number of excitations
        nex_a = nOa * nVa
        nex_b = nOb * nVb

        assert n == nOb + nVb
        assert t2aa.shape == (nOa, nOa, nVa, nVa)
        assert t2ab.shape == (nOa, nOb, nVa, nVb)
        assert t2bb.shape == (nOb, nOb, nVb, nVb)

        # t2(i,j,a,b) -> t2(ai,bj)
        t2aa = jnp.einsum("ijab->aibj", t2aa)
        t2ab = jnp.einsum("ijab->aibj", t2ab)
        t2bb = jnp.einsum("ijab->aibj", t2bb)

        t2aa = t2aa.reshape(nex_a, nex_a)
        t2ab = t2ab.reshape(nex_a, nex_b)
        t2bb = t2bb.reshape(nex_b, nex_b)

        # Symmetric t2 =
        # t2aa/2 t2ab
        # t2ab^T t2bb
        t2 = np.zeros((nex_a + nex_b, nex_a + nex_b))
        t2[:nex_a, :nex_a] = 0.5 * t2aa
        t2[nex_a:, :nex_a] = t2ab.T
        t2[:nex_a, nex_a:] = t2ab
        t2[nex_a:, nex_a:] = 0.5 * t2bb

        # t2 = LL^T
        e_val, e_vec = jnp.linalg.eigh(t2)
        L = e_vec @ jnp.diag(np.sqrt(e_val + 0.0j))
        assert abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

        # alpha/beta operators for HS
        # Summation on the left to have a list of operators
        La = L[:nex_a, :]
        Lb = L[nex_a:, :]
        La = La.T.reshape(nex_a + nex_b, nVa, nOa)
        Lb = Lb.T.reshape(nex_a + nex_b, nVb, nOb)

        wave_data["T2_La"] = La
        wave_data["T2_Lb"] = Lb

        return wave_data

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def get_init_walkers(
        self, wave_data: dict, n_walkers: int, walker_type: str
    ) -> walker_batch:

        nOa, nOb = self.nocc
        nVa, nVb = self.nvirt
        n = self.norb

        nex_a = nOa * nVa
        nex_b = nOb * nVb

        t1a = wave_data["t1a"]
        t1b = wave_data["t1b"]

        Ca_occ, Ca_vir = jnp.split(wave_data["mo_coeff"][0], [nOa], axis=1)
        Cb_occ, Cb_vir = jnp.split(wave_data["mo_coeff"][1], [nOb], axis=1)

        # e^T1
        e_t1a = t1a.T + 0.0j
        e_t1b = t1b.T + 0.0j

        ops_a = jnp.array([e_t1a] * n_walkers)
        ops_b = jnp.array([e_t1b] * n_walkers)

        La = wave_data["T2_La"]
        Lb = wave_data["T2_Lb"]

        wave_data["key"], subkey = random.split(wave_data["key"])
        fields = random.normal(
            subkey,
            shape=(
                n_walkers,
                La.shape[0],
            ),
        )

        # e^{T1+T2}
        ops_a = ops_a + jnp.einsum("wg,gai->wai", fields, La)
        ops_b = ops_b + jnp.einsum("wg,gai->wai", fields, Lb)

        # Initial walkers
        rdm1 = self.get_rdm1(wave_data)
        natorbs_up = jnp.linalg.eigh(rdm1[0])[1][:, ::-1][:, :nOa]
        natorbs_dn = jnp.linalg.eigh(rdm1[1])[1][:, ::-1][:, :nOb]

        walkers_a = jnp.array([natorbs_up + 0.0j] * n_walkers)
        walkers_b = jnp.array([natorbs_dn + 0.0j] * n_walkers)

        id_a = jnp.array([np.identity(n) + 0.0j] * n_walkers)
        id_b = jnp.array([np.identity(n) + 0.0j] * n_walkers)

        # e^{T1+T2} \ket{\phi}
        walkers_a = (
            id_a + jnp.einsum("pa,wai,iq -> wpq", Ca_vir, ops_a, Ca_occ.T)
        ) @ walkers_a
        walkers_b = (
            id_b + jnp.einsum("pa,wai,iq -> wpq", Cb_vir, ops_b, Cb_occ.T)
        ) @ walkers_b

        walkers_a = jnp.array(walkers_a)
        walkers_b = jnp.array(walkers_b)

        walkers = UHFWalkers([walkers_a, walkers_b])
        return walkers

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class gcisd_complex(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_green_generalized(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_generalized(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = self._calc_green_generalized(walker)
        o0 = jnp.linalg.det(walker[: walker.shape[1], :])
        o1 = jnp.einsum("ia,ia", ci1.conj(), GF[:, nocc:])
        o2 = 2.0 * jnp.einsum("iajb, ia, jb", ci2.conj(), GF[:, nocc:], GF[:, nocc:])
        o = (1.0 + o1 + 0.25 * o2) * o0
        return o

    @partial(jit, static_argnums=0)
    def _calc_force_bias_generalized(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0] + self.nelec[1]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, :nocc, :]

        # Ref
        nu0 = jnp.einsum("gpq,pq->g", chol[:, :nocc, :], green)

        # Single excitations
        nu1 = jnp.einsum(
            "gpq,ia,pq,ia->g", chol[:, :nocc, :], ci1.conj(), green, green_occ
        )
        nu1 -= jnp.einsum("gpq,ia,iq,pa->g", chol, ci1.conj(), green, greenp)

        # nu1  = jnp.einsum("grq,rq,ia,ia->g", rot_chol[:,:,:], green, ci1.conj(), green_occ)
        # nu1 -= jnp.einsum("gpq,ia,iq,pa->g", chol[:,:,:], ci1.conj(), green, greenp)

        # Double excitations
        nu2 = 2.0 * jnp.einsum(
            "gpq,iajb,pq,ia,jb->g",
            chol[:, :nocc, :],
            ci2.conj(),
            green,
            green_occ,
            green_occ,
        )
        # nu2 -= jnp.einsum("gpq,iajb,pq,ib,ja->g", chol[:,:nocc,:], ci2.conj(), green, green_occ, green_occ)
        nu2 -= 4.0 * jnp.einsum(
            "gpq,iajb,pa,iq,jb->g", chol[:, :, :], ci2.conj(), greenp, green, green_occ
        )
        # nu2 += 2.0*jnp.einsum("gpq,iajb,pa,ib,jq->g", chol[:,:,:], ci2.conj(), greenp, green_occ, green)
        # nu2 += jnp.einsum("gpq,iajb,pb,iq,ja->g", chol[:,:,:], ci2.conj(), greenp, green, green_occ)
        # nu2 -= jnp.einsum("gpq,iajb,pb,ia,jq->g", chol[:,:,:], ci2.conj(), greenp, green_occ, green)

        # V2
        # nu2 = 2.0*jnp.einsum("grq,rq,iajb,ia,jb->g", rot_chol[:,:,:], green, ci2.conj(), green_occ, green_occ)
        # nu2 -= 4.0*jnp.einsum("gpq,iajb,pa,iq,jb->g", chol, ci2.conj(), greenp, green, green_occ)
        nu2 *= 0.25

        nu = nu0 + nu1 + nu2
        o1 = jnp.einsum("ia,ia->", ci1.conj(), green_occ)
        o2 = 0.25 * 2.0 * jnp.einsum("iajb, ia, jb->", ci2.conj(), green_occ, green_occ)
        overlap = 1.0 + o1 + o2
        nu = nu / overlap
        return nu

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0] + self.nelec[1]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, :nocc, :]
        h1 = ham_data["h1"][0]
        rot_h1 = h1[:nocc, :]

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = jnp.einsum("pq,pq->", rot_h1, green)

        # single excitations
        e1_1 = jnp.einsum(
            "pq,ia,pq,ia->", rot_h1, ci1.conj(), green, green_occ, optimize="optimal"
        )
        e1_1 -= jnp.einsum(
            "pq,ia,iq,pa->", h1, ci1.conj(), green, greenp, optimize="optimal"
        )

        # double excitations
        # e1_2  = jnp.einsum("pq,iajb,pq,ia,jb->", h1[nocc:,:], ci2.conj(), green, green_occ, green_occ)
        # e1_2 -= jnp.einsum("pq,iajb,pq,ib,ja->", h1[nocc:,:], ci2.conj(), green, green_occ, green_occ)
        # e1_2 -= jnp.einsum("pq,iajb,pa,iq,jb->", h1, ci2.conj(), greenp, green, green_occ)
        # e1_2 += jnp.einsum("pq,iajb,pa,ib,jq->", h1, ci2.conj(), greenp, green_occ, green)
        # e1_2 += jnp.einsum("pq,iajb,pb,iq,ja->", h1, ci2.conj(), greenp, green, green_occ)
        # e1_2 -= jnp.einsum("pq,iajb,pb,ia,jq->", h1, ci2.conj(), greenp, green_occ, green)

        e1_2 = 2.0 * jnp.einsum(
            "rq,rq,iajb,ia,jb", rot_h1, green, ci2.conj(), green_occ, green_occ
        )
        e1_2 -= 4.0 * jnp.einsum(
            "pq,iajb,pa,iq,jb", h1, ci2.conj(), greenp, green, green_occ
        )
        e1_2 *= 0.25

        # 2 body energy
        # ref
        f = jnp.einsum("gij,jk->gik", rot_chol, green.T, optimize="optimal")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        e2_0 = (jnp.sum(c * c) - exc) / 2.0

        # single excitations
        e2_1 = jnp.einsum(
            "gpr,gqs,ia,ir,ps,qa->",
            chol[:, :nocc, :],
            chol[:, :, :],
            ci1.conj(),
            green,
            green,
            greenp,
        )
        e2_1 -= jnp.einsum(
            "gpr,gqs,ia,ir,pa,qs->",
            chol[:, :, :],
            chol[:, :nocc, :],
            ci1.conj(),
            green,
            greenp,
            green,
        )
        e2_1 -= jnp.einsum(
            "gpr,gqs,ia,pr,is,qa->",
            chol[:, :nocc, :],
            chol[:, :, :],
            ci1.conj(),
            green,
            green,
            greenp,
        )
        e2_1 += jnp.einsum(
            "gpr,gqs,ia,pr,ia,qs->",
            chol[:, :nocc, :],
            chol[:, :nocc, :],
            ci1.conj(),
            green,
            green_occ,
            green,
        )
        e2_1 += jnp.einsum(
            "gpr,gqs,ia,qr,is,pa->",
            chol[:, :, :],
            chol[:, :nocc, :],
            ci1.conj(),
            green,
            green,
            greenp,
        )
        e2_1 -= jnp.einsum(
            "gpr,gqs,ia,qr,ia,ps->",
            chol[:, :nocc, :],
            chol[:, :nocc, :],
            ci1.conj(),
            green,
            green_occ,
            green,
        )
        e2_1 *= 0.5

        # double excitations
        e2_2 = 2.0 * jnp.einsum(
            "gpr,gqs,iajb,ir,js,pa,qb->",
            chol,
            chol,
            ci2.conj(),
            green,
            green,
            greenp,
            greenp,
        )
        # e2_2 -= jnp.einsum("gpr,gqs,iajb,ir,js,pb,qa->", chol           , chol           , ci2.conj(), green, green, greenp   , greenp)
        e2_2 -= 2.0 * jnp.einsum(
            "gpr,gqs,iajb,ir,ps,ja,qb->",
            chol[:, :nocc, :],
            chol,
            ci2.conj(),
            green,
            green,
            green_occ,
            greenp,
        )
        # e2_2 += jnp.einsum("gpr,gqs,iajb,ir,ps,jb,qa->", chol[:,:nocc,:], chol           , ci2.conj(), green, green, green_occ, greenp)
        e2_2 += 2.0 * jnp.einsum(
            "gpr,gqs,iajb,ir,qs,ja,pb->",
            chol,
            chol[:, :nocc, :],
            ci2.conj(),
            green,
            green,
            green_occ,
            greenp,
        )
        # e2_2 -= jnp.einsum("gpr,gqs,iajb,ir,qs,jb,pa->", chol           , chol[:,:nocc,:], ci2.conj(), green, green, green_occ, greenp)
        # P_ij
        e2_2 *= 2.0

        e2_2 += 4.0 * jnp.einsum(
            "gpr,gqs,iajb,pr,is,ja,qb->",
            chol[:, :nocc, :],
            chol,
            ci2.conj(),
            green,
            green,
            green_occ,
            greenp,
        )
        # e2_2 -= 2.0 * jnp.einsum("gpr,gqs,iajb,pr,is,jb,qa->", chol[:,:nocc,:], chol           , ci2.conj(), green, green, green_occ, greenp   )
        e2_2 += 2.0 * jnp.einsum(
            "gpr,gqs,iajb,pr,qs,ia,jb->",
            chol[:, :nocc, :],
            chol[:, :nocc, :],
            ci2.conj(),
            green,
            green,
            green_occ,
            green_occ,
        )
        # e2_2 -=       jnp.einsum("gpr,gqs,iajb,pr,qs,ib,ja->", chol[:,:nocc,:], chol[:,:nocc,:], ci2.conj(), green, green, green_occ, green_occ)
        # P_pq
        e2_2 -= 4.0 * jnp.einsum(
            "gpr,gqs,iajb,qr,is,ja,pb->",
            chol,
            chol[:, :nocc, :],
            ci2.conj(),
            green,
            green,
            green_occ,
            greenp,
        )
        # e2_2 += 2.0 * jnp.einsum("gpr,gqs,iajb,qr,is,jb,pa->", chol           , chol[:,:nocc,:], ci2.conj(), green, green, green_occ, greenp   )
        e2_2 -= 2.0 * jnp.einsum(
            "gpr,gqs,iajb,qr,ps,ia,jb->",
            chol[:, :nocc, :],
            chol[:, :nocc, :],
            ci2.conj(),
            green,
            green,
            green_occ,
            green_occ,
        )
        # e2_2 +=       jnp.einsum("gpr,gqs,iajb,qr,ps,ib,ja->", chol[:,:nocc,:], chol[:,:nocc,:], ci2.conj(), green, green, green_occ, green_occ)
        e2_2 *= 0.5 * 0.25

        e = e1_0 + e1_1 + e1_2 + e2_0 + e2_1 + e2_2
        o1 = jnp.einsum("ia,ia->", ci1.conj(), green_occ)
        o2 = 0.25 * 2.0 * jnp.einsum("iajb, ia, jb->", ci2.conj(), green_occ, green_occ)
        overlap = 1.0 + o1 + o2
        e = e / overlap

        return e + e0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class UCISD(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_chunks: int = 1
    projector: Optional[str] = None

    @partial(jit, static_argnums=0)
    def _calc_green_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array
    ) -> List[jax.Array]:

        green_up = (walker_up.dot(jnp.linalg.inv(walker_up[: walker_up.shape[1], :]))).T
        green_dn = (walker_dn.dot(jnp.linalg.inv(walker_dn[: walker_dn.shape[1], :]))).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:

        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]
        moA, moB = wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]

        walker_dn_B = moB.T.dot(
            walker_dn[:, :noccB]
        )  # put walker_dn in the basis of alpha reference

        GFA, GFB = self._calc_green_unrestricted(walker_up, walker_dn_B)

        o0 = jnp.linalg.det(walker_up[:noccA, :]) * jnp.linalg.det(
            walker_dn_B[:noccB, :]
        )

        o1 = jnp.einsum("ia,ia", ci1A, GFA[:, noccA:]) + jnp.einsum(
            "ia,ia", ci1B, GFB[:, noccB:]
        )

        # AA
        o2 = 0.5 * jnp.einsum("iajb, ia, jb", ci2AA, GFA[:, noccA:], GFA[:, noccA:])
        # o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2AA, GFA[:, noccA:], GFA[:, noccA:])

        # BB
        o2 += 0.5 * jnp.einsum("iajb, ia, jb", ci2BB, GFB[:, noccB:], GFB[:, noccB:])
        # o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2BB, GFB[:, noccB:], GFB[:, noccB:])

        # AB
        o2 += jnp.einsum("iajb, ia, jb", ci2AB, GFA[:, noccA:], GFB[:, noccB:])
        return (1.0 + o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_overlap_generalized(self, walker: jax.Array, wave_data: dict) -> complex:

        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]

        walker_ = jnp.vstack(
            [
                walker[: self.norb],
                wave_data["mo_coeff"][1].T.dot(walker[self.norb :, :]),
            ]
        )  # put walker_dn in the basis of alpha reference

        Atrial, Btrial = (
            wave_data["mo_coeff"][0][:, :noccA],
            wave_data["mo_coeff"][1][:, :noccB],
        )
        bra = jnp.block([[Atrial, 0 * Btrial], [0 * Atrial, Btrial]])
        o0 = jnp.linalg.det(bra.T.conj() @ walker)

        bra = jnp.block(
            [
                [Atrial, 0 * Btrial],
                [
                    0 * Atrial,
                    (wave_data["mo_coeff"][1].T @ wave_data["mo_coeff"][1])[:, :noccB],
                ],
            ]
        )

        gf = (walker_ @ jnp.linalg.inv(bra.T.conj() @ walker_) @ bra.T.conj()).T
        gfA, gfB = (
            gf[: self.nelec[0], : self.norb],
            gf[self.norb : self.norb + self.nelec[1], self.norb :],
        )
        gfAB, gfBA = (
            gf[: self.nelec[0], self.norb :],
            gf[self.norb : self.norb + self.nelec[1], : self.norb],
        )

        o1 = jnp.einsum("ia,ia", ci1A, gfA[:, noccA:]) + jnp.einsum(
            "ia,ia", ci1B, gfB[:, noccB:]
        )

        # AA
        o2 = jnp.einsum("iajb, ia, jb", ci2AA, gfA[:, noccA:], gfA[:, noccA:])
        # o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2AA, GFA[:, noccA:], GFA[:, noccA:])

        # BB
        o2 += jnp.einsum("iajb, ia, jb", ci2BB, gfB[:, noccB:], gfB[:, noccB:])
        # o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2BB, GFB[:, noccB:], GFB[:, noccB:])

        # AB
        o2 += 2.0 * jnp.einsum("iajb, ia, jb", ci2AB, gfA[:, noccA:], gfB[:, noccB:])
        o2 -= 2.0 * jnp.einsum("iajb, ib, ja", ci2AB, gfAB[:, noccB:], gfBA[:, noccA:])

        return (1.0 + o1 + o2 / 2.0) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class UCISinfD(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array
    ) -> List[jax.Array]:

        green_up = (walker_up.dot(jnp.linalg.inv(walker_up[: walker_up.shape[1], :]))).T
        green_dn = (walker_dn.dot(jnp.linalg.inv(walker_dn[: walker_dn.shape[1], :]))).T
        return [green_up, green_dn]

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        noccA, rA, ci2AA = self.nelec[0], wave_data["rA"], wave_data["ci2AA"]
        noccB, rB, ci2BB = self.nelec[1], wave_data["rB"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]
        _, moB = wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]

        walker_dn_B = moB.T @ (
            walker_dn[:, :noccB]
        )  # put walker_dn in the basis of beta reference

        # rotate with t1
        walker_up_r = rA @ walker_up
        walker_dn_r = rB @ walker_dn_B

        GFA, GFB = self._calc_green_unrestricted(walker_up_r, walker_dn_r)

        o0 = jnp.linalg.det(walker_up_r[:noccA, :]) * jnp.linalg.det(
            walker_dn_r[:noccB, :]
        )

        # AA
        o2 = 0.5 * jnp.einsum("iajb, ia, jb", ci2AA, GFA[:, noccA:], GFA[:, noccA:])
        # o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2AA, GFA[:, noccA:], GFA[:, noccA:])

        # BB
        o2 += 0.5 * jnp.einsum("iajb, ia, jb", ci2BB, GFB[:, noccB:], GFB[:, noccB:])
        # o2 -= 0.25 * jnp.einsum("iajb, ib, ja", ci2BB, GFB[:, noccB:], GFB[:, noccB:])

        # AB
        o2 += jnp.einsum("iajb, ia, jb", ci2AB, GFA[:, noccA:], GFB[:, noccB:])

        return (1 + o2) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class GCISD(wave_function_auto):
    """This class contains functions for the GCISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>
    The wave_data need to store the coefficients C(ia) and C(ia jb) and the GHF mo coefficients
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green_generalized(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        nocc, ci1, ci2 = (
            self.nelec[0] + self.nelec[1],
            wave_data["ci1"],
            wave_data["ci2"],
        )
        walker = jnp.block(
            [
                [walker_up, jnp.zeros_like(walker_dn)],
                [jnp.zeros_like(walker_up), walker_dn],
            ]
        )
        walker = wave_data["mo_coeff"].T @ walker
        GF = self._calc_green_generalized(walker)
        o0 = jnp.linalg.det(walker[: walker.shape[1], :])
        o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
        o2 = jnp.einsum("iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]) - jnp.einsum(
            "iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:]
        )
        return (1.0 + o1 + o2 / 4.0) * o0

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class CISD_THC(wave_function_auto):
    """This class contains functions for the CISD wavefunction
    |0> + c(ia) |ia> + c(ia jb) |ia jb>

    . The wave_data need to store the coefficient C(ia) and C(ia jb)  in the THC format
    i.e. C(i,a,j,b) = X1(P,i) X2(P,a) V(P,Q) X1(P,j) X2(P,b)
    """

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1.0e-4  # finite difference step size in local energy calculations
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _calc_green_restricted(self, walker: jax.Array) -> jax.Array:
        return (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, Xocc, Xvirt, VKL = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["Xocc"],
            wave_data["Xvirt"],
            wave_data["VKL"],
        )
        GF = self._calc_green_restricted(walker)

        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2

        o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])

        # A = jnp.einsum('ia,Pi,Pa->P', GF[:,nocc:], Xocc, Xvirt)
        # o2 = 2*jnp.einsum('P,PQ,Q', A, VKL, A)

        gv = GF[:, nocc:] @ Xvirt.T
        A = jnp.einsum("Pi,iP->P", Xocc, gv)
        o2 = 2 * (A @ VKL).dot(A)

        B = Xocc @ gv
        o2 -= jnp.sum(B * B.T * VKL)

        return (1.0 + 2 * o1 + o2) * o0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cis(wave_function):
    """A manual implementation of the excited state CIS wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1 = walker.shape[1], wave_data["ci1"]
        gf = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = jnp.einsum("ia,ia", ci1, gf[:, nocc:])
        return 2 * o1 * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci1 = wave_data["ci1"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")

        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        ci1gp = jnp.einsum("pt,it->pi", ci1, greenp, optimize="optimal")
        gci1gp = jnp.einsum("pj,pi->ij", green, ci1gp, optimize="optimal")
        fb_1_1 = 4 * ci1g * lg
        fb_1_2 = -2 * jnp.einsum("gij,ij->g", chol, gci1gp, optimize="optimal")
        fb_1 = fb_1_1 + fb_1_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap = overlap_1

        return fb_1 / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1 = wave_data["ci1"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        # rot_chol = ham_data["rot_chol"]
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = jnp.einsum("pj,pj->", h1[:nocc, :], green)

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
        e1_1 = e1_1_1 + e1_1_2

        e1 = e1_1

        # two body energy
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = jnp.einsum("gij,ij->g", chol, ci1_green, optimize="optimal")
        e2_1_2 = -2 * (lci1g @ lg)
        # lci1g1 = jnp.einsum("gij,jk->gik", chol, ci1_green, optimize="optimal")
        # glgpci1 = jnp.einsum(("gpi,iq->gpq"), gl, gpci1, optimize="optimal")
        ci1g1 = ci1 @ green_occ.T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1, lg1, ci1g1, optimize="optimal")
        lci1g = jnp.einsum("gip,qi->gpq", ham_data["lci1"], green, optimize="optimal")
        e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g, lg1, optimize="optimal")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        e2 = e2_1

        # overlap
        overlap_1 = 2 * ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
        overlap = overlap_1
        return (e1 + e2) / overlap + e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["lci1"] = jnp.einsum(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            optimize="optimal",
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd(wave_function):
    """A manual implementation of the restricted CISD wave function.

    The corresponding wave_data contains "ci1" and "ci2" which are the coefficients of the single (c_ia)
    and double (c_iajb) excitations respectively. Mixed precision and memory mode are used to optimize
    the performance of the energy calculations.

    Attributes:
        norb: int
            Number of orbitals.
        nelec: Tuple[int, int]
            Number of electrons in alpha and beta spin channels.
        n_chunks: int
            Number of walker chunks.
        mixed_real_dtype: DTypeLike
            Data type used for mixed precision, double precision by default.
        mixed_complex_dtype: DTypeLike
            Data type used for mixed precision, double precision by default.
        memory_mode: enum
            Memory mode for energy calculations. "high" builds O(XNM) scaling intermediates per walker,
            "low" builds at most O(XN^2).
        _mixed_real_dtype_testing: DTypeLike
            Data type used for testing. Some operations are in single precision by default but
            can be changed to double precision for testing.
        _mixed_complex_dtype_testing: DTypeLike
            Data type used for testing. Some operations are in single precision by default but
            can be changed to double precision for testing.
    """

    norb: int
    nelec: Tuple[int, int]
    n_chunks: int = 1
    projector: Optional[str] = None
    mixed_real_dtype: DTypeLike = jnp.float64
    mixed_complex_dtype: DTypeLike = jnp.complex128
    memory_mode: Literal["high", "low"] = "low"
    _mixed_real_dtype_testing: DTypeLike = jnp.float32
    _mixed_complex_dtype_testing: DTypeLike = jnp.complex64

    def __post_init__(self):
        if self.memory_mode not in {"low", "high"}:
            raise ValueError(
                f"memory_mode must be one of ['low', 'high'], got {self.memory_mode}"
            )

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2 = walker.shape[1], wave_data["ci1"], wave_data["ci2"]
        GF = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        o0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        o1 = jnp.einsum("ia,ia", ci1, GF[:, nocc:])
        o2 = 2 * jnp.einsum(
            "iajb, ia, jb", ci2, GF[:, nocc:], GF[:, nocc:]
        ) - jnp.einsum("iajb, ib, ja", ci2, GF[:, nocc:], GF[:, nocc:])
        return (1.0 + 2 * o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")

        # ref
        fb_0 = 2 * lg

        # single excitations
        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        ci1gp = jnp.einsum("pt,it->pi", ci1, greenp, optimize="optimal")
        gci1gp = jnp.einsum("pj,pi->ij", green, ci1gp, optimize="optimal")
        fb_1_1 = 4 * ci1g * lg
        fb_1_2 = -2 * jnp.einsum(
            "gij,ij->g",
            chol.astype(self.mixed_real_dtype),
            gci1gp.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_c = jnp.einsum(
            "ptqu,pt->qu",
            ci2.astype(self.mixed_real_dtype),
            green_occ.astype(self.mixed_complex_dtype),
        )
        ci2g_e = jnp.einsum(
            "ptqu,pu->qt",
            ci2.astype(self.mixed_real_dtype),
            green_occ.astype(self.mixed_complex_dtype),
        )
        cisd_green_c = (greenp @ ci2g_c.T) @ green
        cisd_green_e = (greenp @ ci2g_e.T) @ green
        cisd_green = -4 * cisd_green_c + 2 * cisd_green_e
        ci2g = 4 * ci2g_c - 2 * ci2g_e
        gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")
        fb_2_1 = lg * gci2g
        fb_2_2 = jnp.einsum(
            "gij,ij->g",
            chol.astype(self.mixed_real_dtype),
            cisd_green.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = 2 * ci1g
        overlap_2 = gci2g / 2.0
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1, ci2 = wave_data["ci1"], wave_data["ci2"]
        nocc = self.nelec[0]
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(self.norb - nocc)))

        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        hg = jnp.einsum("pj,pj->", h1[:nocc, :], green)

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = 2 * hg

        # single excitations
        ci1g = jnp.einsum("pt,pt->", ci1, green_occ, optimize="optimal")
        e1_1_1 = 4 * ci1g * hg
        gpci1 = greenp @ ci1.T
        ci1_green = gpci1 @ green
        e1_1_2 = -2 * jnp.einsum("ij,ij->", h1, ci1_green, optimize="optimal")
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_c = jnp.einsum(
            "ptqu,pt->qu",
            ci2.astype(self.mixed_real_dtype),
            green_occ.astype(self.mixed_complex_dtype),
        )
        ci2g_e = jnp.einsum(
            "ptqu,pu->qt",
            ci2.astype(self.mixed_real_dtype),
            green_occ.astype(self.mixed_complex_dtype),
        )
        ci2_green_c = (greenp @ ci2g_c.T) @ green
        ci2_green_e = (greenp @ ci2g_e.T) @ green
        ci2_green = 2 * ci2_green_c - ci2_green_e
        ci2g = 2 * ci2g_c - ci2g_e
        gci2g = jnp.einsum("qu,qu->", ci2g, green_occ, optimize="optimal")
        e1_2_1 = 2 * hg * gci2g
        e1_2_2 = -2 * jnp.einsum("ij,ij->", h1, ci2_green, optimize="optimal")
        e1_2 = e1_2_1 + e1_2_2
        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        # lg1 = jnp.einsum("gpj,pk->gjk", rot_chol, green, optimize="optimal")
        lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        e2_0_1 = 2 * lg @ lg
        e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = 2 * e2_0 * ci1g
        lci1g = jnp.einsum(
            "gij,ij->g",
            chol.astype(self.mixed_real_dtype),
            ci1_green.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        e2_1_2 = -2 * (lci1g @ lg)
        # lci1g1 = jnp.einsum("gij,jk->gik", chol, ci1_green, optimize="optimal")
        # glgpci1 = jnp.einsum(("gpi,iq->gpq"), gl, gpci1, optimize="optimal")
        ci1g1 = ci1 @ green_occ.T
        # e2_1_3 = jnp.einsum("gpq,gpq->", glgpci1, lg1, optimize="optimal")
        e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1, lg1, ci1g1, optimize="optimal")
        lci1g = jnp.einsum("gip,qi->gpq", ham_data["lci1"], green, optimize="optimal")
        e2_1_3_2 = -jnp.einsum("gpq,gqp->", lci1g, lg1, optimize="optimal")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g = jnp.einsum(
            "gij,ij->g",
            chol.astype(self.mixed_real_dtype),
            ci2_green.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        e2_2_2_1 = -lci2g @ lg

        if self.memory_mode == "low":

            def scan_over_chol(carry, x):
                chol_i, rot_chol_i = x
                gl_i = jnp.einsum("pj,ji->pi", green, chol_i, optimize="optimal")
                lci2_green_i = jnp.einsum(
                    "pi,ji->pj", rot_chol_i, ci2_green, optimize="optimal"
                )
                carry[0] += 0.5 * jnp.einsum(
                    "pi,pi->", gl_i, lci2_green_i, optimize="optimal"
                )
                glgp_i = jnp.einsum(
                    "pi,it->pt", gl_i, greenp, optimize="optimal"
                ).astype(self._mixed_complex_dtype_testing)
                l2ci2_1 = jnp.einsum(
                    "pt,qu,ptqu->",
                    glgp_i,
                    glgp_i,
                    ci2.astype(self._mixed_real_dtype_testing),
                    optimize="optimal",
                )
                l2ci2_2 = jnp.einsum(
                    "pu,qt,ptqu->",
                    glgp_i,
                    glgp_i,
                    ci2.astype(self._mixed_real_dtype_testing),
                    optimize="optimal",
                )
                carry[1] += 2 * l2ci2_1 - l2ci2_2
                return carry, 0.0

            [e2_2_2_2, e2_2_3], _ = lax.scan(
                scan_over_chol, [0.0, 0.0], (chol, rot_chol)
            )
        else:
            lci2_green = jnp.einsum(
                "gpi,ji->gpj", rot_chol, ci2_green, optimize="optimal"
            )
            gl = jnp.einsum(
                "pj,gji->gpi",
                green.astype(self.mixed_complex_dtype),
                chol.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            e2_2_2_2 = 0.5 * jnp.einsum("gpi,gpi->", gl, lci2_green, optimize="optimal")
            glgp = jnp.einsum("gpi,it->gpt", gl, greenp, optimize="optimal").astype(
                self._mixed_complex_dtype_testing
            )
            l2ci2_1 = jnp.einsum(
                "gpt,gqu,ptqu->g",
                glgp,
                glgp,
                ci2.astype(self._mixed_real_dtype_testing),
                optimize="optimal",
            )
            l2ci2_2 = jnp.einsum(
                "gpu,gqt,ptqu->g",
                glgp,
                glgp,
                ci2.astype(self._mixed_real_dtype_testing),
                optimize="optimal",
            )
            e2_2_3 = 2 * l2ci2_1.sum() - l2ci2_2.sum()

        e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = 2 * ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2

        return (e1 + e2) / overlap + e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        ham_data["lci1"] = jnp.einsum(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1"],
            optimize="optimal",
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class ucisd(wave_function):
    """A manual implementation of the UCISD wave function."""

    norb: int
    nelec: Tuple[int, int]
    n_chunks: int = 1
    projector: Optional[str] = None
    mixed_real_dtype: DTypeLike = jnp.float64
    mixed_complex_dtype: DTypeLike = jnp.complex128
    memory_mode: Literal["high", "low"] = "low"
    _mixed_real_dtype_testing: DTypeLike = jnp.float32
    _mixed_complex_dtype_testing: DTypeLike = jnp.complex64

    @partial(jit, static_argnums=0)
    def _calc_overlap_unrestricted(
        self, walker_up: jax.Array, walker_dn: jax.Array, wave_data: dict
    ) -> complex:
        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]
        _, moB = wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]
        walker_dn_B = moB.T.dot(walker_dn[:, :noccB])
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[: walker_up.shape[1], :]))).T
        green_b = (
            walker_dn_B.dot(jnp.linalg.inv(walker_dn_B[: walker_dn_B.shape[1], :]))
        ).T
        green_a, green_b = green_a[:, noccA:], green_b[:, noccB:]
        o0 = jnp.linalg.det(walker_up[:noccA, :]) * jnp.linalg.det(
            walker_dn_B[:noccB, :]
        )
        o1 = jnp.einsum("ia,ia", ci1A, green_a) + jnp.einsum("ia,ia", ci1B, green_b)
        o2 = (
            0.5 * jnp.einsum("iajb, ia, jb", ci2AA, green_a, green_a)
            + 0.5 * jnp.einsum("iajb, ia, jb", ci2BB, green_b, green_b)
            + jnp.einsum("iajb, ia, jb", ci2AB, green_a, green_b)
        )
        return (1.0 + o1 + o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_overlap_generalized(self, walker: jax.Array, wave_data: dict) -> complex:

        noccA, ci1A, ci2AA = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        noccB, ci1B, ci2BB = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]

        walker_ = jnp.vstack(
            [
                walker[: self.norb],
                wave_data["mo_coeff"][1].T.dot(walker[self.norb :, :]),
            ]
        )  # put walker_dn in the basis of alpha reference

        Atrial, Btrial = (
            wave_data["mo_coeff"][0][:, :noccA],
            wave_data["mo_coeff"][1][:, :noccB],
        )
        bra = jnp.block([[Atrial, 0 * Btrial], [0 * Atrial, Btrial]])
        o0 = jnp.linalg.det(bra.T.conj() @ walker)

        bra = jnp.block(
            [
                [Atrial, 0 * Btrial],
                [
                    0 * Atrial,
                    (wave_data["mo_coeff"][1].T @ wave_data["mo_coeff"][1])[:, :noccB],
                ],
            ]
        )

        gf = (walker_ @ jnp.linalg.inv(bra.T.conj() @ walker_) @ bra.T.conj()).T

        gfA, gfB = (
            gf[: self.nelec[0], : self.norb],
            gf[self.norb : self.norb + self.nelec[1], self.norb :],
        )
        gfAB, gfBA = (
            gf[: self.nelec[0], self.norb :],
            gf[self.norb : self.norb + self.nelec[1], : self.norb],
        )

        o1 = jnp.einsum("ia,ia", ci1A, gfA[:, noccA:]) + jnp.einsum(
            "ia,ia", ci1B, gfB[:, noccB:]
        )

        # AA
        o2 = jnp.einsum("iajb, ia, jb", ci2AA, gfA[:, noccA:], gfA[:, noccA:])

        # BB
        o2 += jnp.einsum("iajb, ia, jb", ci2BB, gfB[:, noccB:], gfB[:, noccB:])

        # AB
        o2 += 2.0 * jnp.einsum("iajb, ia, jb", ci2AB, gfA[:, noccA:], gfB[:, noccB:])
        o2 -= 2.0 * jnp.einsum("iajb, ib, ja", ci2AB, gfAB[:, noccB:], gfBA[:, noccA:])

        return (1.0 + o1 + 0.5 * o2) * o0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> jax.Array:
        """Calculates force bias < psi_T | chol_gamma | walker > / < psi_T | walker >"""
        nocc_a, ci1_a, ci2_aa = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        nocc_b, ci1_b, ci2_bb = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2_ab = wave_data["ci2AB"]
        walker_dn_b = wave_data["mo_coeff"][1].T.dot(walker_dn[:, :nocc_b])
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn_b.dot(jnp.linalg.inv(walker_dn_b[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol_b"].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, : self.nelec[0], :]
        rot_chol_b = chol_b[:, : self.nelec[1], :]
        lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
        lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
        lg = lg_a + lg_b

        # ref
        fb_0 = lg_a + lg_b

        # single excitations
        ci1g_a = jnp.einsum("pt,pt->", ci1_a, green_occ_a, optimize="optimal")
        ci1g_b = jnp.einsum("pt,pt->", ci1_b, green_occ_b, optimize="optimal")
        ci1g = ci1g_a + ci1g_b
        fb_1_1 = ci1g * lg
        ci1gp_a = jnp.einsum("pt,it->pi", ci1_a, greenp_a, optimize="optimal")
        ci1gp_b = jnp.einsum("pt,it->pi", ci1_b, greenp_b, optimize="optimal")
        gci1gp_a = jnp.einsum("pj,pi->ij", green_a, ci1gp_a, optimize="optimal")
        gci1gp_b = jnp.einsum("pj,pi->ij", green_b, ci1gp_b, optimize="optimal")
        fb_1_2 = -jnp.einsum(
            "gij,ij->g",
            chol_a.astype(self.mixed_real_dtype),
            gci1gp_a.astype(self.mixed_complex_dtype),
            optimize="optimal",
        ) - jnp.einsum(
            "gij,ij->g",
            chol_b.astype(self.mixed_real_dtype),
            gci1gp_b.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        fb_1 = fb_1_1 + fb_1_2

        # double excitations
        ci2g_a = jnp.einsum(
            "ptqu,pt->qu",
            ci2_aa.astype(self.mixed_real_dtype),
            green_occ_a.astype(self.mixed_complex_dtype),
        )
        ci2g_b = jnp.einsum(
            "ptqu,pt->qu",
            ci2_bb.astype(self.mixed_real_dtype),
            green_occ_b.astype(self.mixed_complex_dtype),
        )
        ci2g_ab_a = jnp.einsum(
            "ptqu,qu->pt",
            ci2_ab.astype(self.mixed_real_dtype),
            green_occ_b.astype(self.mixed_complex_dtype),
        )
        ci2g_ab_b = jnp.einsum(
            "ptqu,pt->qu",
            ci2_ab.astype(self.mixed_real_dtype),
            green_occ_a.astype(self.mixed_complex_dtype),
        )
        gci2g_a = 0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
        gci2g_b = 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
        gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
        gci2g = gci2g_a + gci2g_b + gci2g_ab
        fb_2_1 = lg * gci2g
        ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
        ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
        fb_2_2_a = -jnp.einsum(
            "gij,ij->g",
            chol_a.astype(self.mixed_real_dtype),
            ci2_green_a.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        fb_2_2_b = -jnp.einsum(
            "gij,ij->g",
            chol_b.astype(self.mixed_real_dtype),
            ci2_green_b.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        fb_2_2 = fb_2_2_a + fb_2_2_b
        fb_2 = fb_2_1 + fb_2_2

        # overlap
        overlap_1 = ci1g
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2

        return (fb_0 + fb_1 + fb_2) / overlap

    @partial(jit, static_argnums=0)
    def _calc_energy_unrestricted(
        self,
        walker_up: jax.Array,
        walker_dn: jax.Array,
        ham_data: dict,
        wave_data: dict,
    ) -> complex:
        nocc_a, ci1_a, ci2_aa = self.nelec[0], wave_data["ci1A"], wave_data["ci2AA"]
        nocc_b, ci1_b, ci2_bb = self.nelec[1], wave_data["ci1B"], wave_data["ci2BB"]
        ci2_ab = wave_data["ci2AB"]
        walker_dn_b = wave_data["mo_coeff"][1].T.dot(walker_dn[:, :nocc_b])
        green_a = (walker_up.dot(jnp.linalg.inv(walker_up[:nocc_a, :]))).T
        green_b = (walker_dn_b.dot(jnp.linalg.inv(walker_dn_b[:nocc_b, :]))).T
        green_occ_a = green_a[:, nocc_a:].copy()
        green_occ_b = green_b[:, nocc_b:].copy()
        greenp_a = jnp.vstack((green_occ_a, -jnp.eye(self.norb - nocc_a)))
        greenp_b = jnp.vstack((green_occ_b, -jnp.eye(self.norb - nocc_b)))

        chol_a = ham_data["chol"].reshape(-1, self.norb, self.norb)
        chol_b = ham_data["chol_b"].reshape(-1, self.norb, self.norb)
        rot_chol_a = chol_a[:, :nocc_a, :]
        rot_chol_b = chol_b[:, :nocc_b, :]
        h1_a = (ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0
        h1_b = ham_data["h1_b"]
        hg_a = jnp.einsum("pj,pj->", h1_a[:nocc_a, :], green_a)
        hg_b = jnp.einsum("pj,pj->", h1_b[:nocc_b, :], green_b)
        hg = hg_a + hg_b

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = hg

        # single excitations
        ci1g_a = jnp.einsum("pt,pt->", ci1_a, green_occ_a, optimize="optimal")
        ci1g_b = jnp.einsum("pt,pt->", ci1_b, green_occ_b, optimize="optimal")
        ci1g = ci1g_a + ci1g_b
        e1_1_1 = ci1g * hg
        gpci1_a = greenp_a @ ci1_a.T
        gpci1_b = greenp_b @ ci1_b.T
        ci1_green_a = gpci1_a @ green_a
        ci1_green_b = gpci1_b @ green_b
        e1_1_2 = -(
            jnp.einsum("ij,ij->", h1_a, ci1_green_a, optimize="optimal")
            + jnp.einsum("ij,ij->", h1_b, ci1_green_b, optimize="optimal")
        )
        e1_1 = e1_1_1 + e1_1_2

        # double excitations
        ci2g_a = (
            jnp.einsum(
                "ptqu,pt->qu",
                ci2_aa.astype(self.mixed_real_dtype),
                green_occ_a.astype(self.mixed_complex_dtype),
            )
            / 4
        )
        ci2g_b = (
            jnp.einsum(
                "ptqu,pt->qu",
                ci2_bb.astype(self.mixed_real_dtype),
                green_occ_b.astype(self.mixed_complex_dtype),
            )
            / 4
        )
        ci2g_ab_a = jnp.einsum(
            "ptqu,qu->pt",
            ci2_ab.astype(self.mixed_real_dtype),
            green_occ_b.astype(self.mixed_complex_dtype),
        )
        ci2g_ab_b = jnp.einsum(
            "ptqu,pt->qu",
            ci2_ab.astype(self.mixed_real_dtype),
            green_occ_a.astype(self.mixed_complex_dtype),
        )
        gci2g_a = jnp.einsum("qu,qu->", ci2g_a, green_occ_a, optimize="optimal")
        gci2g_b = jnp.einsum("qu,qu->", ci2g_b, green_occ_b, optimize="optimal")
        gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a, optimize="optimal")
        gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
        e1_2_1 = hg * gci2g
        ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
        ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
        ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
        ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
        e1_2_2_a = -jnp.einsum(
            "ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a, optimize="optimal"
        )
        e1_2_2_b = -jnp.einsum(
            "ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b, optimize="optimal"
        )
        e1_2_2 = e1_2_2_a + e1_2_2_b
        e1_2 = e1_2_1 + e1_2_2

        e1 = e1_0 + e1_1 + e1_2

        # two body energy
        # ref
        lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a, optimize="optimal")
        lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b, optimize="optimal")
        e2_0_1 = ((lg_a + lg_b) @ (lg_a + lg_b)) / 2.0
        lg1_a = jnp.einsum("gpj,qj->gpq", rot_chol_a, green_a, optimize="optimal")
        lg1_b = jnp.einsum("gpj,qj->gpq", rot_chol_b, green_b, optimize="optimal")
        e2_0_2 = (
            -(
                jnp.sum(vmap(lambda x: x * x.T)(lg1_a))
                + jnp.sum(vmap(lambda x: x * x.T)(lg1_b))
            )
            / 2.0
        )
        e2_0 = e2_0_1 + e2_0_2

        # single excitations
        e2_1_1 = e2_0 * ci1g
        lci1g_a = jnp.einsum(
            "gij,ij->g",
            chol_a.astype(self.mixed_real_dtype),
            ci1_green_a.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        lci1g_b = jnp.einsum(
            "gij,ij->g",
            chol_b.astype(self.mixed_real_dtype),
            ci1_green_b.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        e2_1_2 = -((lci1g_a + lci1g_b) @ (lg_a + lg_b))
        ci1g1_a = ci1_a @ green_occ_a.T
        ci1g1_b = ci1_b @ green_occ_b.T
        e2_1_3_1 = jnp.einsum(
            "gpq,gqr,rp->", lg1_a, lg1_a, ci1g1_a, optimize="optimal"
        ) + jnp.einsum("gpq,gqr,rp->", lg1_b, lg1_b, ci1g1_b, optimize="optimal")
        lci1g_a = jnp.einsum(
            "gip,qi->gpq", ham_data["lci1_a"], green_a, optimize="optimal"
        )
        lci1g_b = jnp.einsum(
            "gip,qi->gpq", ham_data["lci1_b"], green_b, optimize="optimal"
        )
        e2_1_3_2 = -jnp.einsum(
            "gpq,gqp->", lci1g_a, lg1_a, optimize="optimal"
        ) - jnp.einsum("gpq,gqp->", lci1g_b, lg1_b, optimize="optimal")
        e2_1_3 = e2_1_3_1 + e2_1_3_2
        e2_1 = e2_1_1 + e2_1_2 + e2_1_3

        # double excitations
        e2_2_1 = e2_0 * gci2g
        lci2g_a = jnp.einsum(
            "gij,ij->g",
            chol_a.astype(self.mixed_real_dtype),
            8 * ci2_green_a.astype(self.mixed_complex_dtype)
            + 2 * ci2_green_ab_a.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        lci2g_b = jnp.einsum(
            "gij,ij->g",
            chol_b.astype(self.mixed_real_dtype),
            8 * ci2_green_b.astype(self.mixed_complex_dtype)
            + 2 * ci2_green_ab_b.astype(self.mixed_complex_dtype),
            optimize="optimal",
        )
        e2_2_2_1 = -((lci2g_a + lci2g_b) @ (lg_a + lg_b)) / 2.0

        if self.memory_mode == "low":

            def scan_over_chol(carry, x):
                chol_a_i, rot_chol_a_i, chol_b_i, rot_chol_b_i = x
                gl_a_i = jnp.einsum("pj,ji->pi", green_a, chol_a_i, optimize="optimal")
                gl_b_i = jnp.einsum("pj,ji->pi", green_b, chol_b_i, optimize="optimal")
                lci2_green_a_i = jnp.einsum(
                    "pi,ji->pj",
                    rot_chol_a_i,
                    8 * ci2_green_a + 2 * ci2_green_ab_a,
                    optimize="optimal",
                )
                lci2_green_b_i = jnp.einsum(
                    "pi,ji->pj",
                    rot_chol_b_i,
                    8 * ci2_green_b + 2 * ci2_green_ab_b,
                    optimize="optimal",
                )
                carry[0] += 0.5 * (
                    jnp.einsum("pi,pi->", gl_a_i, lci2_green_a_i, optimize="optimal")
                    + jnp.einsum("pi,pi->", gl_b_i, lci2_green_b_i, optimize="optimal")
                )
                glgp_a_i = jnp.einsum(
                    "pi,it->pt", gl_a_i, greenp_a, optimize="optimal"
                ).astype(self._mixed_complex_dtype_testing)
                glgp_b_i = jnp.einsum(
                    "pi,it->pt", gl_b_i, greenp_b, optimize="optimal"
                ).astype(self._mixed_complex_dtype_testing)
                l2ci2_a = 0.5 * jnp.einsum(
                    "pt,qu,ptqu->",
                    glgp_a_i,
                    glgp_a_i,
                    ci2_aa.astype(self._mixed_real_dtype_testing),
                    optimize="optimal",
                )
                l2ci2_b = 0.5 * jnp.einsum(
                    "pt,qu,ptqu->",
                    glgp_b_i,
                    glgp_b_i,
                    ci2_bb.astype(self._mixed_real_dtype_testing),
                    optimize="optimal",
                )
                l2ci2_ab = jnp.einsum(
                    "pt,qu,ptqu->",
                    glgp_a_i,
                    glgp_b_i,
                    ci2_ab.astype(self._mixed_real_dtype_testing),
                    optimize="optimal",
                )
                carry[1] += l2ci2_a + l2ci2_b + l2ci2_ab
                return carry, 0.0

            [e2_2_2_2, e2_2_3], _ = lax.scan(
                scan_over_chol, [0.0, 0.0], (chol_a, rot_chol_a, chol_b, rot_chol_b)
            )
        else:
            gl_a = jnp.einsum(
                "pj,gji->gpi",
                green_a.astype(self.mixed_complex_dtype),
                chol_a.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            gl_b = jnp.einsum(
                "pj,gji->gpi",
                green_b.astype(self.mixed_complex_dtype),
                chol_b.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lci2_green_a = jnp.einsum(
                "gpi,ji->gpj",
                rot_chol_a,
                8 * ci2_green_a + 2 * ci2_green_ab_a,
                optimize="optimal",
            )
            lci2_green_b = jnp.einsum(
                "gpi,ji->gpj",
                rot_chol_b,
                8 * ci2_green_b + 2 * ci2_green_ab_b,
                optimize="optimal",
            )
            e2_2_2_2 = 0.5 * (
                jnp.einsum("gpi,gpi->", gl_a, lci2_green_a, optimize="optimal")
                + jnp.einsum("gpi,gpi->", gl_b, lci2_green_b, optimize="optimal")
            )
            glgp_a = jnp.einsum(
                "gpi,it->gpt", gl_a, greenp_a, optimize="optimal"
            ).astype(self._mixed_complex_dtype_testing)
            glgp_b = jnp.einsum(
                "gpi,it->gpt", gl_b, greenp_b, optimize="optimal"
            ).astype(self._mixed_complex_dtype_testing)
            l2ci2_a = 0.5 * jnp.einsum(
                "gpt,gqu,ptqu->g",
                glgp_a,
                glgp_a,
                ci2_aa.astype(self._mixed_real_dtype_testing),
                optimize="optimal",
            )
            l2ci2_b = 0.5 * jnp.einsum(
                "gpt,gqu,ptqu->g",
                glgp_b,
                glgp_b,
                ci2_bb.astype(self._mixed_real_dtype_testing),
                optimize="optimal",
            )
            l2ci2_ab = jnp.einsum(
                "gpt,gqu,ptqu->g",
                glgp_a,
                glgp_b,
                ci2_ab.astype(self._mixed_real_dtype_testing),
                optimize="optimal",
            )
            e2_2_3 = l2ci2_a.sum() + l2ci2_b.sum() + l2ci2_ab.sum()

        e2_2_2 = e2_2_2_1 + e2_2_2_2
        e2_2 = e2_2_1 + e2_2_2 + e2_2_3

        e2 = e2_0 + e2_1 + e2_2

        # overlap
        overlap_1 = ci1g  # jnp.einsum("ia,ia", ci1, green_occ)
        overlap_2 = gci2g
        overlap = 1.0 + overlap_1 + overlap_2
        return (e1 + e2) / overlap + e0

    @partial(jit, static_argnums=0)
    def _calc_energy_generalized(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        ci1A = wave_data["ci1A"]
        ci1B = wave_data["ci1B"]
        ci2AA = wave_data["ci2AA"]
        ci2BB = wave_data["ci2BB"]
        ci2AB = wave_data["ci2AB"]

        n_mo = self.norb
        n_oa = self.nelec[0]
        n_ob = self.nelec[1]
        n_va = n_mo - n_oa
        n_vb = n_mo - n_ob

        walker_ = jnp.vstack(
            [walker[:n_mo], wave_data["mo_coeff"][1].T.dot(walker[n_mo:, :])]
        )  # put walker_dn in the basis of alpha reference

        Atrial, Btrial = (
            wave_data["mo_coeff"][0][:, :n_oa],
            wave_data["mo_coeff"][1][:, :n_ob],
        )

        bra = jnp.block(
            [
                [Atrial, 0 * Btrial],
                [
                    0 * Atrial,
                    (wave_data["mo_coeff"][1].T @ wave_data["mo_coeff"][1])[:, :n_ob],
                ],
            ]
        )

        # Half green function (U (V^\dag U)^{-1})^T
        #        n_oa n_va n_ob n_vb
        # n_oa (  1    2    3    4  )
        # n_ob (  5    6    7    8  )
        #
        green = (walker_ @ jnp.linalg.inv(bra.T.conj() @ walker_)).T

        # (1, 2)
        green_aa = green[:n_oa, :n_mo]
        # (7, 8)
        green_bb = green[n_oa:, n_mo:]
        # (3, 4)
        green_ab = green[:n_oa, n_mo:]
        # (5, 6)
        green_ba = green[n_oa:, :n_mo]

        # (2)
        green_occ_aa = green_aa[:, n_oa:]
        # (8)
        green_occ_bb = green_bb[:, n_ob:]
        # (4)
        green_occ_ab = green_ab[:, n_ob:]
        # (6)
        green_occ_ba = green_ba[:, n_oa:]

        green_occ = jnp.block(
            [[green_occ_aa, green_occ_ab], [green_occ_ba, green_occ_bb]]
        )

        greenp_aa = jnp.vstack((green_occ_aa, -jnp.eye(n_va)))
        greenp_bb = jnp.vstack((green_occ_bb, -jnp.eye(n_vb)))
        greenp_ab = jnp.vstack((green_occ_ab, -jnp.zeros((n_va, n_vb))))
        greenp_ba = jnp.vstack((green_occ_ba, -jnp.zeros((n_vb, n_va))))

        greenp = jnp.block([[greenp_aa, greenp_ab], [greenp_ba, greenp_bb]])

        h1_aa = (ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0
        h1_bb = ham_data["h1_b"]
        h1 = la.block_diag(h1_aa, h1_bb)

        rot_h1_aa = h1_aa[:n_oa, :]
        rot_h1_bb = h1_bb[:n_ob, :]
        rot_h1 = la.block_diag(rot_h1_aa, rot_h1_bb)

        chol_aa = ham_data["chol"].reshape(-1, n_mo, n_mo)
        chol_bb = ham_data["chol_b"].reshape(-1, n_mo, n_mo)
        nchol = jnp.shape(chol_aa)[0]

        def chol_block(i):
            return la.block_diag(chol_aa[i], chol_bb[i])

        chol = jax.vmap(chol_block)(jnp.arange(nchol))

        rot_chol_aa = chol_aa[:, :n_oa, :]
        rot_chol_bb = chol_bb[:, :n_ob, :]

        def rot_chol_block(i):
            return la.block_diag(rot_chol_aa[i], rot_chol_bb[i])

        rot_chol = jax.vmap(rot_chol_block)(jnp.arange(nchol))

        ci1 = la.block_diag(ci1A, ci1B)

        ci2 = jnp.zeros((n_oa + n_ob, n_va + n_vb, n_oa + n_ob, n_va + n_vb))
        ci2 = lax.dynamic_update_slice(ci2, ci2AA, (0, 0, 0, 0))
        ci2 = lax.dynamic_update_slice(ci2, ci2BB, (n_oa, n_va, n_oa, n_va))
        ci2 = lax.dynamic_update_slice(ci2, ci2AB, (0, 0, n_oa, n_va))
        ci2 = lax.dynamic_update_slice(
            ci2, -jnp.einsum("iajb->jaib", ci2AB), (n_oa, 0, 0, n_va)
        )
        ci2 = lax.dynamic_update_slice(
            ci2, -jnp.einsum("iajb->ibja", ci2AB), (0, n_va, n_oa, 0)
        )
        ci2 = lax.dynamic_update_slice(
            ci2, jnp.einsum("iajb->jbia", ci2AB), (n_oa, n_va, 0, 0)
        )

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # ref
        e1_0 = jnp.einsum("pq,pq->", rot_h1, green)

        # single excitations
        # e1_1 = jnp.einsum("pq,ia,pq,ia->", rot_h1, ci1.conj(), green, green_occ)
        e1_1_0 = jnp.einsum("pq,ia,pq,ia->", rot_h1, ci1, green, green_occ)

        # e1_1 -= jnp.einsum("pq,ia,iq,pa->", h1, ci1.conj(), green, greenp)
        e1_1_1 = -jnp.einsum("pq,ia,iq,pa->", h1, ci1, green, greenp)

        e1_1 = e1_1_0 + e1_1_1

        ## double excitations
        # e1_2 = 2.0 * jnp.einsum("rq,rq,iajb,ia,jb", rot_h1, green, ci2.conj(), green_occ, green_occ)
        e1_2_0 = 2.0 * jnp.einsum(
            "rq,rq,iajb,ia,jb", rot_h1, green, ci2, green_occ, green_occ
        )

        # e1_2 -= 4.0 * jnp.einsum("pq,iajb,pa,iq,jb", h1, ci2.conj(), greenp, green, green_occ)
        e1_2_1 = -4.0 * jnp.einsum(
            "pq,iajb,pa,iq,jb", h1, ci2.conj(), greenp, green, green_occ
        )

        e1_2 = e1_2_0 + e1_2_1
        e1_2 *= 0.25

        # 2 body energy
        # ref
        f = jnp.einsum("gij,jk->gik", rot_chol, green.T, optimize="optimal")
        c = vmap(jnp.trace)(f)
        exc = jnp.sum(vmap(lambda x: x * x.T)(f))
        e2_0 = (jnp.sum(c * c) - exc) / 2.0

        # single excitations
        # e2_1 = jnp.einsum( "gpr,gqs,ia,ir,ps,qa->", chol[:, :nocc, :], chol[:, :, :], ci1.conj(), green, green, greenp)
        e2_1 = jnp.einsum(
            "gpr,gqs,ia,ir,ps,qa->", rot_chol, chol, ci1, green, green, greenp
        )

        # e2_1 -= jnp.einsum( "gpr,gqs,ia,pr,is,qa->", chol[:, :nocc, :], chol[:, :, :], ci1.conj(), green, green, greenp)
        e2_1 -= jnp.einsum(
            "gpr,gqs,ia,pr,is,qa->", rot_chol, chol, ci1, green, green, greenp
        )

        # e2_1 -= jnp.einsum( "gpr,gqs,ia,ir,pa,qs->", chol[:, :, :], chol[:, :nocc, :], ci1.conj(), green, greenp, green)
        e2_1 -= jnp.einsum(
            "gpr,gqs,ia,ir,pa,qs->", chol, rot_chol, ci1, green, greenp, green
        )

        # e2_1 += jnp.einsum( "gpr,gqs,ia,qr,is,pa->", chol[:, :, :], chol[:, :nocc, :], ci1.conj(), green, green, greenp)
        e2_1 += jnp.einsum(
            "gpr,gqs,ia,qr,is,pa->", chol, rot_chol, ci1, green, green, greenp
        )

        # e2_1 += jnp.einsum( "gpr,gqs,ia,pr,ia,qs->", chol[:, :nocc, :], chol[:, :nocc, :], ci1.conj(), green, green_occ, green)
        e2_1 += jnp.einsum(
            "gpr,gqs,ia,pr,ia,qs->", rot_chol, rot_chol, ci1, green, green_occ, green
        )

        # e2_1 -= jnp.einsum( "gpr,gqs,ia,qr,ia,ps->", chol[:, :nocc, :], chol[:, :nocc, :], ci1.conj(), green, green_occ, green)
        e2_1 -= jnp.einsum(
            "gpr,gqs,ia,qr,ia,ps->", rot_chol, rot_chol, ci1, green, green_occ, green
        )

        e2_1 *= 0.5

        ## double excitations
        # e2_2 = 2.0 * jnp.einsum("gpr,gqs,iajb,ir,js,pa,qb->", chol, chol, ci2.conj(), green, green, greenp, greenp)
        e2_2 = 2.0 * jnp.einsum(
            "gpr,gqs,iajb,ir,js,pa,qb->", chol, chol, ci2, green, green, greenp, greenp
        )

        # e2_2 -= 2.0 * jnp.einsum("gpr,gqs,iajb,ir,ps,ja,qb->", chol[:, :nocc, :], chol, ci2.conj(), green, green, green_occ, greenp)
        e2_2 -= 2.0 * jnp.einsum(
            "gpr,gqs,iajb,ir,ps,ja,qb->",
            rot_chol,
            chol,
            ci2,
            green,
            green,
            green_occ,
            greenp,
        )

        # e2_2 += 2.0 * jnp.einsum("gpr,gqs,iajb,ir,qs,ja,pb->", chol, chol[:, :nocc, :], ci2.conj(), green, green, green_occ, greenp)
        e2_2 += 2.0 * jnp.einsum(
            "gpr,gqs,iajb,ir,qs,ja,pb->",
            chol,
            rot_chol,
            ci2,
            green,
            green,
            green_occ,
            greenp,
        )

        ## P_ij
        e2_2 *= 2.0

        # e2_2 += 4.0 * jnp.einsum("gpr,gqs,iajb,pr,is,ja,qb->", chol[:, :nocc, :], chol, ci2.conj(), green, green, green_occ, greenp)
        e2_2 += 4.0 * jnp.einsum(
            "gpr,gqs,iajb,pr,is,ja,qb->",
            rot_chol,
            chol,
            ci2,
            green,
            green,
            green_occ,
            greenp,
        )

        # e2_2 += 2.0 * jnp.einsum("gpr,gqs,iajb,pr,qs,ia,jb->", chol[:, :nocc, :], chol[:, :nocc, :], ci2.conj(), green, green, green_occ, green_occ)
        e2_2 += 2.0 * jnp.einsum(
            "gpr,gqs,iajb,pr,qs,ia,jb->",
            rot_chol,
            rot_chol,
            ci2,
            green,
            green,
            green_occ,
            green_occ,
        )

        ## P_pq
        # e2_2 -= 4.0 * jnp.einsum("gpr,gqs,iajb,qr,is,ja,pb->", chol, chol[:, :nocc, :], ci2.conj(), green, green, green_occ, greenp)
        e2_2 -= 4.0 * jnp.einsum(
            "gpr,gqs,iajb,qr,is,ja,pb->",
            chol,
            rot_chol,
            ci2.conj(),
            green,
            green,
            green_occ,
            greenp,
        )

        # e2_2 -= 2.0 * jnp.einsum("gpr,gqs,iajb,qr,ps,ia,jb->", chol[:, :nocc, :], chol[:, :nocc, :], ci2.conj(), green, green, green_occ, green_occ)
        e2_2 -= 2.0 * jnp.einsum(
            "gpr,gqs,iajb,qr,ps,ia,jb->",
            rot_chol,
            rot_chol,
            ci2,
            green,
            green,
            green_occ,
            green_occ,
        )

        e2_2 *= 0.5 * 0.25

        e = e1_0 + e1_1 + e1_2 + e2_0 + e2_1 + e2_2
        o1 = jnp.einsum("ia,ia", ci1A, green_occ_aa) + jnp.einsum(
            "ia,ia", ci1B, green_occ_bb
        )
        o2 = jnp.einsum("iajb, ia, jb", ci2AA, green_occ_aa, green_occ_aa)
        o2 += jnp.einsum("iajb, ia, jb", ci2BB, green_occ_bb, green_occ_bb)
        o2 += 2.0 * jnp.einsum("iajb, ia, jb", ci2AB, green_occ_aa, green_occ_bb)
        o2 -= 2.0 * jnp.einsum("iajb, ib, ja", ci2AB, green_occ_ab, green_occ_ba)
        overlap = 1.0 + o1 + 0.5 * o2
        e = e / overlap

        return e + e0

    @partial(jit, static_argnums=0)
    def _build_measurement_intermediates(self, ham_data: dict, wave_data: dict) -> dict:
        mo_coeff_b = wave_data["mo_coeff"][1]
        ham_data["h1_b"] = (
            mo_coeff_b.T @ (ham_data["h1"][1] + ham_data["h1"][1].T) @ mo_coeff_b
        ) / 2
        ham_data["chol_b"] = jnp.einsum(
            "pi,gij,jq->gpq",
            mo_coeff_b.T,
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            mo_coeff_b,
        )
        ham_data["lci1_a"] = jnp.einsum(
            "git,pt->gip",
            ham_data["chol"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[0] :],
            wave_data["ci1A"],
            optimize="optimal",
        )
        ham_data["lci1_b"] = jnp.einsum(
            "git,pt->gip",
            ham_data["chol_b"].reshape(-1, self.norb, self.norb)[:, :, self.nelec[1] :],
            wave_data["ci1B"],
            optimize="optimal",
        )
        return ham_data

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_eom_t_auto(wave_function_auto):
    """(r1 + r2 + r1 c1 + r1 c2 + r2 c1) |0>, ad implementation"""

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1e-4
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2, r1, r2 = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["ci2"],
            wave_data["r1"],
            wave_data["r2"],
        )
        green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        green_occ = green[:, nocc:]
        # r1 terms
        # r1 1
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        r1_1 = r1g
        # r1 c1
        c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
        r1_c1_1 = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = ci1 @ green_occ.T
        r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1_c1 = r1_c1_1 + r1_c1_2
        # r1 c2
        c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", ci2, green_occ, green_occ
        )
        r1_c2_1 = r1g * c2g2
        c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
        c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        gc2_g = (c2g_1 - c2g_2) @ green_occ.T
        r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
        r1_c2 = r1_c2_1 + r1_c2_2

        # r2 terms
        # r2 1
        r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", r2, green_occ, green_occ
        )
        r2_1 = r2g2
        # r2 c1
        r2_c1_1 = r2g2 * c1g
        r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
        r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
        gr2_g = (r2g_1 - r2g_2) @ green_occ.T
        r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
        r2_c1 = r2_c1_1 + r2_c1_2

        # # r2 c2
        # r2_c2_1 = r2g2 * c2g2
        # r2_c2_2 = -8 * jnp.einsum("pq,qp", gr2_g, gc2_g)
        # r2_int = jnp.einsum("ptqu,rt,su->prqs", r2, green_occ, green_occ)
        # c2_int = jnp.einsum("ptqu,rt,su->prqs", ci2, green_occ, green_occ)
        # r2_c2_3 = 2 * jnp.einsum("prqs,rpsq", r2_int, c2_int) - jnp.einsum(
        #     "prqs,rqsp", r2_int, c2_int
        # )
        # r2_c2 = r2_c2_1 + r2_c2_2 + r2_c2_3

        overlap_0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1) * overlap_0
        # return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1 + r2_c2) * overlap_0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_eom_auto(wave_function_auto):
    """(r1 + r2) (1 + c1 + c2) |0>, ad implementation"""

    norb: int
    nelec: Tuple[int, int]
    eps: float = 1e-4
    n_chunks: int = 1

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2, r1, r2 = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["ci2"],
            wave_data["r1"],
            wave_data["r2"],
        )
        green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        green_occ = green[:, nocc:]
        # r1 terms
        # r1 1
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        r1_1 = r1g
        # r1 c1
        c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
        r1_c1_1 = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = ci1 @ green_occ.T
        r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1_c1 = r1_c1_1 + r1_c1_2
        # r1 c2
        c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", ci2, green_occ, green_occ
        )
        r1_c2_1 = r1g * c2g2
        c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
        c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        gc2_g = (c2g_1 - c2g_2) @ green_occ.T
        r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
        r1_c2 = r1_c2_1 + r1_c2_2

        # r2 terms
        # r2 1
        r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", r2, green_occ, green_occ
        )
        r2_1 = r2g2
        # r2 c1
        r2_c1_1 = r2g2 * c1g
        r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
        r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
        gr2_g = (r2g_1 - r2g_2) @ green_occ.T
        r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
        r2_c1 = r2_c1_1 + r2_c1_2

        # r2 c2
        r2_c2_1 = r2g2 * c2g2
        r2_c2_2 = -8 * jnp.einsum("pq,qp", gr2_g, gc2_g)
        r2_int = jnp.einsum("ptqu,rt,su->prqs", r2, green_occ, green_occ)
        c2_int = jnp.einsum("ptqu,rt,su->prqs", ci2, green_occ, green_occ)
        r2_c2_3 = 2 * jnp.einsum("prqs,rpsq", r2_int, c2_int) - jnp.einsum(
            "prqs,rqsp", r2_int, c2_int
        )
        r2_c2 = r2_c2_1 + r2_c2_2 + r2_c2_3

        overlap_0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1 + r2_c2) * overlap_0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_eom_t(wave_function):
    """(r1 + r2 + r1 c1 + r1 c2 + r2 c1) |0>, manual implementation"""

    norb: int
    nelec: Tuple[int, int]
    n_chunks: int = 1
    mixed_real_dtype: DTypeLike = jnp.float32
    mixed_complex_dtype: DTypeLike = jnp.complex64

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2, r1, r2 = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["ci2"],
            wave_data["r1"],
            wave_data["r2"],
        )
        green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        green_occ = green[:, nocc:]
        # r1 terms
        # r1 1
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        r1_1 = r1g
        # r1 c1
        c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
        r1_c1_1 = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = ci1 @ green_occ.T
        r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1_c1 = r1_c1_1 + r1_c1_2
        # r1 c2
        c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", ci2, green_occ, green_occ
        )
        r1_c2_1 = r1g * c2g2
        c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
        c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        gc2_g = (c2g_1 - c2g_2) @ green_occ.T
        r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
        r1_c2 = r1_c2_1 + r1_c2_2

        # r2 terms
        # r2 1
        r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", r2, green_occ, green_occ
        )
        r2_1 = r2g2
        # r2 c1
        r2_c1_1 = r2g2 * c1g
        r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
        r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
        gr2_g = (r2g_1 - r2g_2) @ green_occ.T
        r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
        r2_c1 = r2_c1_1 + r2_c1_2

        overlap_0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1) * overlap_0
        # return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1 + r2_c2) * overlap_0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        c1, c2, r1, r2 = (
            jnp.array(wave_data["ci1"]),
            jnp.array(wave_data["ci2"]),
            jnp.array(wave_data["r1"]),
            jnp.array(wave_data["r2"]),
        )
        nocc = self.nelec[0]
        nvirt = self.norb - nocc
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(nvirt)))
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]

        # r1
        # 2: spin
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        # 2: spin
        lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        fb_r1_1 = r1g * lg
        g_r1_gp = green.T @ (r1 @ greenp.T)
        # 2: spin
        fb_r1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_gp)
        fb_r1 = fb_r1_1 + fb_r1_2

        # r1 c1
        # 2: spin
        c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
        r1c1_c = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = c1 @ green_occ.T
        # 2: spin
        r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1c1 = r1c1_c + r1c1_e
        fb_r1c1_1 = r1c1 * lg
        r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
        g_r1_c1_gp = green.T @ (r1_c1 @ greenp.T)
        # 2: spin
        fb_r1c1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_c1_gp)
        fb_r1c1 = fb_r1c1_1 + fb_r1c1_2

        # r2
        # 2: spin, 0.5: r2, 2: permutation
        r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
        # 0.5: r2, 2: permutation
        r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
        r2g = r2g_c - r2g_e
        # 2: spin, 0.5: no permuation
        r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
        fb_r2_1 = lg * r2g2
        g_r2g_gp = green.T @ (r2g @ greenp.T)
        # 2: spin
        fb_r2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2g_gp)
        fb_r2 = fb_r2_1 + fb_r2_2

        # r2 c1
        r2c1_c = r2g2 * c1g
        r2g_g = r2g @ green_occ.T
        # 2: spin
        r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
        r2c1 = r2c1_c + r2c1_e
        fb_r2c1_1 = lg * r2c1
        r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
        g_c1_g = green_occ.T @ c1_g
        # 2: spin, 2: permutation, 0.5: r2
        r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
        # 2: permutation, 0.5: r2
        r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
        g_r2_c1_gp = green.T @ (r2_c1 @ greenp.T)
        # 2: spin
        fb_r2c1_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2_c1_gp)
        fb_r2c1 = fb_r2c1_1 + fb_r2c1_2

        # r1 c2
        # 2: spin, 0.5: c2, 2: permutation
        c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
        # 0.5: c2, 2: permutation
        c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
        c2g = c2g_c - c2g_e
        # 2: spin, 0.5: no permuation
        c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
        r1c2_c = r1g * c2g2
        c2g_g = c2g @ green_occ.T
        # 2: spin
        r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
        r1c2 = r1c2_c + r1c2_e
        fb_r1c2_1 = lg * r1c2
        r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
        g_r1_g = green_occ.T @ r1_g
        # 2: spin, 2: permutation, 0.5: c2
        r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
        # 2: permutation, 0.5: c2
        r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
        g_r1_c2_gp = green.T @ (r1_c2 @ greenp.T)
        # 2: spin
        fb_r1c2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r1_c2_gp)
        fb_r1c2 = fb_r1c2_1 + fb_r1c2_2

        overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2
        fb = (fb_r1 + fb_r1c1 + fb_r2 + fb_r2c1 + fb_r1c2) / overlap
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        c1, c2, r1, r2 = (
            jnp.array(wave_data["ci1"]),
            jnp.array(wave_data["ci2"]),
            jnp.array(wave_data["r1"]),
            jnp.array(wave_data["r2"]),
        )
        nocc = self.nelec[0]
        nvirt = self.norb - nocc
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(nvirt)))
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # r1
        # 2: spin
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        # 2: spin
        h1g = 2 * jnp.einsum("pt,pt", h1[:nocc, :], green)
        e1_r1_1 = r1g * h1g
        gp_h_g = greenp.T @ (h1 @ green.T)
        # 2: spin
        e1_r1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1)
        e1_r1 = e1_r1_1 + e1_r1_2

        # r1 c1
        # 2: spin
        c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
        r1c1_c = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = c1 @ green_occ.T
        # 2: spin
        r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1c1 = r1c1_c + r1c1_e
        e1_r1c1_1 = r1c1 * h1g

        r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
        # 2: spin
        e1_r1c1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1_c1)
        e1_r1c1 = e1_r1c1_1 + e1_r1c1_2

        # r2
        # 2: spin, 0.5: r2, 2: permutation
        r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
        # 0.5: r2, 2: permutation
        r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
        r2g = r2g_c - r2g_e
        # 2: spin, 0.5: no permuation
        r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
        e1_r2_1 = h1g * r2g2
        # 2: spin
        e1_r2_2 = -2.0 * jnp.einsum("pt,tp", r2g, gp_h_g, optimize="optimal")
        e1_r2 = e1_r2_1 + e1_r2_2

        # r2 c1
        r2c1_c = r2g2 * c1g
        r2g_g = r2g @ green_occ.T
        # 2: spin
        r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
        r2c1 = r2c1_c + r2c1_e
        e1_r2c1_1 = h1g * r2c1
        r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
        g_c1_g = green_occ.T @ c1_g
        # 2: spin, 2: permutation, 0.5: r2
        r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
        # 2: permutation, 0.5: r2
        r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
        # 2: spin
        e1_r2c1_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r2_c1, optimize="optimal")
        e1_r2c1 = e1_r2c1_1 + e1_r2c1_2

        # r1 c2
        # 2: spin, 0.5: c2, 2: permutation
        c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
        # 0.5: c2, 2: permutation
        c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
        c2g = c2g_c - c2g_e
        # 2: spin, 0.5: no permuation
        c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
        r1c2_c = r1g * c2g2
        c2g_g = c2g @ green_occ.T
        # 2: spin
        r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
        r1c2 = r1c2_c + r1c2_e
        e1_r1c2_1 = h1g * r1c2
        r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
        g_r1_g = green_occ.T @ r1_g
        # 2: spin, 2: permutation, 0.5: c2
        r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
        # 2: permutation, 0.5: c2
        r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
        # 2: spin
        e1_r1c2_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r1_c2, optimize="optimal")
        e1_r1c2 = e1_r1c2_1 + e1_r1c2_2

        e1 = e1_r1 + e1_r1c1 + e1_r2 + e1_r2c1 + e1_r1c2  # + e1_r2c2

        # 2 body energy
        # 2: spin
        lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        l_g = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        # 0.5: coulomb
        l2g2_c = 0.5 * (lg @ lg)
        l2g2_e = -jnp.sum(vmap(lambda x: x * x.T)(l_g))
        l2g2 = l2g2_c + l2g2_e

        # doing this first to build intermediates
        # r2
        e2_r2_1 = l2g2 * r2g2

        # carry: [e2_r2_, e2_c2_, e2_r1c1_, e2_r2c1_, e2_r1c2_, l2g]
        def loop_over_chol(carry, x):
            chol_i, lg_i, l_g_i = x
            # build intermediate
            gp_l_g_i = greenp.T @ (chol_i @ green.T)
            # 0.5: coulomb, 2: permutation
            l2g_i_c = gp_l_g_i * lg_i
            l2g_i_e = gp_l_g_i @ l_g_i
            l2g_i = l2g_i_c - l2g_i_e
            carry[5] += l2g_i

            gp_l_g_i = gp_l_g_i.astype(self.mixed_complex_dtype)
            # evaluate energy
            # r2
            # 4: spin, 2: permutation, 0.5: r2, 0.5: coulomb
            l2r2_c = 2.0 * jnp.einsum(
                "tp,uq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 2: spin, 2: permutation, 0.5: r2, 0.5: coulomb
            l2r2_e = jnp.einsum(
                "up,tq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r2 = l2r2_c - l2r2_e
            carry[0] += l2r2

            # c2
            # 4: spin, 2: permutation, 0.5: c2, 0.5: coulomb
            l2c2_c = 2.0 * jnp.einsum(
                "tp,uq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 2: spin, 2: permutation, 0.5: c2, 0.5: coulomb
            l2c2_e = jnp.einsum(
                "up,tq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2c2 = l2c2_c - l2c2_e
            carry[1] += l2c2

            # r1 c1
            # 2: spin
            lr1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r1, optimize="optimal")
            lc1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c1, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r1c1_c = lr1 * lc1
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r1.astype(self.mixed_real_dtype),
                c1.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r1c1 = l2r1c1_c - l2r1c1_e
            carry[2] += l2r1c1

            # r2 c1
            # 2: spin
            lr2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r2g, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r2c1_1_c = lc1 * lr2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c1_1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r2g.astype(self.mixed_complex_dtype),
                c1.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r2c1_1 = l2r2c1_1_c - l2r2c1_1_e

            # 2: spin, 0.5: r2, 2: permutation
            lr2_c = 2.0 * jnp.einsum(
                "tp,ptqu->qu",
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 0.5: r2, 2: permutation
            lr2_e = jnp.einsum(
                "tp,puqt->qu",
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lr2 = lr2_c - lr2_e
            lr2_c1 = (lr2 @ green_occ.T) @ c1
            c1_lr2 = c1_g @ lr2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c1_2 = -2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lr2_c1 + c1_lr2, optimize="optimal"
            )
            l2r2c1 = l2r2c1_1 + l2r2c1_2
            carry[3] += l2r2c1

            # r1 c2
            # 2: spin
            lc2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c2g, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r1c2_1_c = lr1 * lc2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c2_1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r1.astype(self.mixed_real_dtype),
                c2g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r1c2_1 = l2r1c2_1_c - l2r1c2_1_e

            # 2: spin, 0.5: c2, 2: permutation
            lc2_c = 2.0 * jnp.einsum(
                "tp,ptqu->qu",
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 0.5: c2, 2: permutation
            lc2_e = jnp.einsum(
                "tp,puqt->qu",
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lc2 = lc2_c - lc2_e
            lc2_r1 = (lc2 @ green_occ.T) @ r1
            r1_lc2 = r1_g @ lc2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c2_2 = -2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lc2_r1 + r1_lc2, optimize="optimal"
            )
            l2r1c2 = l2r1c2_1 + l2r1c2_2
            carry[4] += l2r1c2

            return carry, 0.0

        l2g = jnp.zeros((nvirt, nocc)) + 0.0j
        [e2_r2_3, e2_c2_3, e2_r1c1_3, e2_r2c1_3, e2_r1c2_3, l2g], _ = lax.scan(
            loop_over_chol, [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, l2g], (chol, lg, l_g)
        )
        e2_r2_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r2g, optimize="optimal")
        e2_r2 = e2_r2_1 + e2_r2_2 + e2_r2_3

        # r1
        e2_r1_1 = l2g2 * r1g
        # 2: spin
        e2_r1_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r1, optimize="optimal")
        e2_r1 = e2_r1_1 + e2_r1_2 + e2_r1c1_3

        # r1 c1
        e2_r1c1_1 = l2g2 * r1c1
        e2_r1c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c1, optimize="optimal")
        e2_r1c1 = e2_r1c1_1 + e2_r1c1_2

        # r2 c1
        e2_r2c1_1 = l2g2 * r2c1
        e2_r2c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r2_c1, optimize="optimal")
        e2_r2c1_3 += e2_r2_3 * c1g
        e2_r2c1 = e2_r2c1_1 + e2_r2c1_2 + e2_r2c1_3

        # r1 c2
        e2_r1c2_1 = l2g2 * r1c2
        e2_r1c2_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c2, optimize="optimal")
        e2_r1c2_3 += r1g * e2_c2_3
        e2_r1c2 = e2_r1c2_1 + e2_r1c2_2 + e2_r1c2_3

        e2 = e2_r1 + e2_r2 + e2_r1c1 + e2_r2c1 + e2_r1c2

        overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2  # + r2c2
        return (e1 + e2) / overlap + e0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class cisd_eom(wave_function):
    """(r1 + r2) (1 + c1 + c2) |0>, manual implementation

    Mixed precision is only used for N^6 scaling terms in the energy calculation. Can be turned off by setting
    mixed_real_dtype and mixed_complex_dtype to jnp.float64 and jnp.complex128 respectively.

    Attributes:
        norb: number of orbitals
        nelec: number of electrons as tuple (alpha, beta)
        n_chunks: number of walkers in a batch
        mixed_real_dtype: real dtype of the mixed precision
        mixed_complex_dtype: complex dtype of the mixed precision
    """

    norb: int
    nelec: Tuple[int, int]
    n_chunks: int = 1
    mixed_real_dtype: DTypeLike = jnp.float32
    mixed_complex_dtype: DTypeLike = jnp.complex64

    @partial(jit, static_argnums=0)
    def _calc_overlap_restricted(self, walker: jax.Array, wave_data: dict) -> complex:
        nocc, ci1, ci2, r1, r2 = (
            walker.shape[1],
            wave_data["ci1"],
            wave_data["ci2"],
            wave_data["r1"],
            wave_data["r2"],
        )
        green = (walker.dot(jnp.linalg.inv(walker[: walker.shape[1], :]))).T
        green_occ = green[:, nocc:]
        # r1 terms
        # r1 1
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        r1_1 = r1g
        # r1 c1
        c1g = 2 * jnp.einsum("pt,pt", ci1, green_occ)
        r1_c1_1 = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = ci1 @ green_occ.T
        r1_c1_2 = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1_c1 = r1_c1_1 + r1_c1_2
        # r1 c2
        c2g2 = 2 * jnp.einsum("ptqu,pt,qu", ci2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", ci2, green_occ, green_occ
        )
        r1_c2_1 = r1g * c2g2
        c2g_1 = jnp.einsum("ptqu,qu->pt", ci2, green_occ)
        c2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", ci2, green_occ)
        gc2_g = (c2g_1 - c2g_2) @ green_occ.T
        r1_c2_2 = -4 * jnp.einsum("pq,qp", r1_g, gc2_g)
        r1_c2 = r1_c2_1 + r1_c2_2

        # r2 terms
        # r2 1
        r2g2 = 2 * jnp.einsum("ptqu,pt,qu", r2, green_occ, green_occ) - jnp.einsum(
            "ptqu,pu,qt", r2, green_occ, green_occ
        )
        r2_1 = r2g2
        # r2 c1
        r2_c1_1 = r2g2 * c1g
        r2g_1 = jnp.einsum("ptqu,qu->pt", r2, green_occ)
        r2g_2 = 0.5 * jnp.einsum("ptqu,pu->qt", r2, green_occ)
        gr2_g = (r2g_1 - r2g_2) @ green_occ.T
        r2_c1_2 = -4 * jnp.einsum("pq,qp", gr2_g, c1_g)
        r2_c1 = r2_c1_1 + r2_c1_2

        # r2 c2
        r2_c2_1 = r2g2 * c2g2
        r2_c2_2 = -8 * jnp.einsum("pq,qp", gr2_g, gc2_g)
        r2_int = jnp.einsum("ptqu,rt,su->prqs", r2, green_occ, green_occ)
        c2_int = jnp.einsum("ptqu,rt,su->prqs", ci2, green_occ, green_occ)
        r2_c2_3 = 2 * jnp.einsum("prqs,rpsq", r2_int, c2_int) - jnp.einsum(
            "prqs,rqsp", r2_int, c2_int
        )
        r2_c2 = r2_c2_1 + r2_c2_2 + r2_c2_3

        overlap_0 = jnp.linalg.det(walker[: walker.shape[1], :]) ** 2
        return (r1_1 + r1_c1 + r1_c2 + r2_1 + r2_c1 + r2_c2) * overlap_0

    @partial(jit, static_argnums=0)
    def _calc_force_bias_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        c1, c2, r1, r2 = (
            jnp.array(wave_data["ci1"]),
            jnp.array(wave_data["ci2"]),
            jnp.array(wave_data["r1"]),
            jnp.array(wave_data["r2"]),
        )
        nocc = self.nelec[0]
        nvirt = self.norb - nocc
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(nvirt)))
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]

        # r1
        # 2: spin
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        # 2: spin
        lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        fb_r1_1 = r1g * lg
        g_r1_gp = green.T @ (r1 @ greenp.T)
        # 2: spin
        fb_r1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_gp)
        fb_r1 = fb_r1_1 + fb_r1_2

        # r1 c1
        # 2: spin
        c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
        r1c1_c = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = c1 @ green_occ.T
        # 2: spin
        r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1c1 = r1c1_c + r1c1_e
        fb_r1c1_1 = r1c1 * lg
        r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
        g_r1_c1_gp = green.T @ (r1_c1 @ greenp.T)
        # 2: spin
        fb_r1c1_2 = -2 * jnp.einsum("gij,ji", chol, g_r1_c1_gp)
        fb_r1c1 = fb_r1c1_1 + fb_r1c1_2

        # r2
        # 2: spin, 0.5: r2, 2: permutation
        r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
        # 0.5: r2, 2: permutation
        r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
        r2g = r2g_c - r2g_e
        # 2: spin, 0.5: no permuation
        r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
        fb_r2_1 = lg * r2g2
        g_r2g_gp = green.T @ (r2g @ greenp.T)
        # 2: spin
        fb_r2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2g_gp)
        fb_r2 = fb_r2_1 + fb_r2_2

        # r2 c1
        r2c1_c = r2g2 * c1g
        r2g_g = r2g @ green_occ.T
        # 2: spin
        r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
        r2c1 = r2c1_c + r2c1_e
        fb_r2c1_1 = lg * r2c1
        r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
        g_c1_g = green_occ.T @ c1_g
        # 2: spin, 2: permutation, 0.5: r2
        r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
        # 2: permutation, 0.5: r2
        r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
        g_r2_c1_gp = green.T @ (r2_c1 @ greenp.T)
        # 2: spin
        fb_r2c1_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2_c1_gp)
        fb_r2c1 = fb_r2c1_1 + fb_r2c1_2

        # r1 c2
        # 2: spin, 0.5: c2, 2: permutation
        c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
        # 0.5: c2, 2: permutation
        c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
        c2g = c2g_c - c2g_e
        # 2: spin, 0.5: no permuation
        c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
        r1c2_c = r1g * c2g2
        c2g_g = c2g @ green_occ.T
        # 2: spin
        r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
        r1c2 = r1c2_c + r1c2_e
        fb_r1c2_1 = lg * r1c2
        r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
        g_r1_g = green_occ.T @ r1_g
        # 2: spin, 2: permutation, 0.5: c2
        r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
        # 2: permutation, 0.5: c2
        r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
        g_r1_c2_gp = green.T @ (r1_c2 @ greenp.T)
        # 2: spin
        fb_r1c2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r1_c2_gp)
        fb_r1c2 = fb_r1c2_1 + fb_r1c2_2

        # r2 c2
        r2c2_c = r2g2 * c2g2
        # 2: spin
        r2c2_e_1 = -2.0 * jnp.einsum("pq,qp", r2g_g, c2g_g, optimize="optimal")
        # 0.5: r2
        r2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", r2, green_occ)
        r2_g_g = jnp.einsum("prqu,su->prqs", r2_g, green_occ)
        # del r2_g
        # 0.5: c2
        c2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", c2, green_occ)
        c2_g_g = jnp.einsum("prqu,su->prqs", c2_g, green_occ)
        # del c2_g
        # 4: spin, 2: permutation
        r2c2_e_2_c = 8.0 * jnp.einsum("prqs,rpsq", r2_g_g, c2_g_g, optimize="optimal")
        # 2: spin, 2: permutation
        r2c2_e_2_e = -4.0 * jnp.einsum("prqs,rqsp", r2_g_g, c2_g_g, optimize="optimal")
        r2c2_e_2 = r2c2_e_2_c + r2c2_e_2_e
        r2c2 = r2c2_c + r2c2_e_1 + r2c2_e_2
        fb_r2c2_1 = lg * r2c2
        r2_c2 = r2g2 * c2g + r2g * c2g2 - r2g_g @ c2g - c2g_g @ r2g
        # 2: spin, 2: permutation
        r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", r2g_g, c2_g, optimize="optimal")
        r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", c2g_g, r2_g, optimize="optimal")
        # 2: permutation
        r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", r2g_g, c2_g, optimize="optimal")
        r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", c2g_g, r2_g, optimize="optimal")
        # 2: spin, 4: permutation
        r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", r2_g_g, c2_g, optimize="optimal")
        r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", c2_g_g, r2_g, optimize="optimal")
        # 4: permutation
        r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", r2_g_g, c2_g, optimize="optimal")
        r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", c2_g_g, r2_g, optimize="optimal")
        g_r2_c2_gp = green.T @ (r2_c2 @ greenp.T)
        # 2: spin
        fb_r2c2_2 = -2.0 * jnp.einsum("gij,ji", chol, g_r2_c2_gp)
        fb_r2c2 = fb_r2c2_1 + fb_r2c2_2

        overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2 + r2c2
        fb = (fb_r1 + fb_r1c1 + fb_r2 + fb_r2c1 + fb_r1c2 + fb_r2c2) / overlap
        return fb

    @partial(jit, static_argnums=0)
    def _calc_energy_restricted(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> complex:
        c1, c2, r1, r2 = (
            jnp.array(wave_data["ci1"]),
            jnp.array(wave_data["ci2"]),
            jnp.array(wave_data["r1"]),
            jnp.array(wave_data["r2"]),
        )
        nocc = self.nelec[0]
        nvirt = self.norb - nocc
        green = (walker.dot(jnp.linalg.inv(walker[:nocc, :]))).T
        green_occ = green[:, nocc:].copy()
        greenp = jnp.vstack((green_occ, -jnp.eye(nvirt)))
        h1 = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0
        chol = ham_data["chol"].reshape(-1, self.norb, self.norb)
        rot_chol = chol[:, : self.nelec[0], :]

        # 0 body energy
        e0 = ham_data["h0"]

        # 1 body energy
        # r1
        # 2: spin
        r1g = 2 * jnp.einsum("pt,pt", r1, green_occ)
        # 2: spin
        h1g = 2 * jnp.einsum("pt,pt", h1[:nocc, :], green)
        e1_r1_1 = r1g * h1g
        gp_h_g = greenp.T @ (h1 @ green.T)
        # 2: spin
        e1_r1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1)
        e1_r1 = e1_r1_1 + e1_r1_2

        # r1 c1
        # 2: spin
        c1g = 2 * jnp.einsum("pt,pt", c1, green_occ)
        r1c1_c = r1g * c1g
        r1_g = r1 @ green_occ.T
        c1_g = c1 @ green_occ.T
        # 2: spin
        r1c1_e = -2 * jnp.einsum("pq,qp", r1_g, c1_g)
        r1c1 = r1c1_c + r1c1_e
        e1_r1c1_1 = r1c1 * h1g

        r1_c1 = r1 * c1g + r1g * c1 - r1_g @ c1 - c1_g @ r1
        # 2: spin
        e1_r1c1_2 = -2 * jnp.einsum("tp,pt", gp_h_g, r1_c1)
        e1_r1c1 = e1_r1c1_1 + e1_r1c1_2

        # r2
        # 2: spin, 0.5: r2, 2: permutation
        r2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", r2, green_occ)
        # 0.5: r2, 2: permutation
        r2g_e = jnp.einsum("ptqu,pu->qt", r2, green_occ)
        r2g = r2g_c - r2g_e
        # 2: spin, 0.5: no permuation
        r2g2 = jnp.einsum("qu,qu", r2g, green_occ, optimize="optimal")
        e1_r2_1 = h1g * r2g2
        # 2: spin
        e1_r2_2 = -2.0 * jnp.einsum("pt,tp", r2g, gp_h_g, optimize="optimal")
        e1_r2 = e1_r2_1 + e1_r2_2

        # r2 c1
        r2c1_c = r2g2 * c1g
        r2g_g = r2g @ green_occ.T
        # 2: spin
        r2c1_e = -2.0 * jnp.einsum("pq,qp", r2g_g, c1_g, optimize="optimal")
        r2c1 = r2c1_c + r2c1_e
        e1_r2c1_1 = h1g * r2c1
        r2_c1 = r2g * c1g + r2g2 * c1 - r2g_g @ c1 - c1_g @ r2g
        g_c1_g = green_occ.T @ c1_g
        # 2: spin, 2: permutation, 0.5: r2
        r2_c1 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_c1_g, r2, optimize="optimal")
        # 2: permutation, 0.5: r2
        r2_c1 += jnp.einsum("tq,ptqu->pu", g_c1_g, r2, optimize="optimal")
        # 2: spin
        e1_r2c1_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r2_c1, optimize="optimal")
        e1_r2c1 = e1_r2c1_1 + e1_r2c1_2

        # r1 c2
        # 2: spin, 0.5: c2, 2: permutation
        c2g_c = 2.0 * jnp.einsum("ptqu,pt->qu", c2, green_occ)
        # 0.5: c2, 2: permutation
        c2g_e = jnp.einsum("ptqu,pu->qt", c2, green_occ)
        c2g = c2g_c - c2g_e
        # 2: spin, 0.5: no permuation
        c2g2 = jnp.einsum("qu,qu", c2g, green_occ, optimize="optimal")
        r1c2_c = r1g * c2g2
        c2g_g = c2g @ green_occ.T
        # 2: spin
        r1c2_e = -2.0 * jnp.einsum("pq,qp", r1_g, c2g_g, optimize="optimal")
        r1c2 = r1c2_c + r1c2_e
        e1_r1c2_1 = h1g * r1c2
        r1_c2 = r1 * c2g2 + r1g * c2g - r1_g @ c2g - c2g_g @ r1
        g_r1_g = green_occ.T @ r1_g
        # 2: spin, 2: permutation, 0.5: c2
        r1_c2 -= 2.0 * jnp.einsum("tp,ptqu->qu", g_r1_g, c2, optimize="optimal")
        # 2: permutation, 0.5: c2
        r1_c2 += jnp.einsum("tq,ptqu->pu", g_r1_g, c2, optimize="optimal")
        # 2: spin
        e1_r1c2_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r1_c2, optimize="optimal")
        e1_r1c2 = e1_r1c2_1 + e1_r1c2_2

        # r2 c2
        r2c2_c = r2g2 * c2g2
        # 2: spin
        r2c2_e_1 = -2.0 * jnp.einsum("pq,qp", r2g_g, c2g_g, optimize="optimal")
        # 0.5: r2
        r2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", r2, green_occ)
        r2_g_g = jnp.einsum("prqu,su->prqs", r2_g, green_occ)
        # 0.5: c2
        c2_g = 0.5 * jnp.einsum("ptqu,rt->prqu", c2, green_occ, optimize="optimal")
        c2_g_g = jnp.einsum("prqu,su->prqs", c2_g, green_occ, optimize="optimal")
        # 4: spin, 2: permutation
        r2c2_e_2_c = 8.0 * jnp.einsum("prqs,rpsq", r2_g_g, c2_g_g, optimize="optimal")
        # 2: spin, 2: permutation
        r2c2_e_2_e = -4.0 * jnp.einsum("prqs,rqsp", r2_g_g, c2_g_g, optimize="optimal")
        r2c2_e_2 = r2c2_e_2_c + r2c2_e_2_e
        r2c2 = r2c2_c + r2c2_e_1 + r2c2_e_2
        e1_r2c2_1 = h1g * r2c2
        r2_c2 = r2g2 * c2g + r2g * c2g2 - r2g_g @ c2g - c2g_g @ r2g
        # 2: spin, 2: permutation
        r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", r2g_g, c2_g, optimize="optimal")
        r2_c2 -= 4.0 * jnp.einsum("pr,rpqu->qu", c2g_g, r2_g, optimize="optimal")
        # 2: permutation
        r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", r2g_g, c2_g, optimize="optimal")
        r2_c2 += 2.0 * jnp.einsum("pr,qpru->qu", c2g_g, r2_g, optimize="optimal")
        # 2: spin, 4: permutation
        r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", r2_g_g, c2_g, optimize="optimal")
        r2_c2 += 8.0 * jnp.einsum("pqrs,qpst->rt", c2_g_g, r2_g, optimize="optimal")
        # 4: permutation
        r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", r2_g_g, c2_g, optimize="optimal")
        r2_c2 -= 4.0 * jnp.einsum("pqrs,spqt->rt", c2_g_g, r2_g, optimize="optimal")
        e1_r2c2_2 = -2.0 * jnp.einsum("tp,pt", gp_h_g, r2_c2, optimize="optimal")
        e1_r2c2 = e1_r2c2_1 + e1_r2c2_2

        e1 = e1_r1 + e1_r1c1 + e1_r2 + e1_r2c1 + e1_r1c2 + e1_r2c2

        # 2 body energy
        # 2: spin
        lg = 2.0 * jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
        l_g = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
        # 0.5: coulomb
        l2g2_c = 0.5 * (lg @ lg)
        l2g2_e = -jnp.sum(vmap(lambda x: x * x.T)(l_g))
        l2g2 = l2g2_c + l2g2_e

        # doing this first to build intermediates
        # r2
        e2_r2_1 = l2g2 * r2g2

        # carry: [e2_r2_, e2_c2_, e2_r1c1_, e2_r2c1_, e2_r1c2_, e2_r2c2_, l2g]
        def loop_over_chol(carry, x):
            chol_i, lg_i, l_g_i = x
            # build intermediate
            gp_l_g_i = greenp.T @ (chol_i @ green.T)
            # 0.5: coulomb, 2: permutation
            l2g_i_c = gp_l_g_i * lg_i
            l2g_i_e = gp_l_g_i @ l_g_i
            l2g_i = l2g_i_c - l2g_i_e
            carry[6] += l2g_i

            gp_l_g_i = gp_l_g_i.astype(self.mixed_complex_dtype)
            # evaluate energy
            # r2
            # 4: spin, 2: permutation, 0.5: r2, 0.5: coulomb
            l2r2_c = 2.0 * jnp.einsum(
                "tp,uq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 2: spin, 2: permutation, 0.5: r2, 0.5: coulomb
            l2r2_e = jnp.einsum(
                "up,tq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r2 = l2r2_c - l2r2_e
            carry[0] += l2r2

            # c2
            # 4: spin, 2: permutation, 0.5: c2, 0.5: coulomb
            l2c2_c = 2.0 * jnp.einsum(
                "tp,uq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 2: spin, 2: permutation, 0.5: c2, 0.5: coulomb
            l2c2_e = jnp.einsum(
                "up,tq,ptqu",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2c2 = l2c2_c - l2c2_e
            carry[1] += l2c2

            # r1 c1
            # 2: spin
            lr1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r1, optimize="optimal")
            lc1 = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c1, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r1c1_c = lr1 * lc1
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r1.astype(self.mixed_real_dtype),
                c1.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r1c1 = l2r1c1_c - l2r1c1_e
            carry[2] += l2r1c1

            # r2 c1
            # 2: spin
            lr2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, r2g, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r2c1_1_c = lc1 * lr2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c1_1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r2g.astype(self.mixed_complex_dtype),
                c1.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            l2r2c1_1 = l2r2c1_1_c - l2r2c1_1_e

            # 2: spin, 0.5: r2, 2: permutation
            lr2_c = -2.0 * jnp.einsum(
                "tp,ptqu->qu",
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 0.5: r2, 2: permutation
            lr2_e = jnp.einsum(
                "tp,puqt->qu",
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lr2 = lr2_c + lr2_e
            lr2_c1 = (lr2 @ green_occ.T) @ c1
            c1_lr2 = c1_g @ lr2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c1_2 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lr2_c1 + c1_lr2, optimize="optimal"
            )
            l2r2c1 = l2r2c1_1 + l2r2c1_2
            carry[3] += l2r2c1

            # r1 c2
            # 2: spin
            lc2g = -2.0 * jnp.einsum("tp,pt", gp_l_g_i, c2g, optimize="optimal")
            # 2: permutation, 0.5: coulomb
            l2r1c2_1_c = lr1 * lc2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c2_1_e = 2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r1.astype(self.mixed_real_dtype),
                c2g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r1c2_1 = l2r1c2_1_c - l2r1c2_1_e

            # 2: spin, 0.5: c2, 2: permutation
            lc2_c = -2.0 * jnp.einsum(
                "tp,ptqu->qu",
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            # 0.5: c2, 2: permutation
            lc2_e = jnp.einsum(
                "tp,puqt->qu",
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                optimize="optimal",
            )
            lc2 = lc2_c + lc2_e
            lc2_r1 = (lc2 @ green_occ.T) @ r1
            r1_lc2 = r1_g @ lc2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r1c2_2 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lc2_r1 + r1_lc2, optimize="optimal"
            )
            l2r1c2 = l2r1c2_1 + l2r1c2_2
            carry[4] += l2r1c2

            # r2 c2
            # 2: permutation, 0.5: coulomb
            l2r2c2_1_c = lr2g * lc2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_1_e = -2.0 * jnp.einsum(
                "up,tq,pt,qu",
                gp_l_g_i,
                gp_l_g_i,
                r2g.astype(self.mixed_complex_dtype),
                c2g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_1 = l2r2c2_1_c + l2r2c2_1_e

            lr2_g = lr2 @ green_occ.T
            lr2_c2g = lr2_g @ c2g
            # 2: spin, 2: permutaion, 0.5: coulomb
            l2r2c2_2_1 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lr2_c2g, optimize="optimal"
            )
            c2g_lr2 = c2g_g @ lr2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_2_2 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, c2g_lr2, optimize="optimal"
            )
            lc2_g = lc2 @ green_occ.T
            lc2_r2g = lc2_g @ r2g
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_2_3 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, lc2_r2g, optimize="optimal"
            )
            r2g_lc2 = r2g_g @ lc2
            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_2_4 = 2.0 * jnp.einsum(
                "tp,pt", gp_l_g_i, r2g_lc2, optimize="optimal"
            )
            l2r2c2_2 = l2r2c2_2_1 + l2r2c2_2_2 + l2r2c2_2_3 + l2r2c2_2_4

            # 4: spin, 0.5: coulomb, 0.5: r2, 4: permutation
            l2r2c2_3_1_1_c = 4.0 * jnp.einsum(
                "vp,wq,rvsw,prqs",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                c2_g_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            # 2: spin, 0.5: coulomb, 0.5; r2, 4: permutation
            l2r2c2_3_1_1_e = -2.0 * jnp.einsum(
                "vp,wq,svrw,prqs",
                gp_l_g_i,
                gp_l_g_i,
                r2.astype(self.mixed_real_dtype),
                c2_g_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_3_1_1 = l2r2c2_3_1_1_c + l2r2c2_3_1_1_e
            # 4: spin, 0.5: coulomb, 0.5: c2, 4: permutation
            l2r2c2_3_1_2_c = 4.0 * jnp.einsum(
                "vp,wq,rvsw,prqs",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                r2_g_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            # 2: spin, 0.5: coulomb, 0.5: c2, 4: permutation
            l2r2c2_3_1_2_e = -2.0 * jnp.einsum(
                "vp,wq,svrw,prqs",
                gp_l_g_i,
                gp_l_g_i,
                c2.astype(self.mixed_real_dtype),
                r2_g_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_3_1_2 = l2r2c2_3_1_2_c + l2r2c2_3_1_2_e
            l2r2c2_3_1 = l2r2c2_3_1_1 + l2r2c2_3_1_2

            # 4: spin, 8: permutation, 0.5: coulomb
            l2r2c2_3_2 = 16.0 * jnp.einsum(
                "vq,us,rpsv,prqu",
                gp_l_g_i,
                gp_l_g_i,
                r2_g.astype(self.mixed_complex_dtype),
                c2_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )

            # 2: spin, 8: permutation, 0.5: coulomb
            l2r2c2_3_3_1 = -8.0 * jnp.einsum(
                "vq,us,sprv,prqu",
                gp_l_g_i,
                gp_l_g_i,
                r2_g.astype(self.mixed_complex_dtype),
                c2_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_3_3_2 = -8.0 * jnp.einsum(
                "vq,us,sprv,prqu",
                gp_l_g_i,
                gp_l_g_i,
                c2_g.astype(self.mixed_complex_dtype),
                r2_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )
            l2r2c2_3_3 = l2r2c2_3_3_1 + l2r2c2_3_3_2

            # 4: spin, 8: permutation, 0.5: coulomb
            l2r2c2_3_4 = 16.0 * jnp.einsum(
                "vp,us,sqrv,prqu",
                gp_l_g_i,
                gp_l_g_i,
                r2_g.astype(self.mixed_complex_dtype),
                c2_g.astype(self.mixed_complex_dtype),
                optimize="optimal",
            )

            l2r2c2_3 = l2r2c2_3_1 + l2r2c2_3_2 + l2r2c2_3_3 + l2r2c2_3_4

            # 2: spin, 2: permutation, 0.5: coulomb
            l2r2c2_4 = -2.0 * jnp.einsum("pq,qp", lr2_g, lc2_g, optimize="optimal")

            l2r2c2 = l2r2c2_1 + l2r2c2_2 + l2r2c2_3 + l2r2c2_4
            carry[5] += l2r2c2

            return carry, 0.0

        l2g = jnp.zeros((nvirt, nocc)) + 0.0j
        [e2_r2_3, e2_c2_3, e2_r1c1_3, e2_r2c1_3, e2_r1c2_3, e2_r2c2_3, l2g], _ = (
            lax.scan(
                loop_over_chol,
                [0.0j, 0.0j, 0.0j, 0.0j, 0.0j, 0.0j, l2g],
                (chol, lg, l_g),
            )
        )
        e2_r2_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r2g, optimize="optimal")
        e2_r2 = e2_r2_1 + e2_r2_2 + e2_r2_3

        # r1
        e2_r1_1 = l2g2 * r1g
        # 2: spin
        e2_r1_2 = -2.0 * jnp.einsum("tp,pt->", l2g, r1, optimize="optimal")
        e2_r1 = e2_r1_1 + e2_r1_2 + e2_r1c1_3

        # r1 c1
        e2_r1c1_1 = l2g2 * r1c1
        e2_r1c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c1, optimize="optimal")
        e2_r1c1 = e2_r1c1_1 + e2_r1c1_2

        # r2 c1
        e2_r2c1_1 = l2g2 * r2c1
        e2_r2c1_2 = -2.0 * jnp.einsum("tp,pt", l2g, r2_c1, optimize="optimal")
        e2_r2c1_3 += e2_r2_3 * c1g
        e2_r2c1 = e2_r2c1_1 + e2_r2c1_2 + e2_r2c1_3

        # r1 c2
        e2_r1c2_1 = l2g2 * r1c2
        e2_r1c2_2 = -2.0 * jnp.einsum("tp,pt", l2g, r1_c2, optimize="optimal")
        e2_r1c2_3 += r1g * e2_c2_3
        e2_r1c2 = e2_r1c2_1 + e2_r1c2_2 + e2_r1c2_3

        # r2 c2
        e2_r2c2_1 = l2g2 * r2c2
        e2_r2c2_2 = -2.0 * jnp.einsum("tp,pt", l2g, r2_c2, optimize="optimal")
        e2_r2c2_3 += r2g2 * e2_c2_3
        e2_r2c2_3 += c2g2 * e2_r2_3
        e2_r2c2 = e2_r2c2_1 + e2_r2c2_2 + e2_r2c2_3

        e2 = e2_r1 + e2_r2 + e2_r1c1 + e2_r2c1 + e2_r1c2 + e2_r2c2

        overlap = r1g + r1c1 + r2g2 + r2c1 + r1c2 + r2c2
        return (e1 + e2) / overlap + e0

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class rhf_lno(rhf, wave_function):
    """Class for the restricted Hartree-Fock wave function with LNO.

    The corresponding wave_data contains "mo_coeff", a list of two jax.Arrays of shape (norb, nelec[sigma]).
    The measurement methods make use of half-rotated integrals which are stored in ham_data.
    ham_data should contain "rot_h1" and "rot_chol" intermediates which are the half-rotated
    one-body and two-body integrals respectively.

    """

    @singledispatchmethod
    def calc_orbenergy(self, walkers, ham_data: dict, wave_data: dict) -> jax.Array:
        """Calculate the energy < psi_T | H | walker > / < psi_T | walker > for a batch of walkers.

        Args:
            walkers : list or jax.Array
                The batched walkers.
            ham_data : dict
                The hamiltonian data.
            wave_data : dict
                The trial wave function data.

        Returns:
            jax.Array: The energy.
        """
        raise NotImplementedError("Walker type not supported")

    @calc_orbenergy.register
    def _(self, walkers: RHFWalkers, ham_data: dict, wave_data: dict) -> jax.Array:
        return walkers.apply_chunked(
            self._calc_orbenergy, self.n_chunks, ham_data, wave_data
        )

    @partial(jit, static_argnums=0)
    def _calc_orbenergy(
        self, walker: jax.Array, ham_data: dict, wave_data: dict
    ) -> jax.Array:
        h0, rot_h1, rot_chol = ham_data["h0"], ham_data["rot_h1"], ham_data["rot_chol"]
        ene0 = 0
        m = jnp.dot(wave_data["prjlo"].T, wave_data["prjlo"])
        nocc = rot_h1.shape[0]
        green_walker = self._calc_green(walker, wave_data)
        f = jnp.einsum(
            "gij,jk->gik",
            rot_chol[:, :nocc, nocc:],
            green_walker.T[nocc:, :nocc],
            optimize="optimal",
        )
        c = vmap(jnp.trace)(f)

        eneo2Jt = jnp.einsum("Gxk,xk,G->", f, m, c) * 2
        eneo2ext = jnp.einsum("Gxy,Gyk,xk->", f, f, m)
        return eneo2Jt - eneo2ext

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
