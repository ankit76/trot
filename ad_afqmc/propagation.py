import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, lax, random, vmap
from jax._src.typing import DTypeLike

from ad_afqmc import wavefunctions, sr
from ad_afqmc.walkers import GHFWalkers, RHFWalkers, UHFWalkers, walker_batch
from ad_afqmc.wavefunctions import wave_function


@dataclass
class propagator(ABC):
    """Abstract base class for propagator classes.

    Attributes:
        dt: time step
        n_walkers: number of walkers
    """

    dt: float = 0.01
    n_walkers: int = 50

    @abstractmethod
    def init_prop_data(
        self,
        trial: wave_function,
        wave_data: Any,
        ham_data: dict,
        seed: int,
        init_walkers: Optional[Union[jax.Array, List]] = None,
    ) -> dict:
        """Initialize propagation data. If walkers are not provided they are generated
        using the trial.

        Args:
            trial: trial wave function handler
            wave_data: dictionary containing the wave function data
            ham_data: dictionary containing the Hamiltonian data
            seed: random seed for key generation
            init_walkers: initial walkers

        Returns:
            prop_data: dictionary containing the propagation data
        """
        pass

    @abstractmethod
    def propagate_constrained(
        self,
        trial: wave_function,
        ham_data: dict,
        prop_data: dict,
        fields: jax.Array,
        wave_data: dict,
    ) -> dict:
        """Constrained propagation.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            fields: auxiliary fields
            wave_data: wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        pass

    def propagate_free(
        self,
        trial: wave_function,
        ham_data: dict,
        prop_data: dict,
        fields: jax.Array,
        wave_data: dict,
    ) -> dict:
        """Free propagation.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            fields: auxiliary fields
            wave_data: wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        raise NotImplementedError(
            "Free propagation is not implemented for this propagator."
        )

    def _build_propagation_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        """Build intermediates for propagation."""
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_afqmc(propagator):
    """Propagator for phaseless AFQMC."""

    dt: float = 0.01
    n_walkers: int = 50
    n_exp_terms: int = 6
    n_chunks: int = 1
    vhs_real_dtype: DTypeLike = jnp.float64
    vhs_complex_dtype: DTypeLike = jnp.complex128
    walker_type: str = "restricted"

    def init_prop_data(
        self,
        trial: wave_function,
        wave_data: dict,
        ham_data: dict,
        seed: int,
        init_walkers: Optional[walker_batch] = None,
    ) -> dict:
        prop_data = {}
        prop_data["weights"] = jnp.ones(self.n_walkers)
        prop_data["key"] = random.PRNGKey(seed)
        if init_walkers is not None:
            prop_data["walkers"] = init_walkers
        else:
            prop_data["walkers"] = trial.get_init_walkers(
                wave_data, self.n_walkers, self.walker_type
            )
        if "e_estimate" in ham_data:
            prop_data["e_estimate"] = ham_data["e_estimate"]
        else:
            energy_samples = jnp.real(
                trial.calc_energy(prop_data["walkers"], ham_data, wave_data)
            )
            e_estimate = jnp.array(jnp.sum(energy_samples) / self.n_walkers)
            prop_data["e_estimate"] = e_estimate
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]
        prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["normed_overlaps"] = prop_data["overlaps"]
        prop_data["norms"] = jnp.ones(self.n_walkers) + 0.0j
        return prop_data

    @partial(jit, static_argnums=(0,))
    def _apply_trotprop_single_det_vhs(
        self, walker_i: jax.Array, vhs_i: jax.Array, exp_h1: jax.Array
    ) -> jax.Array:
        """Apply the Trotterized propagator to a single det."""
        walker_i = exp_h1 @ walker_i
        fact_recip = jnp.array(
            [1.0 / math.factorial(n + 1) for n in range(self.n_exp_terms - 1)]
        )
        fact_recip = fact_recip.reshape(-1, 1, 1)

        def scanned_fun(carry, x):
            carry = vhs_i.astype(self.vhs_complex_dtype) @ carry
            return carry, carry

        _, vhs_n_walker = lax.scan(
            scanned_fun,
            walker_i.astype(self.vhs_complex_dtype),
            jnp.arange(1, self.n_exp_terms),
        )
        walker_i = walker_i + jnp.sum(vhs_n_walker * fact_recip, axis=0)
        walker_i = exp_h1 @ walker_i
        return walker_i

    @partial(jit, static_argnums=(0,))
    def _apply_trotprop_single_det(
        self,
        walker_i: jax.Array,
        fields_i: jax.Array,
        ham_data: dict,
    ) -> jax.Array:
        if (
            self.walker_type == "generalized"
        ):  # mixed precision not used for ghf walkers, would require fixing complex dtype handling
            vhs_i = (
                1.0j
                * jnp.sqrt(self.dt)
                * (fields_i @ ham_data["chol"]).reshape(
                    walker_i.shape[0], walker_i.shape[0]
                )
            )
        else:
            vhs_i = (
                1.0j
                * jnp.sqrt(self.dt)
                * (
                    fields_i.astype(self.vhs_complex_dtype)
                    @ ham_data["chol"].astype(self.vhs_real_dtype)
                ).reshape(walker_i.shape[0], walker_i.shape[0])
            )
        return self._apply_trotprop_single_det_vhs(walker_i, vhs_i, ham_data["exp_h1"])

    @partial(jit, static_argnums=(0,))
    def _apply_trotprop_single_det_uhf(
        self,
        walker_i_up: jax.Array,
        walker_i_dn: jax.Array,
        fields_i: jax.Array,
        ham_data: dict,
    ) -> Tuple[jax.Array, jax.Array]:
        """Apply the Trotterized propagator to a single det for UHF."""
        vhs_i = (
            1.0j
            * jnp.sqrt(self.dt)
            * (
                fields_i.astype(self.vhs_complex_dtype)
                @ ham_data["chol"].astype(self.vhs_real_dtype)
            ).reshape(walker_i_up.shape[0], walker_i_up.shape[0])
        )
        walker_i_up = self._apply_trotprop_single_det_vhs(
            walker_i_up, vhs_i, ham_data["exp_h1"][0]
        )
        walker_i_dn = self._apply_trotprop_single_det_vhs(
            walker_i_dn, vhs_i, ham_data["exp_h1"][1]
        )
        return walker_i_up, walker_i_dn

    @singledispatchmethod
    def _apply_trotprop(
        self, walkers: walker_batch, fields: jax.Array, ham_data: dict
    ) -> walker_batch:
        """Apply the Trotterized propagator to a batch of walkers.

        Args:
            walkers: batch of walkers
            fields: auxiliary fields
            ham_data: dictionary containing the Hamiltonian data

        Returns:
            walkers: batch of walkers after propagation
        """
        raise NotImplementedError(
            f"Propagation not implemented for walker type {type(walkers)}"
        )

    @_apply_trotprop.register(RHFWalkers)
    @_apply_trotprop.register(GHFWalkers)
    def _apply_trotprop_rhf_ghf(
        self, walkers: walker_batch, fields: jax.Array, ham_data: dict
    ) -> walker_batch:
        """Apply the Trotterized propagator to a batch of RHF/GHF walkers."""
        return walkers.apply_chunked_prop(
            self._apply_trotprop_single_det,
            fields,
            self.n_chunks,
            ham_data,
        )

    @_apply_trotprop.register(UHFWalkers)
    def _apply_trotprop_uhf(
        self, walkers: UHFWalkers, fields: jax.Array, ham_data: dict
    ) -> UHFWalkers:
        """Apply the Trotterized propagator to a batch of UHF walkers."""
        return walkers.apply_chunked_prop(
            self._apply_trotprop_single_det_uhf,
            fields,
            self.n_chunks,
            ham_data,
        )

    @partial(jit, static_argnums=(0, 1))
    def propagate_constrained(
        self,
        trial: wave_function,
        ham_data: dict,
        prop_data: dict,
        fields: jax.Array,
        wave_data: dict,
    ) -> dict:
        """Phaseless AFQMC propagation.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            fields: auxiliary fields
            wave_data: wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        force_bias = trial.calc_force_bias(prop_data["walkers"], ham_data, wave_data)
        field_shifts = -jnp.sqrt(self.dt) * (1.0j * force_bias - ham_data["mf_shifts"])
        shifted_fields = fields - field_shifts
        shift_term = jnp.sum(shifted_fields * ham_data["mf_shifts"], axis=1)
        fb_term = jnp.sum(
            fields * field_shifts - field_shifts * field_shifts / 2.0, axis=1
        )
        prop_data["walkers"] = self._apply_trotprop(
            prop_data["walkers"], shifted_fields, ham_data
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        imp_fun = (
            jnp.exp(
                -jnp.sqrt(self.dt) * shift_term
                + fb_term
                + self.dt * (prop_data["pop_control_ene_shift"] + ham_data["h0_prop"])
            )
            * overlaps_new
            / prop_data["overlaps"]
        )
        theta = jnp.angle(
            jnp.exp(-jnp.sqrt(self.dt) * shift_term)
            * overlaps_new
            / prop_data["overlaps"]
        )
        imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
        imp_fun_phaseless = jnp.array(
            jnp.where(jnp.isnan(imp_fun_phaseless), 0.0, imp_fun_phaseless)
        )
        imp_fun_phaseless = jnp.where(
            imp_fun_phaseless < 1.0e-3, 0.0, imp_fun_phaseless
        )
        imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100.0, 0.0, imp_fun_phaseless)
        prop_data["weights"] = imp_fun_phaseless * prop_data["weights"]
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] > 100.0, 0.0, prop_data["weights"])
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        prop_data["overlaps"] = overlaps_new
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate_free(
        self,
        trial: wave_function,
        ham_data,
        prop_data: dict,
        fields: jax.Array,
        wave_data: Sequence,
    ) -> dict:
        """Free propagation."""
        assert self.walker_type in [
            "restricted",
            "unrestricted",
        ], "Free propagation is only implemented for restricted and unrestricted walkers."
        if self.walker_type == "unrestricted":
            shift_term = jnp.einsum("wg,sg->sw", fields, ham_data["mf_shifts_fp"])
            constants = jnp.einsum(
                "sw,s->sw",
                jnp.exp(-jnp.sqrt(self.dt) * shift_term),
                jnp.exp(self.dt * ham_data["h0_prop_fp"]),
            )
        else:
            shift_term = jnp.einsum("wg,g->w", fields, ham_data["mf_shifts_fp"])
            constants = jnp.exp(-jnp.sqrt(self.dt) * shift_term) * jnp.exp(
                self.dt * ham_data["h0_prop_fp"]
            )
        prop_data["walkers"] = self._apply_trotprop(
            prop_data["walkers"], fields, ham_data
        )
        prop_data["walkers"] = prop_data["walkers"].multiply_constants(constants)
        norms = prop_data["walkers"].qr_norm()
        prop_data["weights"] *= norms.real

        prop_data["key"], subkey = random.split(prop_data["key"])
        zeta = random.uniform(subkey)

        # Since we compute only R in the QR we need to divide
        # by norms.real after the SR, requiering to keep track
        # of the indices
        indices = sr.get_sr_indices(prop_data["weights"], zeta)
        norms = norms[indices]

        prop_data["walkers"], prop_data["weights"] = prop_data[
            "walkers"
        ].stochastic_reconfiguration_local(prop_data["weights"], zeta)

        prop_data["weights"] /= norms.real

        return prop_data

    @partial(jit, static_argnums=(0, 2))
    def _build_propagation_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        rdm1 = wave_data["rdm1"]
        if self.walker_type == "generalized":
            rdm1 = rdm1[0]
        else:
            rdm1 = rdm1[0] + rdm1[1]
        ham_data["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(trial.norb, trial.norb) * rdm1)
        )(ham_data["chol"])
        ham_data["h0_prop"] = (
            -ham_data["h0"] - jnp.sum(ham_data["mf_shifts"] ** 2) / 2.0
        )
        if self.walker_type == "unrestricted":
            ham_data["mf_shifts_fp"] = jnp.stack(
                (
                    ham_data["mf_shifts"] / trial.nelec[0] / 2.0,
                    ham_data["mf_shifts"] / trial.nelec[1] / 2.0,
                )
            )
            ham_data["h0_prop_fp"] = jnp.stack(
                (
                    (ham_data["h0_prop"] + ham_data["ene0"]) / trial.nelec[0] / 2.0,
                    (ham_data["h0_prop"] + ham_data["ene0"]) / trial.nelec[1] / 2.0,
                )
            )
        else:
            ham_data["mf_shifts_fp"] = ham_data["mf_shifts"] / 2.0 / trial.nelec[0]
            ham_data["h0_prop_fp"] = (
                (ham_data["h0_prop"] + ham_data["ene0"]) / trial.nelec[0] / 2.0
            )

        v0 = 0.5 * jnp.einsum(
            "gik,gkj->ij",
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
            optimize="optimal",
        )
        mf_shifts_r = (1.0j * ham_data["mf_shifts"]).real
        v1 = jnp.einsum(
            "g,gik->ik",
            mf_shifts_r,
            ham_data["chol"].reshape(-1, trial.norb, trial.norb),
        )
        if self.walker_type == "generalized":
            h1_mod = ham_data["h1"][0] - v0 - v1
            ham_data["exp_h1"] = jsp.linalg.expm(-self.dt * h1_mod / 2.0)
        elif self.walker_type == "unrestricted":
            h1_mod = ham_data["h1"] - jnp.array([v0 + v1, v0 + v1])
            ham_data["exp_h1"] = jnp.array(
                [
                    jsp.linalg.expm(-self.dt * h1_mod[0] / 2.0),
                    jsp.linalg.expm(-self.dt * h1_mod[1] / 2.0),
                ]
            )
        else:
            h1_mod = (ham_data["h1"][0] + ham_data["h1"][1]) / 2.0 - v0 - v1
            ham_data["exp_h1"] = jsp.linalg.expm(-self.dt * h1_mod / 2.0)
        return ham_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_cpmc(propagator_afqmc):
    """CPMC propagator for the Hubbard model with on-site interactions."""

    walker_type: str = "unrestricted"

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham_data: dict,
        seed: int,
        init_walkers: Optional[UHFWalkers] = None,
    ) -> dict:
        prop_data = super().init_prop_data(
            trial, wave_data, ham_data, seed, init_walkers
        )
        prop_data["walkers"].data[0] = prop_data["walkers"].data[0].real
        prop_data["walkers"].data[1] = prop_data["walkers"].data[1].real
        prop_data["overlaps"] = prop_data["overlaps"].real
        try:
            prop_data["greens"] = trial.calc_green_full(
                prop_data["walkers"].data, wave_data
            )
        except:
            pass
        gamma = jnp.arccosh(jnp.exp(self.dt * ham_data["u"] / 2))
        const = jnp.exp(-self.dt * ham_data["u"] / 2)
        prop_data["hs_constant"] = const * jnp.array(
            [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
        )
        prop_data["node_crossings"] = 0
        return prop_data

    @partial(jit, static_argnums=(0, 2))
    def _build_propagation_intermediates(
        self, ham_data: dict, trial: wave_function, wave_data: dict
    ) -> dict:
        # ham_data = super()._build_propagation_intermediates(ham_data, trial, wave_data)
        # no mean field shift
        ham_data["exp_h1"] = jnp.array(
            [
                jsp.linalg.expm(-self.dt * ham_data["h1"][0] / 2.0),
                jsp.linalg.expm(-self.dt * ham_data["h1"][1] / 2.0),
            ]
        )
        return ham_data

    @partial(jit, static_argnums=(0, 1))
    def propagate_one_body(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        wave_data: dict,
    ) -> dict:
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        )
        prop_data["overlaps"] = overlaps_new
        prop_data["greens"] = trial.calc_green_full(
            prop_data["walkers"].data, wave_data
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate_constrained(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data = self.propagate_one_body(trial, ham_data, prop_data, wave_data)

        # two body
        # TODO: define separate sampler that feeds uniform_rns instead of gaussian_rns
        uniform_rns = (jsp.special.erf(gaussian_rns / 2**0.5) + 1) / 2

        # iterate over sites
        def scanned_fun(carry, x):
            # field 1
            ratio_0 = trial.calc_overlap_ratio(
                carry["greens"],
                jnp.array([[0, x], [1, x]]),
                prop_data["hs_constant"][0] - 1,
            )
            ratio_0 = jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0)
            carry["node_crossings"] += jnp.sum(jnp.array(ratio_0) == 0.0)

            # field 2
            ratio_1 = trial.calc_overlap_ratio(
                carry["greens"],
                jnp.array([[0, x], [1, x]]),
                prop_data["hs_constant"][1] - 1,
            )

            ratio_1 = jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1)
            carry["node_crossings"] += jnp.sum(jnp.array(ratio_1) == 0.0)

            # normalize
            prob_0 = ratio_0.real / 2.0
            prob_1 = ratio_1.real / 2.0
            norm = prob_0 + prob_1 + 1e-13
            prob_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask = rns < prob_0
            constants = jnp.where(
                mask.reshape(-1, 1),
                prop_data["hs_constant"][0],
                prop_data["hs_constant"][1],
            )
            new_walkers_up = (
                carry["walkers"].data[0].at[:, x, :].mul(constants[:, 0].reshape(-1, 1))
            )
            new_walkers_dn = (
                carry["walkers"].data[1].at[:, x, :].mul(constants[:, 1].reshape(-1, 1))
            )
            carry["walkers"].data = [new_walkers_up, new_walkers_dn]
            ratios = jnp.where(mask, ratio_0, ratio_1)
            update_constants = constants - 1
            carry["greens"] = trial.update_green(
                carry["greens"],
                ratios,
                jnp.array([[0, x], [1, x]]),
                update_constants,
            )
            carry["overlaps"] = ratios * carry["overlaps"]
            carry["weights"] *= norm
            return carry, x

        prop_data, _ = lax.scan(scanned_fun, prop_data, jnp.arange(trial.norb))

        # one body
        prop_data = self.propagate_one_body(trial, ham_data, prop_data, wave_data)

        prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_cpmc_slow(propagator_cpmc):
    """CPMC propagator for the Hubbard model with on-site interactions."""

    @partial(jit, static_argnums=(0, 1))
    def propagate_constrained(
        self,
        trial: wavefunctions.wave_function,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        )
        prop_data["overlaps"] = overlaps_new

        # two body
        # TODO: define separate sampler that feeds uniform_rns instead of gaussian_rns
        uniform_rns = (jsp.special.erf(gaussian_rns / 2**0.5) + 1) / 2

        # iterate over sites
        # TODO: fast update
        def scanned_fun(carry, x):
            # field 1
            new_walkers_0_up = (
                carry["walkers"].data[0].at[:, x, :].mul(prop_data["hs_constant"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"].data[1].at[:, x, :].mul(prop_data["hs_constant"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                UHFWalkers([jnp.array(new_walkers_0_up), jnp.array(new_walkers_0_dn)]),
                wave_data,
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.array(jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0))
            carry["node_crossings"] += jnp.sum(jnp.array(ratio_0) == 0.0)

            # field 2
            new_walkers_1_up = (
                carry["walkers"].data[0].at[:, x, :].mul(prop_data["hs_constant"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"].data[1].at[:, x, :].mul(prop_data["hs_constant"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                UHFWalkers([jnp.array(new_walkers_1_up), jnp.array(new_walkers_1_dn)]),
                wave_data,
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))
            carry["node_crossings"] += jnp.sum(jnp.array(ratio_1) == 0.0)

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"].data = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new
            return carry, x

        prop_data, _ = lax.scan(scanned_fun, prop_data, jnp.arange(trial.norb))

        # one body
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"])
        )
        prop_data["overlaps"] = overlaps_new

        prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate_1(
        self,
        trial: wavefunctions.wave_function,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm (no importance sampling).

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        # prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        # prop_data["weights"] = jnp.where(
        #     prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        # )
        prop_data["overlaps"] = overlaps_new

        # two body
        # TODO: define separate sampler that feeds uniform_rns instead of gaussian_rns
        uniform_rns = (jsp.special.erf(gaussian_rns / 2**0.5) + 1) / 2

        # iterate over sites
        # TODO: fast update
        def scanned_fun(carry, x):
            # field 1
            new_walkers_0_up = (
                carry["walkers"].data[0].at[:, x, :].mul(prop_data["hs_constant"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"].data[1].at[:, x, :].mul(prop_data["hs_constant"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                [new_walkers_0_up, new_walkers_0_dn], wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real
            ratio_0 = jnp.array(jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0))

            # field 2
            new_walkers_1_up = (
                carry["walkers"].data[0].at[:, x, :].mul(prop_data["hs_constant"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"].data[1].at[:, x, :].mul(prop_data["hs_constant"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                [new_walkers_1_up, new_walkers_1_dn], wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask_0 = (rns < 0.5).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"].data = new_walkers
            # carry["weights"] *= (overlaps_new / carry["overlaps"]).real
            carry["overlaps"] = overlaps_new
            return carry, x

        prop_data, _ = lax.scan(
            scanned_fun, prop_data, jnp.arange(ham_data["chol"].shape[0])
        )  # TODO: chol will be removed from ham_data for hubbard

        # one body
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        ) * jnp.exp(
            self.dt
            * (prop_data["e_estimate"])
            / 2
            / prop_data["walkers"].data[0].shape[-1]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        ) * jnp.exp(
            self.dt
            * (prop_data["e_estimate"])
            / 2
            / prop_data["walkers"].data[1].shape[-1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        # prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        # prop_data["weights"] = jnp.where(
        #     prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        # )
        prop_data["overlaps"] = overlaps_new
        prop_data["weights"] = overlaps_new.real
        # prop_data["weights"] *= jnp.exp(self.dt * (prop_data["e_estimate"]))
        # prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        # prop_data["weights"] = jnp.where(
        #     prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        # )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_cpmc_nn(propagator_cpmc):
    neighbors: Optional[Tuple] = None

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham_data: dict,
        seed: int,
        init_walkers: Optional[UHFWalkers] = None,
    ) -> dict:
        prop_data = super().init_prop_data(
            trial, wave_data, ham_data, seed, init_walkers
        )
        gamma_1 = jnp.arccosh(jnp.exp(self.dt * ham_data["u_1"] / 2))
        const_1 = jnp.exp(-self.dt * ham_data["u_1"] / 2)
        prop_data["hs_constant_nn"] = const_1 * jnp.array(
            [
                [jnp.exp(gamma_1), jnp.exp(-gamma_1)],
                [jnp.exp(-gamma_1), jnp.exp(gamma_1)],
            ]
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate_constrained(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        _gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            _gaussian_rns: Gaussian random numbers (not used, need to define cpmc sampler)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        def update_walkers(carry, trial, rns, spins, sites, hs_constants, tol=1.0e-8):
            update_indices = jnp.column_stack((spins, sites))

            # field 1
            ratio_0 = trial.calc_overlap_ratio(
                carry["greens"],
                update_indices,
                hs_constants[0] - 1,
            )
            ratio_0 = jnp.where(ratio_0 < tol, 0.0, ratio_0)
            carry["node_crossings"] += jnp.sum(jnp.array(ratio_0) == 0.0)

            # field 2
            ratio_1 = trial.calc_overlap_ratio(
                carry["greens"],
                update_indices,
                hs_constants[1] - 1,
            )

            ratio_1 = jnp.where(ratio_1 < tol, 0.0, ratio_1)
            carry["node_crossings"] += jnp.sum(jnp.array(ratio_0) == 0.0)

            # normalize
            prob_0 = ratio_0.real / 2.0
            prob_1 = ratio_1.real / 2.0
            norm = prob_0 + prob_1
            prob_0 /= norm

            # update
            mask = rns < prob_0 # (nwalkers,)
            constants = jnp.where( # (nwalkers, 2) for lambda_{\pm}
                mask.reshape(-1, 1),
                hs_constants[0], # field 1
                hs_constants[1], # field 2
            )
            ratios = jnp.where(mask, ratio_0, ratio_1)

            spin_i, spin_j = spins
            site_i, site_j = sites
            walkers = carry["walkers"]
            walkers_up, walkers_dn = walkers.data[0], walkers.data[1]
            
            if spin_i == 0:
                walkers_up = (
                    walkers_up
                    .at[:, site_i, :]
                    .mul(constants[:, 0].reshape(-1, 1)) # lambda_{+}
                )

            else:
                walkers_dn = (
                    walkers_dn
                    .at[:, site_i, :]
                    .mul(constants[:, 0].reshape(-1, 1)) # lambda_{+}
                )

            if spin_j == 0:
                walkers_up = (
                    walkers_up
                    .at[:, site_j, :]
                    .mul(constants[:, 1].reshape(-1, 1)) # lambda_{-}
                )

            else:
                walkers_dn = (
                    walkers_dn
                    .at[:, site_j, :]
                    .mul(constants[:, 1].reshape(-1, 1)) # lambda_{-}
                )

            carry["walkers"].data = [walkers_up, walkers_dn]
            carry["greens"] = trial.update_green(
                carry["greens"],
                ratios,
                update_indices,
                constants - 1,
            )
            carry["overlaps"] = ratios * carry["overlaps"]
            carry["weights"] *= norm
            return carry
        
        # one body
        prop_data = self.propagate_one_body(trial, ham_data, prop_data, wave_data)

        # two body
        # onsite interactions
        prop_data["key"], subkey = random.split(prop_data["key"])
        uniform_rns = random.uniform(subkey, shape=(self.n_walkers, trial.norb))

        # iterate over sites
        def scanned_fun(carry, x):
            carry = update_walkers(
                carry, 
                trial, 
                uniform_rns[:, x], 
                (0, 1),
                (x, x),
                prop_data["hs_constant"]
            )
            return carry, x

        prop_data, _ = lax.scan(scanned_fun, prop_data, jnp.arange(trial.norb))

        # nearest-neighbor interactions
        prop_data["key"], subkey = random.split(prop_data["key"])
        uniform_rns_nn = random.uniform(
            subkey, shape=(self.n_walkers, 4, jnp.array(self.neighbors).shape[0])
        )

        # iterate over nearest-neighbour bonds
        def scanned_fun_nn(carry, x):
            site_i = jnp.array(self.neighbors)[x][0]
            site_j = jnp.array(self.neighbors)[x][1]

            # up, up
            carry = update_walkers(
                carry, 
                trial, 
                uniform_rns_nn[:, 0, x], 
                (0, 0),
                (site_i, site_j),
                prop_data["hs_constant_nn"]
            )

            # up, dn
            carry = update_walkers(
                carry, 
                trial, 
                uniform_rns_nn[:, 1, x], 
                (0, 1),
                (site_i, site_j),
                prop_data["hs_constant_nn"]
            )

            # dn, up
            carry = update_walkers(
                carry, 
                trial, 
                uniform_rns_nn[:, 2, x], 
                (1, 0),
                (site_i, site_j),
                prop_data["hs_constant_nn"]
            )

            # dn, dn
            carry = update_walkers(
                carry, 
                trial, 
                uniform_rns_nn[:, 3, x], 
                (1, 1),
                (site_i, site_j),
                prop_data["hs_constant_nn"]
            )

            return carry, x

        prop_data, _ = lax.scan(
            scanned_fun_nn, prop_data, jnp.arange(jnp.array(self.neighbors).shape[0])
        )

        # one body
        prop_data = self.propagate_one_body(trial, ham_data, prop_data, wave_data)

        prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_cpmc_nn_slow(propagator_cpmc):
    neighbors: Optional[Tuple] = None

    def init_prop_data(
        self,
        trial: wavefunctions.wave_function_cpmc,
        wave_data: dict,
        ham_data: dict,
        seed: int,
        init_walkers: Optional[UHFWalkers] = None,
    ) -> dict:
        prop_data = super().init_prop_data(
            trial, wave_data, ham_data, seed, init_walkers
        )
        prop_data["greens"] = trial.calc_green_full(
            prop_data["walkers"].data, wave_data
        )
        gamma = jnp.arccosh(jnp.exp(self.dt * ham_data["u"] / 2))
        const = jnp.exp(-self.dt * ham_data["u"] / 2)
        prop_data["hs_constant_onsite"] = const * jnp.array(
            [[jnp.exp(gamma), jnp.exp(-gamma)], [jnp.exp(-gamma), jnp.exp(gamma)]]
        )
        gamma_1 = jnp.arccosh(jnp.exp(self.dt * ham_data["u_1"] / 2))
        const_1 = jnp.exp(-self.dt * ham_data["u_1"] / 2)
        prop_data["hs_constant_nn"] = const_1 * jnp.array(
            [
                [jnp.exp(gamma_1), jnp.exp(-gamma_1)],
                [jnp.exp(-gamma_1), jnp.exp(gamma_1)],
            ]
        )
        return prop_data

    @partial(jit, static_argnums=(0, 1))
    def propagate_constrained(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        _gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers (these are converted to uniform)
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.where(
            prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"]
        )
        prop_data["overlaps"] = overlaps_new

        # two body
        # on site
        prop_data["key"], subkey = random.split(prop_data["key"])
        uniform_rns = random.uniform(subkey, shape=(self.n_walkers, trial.norb))

        # iterate over sites
        # TODO: fast update
        def scanned_fun(carry, x):
            # field 1
            new_walkers_0_up = (
                carry["walkers"]
                .data[0]
                .at[:, x, :]
                .mul(prop_data["hs_constant_onsite"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"]
                .data[1]
                .at[:, x, :]
                .mul(prop_data["hs_constant_onsite"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                UHFWalkers([new_walkers_0_up, new_walkers_0_dn]), wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.array(jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0))

            # field 2
            new_walkers_1_up = (
                carry["walkers"]
                .data[0]
                .at[:, x, :]
                .mul(prop_data["hs_constant_onsite"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"]
                .data[1]
                .at[:, x, :]
                .mul(prop_data["hs_constant_onsite"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                UHFWalkers([new_walkers_1_up, new_walkers_1_dn]), wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns[:, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"].data = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new
            return carry, x

        prop_data, _ = lax.scan(scanned_fun, prop_data, jnp.arange(trial.norb))

        prop_data["key"], subkey = random.split(prop_data["key"])

        # Index 1 has shape 4 = 2x2 spin combinations.
        uniform_rns_1 = random.uniform(
            subkey, shape=(self.n_walkers, 4, jnp.array(self.neighbors).shape[0])
        )

        # neighbors
        def scanned_fun_1(carry, x):
            site_i = jnp.array(self.neighbors)[x][0]
            site_j = jnp.array(self.neighbors)[x][1]

            # up up
            # field 1
            new_walkers_0_up = (
                carry["walkers"]
                .data[0]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][0, 0])
            )
            new_walkers_0_up = (
                new_walkers_0_up
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                UHFWalkers([new_walkers_0_up, carry["walkers"].data[1]]), wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.array(jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0))

            # field 2
            new_walkers_1_up = (
                carry["walkers"]
                .data[0]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][1, 0])
            )
            new_walkers_1_up = (
                new_walkers_1_up
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                UHFWalkers([new_walkers_1_up, carry["walkers"].data[1]]), wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns_1[:, 0, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers = [new_walkers_up, carry["walkers"].data[1]]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"].data = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new

            # up dn
            # field 1
            new_walkers_0_up = (
                carry["walkers"]
                .data[0]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][0, 0])
            )
            new_walkers_0_dn = (
                carry["walkers"]
                .data[1]
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                UHFWalkers([new_walkers_0_up, new_walkers_0_dn]), wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.array(jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0))

            # field 2
            new_walkers_1_up = (
                carry["walkers"]
                .data[0]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][1, 0])
            )
            new_walkers_1_dn = (
                carry["walkers"]
                .data[1]
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                UHFWalkers([new_walkers_1_up, new_walkers_1_dn]), wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns_1[:, 1, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"].data = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new

            # dn up
            # field 1
            new_walkers_0_dn = (
                carry["walkers"]
                .data[1]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][0, 0])
            )
            new_walkers_0_up = (
                carry["walkers"]
                .data[0]
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                UHFWalkers([new_walkers_0_up, new_walkers_0_dn]), wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.array(jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0))

            # field 2
            new_walkers_1_dn = (
                carry["walkers"]
                .data[1]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][1, 0])
            )
            new_walkers_1_up = (
                carry["walkers"]
                .data[0]
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                UHFWalkers([new_walkers_1_up, new_walkers_1_dn]), wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns_1[:, 2, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_up = jnp.where(mask_0, new_walkers_0_up, new_walkers_1_up)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [new_walkers_up, new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"].data = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new

            # dn dn
            # field 1
            new_walkers_0_dn = (
                carry["walkers"]
                .data[1]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][0, 0])
            )
            new_walkers_0_dn = (
                new_walkers_0_dn
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][0, 1])
            )
            overlaps_new_0 = trial.calc_overlap(
                UHFWalkers([carry["walkers"].data[0], new_walkers_0_dn]), wave_data
            )
            ratio_0 = (overlaps_new_0 / carry["overlaps"]).real / 2.0
            ratio_0 = jnp.array(jnp.where(ratio_0 < 1.0e-8, 0.0, ratio_0))

            # field 2
            new_walkers_1_dn = (
                carry["walkers"]
                .data[1]
                .at[:, site_i, :]
                .mul(prop_data["hs_constant_nn"][1, 0])
            )
            new_walkers_1_dn = (
                new_walkers_1_dn
                .at[:, site_j, :]
                .mul(prop_data["hs_constant_nn"][1, 1])
            )
            overlaps_new_1 = trial.calc_overlap(
                UHFWalkers([carry["walkers"].data[0], new_walkers_1_dn]), wave_data
            )
            ratio_1 = (overlaps_new_1 / carry["overlaps"]).real / 2.0
            ratio_1 = jnp.array(jnp.where(ratio_1 < 1.0e-8, 0.0, ratio_1))

            # normalize
            norm = ratio_0 + ratio_1
            ratio_0 /= norm

            # update
            rns = uniform_rns_1[:, 3, x]
            mask_0 = (rns < ratio_0).reshape(-1, 1, 1)
            new_walkers_dn = jnp.where(mask_0, new_walkers_0_dn, new_walkers_1_dn)
            new_walkers = [carry["walkers"].data[0], new_walkers_dn]
            mask_0 = mask_0.reshape(-1)
            overlaps_new = jnp.where(mask_0, overlaps_new_0, overlaps_new_1)
            carry["walkers"].data = new_walkers
            carry["weights"] *= norm
            carry["overlaps"] = overlaps_new

            return carry, x

        prop_data, _ = lax.scan(
            scanned_fun_1, prop_data, jnp.arange(jnp.array(self.neighbors).shape[0])
        )

        # one body
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        prop_data["weights"] *= (overlaps_new / prop_data["overlaps"]).real
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"])
        )
        prop_data["overlaps"] = overlaps_new

        prop_data["weights"] *= jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]))
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))


@dataclass
class propagator_cpmc_continuous(propagator_cpmc):

    @partial(jit, static_argnums=(0, 1))
    def propagate_constrained(
        self,
        trial: wavefunctions.wave_function_cpmc,
        ham_data: dict,
        prop_data: dict,
        gaussian_rns: jnp.array,
        wave_data: dict,
    ) -> dict:
        """
        Propagate the walkers using the CPMC algorithm.

        Args:
            trial: trial wave function handler
            ham_data: dictionary containing the Hamiltonian data
            prop_data: dictionary containing the propagation data
            gaussian_rns: Gaussian random numbers
            wave_data: dictionary containing the wave function data

        Returns:
            prop_data: dictionary containing the updated propagation data
        """
        # one body
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        )

        # two body
        green_diag = trial.calc_green_diagonal(prop_data["walkers"], wave_data)
        force_bias = (
            0.0 * ham_data["hs_constant"] * (green_diag[:, 0] - green_diag[:, 1])
        )
        shifted_fields = gaussian_rns - force_bias
        fb_term = 0.0 * jnp.sum(
            gaussian_rns * force_bias - force_bias * force_bias / 2.0, axis=1
        )
        prop_data["walkers"].data[0] = jnp.einsum(
            "wi,wij->wij",
            jnp.exp(shifted_fields * ham_data["hs_constant"]),
            prop_data["walkers"].data[0],
            optimize="optimal",
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "wi,wij->wij",
            jnp.exp(-shifted_fields * ham_data["hs_constant"]),
            prop_data["walkers"].data[1],
            optimize="optimal",
        )

        # one body
        prop_data["walkers"].data[0] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][0], prop_data["walkers"].data[0]
        )
        prop_data["walkers"].data[1] = jnp.einsum(
            "ij,wjk->wik", ham_data["exp_h1"][1], prop_data["walkers"].data[1]
        )
        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)

        imp_fun = (
            jnp.exp(self.dt * (prop_data["pop_control_ene_shift"]) + fb_term)
            * overlaps_new
            / prop_data["overlaps"]
        )
        prop_data["weights"] *= imp_fun.real
        prop_data["weights"] = jnp.array(
            jnp.where(prop_data["weights"] < 1.0e-8, 0.0, prop_data["weights"])
        )
        prop_data["overlaps"] = overlaps_new
        prop_data["weights"] = jnp.where(
            prop_data["weights"] > 100.0, 0.0, prop_data["weights"]
        )
        # prop_data["weights"] = overlaps_new.real
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt
        )
        return prop_data

    def __hash__(self) -> int:
        return hash(tuple(self.__dict__.values()))
