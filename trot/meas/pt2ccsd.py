from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import EstimatorOps, MeasOps, k_energy
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.pt2ccsd import Pt2ccsdTrial
from ..trial.pt2ccsd import overlap_r
from .. import walkers as wk
from ..prop.types import PropState, QmcParams


@dataclass(frozen=True)
class Pt2ccsdMeasCfg:
    memory_mode: str = "high"  # all Choleskies, as in dense CISD
    chol_batch_size: int = 64  # vectors per contraction in low-memory mode
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64

    def __post_init__(self):
        if self.memory_mode not in {"low", "high"}:
            raise ValueError("PT2 memory_mode must be 'low' or 'high'.")
        if not isinstance(self.chol_batch_size, Integral) or self.chol_batch_size < 1:
            raise ValueError("chol_batch_size must be a positive integer.")


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdMeasCtx:
    cfg: Pt2ccsdMeasCfg  # static
    rot_chol: jax.Array  # (nchol, nocc, norb), input precision

    def tree_flatten(self):
        children = (self.rot_chol,)
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        (rot_chol,) = children
        return cls(cfg=cfg, rot_chol=rot_chol)


def build_meas_ctx(
    ham_data: HamChol, trial_data: Pt2ccsdTrial, cfg: Pt2ccsdMeasCfg = Pt2ccsdMeasCfg()
) -> Pt2ccsdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("pt2CCSD MeasOps currently assumes HamChol.basis == 'restricted'.")

    # Match the dense Green-function transpose convention. The Thouless
    # modes kernel uses its own conjugated-reference convention.
    rot_chol = jnp.einsum("pi,gpq->giq", trial_data.mo_t, ham_data.chol, optimize="optimal")
    return Pt2ccsdMeasCtx(cfg=cfg, rot_chol=rot_chol)


def _greens_restricted(walker: jax.Array, mo_t: jax.Array) -> jax.Array:
    return (walker @ (jnp.linalg.inv(mo_t.T @ walker)) @ mo_t.T).T


def _greenp_from_green(green: jax.Array, nocc: int) -> jax.Array:
    norb = green.shape[0]
    return (green - jnp.eye(norb))[:, nocc:]


def _mixed_t2_contract(subscripts: str, t2: jax.Array, x: jax.Array) -> jax.Array:
    # Do not promote the entire real doubles tensor to complex.
    if not jnp.iscomplexobj(t2) and jnp.iscomplexobj(x):
        return jnp.einsum(subscripts, t2, x.real, optimize="optimal") + 1j * jnp.einsum(
            subscripts, t2, x.imag, optimize="optimal"
        )
    return jnp.einsum(subscripts, t2, x, optimize="optimal")


def _half_green_chol_batched(half_green: jax.Array, chol: jax.Array) -> jax.Array:
    # Avoid promoting the entire real (nchol, norb, norb) array to complex.
    if not jnp.iscomplexobj(chol) and jnp.iscomplexobj(half_green):
        return jnp.einsum("ir,gqr->giq", half_green.real, chol, optimize="optimal") + 1j * jnp.einsum(
            "ir,gqr->giq", half_green.imag, chol, optimize="optimal"
        )
    return jnp.einsum("ir,gqr->giq", half_green, chol, optimize="optimal")


def _chol_dot_batched(chol: jax.Array, matrix: jax.Array) -> jax.Array:
    if not jnp.iscomplexobj(chol) and jnp.iscomplexobj(matrix):
        return jnp.einsum("gpq,pq->g", chol, matrix.real, optimize="optimal") + 1j * jnp.einsum(
            "gpq,pq->g", chol, matrix.imag, optimize="optimal"
        )
    return jnp.einsum("gpq,pq->g", chol, matrix, optimize="optimal")


def _two_body_half_rotated(
    half_green: jax.Array,
    reference_occ: jax.Array,
    greenp: jax.Array,
    t2_green: jax.Array,
    t2: jax.Array,
    chol: jax.Array,
    rot_chol: jax.Array,
    cfg: Pt2ccsdMeasCfg,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """CISD-style batched contractions using occupied-space intermediates.

    High-memory mode contracts all Choleskies together. Low-memory mode uses
    bounded groups with the same contractions, not a scalar Cholesky loop.
    Only T2 applications may use FP32; Green/Cholesky products and scalar
    reductions retain input precision. Real Choleskies stay real.
    """
    dtype_acc = jnp.result_type(half_green, t2, chol)

    def working(array):
        dtype = cfg.mixed_complex_dtype if jnp.iscomplexobj(array) else cfg.mixed_real_dtype
        return array.astype(dtype)

    mixed = cfg.mixed_real_dtype == jnp.float32
    t2_work = working(t2) if mixed else t2
    zero = jnp.zeros((), dtype=dtype_acc)

    def contract(chol_batch, rot_batch):
        lg1 = jnp.einsum("gir,jr->gij", rot_batch, half_green, optimize="optimal")
        lg = jnp.trace(lg1, axis1=-2, axis2=-1)
        e20 = 2 * jnp.sum(lg * lg, dtype=dtype_acc)
        e20 -= jnp.sum(lg1 * jnp.swapaxes(lg1, -1, -2), dtype=dtype_acc)
        lt2g = _chol_dot_batched(chol_batch, t2_green)
        e221 = -jnp.sum(lt2g * lg, dtype=dtype_acc)
        gl_half = _half_green_chol_batched(half_green, chol_batch)
        lt2_half = jnp.einsum("gir,qr->giq", rot_batch, t2_green, optimize="optimal")
        e222 = 0.5 * jnp.sum(gl_half * lt2_half, dtype=dtype_acc)
        gl_occ = jnp.einsum("ji,gip->gjp", reference_occ, gl_half, optimize="optimal")
        x = jnp.einsum("gpi,ia->gpa", gl_occ, greenp, optimize="optimal")
        x_work = working(x) if mixed else x
        tc = _mixed_t2_contract("iajb,gjb->gia", t2_work, x_work)
        te = _mixed_t2_contract("iajb,gja->gib", t2_work, x_work)
        x_acc = x.astype(dtype_acc)
        direct = jnp.sum(x_acc * tc.astype(dtype_acc), dtype=dtype_acc)
        exchange = jnp.sum(x_acc * te.astype(dtype_acc), dtype=dtype_acc)
        return e20, e221, e222, 2 * direct - exchange

    nchol = chol.shape[0]
    if nchol == 0:
        return zero, zero, zero, zero
    if cfg.memory_mode == "high" or nchol <= cfg.chol_batch_size:
        return contract(chol, rot_chol)

    batch_size = cfg.chol_batch_size
    nfull, remainder = divmod(nchol, batch_size)

    def accumulate(index, carry):
        chol_batch = lax.dynamic_slice_in_dim(chol, index * batch_size, batch_size, axis=0)
        rot_batch = lax.dynamic_slice_in_dim(rot_chol, index * batch_size, batch_size, axis=0)
        values = contract(chol_batch, rot_batch)
        return tuple(old + value for old, value in zip(carry, values))

    values = lax.fori_loop(0, nfull, accumulate, (zero, zero, zero, zero))
    if remainder:
        # Pad only the last batch, never the full Hamiltonian.
        padding = ((0, batch_size - remainder), (0, 0), (0, 0))
        tail = contract(
            jnp.pad(chol[nfull * batch_size :], padding),
            jnp.pad(rot_chol[nfull * batch_size :], padding),
        )
        values = tuple(old + value for old, value in zip(values, tail))
    return values


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: Pt2ccsdMeasCtx, trial_data: Pt2ccsdTrial
) -> jax.Array:
    mo_t, t2 = trial_data.mo_t, trial_data.t2
    nocc = trial_data.nocc
    # Preserve the existing dense kernel's transpose convention.
    half_green = (walker @ jnp.linalg.inv(mo_t.T @ walker)).T
    green = mo_t @ half_green
    greenp = _greenp_from_green(green, nocc)
    hg = jnp.einsum("pq,pq->", ham_data.h1, green, optimize="optimal")
    # One-body and overlap components retain input precision.
    t2g_c = jnp.einsum("iajb,ia->jb", t2, green[:nocc, nocc:], optimize="optimal")
    t2g_e = jnp.einsum("iajb,ib->ja", t2, green[:nocc, nocc:], optimize="optimal")
    t2_green_c = (greenp @ t2g_c.T) @ green[:nocc, :]
    t2_green_e = (greenp @ t2g_e.T) @ green[:nocc, :]
    t2_green = 2 * t2_green_c - t2_green_e
    t2g = 2 * t2g_c - t2g_e
    theta = jnp.einsum("ia,ia->", t2g, green[:nocc, nocc:], optimize="optimal")
    e12 = 2 * hg * theta - 2 * jnp.einsum("pq,pq->", ham_data.h1, t2_green, optimize="optimal")
    e20, e221, e222, e23 = _two_body_half_rotated(
        half_green,
        mo_t[:nocc],
        greenp,
        t2_green,
        t2,
        ham_data.chol,
        meas_ctx.rot_chol,
        meas_ctx.cfg,
    )
    e22 = e20 * theta + 4 * (e221 + e222) + e23
    # [<T2>, <H_elec>, <T2 H_elec>], relative to the Thouless reference.
    return jnp.stack([theta, 2 * hg + e20, e12 + e22])


def combine_first_order_energy(h0, components):
    """Combine ``[theta, electronic_0, h_t]`` component ratios."""

    theta = components[..., 0]
    electronic_0 = components[..., 1]
    h_t = components[..., 2]
    return h0 + electronic_0 + h_t - theta * electronic_0


def project_first_order_energy_terms(theta, component_terms):
    """Project sampled ``[delta electronic_0, delta h_t]`` terms onto energy.

    The PT energy is ``h0 + electronic_0 + h_t - theta * electronic_0``.
    Holding the exactly evaluated ``theta`` fixed, its change under sampled
    Cholesky-component fluctuations is therefore
    ``delta h_t + (1 - theta) * delta electronic_0``.
    """

    delta_electronic_0 = component_terms[..., 0]
    delta_h_t = component_terms[..., 1]
    return delta_h_t + (1.0 - theta) * delta_electronic_0


def make_pt2ccsd_meas_ops(
    sys: System,
    memory_mode: str = "high",
    mixed_precision: bool = False,
    testing: bool = False,
    *,
    chol_batch_size: int = 64,
) -> MeasOps:
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"pt2CCSD MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )

    cfg = Pt2ccsdMeasCfg(
        memory_mode=memory_mode,
        chol_batch_size=chol_batch_size,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    return MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel_rw_rh},
    )


def make_pt2ccsd_estimator_ops(
    sys: System,
    memory_mode: str = "high",
    mixed_precision: bool = False,
    testing: bool = False,
    *,
    chol_batch_size: int = 64,
) -> EstimatorOps:
    """Build guide-independent dense pt2CCSD estimator operations."""

    meas_ops = make_pt2ccsd_meas_ops(
        sys,
        memory_mode=memory_mode,
        chol_batch_size=chol_batch_size,
        mixed_precision=mixed_precision,
        testing=testing,
    )
    return EstimatorOps(
        reference_overlap=overlap_r,
        components=energy_kernel_rw_rh,
        combine_energy=combine_first_order_energy,
        component_names=("theta", "electronic_0", "h_t"),
        build_estimator_ctx=meas_ops.build_meas_ctx,
    )


def get_init_pt2trial_energy(
    init_state: PropState,
    ham_data: HamChol,
    trial_data: Pt2ccsdTrial,
    trial_meas_ops: MeasOps,
    trial_meas_ctx: Pt2ccsdMeasCtx,
    params: QmcParams,
):

    walker_0 = wk.take_walkers(init_state.walkers, jnp.array([0]))
    trial_e_kernel = trial_meas_ops.require_kernel(k_energy)
    pt2results = wk.vmap_chunked(trial_e_kernel, n_chunks=1, in_axes=(0, None, None, None))(
        walker_0, ham_data, trial_meas_ctx, trial_data
    )
    t2, e0, e1 = pt2results[:, 0], pt2results[:, 1], pt2results[:, 2]
    trial_overlap = wk.vmap_chunked(
        trial_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walker_0, trial_data)
    guide_overlap = init_state.overlaps[0]
    trial_weights = init_state.weights * trial_overlap / guide_overlap
    trial_energy = combine_first_order_energy(ham_data.h0, pt2results).mean()

    return trial_energy + 0j, jnp.sum(trial_weights)
