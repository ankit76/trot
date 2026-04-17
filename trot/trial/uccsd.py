import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from typing import Any


def build_hs_op(t2: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """
    Builds the Cholesky decomposition of UCCSD T2 amplitudes,
    T2 = LL^T.

    Input:
    t2: UCCSD T2 amplitudes

    Output:
    (La, Lb): alpha and beta parts of the Cholesky vectors
    """
    t2aa, t2ab, t2bb = t2

    t2aa = jnp.asarray(t2aa)
    t2ab = jnp.asarray(t2ab)
    t2bb = jnp.asarray(t2bb)

    nOa, nOb, nVa, nVb = t2ab.shape
    n = nOa + nVa
    assert n == nOb + nVb

    # Number of excitations
    nex_a = nOa * nVa
    nex_b = nOb * nVb

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
    t2 = jnp.zeros((nex_a + nex_b, nex_a + nex_b))
    t2 = jax.lax.dynamic_update_slice(t2, 0.5 * t2aa, (0, 0))
    t2 = jax.lax.dynamic_update_slice(t2, t2ab.T, (nex_a, 0))
    t2 = jax.lax.dynamic_update_slice(t2, t2ab, (0, nex_a))
    t2 = jax.lax.dynamic_update_slice(t2, 0.5 * t2bb, (nex_a, nex_a))

    # t2 = LL^T
    e_val, e_vec = jnp.linalg.eigh(t2)
    L = e_vec @ jnp.diag(jnp.sqrt(e_val + 0.0j))
    assert abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

    # alpha/beta operators for HS
    # Summation on the left to have a list of operators
    La = jnp.array(L[:nex_a, :])
    Lb = jnp.array(L[nex_a:, :])
    La = La.T.reshape(nex_a + nex_b, nVa, nOa)
    Lb = Lb.T.reshape(nex_a + nex_b, nVb, nOb)

    return (La, Lb)


def init_walkers(
    trial_coeff: tuple[ArrayLike, ArrayLike],
    t1: ArrayLike,
    hs_op: tuple[jax.Array, jax.Array],
    subkey: jax.Array,
    n_w: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Builds a stochastic representation of the UCCSD wavefunction using the Hubbard-Stratonovich transformation.

    Input:
    trial_coeff: mo coefficients in the alpha mo basis
    t1         : UCCSD T1 amplitudes
    hs_op      : alpha and beta Cholesky vectors from the UCCSD T2 amplitudes
    subkey     : PRNG key
    n_w        : number of walkers

    Output:
    (w_a, w_b): alpha and beta walkers
    """
    t1a, t1b = t1

    t1a = jnp.asarray(t1a)
    t1b = jnp.asarray(t1b)

    nOa, nVa = t1a.shape
    nOb, nVb = t1b.shape
    n = nOa + nVa
    assert n == nOb + nVb

    nex_a = nOa * nVa
    nex_b = nOb * nVb
    nex = nex_a + nex_b

    La, Lb = hs_op
    assert La.shape == (nex_a + nex_b, nVa, nOa)
    assert Lb.shape == (nex_a + nex_b, nVb, nOb)

    Ca, Cb = trial_coeff
    Ca = jnp.asarray(Ca)
    Cb = jnp.asarray(Cb)
    assert Ca.shape == (n, n)
    assert Cb.shape == (n, n)

    Ca_occ, Ca_vir = jnp.split(Ca, [nOa], axis=1)
    Cb_occ, Cb_vir = jnp.split(Cb, [nOb], axis=1)

    # e^T1
    e_t1a = t1a.T + 0.0j
    e_t1b = t1b.T + 0.0j

    ops_a = jnp.array([e_t1a] * n_w)
    ops_b = jnp.array([e_t1b] * n_w)

    fields = jax.random.normal(subkey, shape=(n_w, nex))

    # e^{T1+T2}
    ops_a = ops_a + jnp.einsum("wg,gai->wai", fields, La)
    ops_b = ops_b + jnp.einsum("wg,gai->wai", fields, Lb)

    # Initial walkers
    dm_a = Ca[:, :nOa] @ Ca[:, :nOa].conj().T
    dm_b = Cb[:, :nOb] @ Cb[:, :nOb].conj().T
    nos_a = jnp.linalg.eigh(dm_a)[1][:, ::-1][:, :nOa]
    nos_b = jnp.linalg.eigh(dm_b)[1][:, ::-1][:, :nOb]

    w_a = jnp.array([nos_a + 0.0j] * n_w)
    w_b = jnp.array([nos_b + 0.0j] * n_w)

    id_a = jnp.array([jnp.identity(n) + 0.0j] * n_w)
    id_b = jnp.array([jnp.identity(n) + 0.0j] * n_w)

    # e^{T1+T2} \ket{\phi}
    w_a = (id_a + jnp.einsum("pa,wai,iq -> wpq", Ca_vir, ops_a, Ca_occ.T)) @ w_a
    w_b = (id_b + jnp.einsum("pa,wai,iq -> wpq", Cb_vir, ops_b, Cb_occ.T)) @ w_b

    return (w_a, w_b)


def make_init_prop_state(trial_coeff: tuple[ArrayLike, ArrayLike], t1: ArrayLike, t2: ArrayLike):
    from jax.sharding import Mesh
    from trot import walkers as wk
    from trot.core.ops import MeasOps, TrialOps, k_energy
    from trot.core.system import System
    from trot.ham.chol import HamChol
    from trot.sharding import shard_prop_state
    from trot.prop.types import PropState, QmcParamsBase

    hs_op = build_hs_op(t2)

    def init_prop_state(
        *,
        sys: System,
        ham_data: HamChol,
        trial_ops: TrialOps,
        trial_data: Any,
        meas_ops: MeasOps,
        params: QmcParamsBase,
        initial_walkers: Any | None = None,
        initial_e_estimate: jax.Array | None = None,
        rdm1: jax.Array | None = None,
        mesh: Mesh | None = None,
    ) -> PropState:
        """
        Initialize AFQMC propagation state.
        """
        assert sys.walker_kind == "unrestricted"
        n_walkers = params.n_walkers
        seed = params.seed
        key = jax.random.PRNGKey(int(seed))
        weights = jnp.ones((n_walkers,))

        initial_walkers = init_walkers(trial_coeff, t1, hs_op, key, n_walkers)

        overlaps = wk.vmap_chunked(meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None))(
            initial_walkers, trial_data
        )

        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        e_kernel = meas_ops.require_kernel(k_energy)
        e_samples = jnp.real(
            wk.vmap_chunked(e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None))(
                initial_walkers, ham_data, meas_ctx, trial_data
            )
        )
        e_est = jnp.mean(e_samples)

        pop_shift = e_est

        node_encounters = jnp.asarray(0)

        state = PropState(
            walkers=initial_walkers,
            weights=weights,
            overlaps=overlaps,
            rng_key=key,
            pop_control_ene_shift=pop_shift,
            e_estimate=e_est,
            node_encounters=node_encounters,
        )
        return shard_prop_state(state, mesh)

    return init_prop_state
