import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from typing import Any


def build_hs_op(t2: ArrayLike) -> jax.Array:
    """
    Builds the Cholesky decomposition of CCSD T2 amplitudes,
    T2 = LL^T.

    Input:
    t2: CCSD T2 amplitudes

    Output:
    L: Cholesky vectors of the CCSD T2 amplitudes
    """
    t2 = jnp.asarray(t2)

    nO, _, nV, _ = t2.shape

    # Number of excitations
    nex = nO * nV

    assert t2.shape == (nO, nO, nV, nV)

    # t2(i,j,a,b) -> t2(ai,bj)
    t2 = jnp.einsum("ijab->aibj", t2)
    t2 = t2.reshape(nex, nex)

    # t2 = LL^T
    e_val, e_vec = jnp.linalg.eigh(t2)
    L = e_vec @ jnp.diag(jnp.sqrt(e_val + 0.0j))
    assert abs(jnp.linalg.norm(t2 - L @ L.T)) < 1e-12

    # Summation on the left to have a list of operators
    L = L.T.reshape(nex, nV, nO)

    return L


def init_walkers(
    trial_coeff: ArrayLike,
    t1: ArrayLike,
    hs_op: jax.Array,
    subkey: jax.Array,
    n_w: int,
) -> jax.Array:
    """
    Builds a stochastic representation of the UCCSD wavefunction using the Hubbard-Stratonovich transformation.

    Input:
    trial_coeff: mo coefficients in the alpha mo basis
    t1         : CCSD T1 amplitudes
    hs_op      : Cholesky vectors of the CCSD T2 amplitudes
    subkey     : PRNG key
    n_w        : number of walkers

    Output:
    w: walkers
    """
    t1 = jnp.asarray(t1)

    nO, nV = t1.shape
    n = nO + nV
    nex = nO * nV

    L = hs_op
    assert L.shape == (nex, nV, nO)

    C = trial_coeff
    C = jnp.asarray(C)
    assert C.shape == (n, n)

    C_occ, C_vir = jnp.split(C, [nO], axis=1)

    # e^T1
    e_t1 = t1.T + 0.0j

    ops = jnp.array([e_t1] * n_w)

    fields = jax.random.normal(subkey, shape=(n_w, nex))

    # e^{T1+T2}
    ops = ops + jnp.einsum("wg,gai->wai", fields, L)

    # Initial walkers
    dm = C[:, :nO] @ C[:, :nO].conj().T
    nos = jnp.linalg.eigh(dm)[1][:, ::-1][:, :nO]

    w = jnp.array([nos + 0.0j] * n_w)

    id_ = jnp.array([jnp.identity(n) + 0.0j] * n_w)

    # e^{T1+T2} \ket{\phi}
    w = (id_ + jnp.einsum("pa,wai,iq -> wpq", C_vir, ops, C_occ.T)) @ w

    return w


def make_init_prop_state(trial_coeff: ArrayLike, t1: ArrayLike, t2: ArrayLike):
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
        assert sys.walker_kind == "restricted"
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
