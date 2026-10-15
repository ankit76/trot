from dataclasses import replace

from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot import testing
from trot.core.ops import (
    BlockEnergyEstimate,
    d_energy_head_guard_count,
    d_energy_head_guard_weight,
    d_energy_sampling_noise,
    d_energy_walker_guide_ess,
    d_energy_walker_guide_max_correction,
    k_energy,
    k_energy_init,
    k_force_bias,
)
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.cisd import (
    CisdMeasCfg,
    build_meas_ctx as build_dense_meas_ctx,
    energy_kernel_rw_rh as dense_energy_kernel,
    force_bias_kernel_rw_rh_high as dense_force_bias_kernel,
)
from trot.meas.cisd_modes import (
    CisdModePairTuningCfg,
    CisdModePopulationStats,
    CisdModePairSamplingCfg,
    _cisd_mode_chol_index_sum_for_walkers,
    _cisd_mode_chol_index_moments_for_walkers,
    _cisd_mode_chol_pair_terms,
    _cisd_mode_chol_terms_for_walkers,
    _cisd_mode_energy_common,
    average_cisd_mode_population_statistics,
    build_meas_ctx as build_mode_meas_ctx,
    configure_cisd_mode_pair_sampling,
    energy_kernel_rw_rh as mode_energy_kernel,
    force_bias_kernel_rw_rh as mode_force_bias_kernel,
    get_cisd_mode_meas_cfg,
    initial_energy_kernel_rw_rh as mode_initial_energy_kernel,
    make_cisd_mode_meas_ops,
    pair_sampled_block_energy,
    retune_cisd_mode_pair_sampling,
    select_cisd_mode_pair_sampling,
    stream_cisd_mode_population_statistics,
)
from trot.prop.types import PropState
from trot.trial import cisd_modes as cisd_modes_module
from trot.trial.cisd import CisdTrial, overlap_r as dense_overlap_r
from trot.trial.cisd_modes import (
    CisdModeTrial,
    factorize_cisd_k_modes,
    make_cisd_mode_trial_data,
    make_cisd_mode_trial_ops,
    mode_apply,
    mode_projections,
    mode_quadratic,
    overlap_r as mode_overlap_r,
)


def _make_dense_and_mode_trials(
    *,
    seed: int = 719,
    nocc: int = 3,
    nvir: int = 4,
    nocc_t_core: int = 0,
    nvir_t_outer: int = 0,
    mode_dtype=jnp.float64,
) -> tuple[CisdTrial, CisdModeTrial, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ci1 = 0.05 * rng.standard_normal((nocc, nvir))

    pair_dim = nocc * nvir
    ci2_pair = 0.02 * rng.standard_normal((pair_dim, pair_dim))
    ci2_pair = 0.5 * (ci2_pair + ci2_pair.T)
    ci2 = ci2_pair.reshape(nocc, nvir, nocc, nvir)

    direct = ci2.reshape(pair_dim, pair_dim)
    exchange = np.transpose(ci2, (0, 3, 2, 1)).reshape(pair_dim, pair_dim)
    kernel = 2.0 * direct - exchange
    np.testing.assert_allclose(kernel, kernel.T, rtol=0.0, atol=1.0e-14)

    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    modes = eigenvectors.T.reshape(pair_dim, nocc, nvir)
    dense_trial = CisdTrial(
        ci1=jnp.asarray(ci1, dtype=jnp.float64),
        ci2=jnp.asarray(ci2, dtype=jnp.float64),
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    mode_trial = CisdModeTrial(
        ci1=jnp.asarray(ci1, dtype=jnp.float64),
        eigenvalues=jnp.asarray(eigenvalues, dtype=jnp.float64),
        modes=jnp.asarray(modes, dtype=mode_dtype),
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    return dense_trial, mode_trial, kernel, eigenvectors


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_full_rank_double_mode_overlap_matches_dense_cisd(nocc_t_core, nvir_t_outer):
    dense_trial, mode_trial, _, _ = _make_dense_and_mode_trials(
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(811),
        mode_trial.norb,
        mode_trial.nocc_full,
        mix=0.25,
    )

    dense_overlap = dense_overlap_r(walker, dense_trial)
    mode_overlap = mode_overlap_r(walker, mode_trial)
    mode_overlap_jit = jax.jit(mode_overlap_r)(walker, mode_trial)
    mode_overlap_batch = jax.vmap(mode_overlap_r, in_axes=(0, None))(
        jnp.stack((walker, walker)), mode_trial
    )
    np.testing.assert_allclose(mode_overlap, dense_overlap, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(mode_overlap_jit, dense_overlap, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        mode_overlap_batch,
        jnp.stack((dense_overlap, dense_overlap)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_mode_helpers_match_explicit_k_contractions():
    _, trial, kernel, _ = _make_dense_and_mode_trials()
    key_r, key_i = jax.random.split(jax.random.PRNGKey(827))
    matrix = jax.random.normal(key_r, (trial.nocc, trial.nvir), dtype=jnp.float64)
    matrix = matrix + 1.0j * jax.random.normal(key_i, (trial.nocc, trial.nvir), dtype=jnp.float64)

    projections = mode_projections(trial, matrix)
    projections_from_apply, applied = mode_apply(trial, matrix)
    quadratic = mode_quadratic(trial, matrix, projections)

    matrix_np = np.asarray(matrix).reshape(-1)
    expected_applied = (kernel @ matrix_np).reshape(trial.nocc, trial.nvir)
    expected_quadratic = matrix_np @ kernel @ matrix_np
    np.testing.assert_allclose(projections_from_apply, projections, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(applied, expected_applied, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(quadratic, expected_quadratic, rtol=1.0e-12, atol=1.0e-12)


def test_restricted_dense_and_lanczos_factorizations_match():
    dense_trial, _, kernel, _ = _make_dense_and_mode_trials(seed=821)
    magnitudes = np.sort(np.abs(np.linalg.eigvalsh(kernel)))[::-1]
    threshold = float(0.5 * (magnitudes[3] + magnitudes[4]))

    dense = factorize_cisd_k_modes(
        np.asarray(dense_trial.ci2),
        threshold=threshold,
        solver="dense",
    )
    lanczos = factorize_cisd_k_modes(
        np.asarray(dense_trial.ci2),
        threshold=threshold,
        solver="lanczos",
        lanczos_initial_rank=5,
        lanczos_tol=1.0e-12,
    )

    dense_modes = dense.modes.reshape(dense.rank, -1)
    dense_kernel = (dense_modes.T * dense.eigenvalues) @ dense_modes
    lanczos_kernel = (
        lanczos.modes.reshape(lanczos.rank, -1).T * lanczos.eigenvalues
    ) @ lanczos.modes.reshape(lanczos.rank, -1)
    assert dense.rank == 4
    assert lanczos.rank == dense.rank
    assert dense.solver == "dense"
    assert dense.dense_driver == "evr"
    assert lanczos.dense_driver is None
    np.testing.assert_allclose(lanczos_kernel, dense_kernel, rtol=2.0e-10, atol=2.0e-12)


def test_restricted_dense_kernel_is_fortran_contiguous():
    dense_trial, _, expected, _ = _make_dense_and_mode_trials(seed=822)

    kernel, _, _ = cisd_modes_module._restricted_k_matrix(
        np.asarray(dense_trial.ci2)
    )

    assert kernel.flags.f_contiguous
    np.testing.assert_allclose(kernel, expected, rtol=0.0, atol=0.0)


def test_restricted_auto_solver_respects_available_host_memory(monkeypatch):
    dense_trial, _, _, _ = _make_dense_and_mode_trials(seed=823)
    amplitudes = np.asarray(dense_trial.ci2)

    monkeypatch.setattr(
        cisd_modes_module,
        "format_dense_memory_selection",
        lambda dimension: (False, f"mock insufficient memory for {dimension}"),
    )
    lanczos = factorize_cisd_k_modes(
        amplitudes,
        threshold=None,
        discarded_norm_target=0.5,
        solver="auto",
        dense_max_dim=1,
        lanczos_initial_rank=5,
        lanczos_tol=1.0e-12,
    )

    monkeypatch.setattr(
        cisd_modes_module,
        "format_dense_memory_selection",
        lambda dimension: (True, f"mock sufficient memory for {dimension}"),
    )
    dense = factorize_cisd_k_modes(
        amplitudes,
        threshold=None,
        discarded_norm_target=0.5,
        solver="auto",
        dense_max_dim=1,
    )

    assert lanczos.solver == "lanczos"
    assert dense.solver == "dense"
    assert lanczos.dense_driver is None
    assert dense.dense_driver == "evr"


def test_restricted_auto_solver_retries_lanczos_after_dense_memory_error(monkeypatch):
    dense_trial, _, _, _ = _make_dense_and_mode_trials(seed=824)
    amplitudes = np.asarray(dense_trial.ci2)
    dense_attempts = 0

    monkeypatch.setattr(
        cisd_modes_module,
        "format_dense_memory_selection",
        lambda dimension: (True, f"mock sufficient memory for {dimension}"),
    )

    def fail_dense(*args, **kwargs):
        nonlocal dense_attempts
        dense_attempts += 1
        raise MemoryError("mock dense allocation failure")

    monkeypatch.setattr(cisd_modes_module, "dense_symmetric_eigh", fail_dense)
    factorization = factorize_cisd_k_modes(
        amplitudes,
        threshold=None,
        discarded_norm_target=0.5,
        solver="auto",
        lanczos_initial_rank=5,
        lanczos_tol=1.0e-12,
    )

    assert dense_attempts == 1
    assert factorization.solver == "lanczos"


def test_mixed_trial_data_uses_lambda64_vectors32_and_dp_reductions():
    dense_trial, mode_dp, _, eigenvectors = _make_dense_and_mode_trials()
    sys = System(
        norb=dense_trial.norb,
        nelec=(dense_trial.nocc_full, dense_trial.nocc_full),
        walker_kind="restricted",
    )
    data = {
        "ci1": np.asarray(dense_trial.ci1),
        "eigenvalues": np.asarray(mode_dp.eigenvalues),
        "eigenvectors": eigenvectors,
    }
    mode_mixed = make_cisd_mode_trial_data(data, sys, mixed_precision=True)
    mode_double = make_cisd_mode_trial_data(data, sys, mixed_precision=False)

    assert mode_mixed.ci1.dtype == jnp.float64
    assert mode_mixed.eigenvalues.dtype == jnp.float64
    assert mode_mixed.modes.dtype == jnp.float32
    assert mode_double.modes.dtype == jnp.float64

    matrix = jnp.asarray(
        np.random.default_rng(839).standard_normal((mode_mixed.nocc, mode_mixed.nvir)),
        dtype=jnp.complex128,
    )
    assert mode_projections(mode_mixed, matrix).dtype == jnp.complex128
    assert mode_quadratic(mode_mixed, matrix).dtype == jnp.complex128
    assert mode_apply(mode_mixed, matrix)[1].dtype == jnp.complex128

    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(853), mode_mixed.norb, mode_mixed.nocc_full, mix=0.25
    )
    dense_overlap = dense_overlap_r(walker, dense_trial)
    mixed_overlap = mode_overlap_r(walker, mode_mixed)
    relative_error = float(jnp.abs(mixed_overlap - dense_overlap) / jnp.abs(dense_overlap))
    assert relative_error < 1.0e-5


def test_mode_trial_data_can_factor_dense_rcisd_amplitudes():
    dense_trial, mode_dp, _, _ = _make_dense_and_mode_trials(seed=841)
    sys = System(
        norb=dense_trial.norb,
        nelec=(dense_trial.nocc_full, dense_trial.nocc_full),
        walker_kind="restricted",
    )
    loaded = make_cisd_mode_trial_data(
        {
            "ci1": np.asarray(dense_trial.ci1),
            "ci2": np.asarray(dense_trial.ci2),
        },
        sys,
        mixed_precision=False,
        mode_solver="dense",
    )

    assert loaded.mode_rank == mode_dp.mode_rank
    reconstructed = (
        loaded.modes.reshape(loaded.mode_rank, -1).T * loaded.eigenvalues
    ) @ loaded.modes.reshape(loaded.mode_rank, -1)
    expected = (
        mode_dp.modes.reshape(mode_dp.mode_rank, -1).T * mode_dp.eigenvalues
    ) @ mode_dp.modes.reshape(mode_dp.mode_rank, -1)
    np.testing.assert_allclose(reconstructed, expected, rtol=2.0e-12, atol=2.0e-12)


def test_mode_trial_is_a_pytree_and_accepts_consistent_truncated_storage():
    dense_trial, trial, kernel, eigenvectors = _make_dense_and_mode_trials(
        nocc_t_core=1, nvir_t_outer=2
    )
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.nocc_t_core == 1
    assert restored.nvir_t_outer == 2
    np.testing.assert_array_equal(restored.eigenvalues, trial.eigenvalues)

    mode_rank = trial.mode_rank - 2
    truncated = CisdModeTrial(
        ci1=trial.ci1,
        eigenvalues=trial.eigenvalues[:mode_rank],
        modes=trial.modes[:mode_rank],
        nocc_t_core=trial.nocc_t_core,
        nvir_t_outer=trial.nvir_t_outer,
    )
    assert truncated.mode_rank == mode_rank

    sys = System(
        norb=dense_trial.norb,
        nelec=(dense_trial.nocc_full, dense_trial.nocc_full),
        walker_kind="restricted",
    )
    loaded = make_cisd_mode_trial_data(
        {
            "ci1": np.asarray(trial.ci1),
            "eigenvalues": np.asarray(trial.eigenvalues[:mode_rank]),
            "eigenvectors": eigenvectors[:, :mode_rank],
            "nocc_t_core": trial.nocc_t_core,
            "nvir_t_outer": trial.nvir_t_outer,
        },
        sys,
        mixed_precision=False,
    )
    assert loaded.modes.shape == (mode_rank, trial.nocc, trial.nvir)
    np.testing.assert_allclose(loaded.eigenvalues, truncated.eigenvalues)
    np.testing.assert_allclose(loaded.modes, truncated.modes)

    matrix = np.random.default_rng(863).standard_normal((trial.nocc, trial.nvir))
    matrix = jnp.asarray(matrix, dtype=jnp.complex128)
    truncated_modes = np.asarray(truncated.modes).reshape(mode_rank, -1)
    truncated_kernel = (
        truncated_modes.T @ np.diag(np.asarray(truncated.eigenvalues)) @ truncated_modes
    )
    expected_applied = (truncated_kernel @ np.asarray(matrix).reshape(-1)).reshape(matrix.shape)
    _, applied = mode_apply(truncated, matrix)
    expected_quadratic = (
        np.asarray(matrix).reshape(-1) @ truncated_kernel @ np.asarray(matrix).reshape(-1)
    )
    np.testing.assert_allclose(applied, expected_applied, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        mode_quadratic(truncated, matrix),
        expected_quadratic,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert kernel.shape == (trial.mode_rank, trial.mode_rank)

    with pytest.raises(ValueError, match="eigenvalues must have shape"):
        CisdModeTrial(
            ci1=trial.ci1,
            eigenvalues=trial.eigenvalues[: mode_rank - 1],
            modes=trial.modes[:mode_rank],
        )


def test_mode_trial_ops_require_restricted_closed_shell_walkers():
    restricted = System(norb=7, nelec=(2, 2), walker_kind="restricted")
    ops = make_cisd_mode_trial_ops(restricted)
    assert ops.overlap is mode_overlap_r

    with pytest.raises(ValueError, match="nup == ndn"):
        make_cisd_mode_trial_ops(System(norb=7, nelec=(3, 2), walker_kind="restricted"))
    with pytest.raises(ValueError, match="restricted walkers"):
        make_cisd_mode_trial_ops(System(norb=7, nelec=(2, 2), walker_kind="unrestricted"))


def test_truncated_overlap_force_bias_and_energy_match_zero_padded_modes():
    _, full_trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    mode_rank = 3
    truncated_trial = CisdModeTrial(
        ci1=full_trial.ci1,
        eigenvalues=full_trial.eigenvalues[:mode_rank],
        modes=full_trial.modes[:mode_rank],
    )
    zero_padded_trial = CisdModeTrial(
        ci1=full_trial.ci1,
        eigenvalues=full_trial.eigenvalues.at[mode_rank:].set(0.0),
        modes=full_trial.modes,
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(869),
        norb=full_trial.norb,
        n_chol=7,
        basis="restricted",
    )
    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    truncated_ctx = build_mode_meas_ctx(ham, truncated_trial, cfg=cfg, n_mode_chunks=2)
    padded_ctx = build_mode_meas_ctx(ham, zero_padded_trial, cfg=cfg, n_mode_chunks=2)
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(871),
        full_trial.norb,
        full_trial.nocc_full,
        mix=0.25,
    )

    truncated_overlap = jax.jit(mode_overlap_r)(walker, truncated_trial)
    padded_overlap = jax.jit(mode_overlap_r)(walker, zero_padded_trial)
    truncated_fb = jax.jit(mode_force_bias_kernel)(walker, ham, truncated_ctx, truncated_trial)
    padded_fb = jax.jit(mode_force_bias_kernel)(walker, ham, padded_ctx, zero_padded_trial)
    truncated_energy = jax.jit(mode_energy_kernel)(walker, ham, truncated_ctx, truncated_trial)
    padded_energy = jax.jit(mode_energy_kernel)(walker, ham, padded_ctx, zero_padded_trial)

    np.testing.assert_allclose(truncated_overlap, padded_overlap, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(truncated_fb, padded_fb, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(truncated_energy, padded_energy, rtol=2.0e-12, atol=2.0e-12)


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
@pytest.mark.parametrize("n_chol", [0, 1, 256, 257, 513])
def test_measurement_context_singles_across_cholesky_batches(
    nocc_t_core, nvir_t_outer, n_chol
):
    _, trial, _, _ = _make_dense_and_mode_trials(
        nocc_t_core=nocc_t_core, nvir_t_outer=nvir_t_outer
    )
    rng = np.random.default_rng(872)
    chol = 0.03 * rng.normal(size=(n_chol, trial.norb, trial.norb))
    ham = HamChol(
        h0=jnp.asarray(0.0), h1=jnp.eye(trial.norb), chol=jnp.asarray(chol), basis="restricted"
    )
    ctx = build_mode_meas_ctx(ham, trial)
    expected = chol[:, :, trial.vir_act_slice] @ np.asarray(trial.ci1).T
    np.testing.assert_allclose(ctx.lci1, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_array_equal(ctx.rot_chol, chol[:, : trial.nocc_full, :])
    assert ctx.lci1.dtype == expected.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_batched_singles_preserves_dtype_and_bilinear_contraction(dtype):
    from trot.meas.cisd_modes import _build_lci1

    rng = np.random.default_rng(874)
    chol = rng.normal(size=(257, 5, 5))
    ci1 = rng.normal(size=(2, 2))
    if np.issubdtype(dtype, np.complexfloating):
        chol = chol + 1.0j * rng.normal(size=chol.shape)
        ci1 = ci1 + 1.0j * rng.normal(size=ci1.shape)
    chol, ci1 = chol.astype(dtype), ci1.astype(dtype)
    expected = chol[:, :, 2:4] @ ci1.T
    actual = _build_lci1(jnp.asarray(chol), jnp.asarray(ci1), vir_start=2, vir_stop=4)
    tolerance = 2.0e-6 if dtype in (np.float32, np.complex64) else 1.0e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_full_rank_double_mode_force_bias_and_energy_match_dense(
    nocc_t_core,
    nvir_t_outer,
):
    dense_trial, mode_trial, _, _ = _make_dense_and_mode_trials(
        nocc=2,
        nvir=3,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(877),
        norb=mode_trial.norb,
        n_chol=7,
        basis="restricted",
    )
    dp_cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=dp_cfg)
    mode_ctx = build_mode_meas_ctx(
        ham,
        mode_trial,
        cfg=dp_cfg,
        n_mode_chunks=1,
    )
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(881),
        mode_trial.norb,
        mode_trial.nocc_full,
        mix=0.25,
    )

    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense_trial)
    mode_fb = jax.jit(mode_force_bias_kernel)(walker, ham, mode_ctx, mode_trial)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense_trial)
    mode_energy = jax.jit(mode_energy_kernel)(walker, ham, mode_ctx, mode_trial)

    np.testing.assert_allclose(mode_fb, dense_fb, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(mode_energy, dense_energy, rtol=2.0e-12, atol=2.0e-12)


@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("n_chol", [0, 1, 256, 257, 513])
@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_initial_energy_cholesky_batches_match_dense(
    mixed_precision, n_chol, nocc_t_core, nvir_t_outer
):
    dense_trial, mode_trial, _, _ = _make_dense_and_mode_trials(
        nocc=2, nvir=3, nocc_t_core=nocc_t_core, nvir_t_outer=nvir_t_outer,
        mode_dtype=jnp.float32 if mixed_precision else jnp.float64,
    )
    rng = np.random.default_rng(875)
    chol = 0.03 * rng.normal(size=(n_chol, mode_trial.norb, mode_trial.norb))
    ham = HamChol(
        h0=jnp.asarray(0.25), h1=jnp.eye(mode_trial.norb),
        chol=jnp.asarray(chol), basis="restricted",
    )
    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype_testing=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    dense_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=cfg)
    mode_ctx = build_mode_meas_ctx(ham, mode_trial, cfg=cfg, n_mode_chunks=2)
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(876), mode_trial.norb, mode_trial.nocc_full, mix=0.2
    )
    expected = jax.jit(dense_energy_kernel)(walker, ham, dense_ctx, dense_trial)
    actual = jax.jit(mode_initial_energy_kernel)(walker, ham, mode_ctx, mode_trial)
    tolerance = 2.0e-6 if mixed_precision else 2.0e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_initial_energy_matches_first_walker_deterministic_cisd_energy(
    mixed_precision, nocc_t_core, nvir_t_outer
):
    from trot.prop.afqmc import init_prop_state
    from trot.prop.types import QmcParams

    _, trial, _, _ = _make_dense_and_mode_trials(
        nocc=2,
        nvir=3,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
        mode_dtype=jnp.float32 if mixed_precision else jnp.float64,
    )
    sys = System(
        norb=trial.norb,
        nelec=(trial.nocc_full, trial.nocc_full),
        walker_kind="restricted",
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(877), norb=trial.norb, n_chol=257, basis="restricted"
    )
    meas_ops = make_cisd_mode_meas_ops(
        sys,
        mixed_precision=mixed_precision,
        n_mode_chunks=2,
        energy_sampling=CisdModePairSamplingCfg(
            chol_head_size=2, pair_sample_size=16, guide_chol_batch_size=2
        ),
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed), trial.norb, trial.nocc_full, mix=0.2
            )
            for seed in (878, 879)
        ]
    )
    ctx = meas_ops.build_meas_ctx(ham, trial)
    assert meas_ops.require_kernel(k_energy) is mode_energy_kernel
    assert meas_ops.require_kernel(k_energy_init) is mode_initial_energy_kernel
    expected = jnp.real(mode_energy_kernel(walkers[0], ham, ctx, trial))
    state = init_prop_state(
        sys=sys,
        ham_data=ham,
        trial_ops=make_cisd_mode_trial_ops(sys),
        trial_data=trial,
        meas_ops=meas_ops,
        params=QmcParams(n_walkers=2, n_chunks=2, seed=880),
        initial_walkers=walkers,
    )
    tolerance = 2.0e-6 if mixed_precision else 2.0e-12
    np.testing.assert_allclose(state.e_estimate, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        state.pop_control_ene_shift, expected, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_array_equal(state.walkers, walkers)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_runtime_initialization_builds_cisd_measurement_context_once(mixed_precision):
    from types import SimpleNamespace

    from trot.prop.afqmc import make_prop_ops
    from trot.prop.types import QmcParams
    from trot.runtime_layout import DefaultRuntimeLayout

    _, trial, _, _ = _make_dense_and_mode_trials(
        nocc=2, nvir=3, nocc_t_core=1, nvir_t_outer=2,
        mode_dtype=jnp.float32 if mixed_precision else jnp.float64,
    )
    sys = System(
        norb=trial.norb, nelec=(trial.nocc_full, trial.nocc_full), walker_kind="restricted"
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(881), norb=trial.norb, n_chol=7, basis="restricted"
    )
    meas_ops = make_cisd_mode_meas_ops(
        sys, mixed_precision=mixed_precision, n_mode_chunks=2,
        energy_sampling=CisdModePairSamplingCfg(
            chol_head_size=2, pair_sample_size=16, guide_chol_batch_size=2
        ),
    )
    built_contexts = []

    def build_context(ham_data, trial_data):
        assert not built_contexts, "Runtime initialization rebuilt the measurement context."
        ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        built_contexts.append(ctx)
        return ctx

    job = SimpleNamespace(
        sys=sys, ham_data=ham, trial_data=trial,
        trial_ops=make_cisd_mode_trial_ops(sys),
        meas_ops=replace(meas_ops, build_meas_ctx=build_context),
        prop_ops=make_prop_ops("restricted", "restricted"),
        params=QmcParams(n_walkers=2, n_chunks=2, seed=882),
        params_cls=QmcParams, mesh=None,
    )
    layout = DefaultRuntimeLayout()
    prepared = layout.prepare(job)
    assert len(built_contexts) == 1
    assert prepared.meas_ctx is built_contexts[0]
    expected = jnp.real(mode_energy_kernel(prepared.state.walkers[0], ham, prepared.meas_ctx, trial))
    tolerance = 2.0e-6 if mixed_precision else 2.0e-12
    np.testing.assert_allclose(prepared.state.e_estimate, expected, rtol=tolerance, atol=tolerance)

    reused = layout.prepare(job, meas_ctx=prepared.meas_ctx, prop_ctx=prepared.prop_ctx)
    assert len(built_contexts) == 1
    assert reused.meas_ctx is prepared.meas_ctx
    np.testing.assert_allclose(reused.state.e_estimate, expected, rtol=tolerance, atol=tolerance)


def test_mode_measurement_mixed_precision_policy_and_accuracy():
    dense_trial, mode_double, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    mode_mixed = CisdModeTrial(
        ci1=mode_double.ci1,
        eigenvalues=mode_double.eigenvalues,
        modes=mode_double.modes.astype(jnp.float32),
    )
    sys = System(
        norb=mode_mixed.norb,
        nelec=(mode_mixed.nocc_full, mode_mixed.nocc_full),
        walker_kind="restricted",
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(883),
        norb=mode_mixed.norb,
        n_chol=7,
        basis="restricted",
    )
    dp_cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=dp_cfg)
    mode_ops = make_cisd_mode_meas_ops(sys, mixed_precision=True, n_mode_chunks=2)
    mode_ctx = mode_ops.build_meas_ctx(ham, mode_mixed)
    cfg = get_cisd_mode_meas_cfg(mode_ops)
    assert cfg is not None
    assert cfg.mixed_real_dtype == jnp.float32
    assert cfg.mixed_complex_dtype == jnp.complex64

    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(887), mode_mixed.norb, mode_mixed.nocc_full, mix=0.25
    )
    dense_fb = dense_force_bias_kernel(walker, ham, dense_ctx, dense_trial)
    mixed_fb = mode_ops.require_kernel(k_force_bias)(walker, ham, mode_ctx, mode_mixed)
    dense_energy = dense_energy_kernel(walker, ham, dense_ctx, dense_trial)
    mixed_energy = mode_ops.require_kernel(k_energy)(walker, ham, mode_ctx, mode_mixed)

    fb_relative_error = float(jnp.linalg.norm(mixed_fb - dense_fb) / jnp.linalg.norm(dense_fb))
    energy_absolute_error = float(jnp.abs(mixed_energy - dense_energy))
    assert fb_relative_error < 1.0e-5
    assert energy_absolute_error < 1.0e-4


def test_n_mode_chunks_validation_and_rank_cap():
    _, mode_trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(899),
        norb=mode_trial.norb,
        n_chol=5,
        basis="restricted",
    )

    with pytest.raises(ValueError, match="n_mode_chunks must be positive"):
        build_mode_meas_ctx(ham, mode_trial, n_mode_chunks=0)

    mode_ctx = build_mode_meas_ctx(ham, mode_trial, n_mode_chunks=100)
    assert mode_ctx.n_mode_chunks == mode_trial.mode_rank


def test_mode_energy_is_independent_of_n_mode_chunks_in_double_precision():
    dense_trial, mode_trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(907),
        norb=mode_trial.norb,
        n_chol=5,
        basis="restricted",
    )
    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float64,
        mixed_complex_dtype=jnp.complex128,
        mixed_real_dtype_testing=jnp.float64,
        mixed_complex_dtype_testing=jnp.complex128,
    )
    dense_ctx = build_dense_meas_ctx(ham, dense_trial, cfg=cfg)
    walker = testing.make_restricted_walker_near_ref(
        jax.random.PRNGKey(911), mode_trial.norb, mode_trial.nocc_full, mix=0.25
    )
    reference = dense_energy_kernel(walker, ham, dense_ctx, dense_trial)

    for n_mode_chunks in (1, 2, 4, 32):
        mode_ctx = build_mode_meas_ctx(
            ham,
            mode_trial,
            cfg=cfg,
            n_mode_chunks=n_mode_chunks,
        )
        candidate = jax.jit(mode_energy_kernel)(walker, ham, mode_ctx, mode_trial)
        np.testing.assert_allclose(candidate, reference, rtol=2.0e-12, atol=2.0e-12)


def test_pair_sampling_config_validation_and_factory_opt_in():
    with pytest.raises(ValueError, match="chol_head_size must be nonnegative"):
        CisdModePairSamplingCfg(chol_head_size=-1, pair_sample_size=8)
    with pytest.raises(ValueError, match="pair_sample_size must be positive"):
        CisdModePairSamplingCfg(chol_head_size=0, pair_sample_size=0)
    with pytest.raises(ValueError, match="uniform_mix must lie"):
        CisdModePairSamplingCfg(
            chol_head_size=0,
            pair_sample_size=8,
            tail_probability_uniform_mix=1.1,
        )
    with pytest.raises(ValueError, match="at least two"):
        CisdModePairSamplingCfg(
            chol_head_size=0,
            pair_sample_size=1,
            track_half_sample_diagnostic=True,
        )
    with pytest.raises(ValueError, match="walker_guide_policy"):
        CisdModePairSamplingCfg(
            chol_head_size=0,
            pair_sample_size=8,
            walker_guide_policy="invalid",  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(ValueError, match="walker_guide_weight_mix"):
        CisdModePairSamplingCfg(
            chol_head_size=0,
            pair_sample_size=8,
            walker_guide_weight_mix=0.0,
        )
    with pytest.raises(ValueError, match="target_tail_std_fraction"):
        CisdModePairTuningCfg(target_tail_std_fraction=0.0)
    with pytest.raises(ValueError, match="final_error_target_ha"):
        CisdModePairTuningCfg(final_error_target_ha=0.0)
    with pytest.raises(ValueError, match="final_error_sampling_fraction"):
        CisdModePairTuningCfg(final_error_sampling_fraction=0.0)
    with pytest.raises(ValueError, match="cross_validation_quantile"):
        CisdModePairTuningCfg(cross_validation_quantile=0.0)
    with pytest.raises(ValueError, match="guide_policy"):
        CisdModePairTuningCfg(guide_policy="invalid")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="tuning_population_count"):
        CisdModePairTuningCfg(tuning_population_count=0)
    with pytest.raises(ValueError, match="tuning_population_spacing_blocks"):
        CisdModePairTuningCfg(tuning_population_spacing_blocks=0)
    with pytest.raises(ValueError, match="walker_guide_policy"):
        CisdModePairTuningCfg(
            walker_guide_policy="invalid"  # pyright: ignore[reportArgumentType]
        )

    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    sys = System(
        norb=trial.norb,
        nelec=(trial.nocc_full, trial.nocc_full),
        walker_kind="restricted",
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(919),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    sampling = CisdModePairSamplingCfg(chol_head_size=2, pair_sample_size=16)
    deterministic_ops = make_cisd_mode_meas_ops(sys, mixed_precision=False)
    sampled_ops = make_cisd_mode_meas_ops(
        sys,
        mixed_precision=False,
        energy_sampling=sampling,
    )
    assert deterministic_ops.block_energy is None
    assert sampled_ops.block_energy is pair_sampled_block_energy
    assert deterministic_ops.retune_block_energy is None
    assert sampled_ops.retune_block_energy is None

    tuning = CisdModePairTuningCfg()
    with pytest.raises(ValueError, match="requires an equilibration"):
        make_cisd_mode_meas_ops(sys, mixed_precision=False, energy_tuning=tuning)
    tuned_ops = make_cisd_mode_meas_ops(
        sys,
        mixed_precision=False,
        energy_sampling=sampling,
        energy_tuning=tuning,
    )
    assert tuned_ops.retune_block_energy is not None

    sampled_ctx = sampled_ops.build_meas_ctx(ham, trial)
    assert sampled_ctx.energy_sampling == sampling
    assert sampled_ctx.reference_chol_scores.shape == (5,)
    assert sampled_ctx.chol_tail_prob.shape == (3,)
    np.testing.assert_allclose(jnp.sum(sampled_ctx.chol_tail_prob), 1.0, atol=1.0e-14)
    assert bool(jnp.all(sampled_ctx.chol_tail_prob > 0.0))

    invalid_sampling = CisdModePairSamplingCfg(chol_head_size=6, pair_sample_size=16)
    with pytest.raises(ValueError, match="must not exceed"):
        build_mode_meas_ctx(ham, trial, energy_sampling=invalid_sampling)


@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
@pytest.mark.parametrize("chol_batch_size", [4, 16])
def test_compiled_reference_scores_match_direct_cholesky_terms(
    mixed_precision, nocc_t_core, nvir_t_outer, chol_batch_size
):
    from trot.meas.cisd_modes import _build_reference_chol_scores, _cisd_mode_chol_terms

    _, trial, _, _ = _make_dense_and_mode_trials(
        nocc=2,
        nvir=3,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
        mode_dtype=jnp.float32 if mixed_precision else jnp.float64,
    )
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(920), norb=trial.norb, n_chol=7, basis="restricted"
    )
    cfg = CisdMeasCfg(
        memory_mode="high",
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
    )
    ctx = build_mode_meas_ctx(ham, trial, cfg=cfg, n_mode_chunks=2)
    reference_walker = jnp.eye(trial.norb, trial.nocc_full, dtype=jnp.complex128)
    common = _cisd_mode_energy_common(reference_walker, ham, ctx, trial)
    direct = _cisd_mode_chol_terms(common, ham.chol, ctx.rot_chol, ctx.lci1, ctx, trial)
    expected = jnp.maximum(jnp.abs(direct).astype(jnp.float64), 1.0e-300)
    actual = _build_reference_chol_scores(ham, ctx, trial, chol_batch_size=chol_batch_size)
    tolerance = 2.0e-6 if mixed_precision else 2.0e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == jnp.float64


def test_ranked_arbitrary_head_uses_indices_without_reordering_cholesky_storage():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(921),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    base_ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
    )
    sampling = CisdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=16,
        rank_head_by_guide=True,
        head_chol_batch_size=1,
    )
    scores = jnp.asarray([1.0, 9.0, 2.0, 8.0, 3.0], dtype=jnp.float64)
    ctx = configure_cisd_mode_pair_sampling(base_ctx, sampling, scores)
    np.testing.assert_array_equal(ctx.chol_head_indices, np.asarray([1, 3]))
    np.testing.assert_array_equal(ctx.chol_tail_indices, np.asarray([0, 2, 4]))
    np.testing.assert_allclose(
        ctx.chol_tail_prob,
        np.asarray([1.0, 2.0, 3.0]) / 6.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    mixed_sampling = replace(sampling, tail_probability_uniform_mix=0.2)
    mixed_ctx = configure_cisd_mode_pair_sampling(base_ctx, mixed_sampling, scores)
    expected_mixed_prob = 0.8 * np.asarray([1.0, 2.0, 3.0]) / 6.0 + 0.2 / 3.0
    np.testing.assert_allclose(
        mixed_ctx.chol_tail_prob,
        expected_mixed_prob,
        rtol=0.0,
        atol=1.0e-14,
    )

    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.25,
            )
            for seed in (922, 924, 926)
        ]
    )
    common = jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    all_terms = _cisd_mode_chol_terms_for_walkers(
        common,
        ham.chol,
        ctx.rot_chol,
        ctx.lci1,
        ctx,
        trial,
    )
    candidate = jax.jit(
        lambda common_i: _cisd_mode_chol_index_sum_for_walkers(
            common_i,
            ctx.chol_head_indices,
            ham,
            ctx,
            trial,
            n_walker_chunks=2,
            chol_batch_size=1,
        )
    )(common)
    expected = jnp.sum(all_terms[:, jnp.asarray([1, 3])], axis=1)
    np.testing.assert_allclose(candidate, expected, rtol=2.0e-12, atol=2.0e-12)

    candidate_sum, candidate_squared_norm = jax.jit(
        lambda common_i: _cisd_mode_chol_index_moments_for_walkers(
            common_i,
            ctx.chol_head_indices,
            ham,
            ctx,
            trial,
            n_walker_chunks=2,
            chol_batch_size=1,
        )
    )(common)
    expected_terms = all_terms[:, jnp.asarray([1, 3])]
    np.testing.assert_allclose(candidate_sum, expected, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        candidate_squared_norm,
        jnp.sum(jnp.real(expected_terms) ** 2, axis=1),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_sampled_pair_terms_gather_inside_chunks_matches_full_pair_matrix():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(923),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.25,
            )
            for seed in (927, 929, 937)
        ]
    )
    common = jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    all_terms = _cisd_mode_chol_terms_for_walkers(
        common,
        ham.chol,
        ctx.rot_chol,
        ctx.lci1,
        ctx,
        trial,
    )
    sample_walker = jnp.asarray([2, 0, 1, 2, 1, 0, 2], dtype=jnp.int32)
    sample_chol = jnp.asarray([4, 1, 3, 0, 2, 4, 2], dtype=jnp.int32)
    expected = all_terms[sample_walker, sample_chol]

    candidate = jax.jit(
        lambda walker_indices, chol_indices: _cisd_mode_chol_pair_terms(
            common,
            walker_indices,
            chol_indices,
            ham,
            ctx,
            trial,
            n_chunks=3,
        )
    )(sample_walker, sample_chol)

    np.testing.assert_allclose(candidate, expected, rtol=2.0e-12, atol=2.0e-12)


def test_pair_sampled_block_energy_full_head_matches_weighted_deterministic_energy():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(929),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    sampling = CisdModePairSamplingCfg(chol_head_size=5, pair_sample_size=8)
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
        energy_sampling=sampling,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.25,
            )
            for seed in (937, 941, 947)
        ]
    )
    weights = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64)
    exact = jax.vmap(mode_energy_kernel, in_axes=(0, None, None, None))(
        walkers,
        ham,
        ctx,
        trial,
    )
    expected = jnp.sum(weights * jnp.real(exact)) / jnp.sum(weights)
    candidate = jax.jit(pair_sampled_block_energy, static_argnums=4)(
        walkers,
        weights,
        jnp.ones_like(weights, dtype=jnp.complex128),
        jax.random.PRNGKey(953),
        2,
        ham,
        ctx,
        trial,
        jnp.asarray(0.0),
        jnp.asarray(20.0),
    )
    assert ctx.chol_tail_prob.shape == (0,)
    np.testing.assert_allclose(candidate, expected, rtol=2.0e-12, atol=2.0e-12)


def test_pair_sampled_head_guard_replaces_flagged_walkers_and_reports_weight():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(954),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    sampling = CisdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=32768,
        guard_head_deviations=True,
    )
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
        energy_sampling=sampling,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=mix,
            )
            for seed, mix in ((955, 0.1), (957, 0.25), (961, 0.65))
        ]
    )
    weights = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64)
    norm_weights = weights / jnp.sum(weights)
    common = jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    all_terms = jnp.real(
        _cisd_mode_chol_terms_for_walkers(
            common,
            ham.chol,
            ctx.rot_chol,
            ctx.lci1,
            ctx,
            trial,
        )
    )
    head_energy = jnp.real(common.base) + jnp.sum(
        all_terms[:, ctx.chol_head_indices],
        axis=1,
    )
    head_center = jnp.sum(norm_weights * head_energy)
    deviations = jnp.abs(head_energy - head_center)
    sorted_deviations = jnp.sort(deviations)
    threshold = 0.5 * (sorted_deviations[-2] + sorted_deviations[-1])
    guarded = deviations > threshold
    e_ref = jnp.asarray(-7.5, dtype=jnp.float64)
    tail_terms = all_terms[:, ctx.chol_tail_indices]
    full_energy = head_energy + jnp.sum(tail_terms, axis=1)
    expected = jnp.sum(norm_weights * jnp.where(guarded, e_ref, full_energy))
    accepted_weight = jnp.sum(jnp.where(guarded, 0.0, norm_weights))
    accepted_prob = jnp.where(guarded, 0.0, norm_weights) / accepted_weight
    importance_values = accepted_weight * tail_terms / ctx.chol_tail_prob[None, :]
    joint_prob = accepted_prob[:, None] * ctx.chol_tail_prob[None, :]
    tail_mean = jnp.sum(joint_prob * importance_values)
    tail_variance = jnp.sum(joint_prob * (importance_values - tail_mean) ** 2)
    standard_error = jnp.sqrt(tail_variance / sampling.pair_sample_size)

    candidate = jax.jit(pair_sampled_block_energy, static_argnums=4)(
        walkers,
        weights,
        jnp.ones_like(weights, dtype=jnp.complex128),
        jax.random.PRNGKey(963),
        2,
        ham,
        ctx,
        trial,
        e_ref,
        threshold,
    )

    assert isinstance(candidate, BlockEnergyEstimate)
    np.testing.assert_allclose(
        candidate.energy,
        expected,
        rtol=0.0,
        atol=float(6.0 * standard_error + 1.0e-12),
    )
    np.testing.assert_array_equal(
        candidate.diagnostics[d_energy_head_guard_count],
        jnp.sum(guarded),
    )
    np.testing.assert_allclose(
        candidate.diagnostics[d_energy_head_guard_weight],
        jnp.sum(norm_weights * guarded),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_head_rms_walker_importance_uses_exact_ht_correction():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(965),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    pair_sample_size = 128
    weight_mix = 0.2
    sampling = CisdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=pair_sample_size,
        walker_guide_policy="head_rms",
        walker_guide_weight_mix=weight_mix,
        track_half_sample_diagnostic=True,
    )
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
        energy_sampling=sampling,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=mix,
            )
            for seed, mix in ((969, 0.1), (973, 0.3), (975, 0.6))
        ]
    )
    weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    norm_weights = weights / jnp.sum(weights)
    common = jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    all_terms = jnp.real(
        _cisd_mode_chol_terms_for_walkers(
            common,
            ham.chol,
            ctx.rot_chol,
            ctx.lci1,
            ctx,
            trial,
        )
    )
    head_terms = all_terms[:, ctx.chol_head_indices]
    head_energy = jnp.real(common.base) + jnp.sum(head_terms, axis=1)
    head_rms_scores = jnp.sqrt(jnp.sum(head_terms**2, axis=1))
    guided_probabilities = norm_weights * head_rms_scores
    guided_probabilities /= jnp.sum(guided_probabilities)
    walker_probabilities = weight_mix * norm_weights + (1.0 - weight_mix) * guided_probabilities
    walker_corrections = norm_weights / walker_probabilities

    rng_key = jax.random.PRNGKey(979)
    key_walker, key_chol = jax.random.split(rng_key)
    sample_walker = jax.random.choice(
        key_walker,
        weights.shape[0],
        shape=(pair_sample_size,),
        replace=True,
        p=walker_probabilities,
    )
    sample_chol_rel = jax.random.choice(
        key_chol,
        ctx.chol_tail_prob.shape[0],
        shape=(pair_sample_size,),
        replace=True,
        p=ctx.chol_tail_prob,
    )
    sample_values = (
        walker_corrections[sample_walker]
        * all_terms[sample_walker, ctx.chol_tail_indices[sample_chol_rel]]
        / ctx.chol_tail_prob[sample_chol_rel]
    )
    expected_energy = jnp.sum(norm_weights * head_energy) + jnp.mean(sample_values)
    half_size = pair_sample_size // 2
    expected_diagnostic = 0.5 * (
        jnp.mean(sample_values[:half_size]) - jnp.mean(sample_values[half_size:])
    )

    candidate = jax.jit(pair_sampled_block_energy, static_argnums=4)(
        walkers,
        weights,
        jnp.ones_like(weights, dtype=jnp.complex128),
        rng_key,
        2,
        ham,
        ctx,
        trial,
        jnp.asarray(0.0),
        jnp.asarray(20.0),
    )

    assert isinstance(candidate, BlockEnergyEstimate)
    np.testing.assert_allclose(candidate.energy, expected_energy, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        candidate.diagnostics[d_energy_sampling_noise],
        expected_diagnostic,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        candidate.diagnostics[d_energy_walker_guide_ess],
        1.0 / jnp.sum(walker_probabilities**2),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        candidate.diagnostics[d_energy_walker_guide_max_correction],
        jnp.max(walker_corrections),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_pair_sampled_tail_matches_exact_energy_within_analytic_sampling_error():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(967),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    pair_sample_size = 8192
    sampling = CisdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=pair_sample_size,
        track_half_sample_diagnostic=True,
    )
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
        energy_sampling=sampling,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.25,
            )
            for seed in (971, 977, 983)
        ]
    )
    weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    norm_weights = weights / jnp.sum(weights)
    common = jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    all_terms = _cisd_mode_chol_terms_for_walkers(
        common,
        ham.chol,
        ctx.rot_chol,
        ctx.lci1,
        ctx,
        trial,
    )
    exact_per_walker = jnp.real(common.base + jnp.sum(all_terms, axis=1))
    exact_block = jnp.sum(norm_weights * exact_per_walker)

    tail_terms = jnp.real(all_terms[:, sampling.chol_head_size :])
    importance_values = tail_terms / ctx.chol_tail_prob[None, :]
    joint_prob = norm_weights[:, None] * ctx.chol_tail_prob[None, :]
    tail_mean = jnp.sum(joint_prob * importance_values)
    tail_variance = jnp.sum(joint_prob * (importance_values - tail_mean) ** 2)
    standard_error = jnp.sqrt(tail_variance / pair_sample_size)

    rng_key = jax.random.PRNGKey(991)
    candidate_result = jax.jit(pair_sampled_block_energy, static_argnums=4)(
        walkers,
        weights,
        jnp.ones_like(weights, dtype=jnp.complex128),
        rng_key,
        2,
        ham,
        ctx,
        trial,
        jnp.asarray(0.0),
        jnp.asarray(20.0),
    )
    assert isinstance(candidate_result, BlockEnergyEstimate)
    np.testing.assert_allclose(
        candidate_result.energy,
        exact_block,
        rtol=0.0,
        atol=float(6.0 * standard_error + 1.0e-12),
    )

    key_walker, key_chol = jax.random.split(rng_key)
    sample_walker = jax.random.choice(
        key_walker,
        weights.shape[0],
        shape=(pair_sample_size,),
        replace=True,
        p=norm_weights,
    )
    sample_chol_rel = jax.random.choice(
        key_chol,
        ctx.chol_tail_prob.shape[0],
        shape=(pair_sample_size,),
        replace=True,
        p=ctx.chol_tail_prob,
    )
    sampled_values = (
        jnp.real(all_terms[sample_walker, ctx.chol_tail_indices[sample_chol_rel]])
        / ctx.chol_tail_prob[sample_chol_rel]
    )
    half_size = pair_sample_size // 2
    expected_diagnostic = 0.5 * (
        jnp.mean(sampled_values[:half_size]) - jnp.mean(sampled_values[half_size:])
    )
    np.testing.assert_allclose(
        candidate_result.diagnostics[d_energy_sampling_noise],
        expected_diagnostic,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_streaming_population_statistics_match_full_contribution_table():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(997),
        norb=trial.norb,
        n_chol=5,
        basis="restricted",
    )
    ctx = build_mode_meas_ctx(
        ham,
        trial,
        cfg=CisdMeasCfg(memory_mode="high"),
        n_mode_chunks=2,
    )
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.25,
            )
            for seed in (1009, 1013, 1019)
        ]
    )
    weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    norm_weights = np.asarray(weights / jnp.sum(weights))
    common = jax.vmap(
        _cisd_mode_energy_common,
        in_axes=(0, None, None, None),
    )(walkers, ham, ctx, trial)
    terms = np.real(
        np.asarray(
            _cisd_mode_chol_terms_for_walkers(
                common,
                ham.chol,
                ctx.rot_chol,
                ctx.lci1,
                ctx,
                trial,
            )
        )
    )
    base = np.real(np.asarray(common.base))

    stats = stream_cisd_mode_population_statistics(
        walkers,
        weights,
        ham,
        ctx,
        trial,
        n_walker_chunks=2,
        chol_batch_size=2,
    )
    expected_means = np.sum(norm_weights[:, None] * terms, axis=0)
    expected_seconds = np.sum(norm_weights[:, None] * terms**2, axis=0)
    expected_local = base + np.sum(terms, axis=1)
    np.testing.assert_allclose(stats.term_means, expected_means, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        stats.term_second_moments,
        expected_seconds,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(stats.rms_scores, np.sqrt(expected_seconds), rtol=2.0e-12)
    np.testing.assert_allclose(stats.local_energies, expected_local, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(
        stats.exact_block_energy_ha,
        np.sum(norm_weights * expected_local),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_population_tuner_selects_least_work_candidate_meeting_noise_target():
    # The RMS-optimal tail variance after a ranked head is
    # (sum_g score_g)^2 - (sum_g mean_g)^2.
    stats = CisdModePopulationStats(
        term_means=np.zeros(4, dtype=np.float64),
        term_second_moments=np.asarray([16.0, 4.0, 1.0, 0.25]),
        rms_scores=np.asarray([4.0, 2.0, 1.0, 0.5]),
        local_energies=np.zeros(2, dtype=np.float64),
        exact_block_energy_ha=0.0,
        independent_population_std_ha=0.16,
        wall_seconds=0.0,
    )
    cfg = CisdModePairTuningCfg(
        target_tail_std_fraction=0.5,
        safety_factor=1.0,
        candidate_sample_sizes=(64, 256),
        maximum_head_fraction=0.75,
        production_head_chol_batch_size=2,
        settling_blocks=0,
    )
    selected = select_cisd_mode_pair_sampling(
        stats,
        cfg,
        n_walkers=10,
        calibration_std_ha=0.16,
        calibration_source="late equilibration block standard deviation",
    )

    # H=2, S=256 has std 1.5/sqrt(256)=0.09375 and fails. H=3,
    # S=64 has std 0.5/sqrt(64)=0.0625 and is the cheapest feasible cell.
    assert selected.sampling.chol_head_size == 3
    assert selected.sampling.pair_sample_size == 64
    assert selected.sampling.rank_head_by_guide
    assert selected.sampling.head_chol_batch_size == 2
    assert selected.estimated_pair_evaluations == 94
    np.testing.assert_allclose(selected.estimated_tail_std_ha, 0.0625)
    np.testing.assert_allclose(selected.target_tail_std_ha, 0.08)
    assert selected.target_tail_std_source == (
        "0.500 x late equilibration block standard deviation"
    )
    np.testing.assert_allclose(selected.calibration_std_ha, 0.16)


def test_population_tuner_averages_temporal_second_moments_and_conditional_variances():
    def make_stats(means: list[float]) -> CisdModePopulationStats:
        return CisdModePopulationStats(
            term_means=np.asarray(means, dtype=np.float64),
            term_second_moments=np.asarray([2.0, 2.0]),
            rms_scores=np.sqrt(np.asarray([2.0, 2.0])),
            local_energies=np.zeros(2, dtype=np.float64),
            exact_block_energy_ha=float(means[0]),
            independent_population_std_ha=0.2,
            wall_seconds=1.0,
        )

    averaged = average_cisd_mode_population_statistics(
        [make_stats([1.0, 0.0]), make_stats([-1.0, 0.0])]
    )
    np.testing.assert_allclose(averaged.term_means, np.zeros(2))
    np.testing.assert_allclose(averaged.term_second_moments, np.asarray([2.0, 2.0]))
    np.testing.assert_allclose(averaged.rms_scores, np.sqrt(np.asarray([2.0, 2.0])))
    assert averaged.population_term_means is not None
    np.testing.assert_allclose(
        averaged.population_term_means,
        np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
    )
    assert averaged.population_term_second_moments is not None
    np.testing.assert_allclose(
        averaged.population_term_second_moments,
        np.asarray([[2.0, 2.0], [2.0, 2.0]]),
    )
    assert averaged.exact_block_energy_ha == -1.0
    assert averaged.wall_seconds == 2.0

    selected = select_cisd_mode_pair_sampling(
        averaged,
        CisdModePairTuningCfg(
            guide_policy="population_rms",
            target_tail_std_ha=1.0,
            safety_factor=1.0,
            candidate_sample_sizes=(100,),
            maximum_head_fraction=0.0,
            tail_probability_uniform_mix=0.0,
            track_half_sample_diagnostic=False,
        ),
        n_walkers=10,
    )
    # q=(1/2, 1/2), so E[X^2]=2/(1/2)+2/(1/2)=8. The
    # population-conditional tail means are +1 and -1, hence E[E[X|b]^2]=1.
    assert selected.estimated_single_pair_variance_ha2 == 7.0
    assert selected.in_sample_single_pair_variance_ha2 == 7.0
    assert selected.cross_validation_fold_count == 2


def test_population_tuner_uses_held_out_variance_and_population_relative_budget():
    population_seconds = np.asarray(
        [[100.0, 1.0], [1.0, 100.0]],
        dtype=np.float64,
    )
    averaged_seconds = np.mean(population_seconds, axis=0)
    stats = CisdModePopulationStats(
        term_means=np.zeros(2, dtype=np.float64),
        term_second_moments=averaged_seconds,
        rms_scores=np.sqrt(averaged_seconds),
        local_energies=np.zeros(2, dtype=np.float64),
        exact_block_energy_ha=0.0,
        independent_population_std_ha=1.0,
        wall_seconds=0.0,
        population_term_means=np.zeros((2, 2), dtype=np.float64),
        population_term_second_moments=population_seconds,
    )
    cfg = CisdModePairTuningCfg(
        final_error_sampling_fraction=0.2,
        safety_factor=1.0,
        cross_validation_quantile=1.0,
        candidate_sample_sizes=(100_000_000,),
        maximum_head_fraction=0.0,
        tail_probability_uniform_mix=0.0,
        track_half_sample_diagnostic=False,
    )
    selected = select_cisd_mode_pair_sampling(
        stats,
        cfg,
        n_walkers=10,
        final_error_target_ha=7.0e-4,
        n_blocks=1000,
    )

    # The fitted guide is uniform and gives 202 Ha^2. In either held-out fold,
    # the guide from the other population is (1/11, 10/11), giving 1101.1 Ha^2.
    np.testing.assert_allclose(selected.in_sample_single_pair_variance_ha2, 202.0)
    np.testing.assert_allclose(selected.estimated_single_pair_variance_ha2, 1101.1)
    assert selected.cross_validation_fold_count == 2
    np.testing.assert_allclose(
        selected.target_tail_std_ha,
        0.35 * stats.independent_population_std_ha,
    )
    assert selected.target_tail_std_source == (
        "0.350 x independent population standard deviation"
    )


def test_hf_guide_tuner_uses_population_moments_to_select_ranked_estimator():
    stats = CisdModePopulationStats(
        term_means=np.zeros(4, dtype=np.float64),
        term_second_moments=np.asarray([16.0, 4.0, 1.0, 0.25]),
        rms_scores=np.asarray([4.0, 2.0, 1.0, 0.5]),
        local_energies=np.zeros(2, dtype=np.float64),
        exact_block_energy_ha=0.0,
        independent_population_std_ha=1.0,
        wall_seconds=0.0,
    )
    hf_scores = np.asarray([1.0, 4.0, 2.0, 0.5])
    cfg = CisdModePairTuningCfg(
        guide_policy="hf",
        target_tail_std_ha=0.5,
        safety_factor=1.0,
        candidate_sample_sizes=(100,),
        minimum_head_fraction=0.5,
        maximum_head_fraction=0.5,
        tail_probability_uniform_mix=0.0,
        track_half_sample_diagnostic=False,
    )
    selected = select_cisd_mode_pair_sampling(
        stats,
        cfg,
        n_walkers=10,
        reference_guide_scores=hf_scores,
    )

    # HF ranks indices (1, 2) into the exact head. The remaining probabilities
    # for indices (0, 3) are (2/3, 1/3), while the variance uses their measured
    # population second moments: 16/(2/3) + 0.25/(1/3) = 24.75.
    assert selected.guide_policy == "hf"
    assert selected.sampling.chol_head_size == 2
    assert selected.sampling.pair_sample_size == 100
    np.testing.assert_allclose(selected.estimated_single_pair_variance_ha2, 24.75)
    np.testing.assert_allclose(selected.estimated_tail_std_ha, np.sqrt(24.75 / 100.0))

    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(925),
        norb=trial.norb,
        n_chol=4,
        basis="restricted",
    )
    ctx = build_mode_meas_ctx(ham, trial)
    configured = configure_cisd_mode_pair_sampling(
        ctx,
        selected.sampling,
        jnp.asarray(hf_scores),
    )
    np.testing.assert_array_equal(configured.chol_head_indices, np.asarray([1, 2]))
    np.testing.assert_array_equal(configured.chol_tail_indices, np.asarray([0, 3]))
    np.testing.assert_allclose(configured.chol_tail_prob, np.asarray([2.0 / 3.0, 1.0 / 3.0]))


def test_retune_collects_temporally_separated_populations():
    _, trial, _, _ = _make_dense_and_mode_trials(nocc=2, nvir=3)
    ham = testing.make_random_ham_chol(
        jax.random.PRNGKey(927),
        norb=trial.norb,
        n_chol=4,
        basis="restricted",
    )
    equil_sampling = CisdModePairSamplingCfg(
        chol_head_size=2,
        pair_sample_size=16,
        rank_head_by_guide=True,
    )
    ctx = build_mode_meas_ctx(ham, trial, energy_sampling=equil_sampling)
    walkers = jnp.stack(
        [
            testing.make_restricted_walker_near_ref(
                jax.random.PRNGKey(seed),
                trial.norb,
                trial.nocc_full,
                mix=0.1,
            )
            for seed in (931, 933)
        ]
    )
    state = PropState(
        walkers=walkers,
        weights=jnp.ones(2, dtype=jnp.float64),
        overlaps=jnp.ones(2, dtype=jnp.complex128),
        rng_key=jax.random.PRNGKey(935),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    advance_calls = []

    def advance_blocks(state, *, n_blocks: int):
        advance_calls.append(n_blocks)
        return (
            state,
            {
                "energy": jnp.zeros(n_blocks, dtype=jnp.float64),
                "weight": jnp.ones(n_blocks, dtype=jnp.float64),
            },
            (),
        )

    retuned = retune_cisd_mode_pair_sampling(
        state,
        jnp.zeros(4, dtype=jnp.float64),
        jnp.ones(4, dtype=jnp.float64),
        None,
        ham,
        ctx,
        trial,
        advance_blocks=advance_blocks,
        tuning_cfg=CisdModePairTuningCfg(
            guide_policy="population_rms",
            target_tail_std_ha=1.0e6,
            safety_factor=1.0,
            candidate_sample_sizes=(16,),
            tuning_n_chunks=1,
            tuning_chol_batch_size=2,
            tuning_population_count=3,
            tuning_population_spacing_blocks=2,
            tail_probability_uniform_mix=0.0,
            track_half_sample_diagnostic=False,
            settling_blocks=0,
        ),
    )

    assert advance_calls == [2, 2]
    assert retuned.meas_ctx.energy_sampling.chol_head_size == 4
    assert retuned.meas_ctx.energy_sampling.pair_sample_size == 16
    assert retuned.settling_blocks == 0
