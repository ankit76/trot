"""Cheap host-side checks of the shared sampling budget and calibration hooks."""

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

from trot import config

config.configure_once(use_gpu=False)

import jax.numpy as jnp
import numpy as np
import pytest

from trot.prop.types import PropState


@pytest.fixture(params=["cisd", "ucisd_k", "ptccsd", "ptuccsd"])
def tuner(request):
    kind = request.param
    prefix = {
        "cisd": "CisdMode",
        "ucisd_k": "UcisdKMode",
        "ptccsd": "PtccsdMode",
        "ptuccsd": "PtuccsdMode",
    }[kind]
    module = import_module(f"trot.meas.{kind}_modes")
    return SimpleNamespace(
        kind=kind,
        module=module,
        config=getattr(module, f"{prefix}PairTuningCfg"),
        stats=getattr(module, f"{prefix}PopulationStats"),
        select=getattr(module, f"select_{kind}_mode_pair_sampling"),
        retune=getattr(module, f"retune_{kind}_mode_pair_sampling"),
        is_pt=kind.startswith("pt"),
    )


def _statistics(tuner, energy=0.0):
    fields = dict(
        term_means=np.zeros(4),
        term_second_moments=np.array([16.0, 4.0, 1.0, 0.25]),
        rms_scores=np.array([4.0, 2.0, 1.0, 0.5]),
        exact_block_energy_ha=energy,
        wall_seconds=0.0,
    )
    if tuner.is_pt:
        fields["phase_coherence"] = 1.0
    else:
        fields.update(local_energies=np.zeros(10), independent_population_std_ha=0.16)
    return tuner.stats(**fields)


@pytest.mark.parametrize("absolute_target", [None, 0.08])
def test_selection_is_independent_of_length_and_final_error(tuner, absolute_target):
    cfg = tuner.config(
        target_tail_std_ha=absolute_target,
        candidate_sample_sizes=(64, 256),
        maximum_head_fraction=0.75,
    )
    for blocks, stopping_target, legacy_target in (
        (30_000, None, None),
        (30_000, 1.0e-3, None),
        (300_000, 1.0e-3, None),
        (300_000, 1.0e-6, None),
        (300_000, None, 1.0e-3),
    ):
        selected = tuner.select(
            _statistics(tuner),
            replace(cfg, final_error_target_ha=legacy_target),
            n_walkers=10,
            reference_guide_scores=np.ones(4),
            calibration_std_ha=0.16 if absolute_target is None else float("nan"),
            calibration_source="exact calibration-population standard deviation",
            final_error_target_ha=stopping_target,
            n_blocks=blocks,
        )
        np.testing.assert_allclose(
            selected.target_tail_std_ha,
            0.35 * 0.16 if absolute_target is None else absolute_target,
        )
        assert selected.sampling.chol_head_size == 3
        assert selected.sampling.pair_sample_size == (256 if absolute_target is None else 64)
        assert selected.target_tail_std_source == (
            "0.350 x exact calibration-population standard deviation"
            if absolute_target is None else "absolute override"
        )


def test_zero_intrinsic_spread_requires_zero_added_noise(tuner):
    selected = tuner.select(
        _statistics(tuner),
        tuner.config(candidate_sample_sizes=(64,)),
        n_walkers=10,
        reference_guide_scores=np.ones(4),
        calibration_std_ha=0.0,
        calibration_source="exact calibration-population standard deviation",
        final_error_target_ha=1.0e-3,
        n_blocks=300_000,
    )
    assert selected.target_tail_std_ha == 0.0
    assert selected.estimated_tail_std_ha == 0.0
    assert selected.sampling.chol_head_size == 4


@pytest.mark.parametrize("noise_scale", [0.0, 1000.0])
@pytest.mark.parametrize("population_count", [1, 5])
def test_retune_uses_existing_exact_snapshots(tuner, monkeypatch, noise_scale, population_count):
    energies = np.linspace(-1.2, -0.8, population_count)
    streamed_energies = []
    advances = []
    selections = []

    def stream(*args, **kwargs):
        energy = energies[len(streamed_energies)]
        streamed_energies.append(energy)
        return _statistics(tuner, energy)

    def select(*args, **kwargs):
        result = tuner.select(*args, **kwargs)
        selections.append(result)
        return result

    def configure(ctx, sampling, scores):
        return SimpleNamespace(energy_sampling=sampling, component_sampling=sampling)

    monkeypatch.setattr(tuner.module, f"stream_{tuner.kind}_mode_population_statistics", stream)
    monkeypatch.setattr(tuner.module, f"select_{tuner.kind}_mode_pair_sampling", select)
    monkeypatch.setattr(tuner.module, f"configure_{tuner.kind}_mode_pair_sampling", configure)

    state = PropState(
        walkers=jnp.ones((10, 1, 1)),
        weights=jnp.ones(10),
        overlaps=jnp.ones(10, dtype=jnp.complex128),
        rng_key=jnp.array([0, 1], dtype=jnp.uint32),
        pop_control_ene_shift=jnp.asarray(0.0),
        e_estimate=jnp.asarray(0.0),
        node_encounters=jnp.asarray(0),
    )
    ctx = SimpleNamespace(
        model_sampling=None, energy_sampling=None, reference_chol_scores=jnp.ones(4)
    )

    def advance(state_i, *, n_blocks):
        advances.append(n_blocks)
        return state_i, {"energy": noise_scale * jnp.arange(n_blocks)}, ()

    kwargs = {}
    equilibration = noise_scale * jnp.arange(6)
    if tuner.is_pt:
        def overlap(walker, trial):
            return jnp.asarray(1.0)

        reference_name = "det_overlap_thouless_r" if tuner.kind == "ptccsd" else "reference_overlap_r"
        monkeypatch.setattr(tuner.module, reference_name, overlap)
        kwargs.update(
            guide_data=None,
            guide_meas_ops=SimpleNamespace(overlap=overlap),
            guide_meas_ctx=None,
        )
        equilibration = jnp.repeat(equilibration[:, None], 3, axis=1)

    def retune():
        return tuner.retune(
            state, equilibration, jnp.ones(6), SimpleNamespace(n_blocks=300_000),
            None, ctx, None,
            advance_blocks=advance,
            tuning_cfg=tuner.config(
                tuning_population_count=population_count,
                candidate_sample_sizes=(64, 256),
                maximum_head_fraction=0.75,
            ),
            target_error=1.0e-3,
            **kwargs,
        )

    if tuner.is_pt and population_count == 1:
        with pytest.raises(ValueError, match="at least two deterministic calibration"):
            retune()
        assert streamed_energies == []
        assert advances == []
        return

    retune()
    assert len(streamed_energies) == population_count
    assert advances == [2] * (population_count - 1)
    assert len(selections) == 1
    expected_std = np.std(energies, ddof=1) if population_count > 1 else 0.16
    np.testing.assert_allclose(selections[0].target_tail_std_ha, 0.35 * expected_std)
    assert selections[0].target_tail_std_source == (
        "0.350 x exact calibration-population standard deviation"
        if population_count > 1 else "0.350 x independent population standard deviation fallback"
    )
