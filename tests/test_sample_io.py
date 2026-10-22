from __future__ import annotations

import numpy as np
import pytest

from trot import sample_io
from trot.core.system import System
from trot.prop.types import QmcParams


def _writer(directory):
    return sample_io.BlockSampleWriter(
        directory,
        driver="test",
        sys=System(norb=2, nelec=(1, 1), walker_kind="restricted"),
        params=QmcParams(seed=7, n_blocks=4),
        h0=0.5,
    )


def test_sample_snapshots_keep_complex_raw_arrays_and_separate_repeated_runs(tmp_path):
    first = _writer(tmp_path)
    second = _writer(tmp_path)
    raw = np.asarray([[1.0 + 2.0j, 3.0], [np.nan, 1000.0]])
    first.write(2, sampling_complete=False, components=raw, keep_mask=[True, False])
    second.write(1, sampling_complete=False, components=raw[:1])

    assert first.path != second.path
    assert len(list(tmp_path.glob("*.npz"))) == 2
    with np.load(first.path, allow_pickle=False) as saved:
        np.testing.assert_equal(saved["components"], raw)
        np.testing.assert_array_equal(saved["keep_mask"], [True, False])
        assert saved["n_blocks_completed"] == 2
        assert saved["n_blocks_requested"] == 4
        assert not saved["sampling_complete"]
        assert saved["phase"] == "production"


@pytest.mark.parametrize("failure_stage", ["write", "replace"])
def test_failed_snapshot_preserves_previous_readable_archive(tmp_path, monkeypatch, failure_stage):
    writer = _writer(tmp_path)
    writer.write(2, sampling_complete=False, energies=[1.0, 2.0])
    previous = writer.path.read_bytes()

    if failure_stage == "write":
        def fail_write(stream, **arrays):
            stream.write(b"incomplete archive")
            raise OSError("simulated disk failure")

        monkeypatch.setattr(sample_io.np, "savez", fail_write)
    else:
        def fail_replace(source, destination):
            raise OSError("simulated rename failure")

        monkeypatch.setattr(sample_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        writer.write(4, sampling_complete=True, energies=[1.0, 2.0, 3.0, 4.0])
    assert writer.path.read_bytes() == previous
    with np.load(writer.path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["energies"], [1.0, 2.0])
        assert not saved["sampling_complete"]
    assert list(tmp_path.iterdir()) == [writer.path]


def test_disabling_sample_output_does_not_write(monkeypatch):
    def unexpected_write(*args, **kwargs):
        pytest.fail("Disabled sample output attempted to write")

    monkeypatch.setattr(sample_io.tempfile, "NamedTemporaryFile", unexpected_write)
    writer = _writer(None)
    writer.write(2, sampling_complete=False, energies=[1.0, 2.0])
    assert writer.path is None


def test_successful_update_replaces_snapshot_with_complete_history(tmp_path):
    writer = _writer(tmp_path)
    writer.write(2, sampling_complete=False, energies=[1.0, 2.0])
    writer.write(4, sampling_complete=True, energies=[1.0, 2.0, 3.0, 4.0])
    with np.load(writer.path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["energies"], [1.0, 2.0, 3.0, 4.0])
        assert saved["sampling_complete"]
        assert saved["n_blocks_completed"] == 4
    assert list(tmp_path.iterdir()) == [writer.path]
