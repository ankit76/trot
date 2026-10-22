"""Atomic snapshots of raw production-block statistics for offline analysis."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .core.system import System
from .prop.types import QmcParams


class BlockSampleWriter:
    """Keep one cumulative NPZ per run, without overwriting earlier runs.

    Snapshots contain production statistics, not walker restart state. A write
    failure propagates to the caller and leaves the previous snapshot intact.
    Passing ``directory=None`` disables writing.
    """

    def __init__(
        self,
        directory: str | Path | None,
        *,
        driver: str,
        sys: System,
        params: QmcParams,
        h0: Any,
        **metadata: Any,
    ) -> None:
        self.path: Path | None = None
        self.metadata: dict[str, Any] = {}
        if directory is None:
            return

        out_dir = Path(directory).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"samples_{driver}_seed{params.seed}_{uuid4().hex}.npz"
        self.metadata = {
            "format_version": 1,
            "driver": driver,
            "phase": "production",
            "h0": h0,
            "seed": params.seed,
            "dt": params.dt,
            "n_prop_steps": params.n_prop_steps,
            "n_walkers": params.n_walkers,
            "n_chunks": params.n_chunks,
            "n_eql_blocks": params.n_eql_blocks,
            "n_blocks_requested": params.n_blocks,
            "error_method": params.error_method,
            "norb": sys.norb,
            "nelec": sys.nelec,
            "walker_kind": sys.walker_kind,
            **metadata,
        }
        print(f"Raw production samples: {self.path}", flush=True)

    def write(
        self,
        completed_blocks: int,
        *,
        sampling_complete: bool,
        **samples: Any,
    ) -> None:
        if self.path is None:
            return
        arrays = {
            name: np.asarray(value)
            for name, value in {
                **self.metadata,
                "n_blocks_completed": completed_blocks,
                "sampling_complete": sampling_complete,
                **samples,
            }.items()
        }
        # Archives must be readable with np.load(..., allow_pickle=False).
        if any(value.dtype.hasobject for value in arrays.values()):
            raise TypeError("Block sample snapshots cannot contain object arrays.")

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.path.parent, prefix=f".{self.path.name}.",
                suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            # The temporary file is on the same filesystem. Readers see either
            # the preceding complete archive or this complete archive.
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
