from __future__ import annotations

import os
import platform
import shlex
import socket
import sys
import warnings
from dataclasses import dataclass


# An OOM on one GPU can strand the other ranks in Acquire/Initialize clique.
# XLA otherwise leaves these NCCL rendezvous waiting indefinitely. Its timeout
# starts after the initial 10-second warning. Allow another 30 minutes so a
# slow but recoverable communicator initialization does not abort the run.
_GPU_FAILURE_XLA_DEFAULTS = {
    "--xla_gpu_nccl_termination_timeout_seconds": "1800",
    "--xla_gpu_nccl_terminate_on_error": "true",
}


def _set_gpu_failure_defaults() -> None:
    """Bound native GPU synchronization failures; preserve explicit XLA flags."""
    flags = os.environ.get("XLA_FLAGS", "")
    configured = {token.split("=", 1)[0] for token in shlex.split(flags)}
    additions = [
        f"{name}={value}"
        for name, value in _GPU_FAILURE_XLA_DEFAULTS.items()
        if name not in configured
    ]
    if additions:
        os.environ["XLA_FLAGS"] = " ".join([flags, *additions]).strip()


def is_jupyter_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def _parse_visible_devices(value: str | None) -> int | None:
    if value is None:
        return None

    text = value.strip()
    if text == "":
        return 0

    lowered = text.lower()
    if lowered in {"-1", "none", "novisibledevices"}:
        return 0

    return len([item for item in text.split(",") if item.strip()])


def visible_gpu_count() -> int:
    """
    Number of GPUs visible to the current process before JAX initializes.

    Preference order mirrors common CUDA/ROCm launcher conventions.
    If no visibility env var is set, fall back to counting NVIDIA GPUs via
    `nvidia-smi -L`; otherwise return 0 if detection is unavailable.
    """
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        count = _parse_visible_devices(os.getenv(name))
        if count is not None:
            return count

    import shutil
    import subprocess

    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                return sum(1 for line in r.stdout.splitlines() if "GPU" in line)
        except Exception:
            pass

    return 0


@dataclass
class AfqmcConfig:
    """
    Global configuration.

    use_gpu:
      - None  : auto (prefer GPU if available, else CPU)
      - True  : force GPU (error if unavailable)
      - False : force CPU
    matmul_precision:
      JAX matrix-product precision, independent of array storage dtypes.
      Defaults to "default", JAX's faster platform policy. Select "highest"
      for more accurate matrix products without changing array storage dtypes.
    """

    use_gpu: bool | None = None
    single_precision: bool = False
    disable_tf32: bool = False  # Disable TF32 on gpu if true
    quiet: bool = True  # suppress prints
    matmul_precision: str = "default"


afqmc_config = AfqmcConfig()

_configured_once = False


def configure_once(
    *,
    use_gpu: bool | None = None,
    single_precision: bool | None = None,
    disable_tf32: bool | None = None,
    quiet: bool | None = None,
    matmul_precision: str | None = None,
) -> None:
    """
    Configure JAX once, subsequent calls do nothing.
    Use GPU if available by default.

    Matrix products use JAX's faster platform default. Pass
    ``matmul_precision="highest"`` for more accurate matrix products, or another
    JAX-supported precision setting. An explicit argument takes precedence
    over ``JAX_DEFAULT_MATMUL_PRECISION``, which otherwise overrides the
    ``afqmc_config`` default. This does not change array storage dtypes.
    Call before importing ``trot.afqmc`` or tracing any JAX computations.

    GPU configuration also enables fatal asynchronous collective errors and a
    30-minute NCCL rendezvous termination timeout after XLA's initial warning.
    Explicit values in ``XLA_FLAGS`` override these defaults. Configure before
    importing JAX so the native runtime sees the settings at initialization.
    """
    global _configured_once
    if _configured_once:
        return

    assert (
        isinstance(use_gpu, bool) or use_gpu is None
    ), f"Expect a bool | None for 'use_gpu', but got '{type(use_gpu)}'."
    assert (
        isinstance(single_precision, bool) or single_precision is None
    ), f"Expect a bool | None for 'single_precision', but got '{type(single_precision)}'."
    assert (
        isinstance(quiet, bool) or quiet is None
    ), f"Expect a bool | None for 'quiet', but got '{type(quiet)}'."
    assert (
        isinstance(matmul_precision, str) or matmul_precision is None
    ), f"Expect a str | None for 'matmul_precision', but got '{type(matmul_precision)}'."

    if use_gpu is not None:
        afqmc_config.use_gpu = use_gpu
    if single_precision is not None:
        afqmc_config.single_precision = single_precision
    if disable_tf32 is not None:
        afqmc_config.disable_tf32 = disable_tf32
    if quiet is not None:
        afqmc_config.quiet = quiet
    afqmc_config.matmul_precision = (
        matmul_precision
        if matmul_precision is not None
        else os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", afqmc_config.matmul_precision)
    )

    setup_jax(
        use_gpu=afqmc_config.use_gpu,
        single_precision=afqmc_config.single_precision,
        disable_tf32=afqmc_config.disable_tf32,
        quiet=afqmc_config.quiet,
        matmul_precision=afqmc_config.matmul_precision,
    )
    _configured_once = True


def _detect_gpu() -> bool:
    """Detect GPU hardware without importing JAX (NVIDIA or AMD)."""
    if visible_gpu_count() > 0:
        return True

    # AMD ROCm: /dev/kfd is the kernel fusion driver
    if os.path.exists("/dev/kfd"):
        return True

    return False


def setup_jax(
    *,
    use_gpu: bool | None,
    single_precision: bool,
    disable_tf32: bool,
    quiet: bool,
    matmul_precision: str = "default",
) -> None:
    """
    Configure JAX runtime.
    """
    jax_already_imported = "jax" in sys.modules

    # resolve auto-detection before touching JAX
    if use_gpu is None:
        use_gpu = _detect_gpu() and os.environ.get("JAX_PLATFORM_NAME") != "cpu"
    afqmc_config.use_gpu = use_gpu

    if use_gpu:
        _set_gpu_failure_defaults()

    # env vars only take effect if JAX hasn't been imported yet
    if jax_already_imported and use_gpu:
        warnings.warn(
            "JAX was imported before AFQMC configuration; "
            "GPU memory settings and XLA collective-failure settings "
            "may not take effect. "
            "For full control, call config.configure_once(...) before importing jax.",
            stacklevel=3,
        )
    if not jax_already_imported:
        if not single_precision:
            os.environ.setdefault("JAX_ENABLE_X64", "1")
        if disable_tf32:
            os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
        if use_gpu:
            os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
        else:
            os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    from jax import config as jax_config

    # These settings work even after import, before computations are traced.
    jax_config.update("jax_threefry_partitionable", False)
    if not single_precision:
        jax_config.update("jax_enable_x64", True)
    jax_config.update("jax_default_matmul_precision", matmul_precision)

    # platform_name only works before backend init
    if not jax_already_imported:
        if use_gpu:
            jax_config.update("jax_platform_name", "gpu")
        else:
            jax_config.update("jax_platform_name", "cpu")

    # verify GPU actually initialized
    if use_gpu:
        import jax

        platforms = {d.platform for d in jax.devices()}
        if "gpu" not in platforms:
            raise RuntimeError(
                "GPU was detected/requested, but JAX did not initialize a GPU backend. "
                "Ensure jaxlib with CUDA or ROCm support is installed."
            )

    if not quiet and use_gpu:
        _print_host_info()


def _print_host_info() -> None:
    hostname = socket.gethostname()
    uname_info = platform.uname()
    print(f"# Hostname: {hostname}")
    print("# Using GPU (Policy A).")
    print(f"# System: {uname_info.system}")
    print(f"# Node Name: {uname_info.node}")
    print(f"# Release: {uname_info.release}")
    print(f"# Version: {uname_info.version}")
    print(f"# Machine: {uname_info.machine}")
    print(f"# Processor: {uname_info.processor}")
