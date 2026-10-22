"""Check runtime configuration without running AFQMC or requiring GPUs."""
import builtins
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "precision,environment,import_jax_first,expected",
    [
        (None, None, False, "default"),
        (None, None, True, "default"),
        ("default", None, False, "default"),
        ("high", None, False, "high"),
        (None, "default", False, "default"),
        (None, "highest", False, "highest"),
        ("default", "highest", False, "default"),
        ("highest", "default", True, "highest"),
    ],
)
def test_matmul_precision_configuration(precision, environment, import_jax_first, expected):
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu")
    env.pop("JAX_DEFAULT_MATMUL_PRECISION", None)
    if environment is not None:
        env["JAX_DEFAULT_MATMUL_PRECISION"] = environment
    code = f"""
from trot import config
if {import_jax_first!r}:
    import jax
config.configure_once(use_gpu=False, matmul_precision={precision!r})
import jax
import jax.numpy as jnp
assert jax.config.jax_default_matmul_precision == {expected!r}
assert config.afqmc_config.matmul_precision == {expected!r}

# Inspect a production-style contraction's tracing metadata, without
# compiling or executing numerical work on the CPU.
operand = jax.ShapeDtypeStruct((3, 3), jnp.float32)
traced = jax.make_jaxpr(lambda a, b: jnp.einsum('ik,kj->ij', a, b))(operand, operand)
dot, = [eq for eq in traced.jaxpr.eqns if eq.primitive.name == 'dot_general']
assert all(p.name.lower() == {expected!r} for p in dot.params['precision'])
assert traced.out_avals[0].dtype == jnp.dtype('float32')
assert jax.config.jax_enable_x64

# Import-time setup in other modules must preserve the first configuration.
config.configure_once(matmul_precision={'default' if expected == 'highest' else 'highest'!r})
assert jax.config.jax_default_matmul_precision == {expected!r}
assert config.afqmc_config.matmul_precision == {expected!r}
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "flags,timeout,terminate",
    [
        (None, "1800", "true"),
        ("--xla_gpu_enable_command_buffer=", "1800", "true"),
        ('--xla_dump_to="/tmp/path with spaces"', "1800", "true"),
        ("--xla_gpu_nccl_termination_timeout_seconds=600", "600", "true"),
        ("--xla_gpu_nccl_termination_timeout_seconds 600", "600", "true"),
        ("--xla_gpu_nccl_terminate_on_error=false", "1800", "false"),
        (
            "--xla_gpu_nccl_termination_timeout_seconds=-1 "
            "--xla_gpu_nccl_terminate_on_error=false",
            "-1",
            "false",
        ),
    ],
)
def test_gpu_failure_defaults_preserve_launcher_flags(monkeypatch, flags, timeout, terminate):
    from trot import config

    monkeypatch.delenv("XLA_FLAGS", raising=False)
    if flags is not None:
        monkeypatch.setenv("XLA_FLAGS", flags)
    config._set_gpu_failure_defaults()
    configured = os.environ["XLA_FLAGS"]
    if flags:
        assert configured.startswith(flags)
    tokens = shlex.split(configured)
    for name, expected in (
        ("--xla_gpu_nccl_termination_timeout_seconds", timeout),
        ("--xla_gpu_nccl_terminate_on_error", terminate),
    ):
        matching = [i for i, token in enumerate(tokens) if token.split("=", 1)[0] == name]
        assert len(matching) == 1
        i = matching[0]
        actual = tokens[i].split("=", 1)[1] if "=" in tokens[i] else tokens[i + 1]
        assert actual == expected

    config._set_gpu_failure_defaults()
    assert os.environ["XLA_FLAGS"] == configured


@pytest.mark.parametrize("use_gpu,detected_gpu", [(True, False), (None, True), (False, True)])
def test_gpu_failure_defaults_installed_before_jax_import(monkeypatch, use_gpu, detected_gpu):
    from trot import config

    for name in ("XLA_FLAGS", "JAX_PLATFORM_NAME", "JAX_ENABLE_X64", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    monkeypatch.setattr(config, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(config, "_detect_gpu", lambda: detected_gpu)
    monkeypatch.setattr(config.afqmc_config, "use_gpu", None)
    enabled = detected_gpu if use_gpu is None else use_gpu
    observations = []

    def check_flags():
        flags = os.environ["XLA_FLAGS"]
        observations.append(flags)
        if enabled:
            assert "--xla_gpu_nccl_termination_timeout_seconds=1800" in flags
            assert "--xla_gpu_nccl_terminate_on_error=true" in flags
        else:
            assert flags == "--xla_gpu_enable_command_buffer="

    def devices():
        check_flags()
        return [SimpleNamespace(platform="gpu")]

    fake_jax = SimpleNamespace(config=SimpleNamespace(update=lambda *args: None), devices=devices)
    original_import = builtins.__import__

    def import_with_observation(name, *args, **kwargs):
        if name == "jax":
            check_flags()
            return fake_jax
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as imports:
        imports.setattr(builtins, "__import__", import_with_observation)
        config.setup_jax(use_gpu=use_gpu, single_precision=False, disable_tf32=False, quiet=True)
    assert observations


def test_gpu_failure_flags_are_accepted_by_xla():
    # A real backend startup catches unsupported/misspelled native flags. The
    # GPU selection path is checked separately without requiring GPU hardware.
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu")
    env["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
    code = """
from trot import config
config._set_gpu_failure_defaults()
config.configure_once(use_gpu=False)
import jax
import jax.numpy as jnp
assert all(device.platform == 'cpu' for device in jax.devices())
assert int(jax.jit(lambda x: x + 1)(jnp.asarray(1)).block_until_ready()) == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
