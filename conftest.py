import os
from pathlib import Path

import pytest

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def _patch_pyscf_einsum_numpy_path_compat() -> None:
    """
    PySCF's custom einsum expects legacy numpy.einsum_path tuples of length >= 4.
    Newer NumPy can return 3-tuples, which breaks on Python 3.12 environments.
    Patch test runtime to accept both tuple formats.
    """
    try:
        import numpy as np
        from pyscf import lib as pyscf_lib
        from pyscf.lib import numpy_helper as nh
    except Exception:
        return

    if getattr(nh, "_trot_einsum_compat_patched", False):
        return

    try:
        a = np.zeros((1, 1))
        b = np.zeros((1, 1))
        c = np.zeros((1, 1, 1, 1))
        contraction_list = nh._einsum_path("ia,jb,iajb", a, b, c, optimize=True, einsum_call=True)[
            1
        ]
        needs_patch = bool(contraction_list) and len(contraction_list[0]) == 3
    except Exception:
        needs_patch = False

    if not needs_patch:
        return

    _numpy_einsum = getattr(nh, "_numpy_einsum", np.einsum)
    _einsum_path = getattr(nh, "_einsum_path", None)
    if _einsum_path is None:
        return

    def _default_contract(subscripts, *operands, **kwargs):
        return _numpy_einsum(subscripts, *operands, **kwargs)

    tmp = getattr(nh, "_contract", None)
    _contract = tmp if tmp is not None else _default_contract

    def _einsum_compat(subscripts, *tensors, **kwargs):
        contract = kwargs.pop("_contract", _contract)
        subscripts = subscripts.replace(" ", "")
        if len(tensors) <= 1 or "..." in subscripts:
            return _numpy_einsum(subscripts, *tensors, **kwargs)
        if len(tensors) <= 2:
            return contract(subscripts, *tensors, **kwargs)

        optimize = kwargs.pop("optimize", True)
        tensors = list(tensors)
        contraction_list = _einsum_path(subscripts, *tensors, optimize=optimize, einsum_call=True)[
            1
        ]
        out = None
        for contraction in contraction_list:
            if len(contraction) >= 4:
                inds = contraction[0]
                einsum_str = contraction[2]
            elif len(contraction) == 3:
                inds = contraction[0]
                einsum_str = contraction[1]
            else:
                raise ValueError(f"Unexpected einsum_path contraction entry: {contraction!r}")

            tmp_operands = [tensors.pop(x) for x in inds]
            if len(tmp_operands) > 2:
                out = _numpy_einsum(einsum_str, *tmp_operands)
            else:
                out = contract(einsum_str, *tmp_operands)
            tensors.append(out)
        return out

    nh.einsum = _einsum_compat
    pyscf_lib.einsum = _einsum_compat
    setattr(nh, "_trot_einsum_compat_patched", True)


_patch_pyscf_einsum_numpy_path_compat()


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked as slow integration coverage",
    )


def _is_slow_test(item: pytest.Item) -> bool:
    path = Path(str(item.fspath)).name
    test_name = getattr(item, "originalname", item.name.split("[", 1)[0])

    slow_files = {
        "test_rhf_fp.py",
        "test_uhf_fp.py",
        "test_cisd_fp.py",
        "test_ucisd_fp.py",
        "test_pt2ccsd.py",
        "test_upt2ccsd.py",
        "test_io.py",
        "test_cis.py",
        "test_eom_cisd.py",
        "test_eom_t_cisd.py",
        "test_ucisdt_molecular.py",
        "test_ucisd_molecular.py",
    }
    if path in slow_files:
        return True

    if path == "test_lno.py":
        return test_name == "test_calc_rhf"

    if path == "test_ghf.py":
        return test_name == "test_calc_ghf_hamiltonian"

    if path == "test_gcisd.py":
        return test_name == "test_calc_ghf_hamiltonian"

    if path == "test_ucisd.py":
        return test_name == "test_calc_rhf_hamiltonian"

    if path == "test_rhf.py" and test_name == "test_calc_rhf_hamiltonian":
        callspec = getattr(item, "callspec", None)
        return callspec is not None and callspec.params.get("walker_kind") != "restricted"

    if path == "test_uhf.py" and test_name == "test_calc_rhf_hamiltonian":
        callspec = getattr(item, "callspec", None)
        return callspec is not None and callspec.params.get("walker_kind") != "restricted"

    return False


def _is_integration_test(item: pytest.Item) -> bool:
    path = Path(str(item.fspath)).name
    return path != "test_cis.py" and _is_slow_test(item)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    run_slow = config.getoption("--run-slow")

    for item in items:
        if not _is_slow_test(item):
            continue

        item.add_marker(pytest.mark.slow)
        if _is_integration_test(item):
            item.add_marker(pytest.mark.integration)

        if not run_slow:
            item.add_marker(skip_slow)
