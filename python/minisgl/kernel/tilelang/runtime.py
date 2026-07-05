from __future__ import annotations


import sys
import os
from pathlib import Path


def _venv_site_packages() -> Path | None:
    prefix = Path(sys.prefix)
    candidate = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    return candidate if candidate.is_dir() else None


def _promote_path(path: Path) -> None:
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def _prepare_import_path() -> None:
    site_packages = _venv_site_packages()
    if site_packages is not None:
        _promote_path(site_packages)

    root = Path(__file__).resolve().parents[4]
    submodule = root / "dependencies" / "tilelang"
    has_native_libs = (submodule / "tilelang" / "lib").is_dir() or (submodule / "build" / "lib").is_dir()
    if submodule.is_dir() and has_native_libs:
        _promote_path(submodule)


def require_tilelang():
    _prepare_import_path()
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise ImportError(
            "int-w8a8-static requires TileLang. "
            "Install the dependencies/tilelang submodule in editable mode or add it to PYTHONPATH."
        ) from exc
    return tilelang, T


_COMPILED_KERNELS = {}


def compile_tilelang(program_key: tuple, factory, out_idx: tuple[int, ...]):
    if program_key in _COMPILED_KERNELS:
        return _COMPILED_KERNELS[program_key]
    tilelang, _ = require_tilelang()
    pass_configs = None
    if os.environ.get("MINISGL_TL_DATA_RACE_CHECK") != "1":
        pass_configs = {tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True}
    compiled = tilelang.compile(
        factory(),
        out_idx=list(out_idx),
        target="cuda",
        pass_configs=pass_configs,
    )
    _COMPILED_KERNELS[program_key] = compiled
    return compiled
