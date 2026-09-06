"""Guards for the static-only typing contract modules in this directory.

Those modules are never collected by pytest — they are not named `test_*.py`,
and their `check_*` functions are not named `test_*`. They are checked only by
the pyright and mypy steps in `.github/workflows/static-analysis.yaml`, so two
things can silently stop being true: a module can become unimportable while
still type-checking, and a module can be renamed or moved out from under those
CI steps without anything failing.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

TYPING_DIR = Path(__file__).parent


def _contract_paths() -> list[Path]:
    return sorted(
        path
        for path in TYPING_DIR.glob("*.py")
        if not path.name.startswith("test_") and path.name != "__init__.py"
    )


def _import_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestContractPaths:
    def test_finds_every_contract_module(self):
        assert [path.name for path in _contract_paths()] == ["call_annotations.py"]


class TestImportModule:
    @pytest.mark.parametrize(
        "path",
        _contract_paths(),
        ids=[path.stem for path in _contract_paths()],
    )
    def test_contract_module_is_importable(self, path: Path):
        module = _import_module(path)

        assert module.__name__ == path.stem
