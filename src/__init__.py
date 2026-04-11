"""Compatibility shim for projects that expect a top-level `src` package.

This package forwards submodule resolution to `models/g3splat`, so legacy imports
like `from src.visualization...` continue to work after vendoring g3splat under
`models/g3splat`.
"""

from pathlib import Path
import pkgutil

# Keep normal package behavior and allow extending search paths.
__path__ = pkgutil.extend_path(__path__, __name__)

# Add vendored g3splat package root as a fallback import location.
_repo_root = Path(__file__).resolve().parents[1]
_g3splat_root = _repo_root / "models" / "g3splat"
if _g3splat_root.is_dir():
    _path_str = str(_g3splat_root)
    if _path_str not in __path__:
        __path__.append(_path_str)
