"""Runtime environment guard for V4d scripts.

CIANNA4090 keeps the compatible NumPy inside the conda environment. Some lab
shells still add ~/.local site-packages ahead of the env, which can import a
NumPy 2.x wheel incompatible with the current CIANNA extension.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def drop_user_site() -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    try:
        user_paths = site.getusersitepackages()
    except Exception:
        user_paths = []
    if isinstance(user_paths, str):
        user_paths = [user_paths]
    if os.environ.get("PYTHONUSERBASE"):
        user_paths.append(str(Path(os.environ["PYTHONUSERBASE"]).expanduser()))

    resolved_user_paths = []
    for path in user_paths:
        if not path:
            continue
        try:
            resolved_user_paths.append(Path(path).expanduser().resolve())
        except OSError:
            continue

    if not resolved_user_paths:
        return

    filtered = []
    for path_entry in sys.path:
        if not path_entry:
            filtered.append(path_entry)
            continue
        try:
            resolved = Path(path_entry).expanduser().resolve()
        except OSError:
            filtered.append(path_entry)
            continue
        if any(resolved == user_path or user_path in resolved.parents for user_path in resolved_user_paths):
            continue
        filtered.append(path_entry)
    sys.path[:] = filtered
