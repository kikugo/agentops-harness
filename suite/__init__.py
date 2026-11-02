"""Attack suite execution and baseline evaluation package."""

import os
from pathlib import Path

import appdirs

# Ensure PyRIT's default path helper points to a clean writable directory
_pyrit_dir = Path(os.getenv("PYRIT_DATA_PATH", "/tmp/pyrit_data"))
_pyrit_dir.mkdir(parents=True, exist_ok=True)


def _safe_user_data_dir(appname=None, appauthor=None, version=None, roaming=False):
    if appname:
        return str(_pyrit_dir / appname)
    return str(_pyrit_dir)


appdirs.user_data_dir = _safe_user_data_dir
