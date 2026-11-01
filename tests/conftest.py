"""Pytest root configuration and environment fixtures."""

from pathlib import Path

import appdirs

# Redirect PyRIT default data path to /tmp/pyrit_data so tests run cleanly
# without touching ~/Library/Application Support/
_pyrit_data_dir = Path("/tmp/pyrit_data")
_pyrit_data_dir.mkdir(parents=True, exist_ok=True)


def _custom_user_data_dir(appname=None, appauthor=None, version=None, roaming=False):
    if appname:
        return str(_pyrit_data_dir / appname)
    return str(_pyrit_data_dir)


appdirs.user_data_dir = _custom_user_data_dir
