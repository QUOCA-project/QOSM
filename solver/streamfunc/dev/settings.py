"""Chooses which configuration the solver and plotting scripts use.

``config.py`` is the tracked template. Make your own copy once::

    cp solver/streamfunc/dev/config.py solver/streamfunc/dev/config_local.py

and edit ``config_local.py`` from then on. It is untracked, so everyone working
on this code keeps their own paths, run names, and experiments without editing a
shared file. When it exists it is used instead of the config.py template.

Settings that are added to the template later but are missing from your copy are
filled in from the template, and a note is printed saying which ones.
"""

import importlib.util
from pathlib import Path


DEV_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = DEV_DIR / "config.py"
LOCAL_PATH = DEV_DIR / "config_local.py"

_loaded = {}


def _import_file(path):
    if path not in _loaded:
        spec = importlib.util.spec_from_file_location(f"qosm_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _loaded[path] = module
    return _loaded[path]


def load(path=None):
    """Return the active config module, preferring config_local.py."""
    if path is not None:
        return _import_file(Path(path).resolve())
    if not LOCAL_PATH.exists():
        return _import_file(TEMPLATE_PATH)

    config = _import_file(LOCAL_PATH)
    template = _import_file(TEMPLATE_PATH)
    missing = sorted(
        name
        for name in dir(template)
        if name.isupper() and not hasattr(config, name)
    )
    for name in missing:
        setattr(config, name, getattr(template, name))
    if missing:
        print(
            f"{LOCAL_PATH.name} has no value for {', '.join(missing)}; "
            "using the config.py default(s)."
        )
    return config
