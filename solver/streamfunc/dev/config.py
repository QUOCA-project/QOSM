"""Template configuration for the streamfunction solver.

This file is tracked by git, so don't edit it for your own runs. Copy it once::

    cp solver/streamfunc/dev/config.py solver/streamfunc/dev/config_local.py

and edit ``config_local.py`` instead; it is untracked and is used in place of
this file whenever it exists. Edit this template only to change the defaults
everyone starts from.

Run a simulation and plot it from anywhere with::

    python solver/streamfunc/dev/solver.py
    python solver/streamfunc/dev/plotting.py

Everything a run produces lands in one folder, ``dev/runs/<RUN_NAME>/``.
Only the first two sections normally need editing; the rest is physics.
Any path here can be replaced by an absolute path, e.g.
``ERA5_FILE = Path("/store/ATMOS/shared/era5/data_0.nc")``.
"""

import math
from pathlib import Path


CONFIG_FILE = Path(__file__).resolve()  # this file, or your copy of it
DEV_DIR = CONFIG_FILE.parent  # solver/streamfunc/dev
REPO_ROOT = DEV_DIR.parents[2]  # QOSM

DAY = 24.0 * 3600.0
DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# 1. Where this run writes
# ---------------------------------------------------------------------------
# Change RUN_NAME to start a new run without overwriting the previous one.
RUN_NAME = "default"
RUNS_DIR = DEV_DIR / "runs"

RUN_DIR = RUNS_DIR / RUN_NAME
OUTPUT_FILE = RUN_DIR / "streamfunc_output.nc"
FIGURE_DIR = RUN_DIR / "figures"


# ---------------------------------------------------------------------------
# 2. Where the ERA5 temperature data comes from
# ---------------------------------------------------------------------------
# ERA5_FILE is the raw download (needs variables 't' and 'z'). The solver only
# uses its lat/lon/time mean, so the first run saves that profile to
# ERA5_CACHE_FILE and every later run reads the cache instead. ERA5_FILE
# therefore only has to exist when the cache is being built.
ERA5_FILE = REPO_ROOT / "solver" / "ERA5_T_data" / "data_0.nc"
ERA5_CACHE_DIR = DEV_DIR / "era5_cache"
ERA5_CACHE_FILE = ERA5_CACHE_DIR / f"{ERA5_FILE.stem}_profile.nc"

# Set True (or just delete the cache file) to recompute the profile.
REBUILD_ERA5_CACHE = False


# ---------------------------------------------------------------------------
# 3. Grid
# ---------------------------------------------------------------------------
Z_MIN = 5_000.0
Z_MAX = 50_000.0
DZ = 100.0

LAT_MIN = -20.0
LAT_MAX = 21.0
DLAT = 0.5


# ---------------------------------------------------------------------------
# 4. Time (edit the day values; seconds are derived for the solver)
# ---------------------------------------------------------------------------
QBO_PERIOD_DAYS = 840.0
RUN_LENGTH_DAYS = 2.0 * QBO_PERIOD_DAYS
TIMESTEP_DAYS = 0.3

T_START = 0.0
DT = TIMESTEP_DAYS * DAY
T_END = RUN_LENGTH_DAYS * DAY


# ---------------------------------------------------------------------------
# 5. Physics
# ---------------------------------------------------------------------------
# Constants
EARTH_RADIUS = 6.371e6
SCALE_HEIGHT = 7.0e3
GAS_CONSTANT = 287.0
CP = 1005.0
GRAVITY = 9.81
OMEGA_EARTH = 7.2921e-5

# QBO forcing
DTHETA = 10.0 * DEG  # meridional half-width of the forcing
QBO_F = 0.1 / DAY  # peak forcing, m/s per day
QBO_DZ = 6.0e3  # vertical half-width
QBO_Z0 = 30.0e3  # centre height
QBO_M = -(2.0 * math.pi) / 25.0e3  # vertical wavenumber
QBO_PERIOD = QBO_PERIOD_DAYS * DAY
QBO_OMEGA = (2.0 * math.pi) / QBO_PERIOD
L_SCALE = 1.1e6

# Newtonian cooling
RADIATIVE_DAMPING_DAYS = 40.0
ALPHA = 1.0 / (RADIATIVE_DAMPING_DAYS * DAY)

# Static stability: floor S0 at S0_FLOOR, or at S0_FLOOR_FACTOR * min(S0)
# when S0_FLOOR is None.
S0_FLOOR = None
S0_FLOOR_FACTOR = 0.5


# ---------------------------------------------------------------------------
# 6. Solver and output
# ---------------------------------------------------------------------------
SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
}

PROGRESS_EVERY = 50
COMPRESS_NETCDF = True


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------
TARGET_LATS = (0.0, 10.0, 15.0)
SNAPSHOT_DAYS = (0.0, 420.0, 840.0, 1260.0)
SHOW_FIGURES = False
SAVE_FIGURES = True
SAVE_ANIMATION = False


def describe():
    """Human-readable summary of the paths this run will use."""
    if ERA5_CACHE_FILE.exists() and not REBUILD_ERA5_CACHE:
        era5 = f"ERA5 profile : {ERA5_CACHE_FILE} (cached)"
    else:
        era5 = (
            f"ERA5 profile : {ERA5_CACHE_FILE} (to be built)\n"
            f"  from       : {ERA5_FILE}"
        )
    return "\n".join(
        [
            f"Config file  : {CONFIG_FILE}",
            f"Run name     : {RUN_NAME}",
            f"Output file  : {OUTPUT_FILE}",
            f"Figures      : {FIGURE_DIR}",
            era5,
            f"Run length   : {RUN_LENGTH_DAYS:g} days, dt = {TIMESTEP_DAYS:g} days",
        ]
    )


if __name__ == "__main__":
    print(describe())
