"""Simple editable configuration for the development streamfunction solver.

This is deliberately plain Python rather than YAML. To change a run, edit the
values below and run ``python dev/solver.py`` from the repository root.
"""

from pathlib import Path


DEV_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEV_DIR.parent


# Run locations
RUN_NAME = "default"
OUTPUT_DIR = DEV_DIR / "runs" / RUN_NAME
OUTPUT_FILE = OUTPUT_DIR / "streamfunc_output.nc"
FIGURE_DIR = DEV_DIR / "figures" / RUN_NAME


# ERA5 profile cache. If ERA5_PROFILE_PATH exists, the solver loads it directly.
# Set REBUILD_ERA5_PROFILE=True to recalculate it from ERA5_SOURCE_PATH.
ERA5_SOURCE_PATH = PROJECT_ROOT / "solver" / "ERA5_T_data" / "data_0.nc"
ERA5_PROFILE_PATH = DEV_DIR / "era5_T_profile.nc"
REBUILD_ERA5_PROFILE = False


# Physical constants
EARTH_RADIUS = 6.371e6
SCALE_HEIGHT = 7.0e3
GAS_CONSTANT = 287.0
CP = 1005.0
GRAVITY = 9.81
OMEGA_EARTH = 7.2921e-5


# Grid
Z_MIN = 5_000.0
Z_MAX = 50_000.0
DZ = 100.0

LAT_MIN = -20.0
LAT_MAX = 21.0
DLAT = 0.5


# Time
T_START = 0.0
DT = 0.3 * 24.0 * 3600.0
T_END = 840.0 * 2.0 * 24.0 * 3600.0


# QBO forcing parameters
DTHETA = 10.0 * 3.141592653589793 / 180.0
QBO_F = 0.1 / 24.0 / 3600.0
QBO_DZ = 6.0e3
QBO_Z0 = 30.0e3
QBO_M = -(2.0 * 3.141592653589793) / 25.0e3
QBO_PERIOD = 840.0 * 24.0 * 3600.0
QBO_OMEGA = (2.0 * 3.141592653589793) / QBO_PERIOD
L_SCALE = 1.1e6
ALPHA = 1.0 / (40.0 * 24.0 * 3600.0)


# Static-stability handling
S0_FLOOR = None
S0_FLOOR_FACTOR = 0.5


# Firedrake/PETSc solve options
SOLVER_PARAMETERS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
}


# Output and diagnostics
PROGRESS_EVERY = 50
COMPRESS_NETCDF = True


# Plotting
TARGET_LATS = (0.0, 10.0, 15.0)
SNAPSHOT_DAYS = (0.0, 420.0, 840.0, 1260.0)
SHOW_FIGURES = False
SAVE_FIGURES = True
SAVE_ANIMATION = False
