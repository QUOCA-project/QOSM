"""Setup, ERA5 caching, and solving for the development streamfunction model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, interp1d
from scipy.spatial import Delaunay

try:
    from . import config as default_config
except ImportError:  # Allows ``python dev/solver.py``
    import config as default_config


SECONDS_PER_DAY = 24.0 * 3600.0


def build_grid(cfg):
    z = np.arange(cfg.Z_MIN, cfg.Z_MAX, cfg.DZ, dtype=float)
    lat = np.arange(cfg.LAT_MIN, cfg.LAT_MAX, cfg.DLAT, dtype=float)
    return z, lat


def build_times(cfg):
    return np.arange(cfg.T_START, cfg.T_END, cfg.DT, dtype=float)


def forcing(lat, z, t, cfg):
    lat_rad = np.deg2rad(lat)[np.newaxis, :]
    z_col = z[:, np.newaxis]
    meridional = np.exp(-(lat_rad**2) / (2.0 * cfg.DTHETA**2))
    vertical = np.exp(-((z_col - cfg.QBO_Z0) ** 2) / (2.0 * cfg.QBO_DZ**2))
    phase = np.cos(cfg.QBO_M * z_col - cfg.QBO_OMEGA * t)
    return cfg.QBO_F * meridional * vertical * phase


def heating(lat, z, t, temperature_anomaly, cfg):
    _ = lat, z, t
    return -cfg.ALPHA * temperature_anomaly


def calculate_era5_profile(source_path, gravity):
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"ERA5 source file not found: {source_path}. "
            "Set ERA5_SOURCE_PATH in dev/config.py."
        )

    with xr.open_dataset(source_path) as dataset:
        if "t" not in dataset or "z" not in dataset:
            raise KeyError("ERA5 source file must contain variables 't' and 'z'.")

        mean_dims = [
            dim
            for dim in ("latitude", "longitude", "valid_time")
            if dim in dataset["t"].dims
        ]
        temperature = dataset["t"].mean(mean_dims).load()
        height = (dataset["z"].mean(mean_dims) / gravity).load()

    profile = xr.Dataset(
        data_vars={
            "T": temperature,
            "Z": height,
        },
        attrs={
            "description": (
                "ERA5 temperature and geopotential-height profiles averaged "
                "over latitude, longitude, and valid_time where present."
            ),
            "source_file": str(source_path),
        },
    )
    profile["T"].attrs["units"] = "K"
    profile["Z"].attrs["units"] = "m"
    profile["Z"].attrs["description"] = "geopotential height computed as ERA5 z / gravity"
    return profile


def load_era5_profile(profile_path):
    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise FileNotFoundError(
            f"ERA5 profile cache not found: {profile_path}. "
            "Set REBUILD_ERA5_PROFILE=True or provide the cached file."
        )
    with xr.open_dataset(profile_path) as dataset:
        return dataset.load()


def get_era5_profile(cfg):
    profile_path = Path(cfg.ERA5_PROFILE_PATH)
    if profile_path.exists() and not cfg.REBUILD_ERA5_PROFILE:
        print(f"Loading cached ERA5 profile from {profile_path}")
        return load_era5_profile(profile_path)

    print(f"Calculating ERA5 profile from {cfg.ERA5_SOURCE_PATH}")
    profile = calculate_era5_profile(cfg.ERA5_SOURCE_PATH, cfg.GRAVITY)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_netcdf(profile_path)
    print(f"Saved ERA5 profile to {profile_path}")
    print(f"Reloading ERA5 profile from {profile_path}")
    return load_era5_profile(profile_path)


def make_s0_profile(profile, z, cfg):
    temperature = np.asarray(profile["T"].values).squeeze()
    height = np.asarray(profile["Z"].values).squeeze()

    if temperature.ndim != 1 or height.ndim != 1:
        raise ValueError("ERA5 profile variables T and Z must be one-dimensional after averaging.")
    if temperature.size != height.size:
        raise ValueError("ERA5 profile variables T and Z must have the same length.")

    order = np.argsort(height)
    height = height[order]
    temperature = temperature[order]

    t_interp = interp1d(height, temperature, kind="linear", fill_value="extrapolate")
    t_on_z = t_interp(z)
    dtdz = np.gradient(t_on_z, z)
    s0_profile = dtdz + cfg.GRAVITY / cfg.CP
    return s0_profile, t_on_z


def numpy_to_function(data_2d, func, mesh, z, lat, y_nd_min, z_nd_min, y_ref, z_ref, earth_radius):
    interpolator = RegularGridInterpolator(
        (z, lat),
        data_2d,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    coords = mesh.coordinates.dat.data
    y_phys = (coords[:, 0] + y_nd_min) * y_ref / (np.pi / 180.0 * earth_radius)
    z_phys = (coords[:, 1] + z_nd_min) * z_ref
    node_coords = np.column_stack([z_phys, y_phys])
    func.dat.data[:] = interpolator(node_coords).squeeze()


def config_attrs(cfg):
    names = [
        "RUN_NAME",
        "ERA5_SOURCE_PATH",
        "ERA5_PROFILE_PATH",
        "Z_MIN",
        "Z_MAX",
        "DZ",
        "LAT_MIN",
        "LAT_MAX",
        "DLAT",
        "T_START",
        "DT",
        "T_END",
        "QBO_F",
        "QBO_DZ",
        "QBO_Z0",
        "QBO_M",
        "QBO_PERIOD",
        "QBO_OMEGA",
        "ALPHA",
        "S0_FLOOR",
        "S0_FLOOR_FACTOR",
    ]
    attrs = {}
    for name in names:
        value = getattr(cfg, name)
        if value is None:
            attrs[name] = "None"
        elif isinstance(value, Path):
            attrs[name] = str(value)
        else:
            attrs[name] = value
    return attrs


def build_output_dataset(psi_store, u_store, t_store, z, lat, times, s0_profile, t0_profile, cfg):
    return xr.Dataset(
        data_vars={
            "psi": (
                ("time", "z", "lat"),
                psi_store,
                {"units": "m2 s-1", "description": "streamfunction"},
            ),
            "U": (
                ("time", "z", "lat"),
                u_store,
                {"units": "m s-1", "description": "zonal wind anomaly"},
            ),
            "T_anomaly": (
                ("time", "z", "lat"),
                t_store,
                {"units": "K", "description": "temperature anomaly"},
            ),
            "S0": (
                ("z",),
                s0_profile,
                {"units": "K m-1", "description": "static stability profile used by the solver"},
            ),
            "T0_on_model_z": (
                ("z",),
                t0_profile,
                {"units": "K", "description": "ERA5 mean temperature interpolated to model z"},
            ),
        },
        coords={
            "time": ("time", times, {"units": "s"}),
            "time_days": ("time", times / SECONDS_PER_DAY, {"units": "days"}),
            "z": ("z", z, {"units": "m"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
        },
        attrs={"description": "Development streamfunction solver output", **config_attrs(cfg)},
    )


def write_output(dataset, output_file, compress=True):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        encoding = {
            name: {"zlib": True, "complevel": 4}
            for name in dataset.data_vars
            if dataset[name].ndim > 0
        }
        try:
            dataset.to_netcdf(output_file, encoding=encoding)
            return output_file
        except ValueError:
            pass

    dataset.to_netcdf(output_file)
    return output_file


def run_simulation(cfg=default_config):
    import firedrake as fd

    era5_profile = get_era5_profile(cfg)
    z, lat = build_grid(cfg)
    times = build_times(cfg)
    ny, nz = len(lat), len(z)

    s0_profile, t0_on_z = make_s0_profile(era5_profile, z, cfg)

    beta0 = 2.0 * cfg.OMEGA_EARTH / cfg.EARTH_RADIUS
    n0 = np.sqrt(np.mean(cfg.GAS_CONSTANT / cfg.SCALE_HEIGHT * s0_profile))
    y0 = 10.0 * np.pi / 180.0 * cfg.EARTH_RADIUS
    z_ref = cfg.SCALE_HEIGHT
    y_ref = (n0 * z_ref) / (beta0 * y0)

    lat_m_min = cfg.LAT_MIN * np.pi / 180.0 * cfg.EARTH_RADIUS
    lat_m_max = cfg.LAT_MAX * np.pi / 180.0 * cfg.EARTH_RADIUS
    y_nd_min = lat_m_min / y_ref
    y_nd_max = lat_m_max / y_ref
    z_nd_min = cfg.Z_MIN / z_ref
    z_nd_max = cfg.Z_MAX / z_ref

    print(f"Y_ref = {y_ref / 1e3:.0f} km, Z_ref = {z_ref / 1e3:.0f} km")
    print(f"y_nd range: {y_nd_min:.4f} to {y_nd_max:.4f}")
    print(f"z_nd range: {z_nd_min:.4f} to {z_nd_max:.4f}")
    print(f"S0 range: {s0_profile.min():.3e} to {s0_profile.max():.3e} K/m")

    mesh = fd.RectangleMesh(ny - 1, nz - 1, y_nd_max - y_nd_min, z_nd_max - z_nd_min)
    x_nd, z_nd_raw = fd.SpatialCoordinate(mesh)
    y_nd_coord = x_nd + y_nd_min
    _ = z_nd_raw + z_nd_min
    lat_coord_m = y_nd_coord * y_ref

    v_space = fd.FunctionSpace(mesh, "CG", 1)

    r_const = fd.Constant(cfg.GAS_CONSTANT)
    h_const = fd.Constant(cfg.SCALE_HEIGHT)
    omega_const = fd.Constant(cfg.OMEGA_EARTH)
    earth_radius_const = fd.Constant(cfg.EARTH_RADIUS)
    y_ref_const = fd.Constant(y_ref)
    z_ref_const = fd.Constant(z_ref)

    s0_2d = np.tile(s0_profile[:, np.newaxis], (1, ny))
    s0_func = fd.Function(v_space, name="S0")
    numpy_to_function(
        s0_2d,
        s0_func,
        mesh,
        z,
        lat,
        y_nd_min,
        z_nd_min,
        y_ref,
        z_ref,
        cfg.EARTH_RADIUS,
    )

    s0_floor = cfg.S0_FLOOR
    if s0_floor is None:
        s0_floor = float(np.min(s0_profile) * cfg.S0_FLOOR_FACTOR)
    s0_func.dat.data[:] = np.maximum(s0_func.dat.data, s0_floor)
    print(f"S0 floor: {s0_floor:.3e}")
    print(f"S0 on mesh min/max: {s0_func.dat.data.min():.3e} / {s0_func.dat.data.max():.3e}")

    s0_np = np.tile(s0_profile[:, np.newaxis], (1, ny))
    beta = 2.0 * omega_const / earth_radius_const
    n2 = r_const / h_const * s0_func

    f_func = fd.Function(v_space, name="F")
    q_func = fd.Function(v_space, name="Q")

    psi = fd.Function(v_space, name="psi")
    psi_trial = fd.TrialFunction(v_space)
    test_func = fd.TestFunction(v_space)

    n2_ref = float(cfg.GAS_CONSTANT / cfg.SCALE_HEIGHT * np.mean(s0_profile))
    lhs_scale = max((beta0 * y0) ** 2 / z_ref**2, n2_ref / y_ref**2)
    eq_scale = fd.Constant(1.0 / lhs_scale)
    print(f"N2_ref = {n2_ref:.3e}, lhs_scale = {lhs_scale:.3e}, eq_scale = {1.0 / lhs_scale:.3e}")

    lhs = eq_scale * (
        beta**2
        * lat_coord_m**2
        * (1.0 / z_ref_const**2)
        * fd.Dx(test_func, 1)
        * fd.Dx(psi_trial, 1)
        + n2
        * (1.0 / y_ref_const**2)
        * fd.Dx(test_func, 0)
        * fd.Dx(psi_trial, 0)
    ) * fd.dx

    rhs = eq_scale * (
        -test_func
        * (
            beta * lat_coord_m * (1.0 / z_ref_const) * fd.Dx(f_func, 1)
            + (r_const / h_const) * (1.0 / y_ref_const) * fd.Dx(q_func, 0)
        )
    ) * fd.dx

    boundary_conditions = [
        fd.DirichletBC(v_space, fd.Constant(0.0), 1),
        fd.DirichletBC(v_space, fd.Constant(0.0), 2),
    ]

    u_np = np.zeros((nz, ny))
    t_np = np.zeros((nz, ny))
    psi_store = np.zeros((len(times), nz, ny))
    u_store = np.zeros((len(times), nz, ny))
    t_store = np.zeros((len(times), nz, ny))

    beta_np = 2.0 * cfg.OMEGA_EARTH / cfg.EARTH_RADIUS
    zz, ll = np.meshgrid(z, lat, indexing="ij")
    coords = mesh.coordinates.dat.data
    y_phys_dof = (coords[:, 0] + y_nd_min) * y_ref / (np.pi / 180.0 * cfg.EARTH_RADIUS)
    z_phys_dof = (coords[:, 1] + z_nd_min) * z_ref
    psi_points = np.column_stack([z_phys_dof, y_phys_dof])
    psi_triangulation = Delaunay(psi_points)

    lat_m = lat * np.pi / 180.0 * cfg.EARTH_RADIUS
    lattile_np = np.tile(lat_m[np.newaxis, :], (nz, 1))

    psi_store[0] = 0.0
    u_store[0] = u_np
    t_store[0] = t_np

    for k in range(1, len(times)):
        t_mid = float(times[k - 1] + 0.5 * cfg.DT)
        f_np = forcing(lat, z, t_mid, cfg)
        q_np = heating(lat, z, t_mid, t_np, cfg)

        numpy_to_function(f_np, f_func, mesh, z, lat, y_nd_min, z_nd_min, y_ref, z_ref, cfg.EARTH_RADIUS)
        numpy_to_function(q_np, q_func, mesh, z, lat, y_nd_min, z_nd_min, y_ref, z_ref, cfg.EARTH_RADIUS)

        fd.solve(lhs == rhs, psi, bcs=boundary_conditions, solver_parameters=cfg.SOLVER_PARAMETERS)

        psi_interp = LinearNDInterpolator(psi_triangulation, psi.dat.data[:], fill_value=0.0)
        psi_mid_np = psi_interp(zz, ll)

        dpsi_dz = np.gradient(psi_mid_np, z, axis=0)
        dpsi_dy = np.gradient(psi_mid_np, lat_m, axis=1)

        u_np = u_np + cfg.DT * (f_np - beta_np * lattile_np * dpsi_dz)
        t_np = t_np + cfg.DT * (q_np - s0_np * dpsi_dy)

        psi_store[k] = psi_mid_np
        u_store[k] = u_np
        t_store[k] = t_np

        if k == 1:
            print(f"  [k=1] psi_mid_np NaN: {np.isnan(psi_mid_np).any()}")
            print(f"  [k=1] U NaN: {np.isnan(u_np).any()}")
            print(f"  [k=1] T NaN: {np.isnan(t_np).any()}")

        if cfg.PROGRESS_EVERY and k % cfg.PROGRESS_EVERY == 0:
            iy10 = int(np.argmin(abs(lat - 10.0)))
            circulation = abs(beta_np * (10.0 * np.pi / 180.0 * cfg.EARTH_RADIUS) * dpsi_dz[:, iy10]).max()
            forcing_mag = abs(f_np[:, iy10]).max()
            ratio = circulation / forcing_mag if forcing_mag else np.nan
            print(
                f"Step {k}/{len(times)}, t = {times[k] / SECONDS_PER_DAY:.0f} days, "
                f"|psi| = {abs(psi_mid_np).max():.3e}, "
                f"|U| = {abs(u_np).max():.3e}, "
                f"|T| = {abs(t_np).max():.3e}, "
                f"|Q| = {abs(q_np).max():.3e}"
            )
            print(f"  |beta*y*dpsi_dz @ 10 deg| = {circulation:.3e}")
            print(f"  |F_np @ 10 deg|           = {forcing_mag:.3e}")
            print(f"  ratio                     = {ratio:.3e}")

    output_dataset = build_output_dataset(psi_store, u_store, t_store, z, lat, times, s0_profile, t0_on_z, cfg)
    output_path = write_output(output_dataset, cfg.OUTPUT_FILE, compress=cfg.COMPRESS_NETCDF)
    print(f"Saved solver output to {output_path}")
    return output_path


def main():
    run_simulation(default_config)


if __name__ == "__main__":
    main()

