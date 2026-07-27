"""Plotting utilities for development streamfunction solver output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    from . import config as default_config
    from .solver import forcing
except ImportError:  # Allows ``python dev/plotting.py``
    import config as default_config
    from solver import forcing


SECONDS_PER_DAY = 24.0 * 3600.0


def load_output(output_file):
    """Load solver output into memory and close the file handle."""
    with xr.open_dataset(output_file) as dataset:
        return dataset.load()


def symmetric_levels(data, n=41):
    """Return symmetric contour levels around zero."""
    vmax = float(np.nanmax(np.abs(data)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    return np.linspace(-vmax, vmax, n)


def nearest_index(values, target):
    """Return index of value nearest to target."""
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def save_or_show(fig, path, cfg):
    """Save and/or display a matplotlib figure according to config."""
    if cfg.SAVE_FIGURES:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {path}")
    if cfg.SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def build_midpoint_fields(dataset, cfg):
    """Rebuild midpoint F, v, and w fields from solver output."""
    z = dataset["z"].values
    lat = dataset["lat"].values
    time = dataset["time"].values

    midpoint_time = time[:-1] + 0.5 * cfg.DT
    psi_mid = dataset["psi"].values[1:]
    lat_m = lat * np.pi / 180.0 * cfg.EARTH_RADIUS

    f_store = np.empty_like(psi_mid)
    for k, t_mid in enumerate(midpoint_time):
        f_store[k] = forcing(lat, z, float(t_mid), cfg)

    v_store = -np.gradient(psi_mid, z, axis=1)
    w_store = np.gradient(psi_mid, lat_m, axis=2)
    return midpoint_time, f_store, v_store, w_store


def plot_snapshots(dataset, midpoint_time, f_store, v_store, w_store, cfg):
    """Save latitude-height snapshots of F, v, w, U, and T."""
    z = dataset["z"].values
    lat = dataset["lat"].values
    time_days = dataset["time"].values / SECONDS_PER_DAY
    midpoint_days = midpoint_time / SECONDS_PER_DAY
    lat_grid, z_grid = np.meshgrid(lat, z / 1000.0)

    u_store = dataset["U"].values
    t_store = dataset["T_anomaly"].values

    levels = {
        "F": symmetric_levels(f_store),
        "v": symmetric_levels(v_store),
        "w": symmetric_levels(w_store),
        "U": symmetric_levels(u_store),
        "T": symmetric_levels(t_store),
    }

    snapshot_indices = sorted({nearest_index(time_days, day) for day in cfg.SNAPSHOT_DAYS})
    snapshot_indices = [index for index in snapshot_indices if index < len(f_store)]

    for index in snapshot_indices:
        fields = [
            (f_store[index], levels["F"], "F (m/s^2)", f"F at t+dt/2 = {midpoint_days[index]:.0f} days"),
            (v_store[index], levels["v"], "v (m/s)", f"v at t+dt/2 = {midpoint_days[index]:.0f} days"),
            (w_store[index], levels["w"], "w (m/s)", f"w at t+dt/2 = {midpoint_days[index]:.0f} days"),
            (u_store[index], levels["U"], "u (m/s)", f"u at t = {time_days[index]:.0f} days"),
            (t_store[index], levels["T"], "T anomaly (K)", f"T at t = {time_days[index]:.0f} days"),
        ]

        fig, axes = plt.subplots(1, 5, figsize=(23, 5), sharey=True)
        for ax, (data, field_levels, cbar_label, title) in zip(axes, fields):
            contour = ax.contourf(lat_grid, z_grid, data, levels=field_levels, cmap="RdBu_r", extend="both")
            fig.colorbar(contour, ax=ax, label=cbar_label, shrink=0.85)
            ax.set_xlabel("Latitude (deg)")
            ax.set_title(title)

        axes[0].set_ylabel("Height (km)")
        fig.suptitle(f"t = {time_days[index]:.0f} days")
        fig.tight_layout()
        output_path = Path(cfg.FIGURE_DIR) / f"snapshot_{time_days[index]:05.0f}d.png"
        save_or_show(fig, output_path, cfg)


def plot_hovmoller_field(field_name, data, x_days, z, lat, target_lats, levels, cbar_label, cfg):
    """Save time-height diagrams at selected latitudes."""
    target_indices = [nearest_index(lat, target) for target in target_lats]
    time_grid, z_grid = np.meshgrid(x_days, z / 1000.0, indexing="ij")

    fig, axes = plt.subplots(1, len(target_indices), figsize=(5 * len(target_indices), 5), sharey=True)
    if len(target_indices) == 1:
        axes = [axes]

    contour = None
    for ax, target, index in zip(axes, target_lats, target_indices):
        hov = data[:, :, index]
        contour = ax.contourf(time_grid, z_grid, hov, levels=levels, cmap="RdBu_r", extend="both")
        ax.set_xlabel("Time (days)")
        ax.set_title(f"{field_name} at {lat[index]:g} deg")
        print(f"{field_name}: target {target:g} deg -> plotted grid latitude {lat[index]:g} deg")

    axes[0].set_ylabel("Height (km)")
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    fig.colorbar(contour, cax=cbar_ax, orientation="horizontal", label=cbar_label)
    fig.subplots_adjust(bottom=0.2, wspace=0.15)
    fig.suptitle(field_name)
    filename = field_name.lower().replace(" ", "_").replace("/", "_")
    save_or_show(fig, Path(cfg.FIGURE_DIR) / f"hovmoller_{filename}.png", cfg)


def plot_hovmollers(dataset, midpoint_time, f_store, v_store, w_store, cfg):
    """Save Hovmoller plots for F, v, w, U, and T."""
    z = dataset["z"].values
    lat = dataset["lat"].values
    time_days = dataset["time"].values / SECONDS_PER_DAY
    midpoint_days = midpoint_time / SECONDS_PER_DAY
    target_lats = tuple(cfg.TARGET_LATS)

    plot_hovmoller_field(
        "F at t+dt/2",
        f_store,
        midpoint_days,
        z,
        lat,
        target_lats,
        symmetric_levels(f_store),
        "F (m/s^2)",
        cfg,
    )
    plot_hovmoller_field(
        "v at t+dt/2",
        v_store,
        midpoint_days,
        z,
        lat,
        target_lats,
        symmetric_levels(v_store),
        "v (m/s)",
        cfg,
    )
    plot_hovmoller_field(
        "w at t+dt/2",
        w_store,
        midpoint_days,
        z,
        lat,
        target_lats,
        symmetric_levels(w_store),
        "w (m/s)",
        cfg,
    )
    plot_hovmoller_field(
        "u at t",
        dataset["U"].values,
        time_days,
        z,
        lat,
        target_lats,
        symmetric_levels(dataset["U"].values),
        "u (m/s)",
        cfg,
    )
    plot_hovmoller_field(
        "T at t",
        dataset["T_anomaly"].values,
        time_days,
        z,
        lat,
        target_lats,
        symmetric_levels(dataset["T_anomaly"].values),
        "T anomaly (K)",
        cfg,
    )


def plot_all(output_file=None, cfg=default_config):
    """Load solver output and save all standard diagnostic plots."""
    if output_file is None:
        output_file = cfg.OUTPUT_FILE
    dataset = load_output(output_file)
    midpoint_time, f_store, v_store, w_store = build_midpoint_fields(dataset, cfg)
    plot_snapshots(dataset, midpoint_time, f_store, v_store, w_store, cfg)
    plot_hovmollers(dataset, midpoint_time, f_store, v_store, w_store, cfg)


def main():
    plot_all(cfg=default_config)


if __name__ == "__main__":
    main()
