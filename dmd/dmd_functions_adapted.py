import os
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import colors

def hankelize(X2d, d):
    n_features, n_time = X2d.shape

    n_cols = n_time - d + 1
    H = np.empty((d * n_features, n_cols), dtype=X2d.dtype)

    for k in range(d):
        H[k*n_features:(k+1)*n_features, :] = X2d[:, k:k+n_cols]

    return H
    
def save_fig(fig, local_save_loc, anim=False, **kwargs):
    ext = local_save_loc.split('.')[-1]
    if anim:
        fig.save('/home/cwp29/tmp/fig.{0}'.format(ext), **kwargs)
    else:
        fig.savefig('/home/cwp29/tmp/fig.{0}'.format(ext), **kwargs)
    os.system("rsync /home/cwp29/tmp/fig.{0} cwp29@rossby:{1}".format(ext, local_save_loc))
    os.system("rm /home/cwp29/tmp/fig.{0}".format(ext))
    
def get_index(z, a):
    return int(np.floor(np.argmin(np.abs(a - z))))

def find_conjugate_pairs(lam, imag_tol=1e-10, pair_tol=1e-8):
    """
    Find complex-conjugate eigenvalue pairs.

    Parameters
    ----------
    lam : array-like, complex, shape (n_modes,)
        Eigenvalues (typically dmd.eigs).
    imag_tol : float
        Values with |Im(lam)| <= imag_tol are treated as real (unpaired singles).
    pair_tol : float
        Tolerance for matching lam[j] to conj(lam[i]).

    Returns
    -------
    pairs : list of tuple
        List of (i, j) indices such that lam[j] ~ conj(lam[i]) and i < j.
    singles : list of int
        Indices of real (or unpaired) eigenvalues.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    n = lam.size
    used = np.zeros(n, dtype=bool)
    pairs = []
    singles = []

    for i in range(n):
        if used[i]:
            continue

        if np.abs(lam[i].imag) <= imag_tol:
            used[i] = True
            singles.append(i)
            continue

        target = np.conj(lam[i])
        # Find closest match among unused indices
        diffs = np.abs(lam - target)
        diffs[used] = np.inf
        diffs[i] = np.inf
        j = int(np.argmin(diffs))

        if np.isfinite(diffs[j]) and diffs[j] < pair_tol:
            used[i] = True
            used[j] = True
            # store consistently
            a, b = (i, j) if i < j else (j, i)
            pairs.append((a, b))
        else:
            used[i] = True
            singles.append(i)

    return pairs, singles

def get_rank_by_energy(sval, ethresh):
    cum_energy = np.cumsum((sval**2) / np.sum(sval**2))
    return np.searchsorted(cum_energy, ethresh), cum_energy

def dehankel(H, delay, dshape):
        return H.reshape(delay, *dshape)[0]

def dehankel_embedded(Hrec, d, dshape):
    """
    Dehankel a reconstruction Hrec given in stacked-Hankel matrix form:
        Hrec shape = (d*nstate, ntime_eff)  where ntime_eff = ntime - d + 1

    dshape = (nvar, npres, nlat, ntime)

    Returns Xrec with shape (nvar, npres, nlat, ntime).
    """
    Hrec = np.asarray(Hrec)
    nvar, npres, nlat, ntime = dshape
    nstate = nvar * npres * nlat

    if Hrec.ndim != 2:
        raise ValueError(f"Hrec must be 2D (d*nstate, ntime_eff), got {Hrec.shape}")

    if Hrec.shape[0] != d * nstate:
        raise ValueError(
            f"Hrec.shape[0] must be d*nstate = {d}*{nstate}={d*nstate}, got {Hrec.shape[0]}"
        )

    ntime_eff = Hrec.shape[1]
    ntime_expected = ntime_eff + d - 1
    if ntime_expected != ntime:
        raise ValueError(
            f"dshape ntime={ntime} is inconsistent with Hrec.shape[1]={ntime_eff} and d={d}: "
            f"ntime should be {ntime_expected}."
        )

    # Reshape to (d, nstate, ntime_eff) then to (d, nvar, npres, nlat, ntime_eff)
    H = Hrec.reshape(d, nstate, ntime_eff)
    H = H.reshape(d, nvar, npres, nlat, ntime_eff)

    # Proper overlap-add mean: each lag k contributes to time indices t+k
    out = np.zeros((nvar, npres, nlat, ntime), dtype=float)
    cnt = np.zeros((ntime,), dtype=float)

    for k in range(d):
        out[..., k:k+ntime_eff] += H[k].real
        cnt[k:k+ntime_eff] += 1.0

    out /= np.maximum(cnt, 1.0)[None, None, None, :]
    return out
    
########## CHECK THIS FUNCTION
    
def reconstruct_dynamics(dmd, keep, ntime):
    dyn_rec = dmd.amplitudes[keep, None] * (dmd.eigs[keep, None] ** np.arange(ntime)[None, :])
    #pairs, singles = find_conjugate_pairs(dmd.eigs[keep])
    pairs = find_conjugate_pairs(dmd.eigs[keep])
    print(pairs)
    dyn_total = 0
    for i, j in pairs:
        dyn_total += dyn_rec[i] + dyn_rec[j]
        print(f"{i} index i, {j} index j")

    return dyn_total, dyn_rec

def reconstruct_data(dmd, keep, delay, dshape):
    H_rec = (dmd.modes[:, keep] @ dmd.dynamics[keep, :]).real
    try:
        X_rec = dehankel(H_rec, delay, dshape)
    except ValueError:
        X_rec = dehankel_embedded(H_rec, delay, dshape)

    return X_rec
    
def get_harmonics(p0, periods, sampfreq=1.0, nharm=2):
    harmonics = [p0]
    harmonics += [p0/n for n in range(2, nharm+1)] # sub-harmonics
    harmonics += [p0*n for n in range(2, nharm+1)] # super-harmonics

    harmonics = np.array(harmonics, dtype=float)

    return np.any(np.abs(periods[:, None] - harmonics[None, :]) < 2*sampfreq, axis=1), np.unique(harmonics)

def _butter_bandpass(period_band, dt=1.0, order=4):
    """
    period_band: (pmin, pmax) in same time units as dt (months)
    returns filter coefficients for bandpass in cycles per dt-unit
    """
    pmin, pmax = period_band
    fmin = 1.0 / pmax   # cycles / month
    fmax = 1.0 / pmin
    fs = 1.0 / dt
    nyq = 0.5 * fs
    low = fmin / nyq
    high = fmax / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Bad band: low={low}, high={high}. Check dt and period_band.")
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a

def bandpass_time(X, period_band, dt=1.0, axis=-1, order=4):
    """Bandpass filter along time axis using filtfilt."""
    X = np.asarray(X)
    b, a = _butter_bandpass(period_band, dt=dt, order=order)
    # fill NaNs with 0 for filtering, but mask later
    X0 = np.where(np.isfinite(X), X, 0.0)
    Y = filtfilt(b, a, X0, axis=axis)
    return Y

def dirichlet_energy_2d(X, lat, pres, weights=None):
    """
    Dirichlet energy of 2D field X(pres, lat, time):
      E = mean_t sum_{p,lat} w(p,lat) * (|dX/dphi|^2 + |dX/dlogp|^2)

    lat: degrees, shape (nlat,)
    pres: same units but positive, shape (npres,)
    weights: optional (npres, nlat) weights 
    """
    X = np.asarray(X)
    lat = np.asarray(lat)
    pres = np.asarray(pres)

    if X.ndim != 3:
        raise ValueError(f"Expected X (npres,nlat,ntime), got {X.shape}")

    phi = np.deg2rad(lat)                       # radians
    logp = np.log(pres.astype(float))

    # gradients: np.gradient accepts coordinate arrays for spacing
    dX_dphi  = np.gradient(X, phi,  axis=1, edge_order=1)
    dX_dlogp = np.gradient(X, logp, axis=0, edge_order=1)

    g2 = dX_dphi**2 + dX_dlogp**2               # (npres,nlat,ntime)

    if weights is None:
        E_t = np.nanmean(g2, axis=(0, 1))
    else:
        w = np.asarray(weights)
        if w.shape != X.shape[:2]:
            raise ValueError(f"weights must be (npres,nlat) = {X.shape[:2]}, got {w.shape}")
        E_t = np.nanmean(g2 * w[:, :, None], axis=(0, 1)) / np.nanmean(w)

    return float(np.nanmean(E_t))

def _weighted_corr(a, b, w=None):
    """Correlation of flattened arrays, optionally weighted."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    if a.size < 3:
        return np.nan

    if w is None:
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a*a) * np.sum(b*b))
        return float(np.sum(a*b) / denom) if denom > 0 else np.nan

    w = np.asarray(w).ravel()[m]
    w = np.where(np.isfinite(w), w, 0.0)
    ws = np.sum(w)
    if ws <= 0:
        return np.nan
    ma = np.sum(w*a) / ws
    mb = np.sum(w*b) / ws
    a0 = a - ma
    b0 = b - mb
    denom = np.sqrt(np.sum(w*a0*a0) * np.sum(w*b0*b0))
    return float(np.sum(w*a0*b0) / denom) if denom > 0 else np.nan

def qbo_score_2d(
    X_rec, X_in,
    lat, pres,
    dt=1.0,
    qbo_band=(26.0, 30.0),
    weights=None,                 # (npres,nlat) e.g. cos(lat)*dp, optional
    dirichlet_gamma=1.0,          # strength of roughness penalty
    filter_order=4,
):
    """
    QBO score using full 2D field:
      score = corr_band * bandpower_frac * regularity

    - corr_band: weighted correlation between bandpassed recon and input (all space+time)
    - bandpower_frac: fraction of recon power within QBO band (space-avg)
    - regularity: exp(-gamma * max(0, E_rec/E_in - 1)) using Dirichlet energy

    Inputs must be (npres, nlat, ntime).
    """
    X_rec = np.asarray(X_rec)
    X_in  = np.asarray(X_in)
    if X_rec.shape != X_in.shape:
        raise ValueError(f"X_rec and X_in must have same shape, got {X_rec.shape} vs {X_in.shape}")
    if X_rec.ndim != 3:
        raise ValueError(f"Expected (npres,nlat,ntime), got {X_rec.shape}")

    npres, nlat, ntime = X_rec.shape

    # Bandpass to QBO band
    Xr_q = bandpass_time(X_rec, period_band=qbo_band, dt=dt, axis=2, order=filter_order)
    Xi_q = bandpass_time(X_in,  period_band=qbo_band, dt=dt, axis=2, order=filter_order)

    # --- Pattern correlation across full 2D+time ---
    if weights is None:
        w_flat = None
    else:
        w_flat = np.repeat(np.asarray(weights)[:, :, None], ntime, axis=2)

    corr_band = _weighted_corr(Xr_q, Xi_q, w=w_flat)

    # --- Bandpower comparison (FFT, space-avg) ---
    
    def qbo_band_power(X):
        X0 = X - np.nanmean(X, axis=2, keepdims=True)
        X0 = np.where(np.isfinite(X0), X0, 0.0)
        F = np.fft.rfft(X0, axis=2)
        P = (np.abs(F)**2)
        freqs = np.fft.rfftfreq(ntime, d=dt)  # cycles / month

        pmin, pmax = qbo_band
        fmin, fmax = 1.0/pmax, 1.0/pmin
        band = (freqs >= fmin) & (freqs <= fmax)

        # integrate band power over frequency, then average over space with optional weights
        Pband = np.sum(P[:, :, band], axis=2)  # (npres,nlat)
        if weights is None:
            return float(np.sum(Pband))
        else:
            w = np.asarray(weights)
            return float(np.sum(w * Pband))

    Pq_rec = qbo_band_power(X_rec)
    Pq_in  = qbo_band_power(X_in)

    if Pq_in > 0 and np.isfinite(Pq_in) and np.isfinite(Pq_rec):
        R_qbo = Pq_rec / Pq_in
        amp_score = float(np.exp(-np.abs(np.log(R_qbo))))
    else:
        R_qbo = np.nan
        amp_score = np.nan

    # --- Dirichlet energy regularity (use bandpassed fields) ---
    E_in  = dirichlet_energy_2d(Xi_q, lat=lat, pres=pres, weights=weights)
    E_rec = dirichlet_energy_2d(Xr_q, lat=lat, pres=pres, weights=weights)

    ratio = (E_rec / E_in) if (E_in > 0 and np.isfinite(E_in) and np.isfinite(E_rec)) else np.nan
    # penalise only if reconstruction is rougher than input
    regularity = float(np.exp(-dirichlet_gamma * max(0.0, ratio - 1.0))) if np.isfinite(ratio) else np.nan

    score = corr_band * amp_score * regularity

    parts = dict(
        corr_band=corr_band,
        Pq_rec=Pq_rec,
        Pq_in=Pq_in,
        R_qbo=R_qbo,
        amp_score=amp_score,
        dirichlet_in=E_in,
        dirichlet_rec=E_rec,
        dirichlet_ratio=ratio,
        regularity=regularity,
    )
    return score, parts

def plot_svd_spectrum(dmd, energy_thresh, set_svd_rank=True):
    
    fig = plt.figure(figsize=(8, 3))
    plt.title("EDMD Singular Value Spectrum (no truncation)")

    s = dmd.operator.svd_vals
    j = np.arange(1, len(s) + 1)
    
    rank, cum_energy = get_rank_by_energy(s, energy_thresh)

    plt.plot(j, s, marker='.', ls='None', color='b')
    plt.yscale('log')
    plt.ylabel(r"$\sigma$")
    plt.xlabel(r"$j$")

    ax2 = plt.gca().twinx()
    ax2.plot(j, cum_energy, color='r')
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Cumulative energy")
    ax2.axhline(energy_thresh, linestyle='--', lw=1, color='gray')
    plt.axvline(rank, linestyle='--', lw=1, color='gray')

    if set_svd_rank:
        return fig, int(rank)
    else:
        return fig

def plot_dmd_summary(dmd, dshape, delay, qbo_band=(26, 30), qbo_period=28):
    dt = dmd.original_time['dt'] # Time step (months)
    lam = np.asarray(dmd.eigs)   # Discrete time eigenvalues
    omega = np.log(lam) / dt     # Continuous time eigenvalues

    freq = np.abs(np.angle(lam)) / (2 * np.pi * dt)         # cycles / month
    period = np.where(freq != 0., 1.0/freq, np.inf)         # months
    r = np.abs(lam)                                         # |lambda|

    # --- pair handling  ---
    # Expect: pairs = [(i,j), ...], singles = [k, ...]
    pairs, singles = find_conjugate_pairs(lam)

    # Build list of modes (each entry is either a pair (i,j) or a single (k,))
    mode_idxs = [(i, j) for (i, j) in pairs] + [(k,) for k in singles]

    # Mode metrics
    per = np.full(len(mode_idxs), np.inf, dtype=float)
    stab = np.zeros(len(mode_idxs), dtype=float)
    energy = np.zeros(len(mode_idxs), dtype=float)

    # modal energy per mode
    phi_norm2 = np.sum(np.abs(dmd.modes)**2, axis=0)
    dyn_power = np.sum(np.abs(dmd.dynamics)**2, axis=1)
    e_mode = phi_norm2 * dyn_power

    for k, idx in enumerate(mode_idxs):
        i = idx[0]
        # period from representative eigenvalue
        per[k] = period[i]
        # stability from |lambda|
        stab[k] = r[i]
        # energy: sum over members of the pair (or single)
        energy[k] = np.sum(e_mode[list(idx)])

    # ---------------------------------------------------
    # Figure layout
    # ---------------------------------------------------
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    axs = fig.subplot_mosaic("AB;DE")

    # ----------------------------------
    # Discrete spectrum (A)
    # ----------------------------------
    # Clip large periods for colour scale readability
    finite_p = np.isfinite(period)
    if np.any(finite_p):
        p_lo = max(1e-3, np.nanpercentile(period[finite_p], 1))
        p_hi = np.nanpercentile(period[finite_p], 99)
    else:
        p_lo, p_hi = 1e-3, 1.0
    period_c = np.clip(period, p_lo, p_hi)

    ax = axs["A"]
    ax.set_title("Discrete spectrum")
    ax.add_patch(plt.Circle((0, 0), 1, color="k", fill=False, linestyle="--", lw=1))

    sc = ax.scatter(
        lam.real, lam.imag,
        c=period_c, s=20, marker="+",
        cmap="viridis", norm=LogNorm(vmin=p_lo, vmax=p_hi)
    )
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)

    ax.text(
        0.02, 0.02,
        f"|λ|<1: {np.sum(r<1.0-0.01)}\n|λ|≈1: {np.sum(np.abs(r-1.0)<0.02)}\n|λ|>1: {np.sum(r>1.0+0.01)}",
        transform=ax.transAxes, fontsize="small", va="bottom"
    )

    # ----------------------------------
    # Continuous spectrum (B)
    # ----------------------------------
    ax = axs["B"]
    ax.set_title("Continuous spectrum")
    ax.set_ylabel(r"Frequency (cycles/month)")
    ax.set_xlabel(r"Growth rate $\mathrm{Re}(\omega)$ [1/month]")

    ax.axhline(0, color='k', lw=1, linestyle='--')
    ax.axvline(0, color='k', lw=1, linestyle='--')

    sc2 = ax.scatter(
        omega.real, omega.imag/(2*np.pi),
        c=period_c, s=20, marker="+",
        cmap="viridis", norm=LogNorm(vmin=p_lo, vmax=p_hi)
    )
    cbar = plt.colorbar(sc2, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Period (months)", fontsize="small")

    # ----------------------------------
    # Period vs stability (D)
    # ----------------------------------
    ax = axs["D"]
    ax.set_title("Period vs. stability")

    ok = np.isfinite(per) & (per > 0)

    scD = ax.scatter(
        per[ok], stab[ok],
        c=energy[ok],
        s=25, marker="o",
        cmap="viridis",
        norm=LogNorm(vmin=max(1e-12, np.nanpercentile(energy[ok], 5)),
                     vmax=np.nanpercentile(energy[ok], 95) if np.any(ok) else 1.0),
        edgecolor="none",
    )
    ax.axhline(1.0, linestyle="--", color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Period (months)")
    ax.set_ylabel(r"$|\lambda|$")
    cbD = plt.colorbar(scD, ax=ax, fraction=0.05, pad=0.02)
    cbD.set_label("Modal energy", fontsize="small")

    # ----------------------------------
    # Period vs modal energy (E)
    # ----------------------------------
    ax = axs["E"]
    ax.set_title("Period vs. modal energy")

    scE = ax.scatter(
        per[ok], energy[ok],
        c=np.clip(stab[ok], 0, 1.05),
        s=25, marker="o",
        cmap="plasma",
        norm=Normalize(vmin=0.9, vmax=1.02),
        edgecolor="none",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Period (months)")
    ax.set_ylabel("Modal energy (normalized)")
    cbE = plt.colorbar(scE, ax=ax, fraction=0.05, pad=0.02)
    cbE.set_label(r"stability $|\lambda|$", fontsize="small")

    # ----------------------------------
    # Spectrum decorations (C/D/E)
    # ----------------------------------
    pmin, pmax = qbo_band
    for ax in [axs["D"], axs["E"]]:
        ax.axvspan(pmin, pmax, alpha=0.15, color='b')
        ax.axvline(qbo_period, ls='--', color='k', lw=1)

    # How many modes in QBO band and near-neutral
    in_band = ok & (per >= pmin) & (per <= pmax)
    near_neutral = in_band & (np.abs(stab - 1.0) < 0.02)
    axs["D"].text(
        0.02, 0.02,
        f"Modes in QBO band: {np.sum(in_band) // 2}\nIn band & |λ|-1<0.02: {np.sum(near_neutral) // 2}",
        transform=axs["D"].transAxes, fontsize="small", va="bottom"
    )

    return fig

def plot_reconstruction_summary(
    dmd,
    keep,
    delay,
    dshape,
    ds,
    std_data,
    stds,
    u_levs,
    T_levs,
    qbo_band=(26, 30),
    lat_band=(-10, 10),
    pres_level=50.0,
    title="DMD reconstruction summary",
):
    """
    Summary plot for a DMD reconstruction based on selected modes.

    Parameters
    ----------
    dmd : fitted DMD object
    keep : boolean array, shape (n_modes,)
        Modes used for reconstruction
    delay : int
        Hankel delay
    dshape : tuple
        (nvar, npres, nlat, ntime)
    ds : xarray.Dataset
        Must contain coords lat, pres, time
    std_data : list of arrays
        Input data, before stacking into data array X
    stds : list of arrays
        Standardisation factors per variable
    qbo_band : (qbo_pmin, qbo_pmax)
        QBO period band
    lat_band : (latmin, latmax)
        Tropical averaging band
    pres_level : float
        Pressure level (hPa) for time series
    title : str
        Figure title
    """

    # -------------------------------------------------
    # Indices and reconstruction
    # -------------------------------------------------
    lat = ds.lat.values
    pres = ds.pres.values
    #time = ds.time.values

    lat_mask = (lat >= lat_band[0]) & (lat <= lat_band[1])
    kpres = get_index(pres_level, pres)

    X_rec = reconstruct_data(dmd, keep, delay, dshape)

    # Undo standardisation
    X_rec_phys = np.zeros_like(X_rec)
    for v in range(X_rec.shape[0]):
        X_rec_phys[v] = (X_rec[v].T / stds[v].T).T

    # Input (ERA5) in physical units
    X_in_phys = []
    for v in range(X_rec.shape[0]):
        Xin = (std_data[v].T / stds[v].T).T
        X_in_phys.append(Xin)
    X_in_phys = np.asarray(X_in_phys)

    # -------------------------------------------------
    # Fix model time axis issue
    # -------------------------------------------------
    dt = dmd.original_time["dt"]  # months per sample (you already use this later)

    ntime_rec = X_rec_phys.shape[-1]
    ntime_in  = X_in_phys.shape[-1]
    ntime = min(ntime_rec, ntime_in)

    # truncate both so they match
    X_rec_phys = X_rec_phys[..., :ntime]
    X_in_phys  = X_in_phys[..., :ntime]

    # robust plotting time axis: months since start (always works)
    tplot = np.arange(ntime) * dt
    
    # -------------------------------------------------
    # Conjugate-pair decomposition for row 4
    # -------------------------------------------------
    keep_idx = np.where(keep)[0]
    pairs, singles = find_conjugate_pairs(dmd.eigs[keep_idx])
    structures = [(i, j) for (i, j) in pairs] + [(i,) for i in singles]

    # -------------------------------------------------
    # Colour by period (structure-level), used for rows 4 and 5
    # -------------------------------------------------
    #dt = dmd.original_time["dt"]

    lam_keep = np.asarray(dmd.eigs)[keep_idx]
    theta_keep = np.angle(lam_keep)
    freq_keep = np.abs(theta_keep) / (2*np.pi*dt)
    period_keep = np.where(freq_keep > 0, 1.0/freq_keep, np.inf)

    period_struct = np.array([period_keep[idx[0]] for idx in structures], dtype=float)

    finite_p = np.isfinite(period_struct) & (period_struct > 0)
    if np.any(finite_p):
        p_lo = max(1e-3, np.nanpercentile(period_struct[finite_p], 1))
        p_hi = np.nanpercentile(period_struct[finite_p], 99)
    else:
        p_lo, p_hi = 1e-3, 1.0

    period_c = np.clip(np.where(np.isfinite(period_struct), period_struct, p_hi), p_lo, p_hi)
    cmap = plt.cm.viridis
    normP = LogNorm(vmin=p_lo, vmax=p_hi)
    colors = cmap(normP(period_c))

    # -------------------------------------------------
    # Modal energy per structure split into u and T (fast)
    # State is stacked as [u, T] for each delay block
    # -------------------------------------------------
    modes = np.asarray(dmd.modes)        # (n_state, n_modes)
    dynamics = np.asarray(dmd.dynamics)  # (n_modes, n_time_eff)

    nvar, npres, nlat, ntime = dshape
    nstate_per_lag = nvar * npres * nlat
    nmodes = modes.shape[1]

    dyn_pow = np.mean(np.abs(dynamics)**2, axis=1)  # (nmodes,)

    # reshape modes to (delay, nvar, npres, nlat, nmodes)
    Phi = modes.reshape(delay, nstate_per_lag, nmodes).reshape(delay, nvar, npres, nlat, nmodes)

    # u / T norms per mode (sum over delay, pres, lat)
    phi_u_norm2 = np.sum(np.abs(Phi[:, 0, :, :, :])**2, axis=(0, 1, 2))  # (nmodes,)
    phi_T_norm2 = np.sum(np.abs(Phi[:, 1, :, :, :])**2, axis=(0, 1, 2))  # (nmodes,)

    energy_mode_u = phi_u_norm2 * dyn_pow
    energy_mode_T = phi_T_norm2 * dyn_pow

    energy_struct_u = np.zeros(len(structures), dtype=float)
    energy_struct_T = np.zeros(len(structures), dtype=float)
    for k, idx in enumerate(structures):
        gi = keep_idx[list(idx)]
        energy_struct_u[k] = np.sum(energy_mode_u[gi])
        energy_struct_T[k] = np.sum(energy_mode_T[gi])

    energy_u_plot = energy_struct_u / np.sum(energy_struct_u) if np.sum(energy_struct_u) > 0 else energy_struct_u
    energy_T_plot = energy_struct_T / np.sum(energy_struct_T) if np.sum(energy_struct_T) > 0 else energy_struct_T

    # -------------------------------------------------
    # Figure layout
    # -------------------------------------------------
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    axs = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        IJ
        """
    )

    var_labels = ["u", "T"]

    for v in range(2):
        levs = u_levs if v == 0 else T_levs
        Xr = X_rec_phys[v]
        Xi = X_in_phys[v]

        # ----------------------------
        # Row 1: time vs pressure
        # ----------------------------
        ax = axs[["A", "B"][v]]
        trop_mean = Xr[:, lat_mask, :].mean(axis=1)

        im = ax.contourf(tplot, pres, trop_mean, levels=levs, cmap="bwr", extend="both")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_title(f"{var_labels[v]}: tropical mean")

        # ----------------------------
        # Row 2: lat vs pressure snapshot at peak zonal wind
        # ----------------------------
        ax = axs[["C", "D"][v]]
        if v == 0:
            t0 = np.argmax(np.abs(trop_mean[kpres]))
        im = ax.contourf(lat, pres, Xr[:, :, t0], levels=levs, cmap="bwr", extend="both")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_title(f"{var_labels[v]}: structure at peak u")
        fig.colorbar(im, ax=ax, orientation="horizontal")

        # ----------------------------
        # Row 3: time series comparison
        # ----------------------------
        ax = axs[["E", "F"][v]]
        ts_rec = Xr[kpres, lat_mask, :].mean(axis=0)
        ts_in  = Xi[kpres, lat_mask, :].mean(axis=0)

        ax.plot(tplot, ts_in, lw=2, color='b', label="Original data")
        ax.plot(tplot, ts_rec, lw=2, color='r', label="Reconstruction")
        r = np.corrcoef(ts_in, ts_rec)[0, 1]
        ax.set_title(f"{var_labels[v]} @ {pres_level:.0f} hPa (r={r:.2f})")
        ax.legend()

        # ----------------------------
        # Row 4: modal contributions (coloured by period, same as row 5)
        # ----------------------------
        ax = axs[["G", "H"][v]]
        for i, (idx, col) in enumerate(zip(structures, colors)):
            gi = keep_idx[list(idx)]
            H = (dmd.modes[:, gi] @ dmd.dynamics[gi, :]).real
            try:
                Xj = dehankel_embedded(H, delay, dshape)
            except ValueError:
                Xj = dehankel(H, delay, dshape)

            Xj = (Xj[v].T / stds[v].T).T
            ts = Xj[kpres, lat_mask, :].mean(axis=0)
            ax.plot(tplot, ts, color=col, alpha=0.6, zorder=period_c[i])

        ax.set_title(f"{var_labels[v]}: modal decomposition at {pres_level:.0f} hPa")

        # ----------------------------
        # Row 5: modal energy
        # ----------------------------
        ax = axs[["I", "J"][v]]
        x = np.arange(len(structures))
        y = energy_u_plot if v == 0 else energy_T_plot

        scE = ax.scatter(period_c, y, c=period_c, cmap=cmap, norm=normP, s=35)
        ax.set_title(f"{var_labels[v]}: modal energy")
        ax.set_xlabel("Period (months)")
        ax.set_ylabel("Modal energy fraction")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)
        ax.grid(True, ls=":", lw=0.6, alpha=0.5)

    fig.suptitle(title)

    # --- QBO score (2D) annotations for u and T (tropics only) ---
    lat_mask = lat_mask
    mask_pres = np.ones_like(pres, dtype=bool)
    w_use = None

    score_u, parts_u = qbo_score_2d(
        # X_rec_phys[0][np.ix_(mask_pres, lat_mask, np.arange(X_rec_phys.shape[-1]))],
        # X_in_phys[0][np.ix_(mask_pres, lat_mask, np.arange(X_in_phys.shape[-1]))],
        X_rec_phys[0][np.ix_(mask_pres, lat_mask, np.arange(ntime))],
        X_in_phys[0][np.ix_(mask_pres, lat_mask, np.arange(ntime))],
        lat=lat[lat_mask],
        pres=pres[mask_pres],
        dt=dt,
        qbo_band=qbo_band,
        weights=w_use,
        dirichlet_gamma=1.0
    )

    score_T, parts_T = qbo_score_2d(
        X_rec_phys[1][np.ix_(mask_pres, lat_mask, np.arange(ntime))],
        X_in_phys[1][np.ix_(mask_pres, lat_mask, np.arange(ntime))],
        lat=lat[lat_mask],
        pres=pres[mask_pres],
        dt=dt,
        qbo_band=qbo_band,
        weights=w_use,
        dirichlet_gamma=1.0
    )

    fig.text(
        0.15, 0.99,
        f"u score: {score_u:.3f}  (corr={parts_u['corr_band']:.2f}, amp={parts_u['amp_score']:.2f}, reg={parts_u['regularity']:.2f})\n",
        ha="center", va="top", fontsize="small"
    )
    
    fig.text(
        0.85, 0.99,
        f"T score: {score_T:.3f}  (corr={parts_T['corr_band']:.2f}, amp={parts_T['amp_score']:.2f}, reg={parts_T['regularity']:.2f})",
        ha="center", va="top", fontsize="small"
    )

    return fig
def plot_reconstruction_summary2(
    dmd,
    keep,
    delay,
    dshape,
    ds,
    std_data,
    stds,
    u_levs,
    T_levs,
    qbo_band=(26, 30),
    lat_band=(-10, 10),
    pres_level=50.0,
    title="DMD reconstruction summary",
    return_outputs=False,   # <--- NEW (default preserves current behaviour)
):
    """
    Summary plot for a DMD reconstruction based on selected modes.
    """

    # -------------------------------------------------
    # Indices and reconstruction
    # -------------------------------------------------
    lat = ds.lat.values
    pres = ds.pres.values
    #time = ds.time.values

    lat_mask = (lat >= lat_band[0]) & (lat <= lat_band[1])
    kpres = get_index(pres_level, pres)

    X_rec = reconstruct_data(dmd, keep, delay, dshape)

    # Undo standardisation
    X_rec_phys = np.zeros_like(X_rec)
    for v in range(X_rec.shape[0]):
        X_rec_phys[v] = (X_rec[v].T / stds[v].T).T

    # Input (ERA5) in physical units
    X_in_phys = []
    for v in range(X_rec.shape[0]):
        Xin = (std_data[v].T / stds[v].T).T
        X_in_phys.append(Xin)
    X_in_phys = np.asarray(X_in_phys)

    # -------------------------------------------------
    # Fix model time axis issue
    # -------------------------------------------------
    dt = dmd.original_time["dt"]  # months per sample

    ntime_rec = X_rec_phys.shape[-1]
    ntime_in  = X_in_phys.shape[-1]
    ntime = min(ntime_rec, ntime_in)

    # truncate both so they match
    X_rec_phys = X_rec_phys[..., :ntime]
    X_in_phys  = X_in_phys[..., :ntime]

    # robust plotting time axis: months since start (always works)
    tplot = np.arange(ntime) * dt

    # -------------------------------------------------
    # Conjugate-pair decomposition for row 4
    # -------------------------------------------------
    keep_idx = np.where(keep)[0]
    pairs, singles = find_conjugate_pairs(dmd.eigs[keep_idx])
    structures = [(i, j) for (i, j) in pairs] + [(i,) for i in singles]

    # -------------------------------------------------
    # Colour by period (structure-level), used for rows 4 and 5
    # -------------------------------------------------
    lam_keep = np.asarray(dmd.eigs)[keep_idx]
    theta_keep = np.angle(lam_keep)
    freq_keep = np.abs(theta_keep) / (2*np.pi*dt)
    period_keep = np.where(freq_keep > 0, 1.0/freq_keep, np.inf)

    period_struct = np.array([period_keep[idx[0]] for idx in structures], dtype=float)

    finite_p = np.isfinite(period_struct) & (period_struct > 0)
    if np.any(finite_p):
        p_lo = max(1e-3, np.nanpercentile(period_struct[finite_p], 1))
        p_hi = np.nanpercentile(period_struct[finite_p], 99)
    else:
        p_lo, p_hi = 1e-3, 1.0

    period_c = np.clip(np.where(np.isfinite(period_struct), period_struct, p_hi), p_lo, p_hi)
    cmap = plt.cm.viridis
    normP = LogNorm(vmin=p_lo, vmax=p_hi)
    colors = cmap(normP(period_c))

    # -------------------------------------------------
    # Modal energy per structure split into u and T (fast)
    # State is stacked as [u, T] for each delay block
    # -------------------------------------------------
    modes = np.asarray(dmd.modes)        # (n_state, n_modes)
    dynamics = np.asarray(dmd.dynamics)  # (n_modes, n_time_eff)

    nvar, npres, nlat, ntime0 = dshape
    nstate_per_lag = nvar * npres * nlat
    nmodes = modes.shape[1]

    dyn_pow = np.mean(np.abs(dynamics)**2, axis=1)  # (nmodes,)

    # reshape modes to (delay, nvar, npres, nlat, nmodes)
    Phi = modes.reshape(delay, nstate_per_lag, nmodes).reshape(delay, nvar, npres, nlat, nmodes)

    # u / T norms per mode (sum over delay, pres, lat)
    phi_u_norm2 = np.sum(np.abs(Phi[:, 0, :, :, :])**2, axis=(0, 1, 2))  # (nmodes,)
    phi_T_norm2 = np.sum(np.abs(Phi[:, 1, :, :, :])**2, axis=(0, 1, 2))  # (nmodes,)

    energy_mode_u = phi_u_norm2 * dyn_pow
    energy_mode_T = phi_T_norm2 * dyn_pow

    energy_struct_u = np.zeros(len(structures), dtype=float)
    energy_struct_T = np.zeros(len(structures), dtype=float)
    for k, idx in enumerate(structures):
        gi = keep_idx[list(idx)]
        energy_struct_u[k] = np.sum(energy_mode_u[gi])
        energy_struct_T[k] = np.sum(energy_mode_T[gi])

    energy_u_plot = energy_struct_u / np.sum(energy_struct_u) if np.sum(energy_struct_u) > 0 else energy_struct_u
    energy_T_plot = energy_struct_T / np.sum(energy_struct_T) if np.sum(energy_struct_T) > 0 else energy_struct_T

    # -------------------------------------------------
    # Figure layout
    # -------------------------------------------------
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    axs = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        IJ
        """
    )

    var_labels = ["u", "T"]

    for v in range(2):
        levs = u_levs if v == 0 else T_levs
        Xr = X_rec_phys[v]
        Xi = X_in_phys[v]

        # ----------------------------
        # Row 1: time vs pressure
        # ----------------------------
        ax = axs[["A", "B"][v]]
        trop_mean = Xr[:, lat_mask, :].mean(axis=1)

        im = ax.contourf(tplot, pres, trop_mean, levels=levs, cmap="bwr", extend="both")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_title(f"{var_labels[v]}: tropical mean")

        # ----------------------------
        # Row 2: lat vs pressure snapshot at peak zonal wind
        # ----------------------------
        ax = axs[["C", "D"][v]]
        if v == 0:
            t0 = np.argmax(np.abs(trop_mean[kpres]))
        im = ax.contourf(lat, pres, Xr[:, :, t0], levels=levs, cmap="bwr", extend="both")
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_title(f"{var_labels[v]}: structure at peak u")
        fig.colorbar(im, ax=ax, orientation="horizontal")

        # ----------------------------
        # Row 3: time series comparison
        # ----------------------------
        ax = axs[["E", "F"][v]]
        ts_rec = Xr[kpres, lat_mask, :].mean(axis=0)
        ts_in  = Xi[kpres, lat_mask, :].mean(axis=0)

        ax.plot(tplot, ts_in, lw=2, color='b', label="Original data")
        ax.plot(tplot, ts_rec, lw=2, color='r', label="Reconstruction")
        r = np.corrcoef(ts_in, ts_rec)[0, 1]
        ax.set_title(f"{var_labels[v]} @ {pres_level:.0f} hPa (r={r:.2f})")
        ax.legend()

        # ----------------------------
        # Row 4: modal contributions (coloured by period, same as row 5)
        # ----------------------------
        ax = axs[["G", "H"][v]]
        for i, (idx, col) in enumerate(zip(structures, colors)):
            gi = keep_idx[list(idx)]
            H = (dmd.modes[:, gi] @ dmd.dynamics[gi, :]).real
            try:
                Xj = dehankel_embedded(H, delay, dshape)
            except ValueError:
                Xj = dehankel(H, delay, dshape)

            Xj = (Xj[v].T / stds[v].T).T
            ts = Xj[kpres, lat_mask, :].mean(axis=0)
            ax.plot(tplot, ts, color=col, alpha=0.6, zorder=period_c[i])

        ax.set_title(f"{var_labels[v]}: modal decomposition at {pres_level:.0f} hPa")

        # ----------------------------
        # Row 5: modal energy
        # ----------------------------
        ax = axs[["I", "J"][v]]
        y = energy_u_plot if v == 0 else energy_T_plot

        ax.scatter(period_c, y, c=period_c, cmap=cmap, norm=normP, s=35)
        ax.set_title(f"{var_labels[v]}: modal energy")
        ax.set_xlabel("Period (months)")
        ax.set_ylabel("Modal energy fraction")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)
        ax.grid(True, ls=":", lw=0.6, alpha=0.5)

    fig.suptitle(title)

    # --- QBO score (2D) annotations for u and T (tropics only) ---
    mask_pres = np.ones_like(pres, dtype=bool)
    w_use = None

    score_u, parts_u = qbo_score_2d(
        X_rec_phys[0][np.ix_(mask_pres, lat_mask, np.arange(ntime))],
        X_in_phys[0][np.ix_(mask_pres, lat_mask, np.arange(ntime))],
        lat=lat[lat_mask],
        pres=pres[mask_pres],
        dt=dt,
        qbo_band=qbo_band,
        weights=w_use,
        dirichlet_gamma=1.0
    )

    score_T, parts_T = qbo_score_2d(
        X_rec_phys[1][np.ix_(mask_pres, lat_mask, np.arange(ntime))],
        X_in_phys[1][np.ix_(mask_pres, lat_mask, np.arange(ntime))],
        lat=lat[lat_mask],
        pres=pres[mask_pres],
        dt=dt,
        qbo_band=qbo_band,
        weights=w_use,
        dirichlet_gamma=1.0
    )

    fig.text(
        0.15, 0.99,
        f"u score: {score_u:.3f}  (corr={parts_u['corr_band']:.2f}, amp={parts_u['amp_score']:.2f}, reg={parts_u['regularity']:.2f})\n",
        ha="center", va="top", fontsize="small"
    )

    fig.text(
        0.85, 0.99,
        f"T score: {score_T:.3f}  (corr={parts_T['corr_band']:.2f}, amp={parts_T['amp_score']:.2f}, reg={parts_T['regularity']:.2f})",
        ha="center", va="top", fontsize="small"
    )

    if not return_outputs:
        return fig

    # Minimal outputs only (as requested)
    outputs = dict(
        # reconstructed and original fields (physical units, time-matched)
        X_rec_phys=X_rec_phys,
        X_in_phys=X_in_phys,

        # time axis
        tplot=tplot,
        dt=dt,
        ntime=ntime,

        # mode selection and structure info
        keep=keep,
        keep_idx=keep_idx,
        pairs=pairs,
        singles=singles,
        structures=structures,

        # eigen / frequency / period info for kept modes
        lam_keep=lam_keep,
        theta_keep=theta_keep,
        freq_keep=freq_keep,
        period_keep=period_keep,

        # structure-level period + colour helpers (used in row 4/5)
        period_struct=period_struct,
        period_c=period_c,
        colors=colors,
        cmap=cmap,
        normP=normP,
    )

    return fig, outputs
