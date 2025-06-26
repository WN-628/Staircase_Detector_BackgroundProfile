import numpy as np
from smooth_temp import smooth_background
from peak_prominence import find_step_peaks

def detect_staircase_peaks_ratio(
    p2d, ct2d, dz,
    Theta=6.0,
    theta_anom=0.04,
    min_prominence=0.0001,   # MATLAB’s 1.8*0.0015
    thr_iface=1.2,
    thr_ml=0.6,
    min_layers=2,
    max_sep=25.0,
    peak_mode='zero',
):
    """
    Run peak-and-gradient-ratio staircase detection on each row of p2d, ct2d.

    Parameters
    ----------
    p2d : numpy.ma.MaskedArray, shape (N, L)
        Pressure profiles, one per row.
    ct2d: numpy.ma.MaskedArray, shape (N, L)
        Conservative Temperature profiles.
    dz : float
        Vertical spacing in meters.
    Theta : float
        Smoothing window width for background (m).
    theta_anom : float
        Anomaly threshold for smooth_background (°C).
    min_prominence : float
        ΔT prominence threshold for peak detection (°C).
    thr_iface : float
        Ratio threshold above which a peak is an interface.
    thr_ml : float
        Ratio threshold below which a peak is a mixed-layer core.
    min_layers : int
        Min alternating layers (peaks) to call a staircase.
    max_sep : float
        Max vertical separation between successive interface peaks (m).
    peak_mode : str
        'zero' or 'prom' for find_step_peaks mode.

    Returns
    -------
    mask_int   : bool array (N, L)
        Interface-peak locations for each profile.
    mask_ml    : bool array (N, L)
        Mixed-layer-core peaks for each profile.
    mask_stair : bool array (N, L)
        Full staircase ranges (filled between first/last interface).
    segments   : list of lists
        `segments[i]` is the list of `(start_idx, end_idx)` tuples for profile `i`.
    ratio2d    : float array (N, L)
        Gradient ratio for every point in every profile.
    ct_bg2d    : numpy.ma.MaskedArray (N, L)
        Smoothed background temperature profiles.
    ct_anom2d  : numpy.ma.MaskedArray (N, L)
        Small-scale temperature anomalies.
    bg_only2d  : numpy.ma.MaskedArray (N, L)
        Background-only profile masked where anomalies exceed threshold.
    """
    N, L = p2d.shape
    ct_bg2d, ct_anom2d, bg_only2d = smooth_background(ct2d, dz,
                                                      Theta=Theta,
                                                      theta=theta_anom)

    mask_int   = np.zeros((N, L), dtype=bool)
    mask_ml    = np.zeros((N, L), dtype=bool)
    mask_stair = np.zeros((N, L), dtype=bool)
    ratio2d    = np.zeros((N, L), dtype=float)
    segments   = [None] * N

    for i in range(N):
        p      = p2d[i]
        ct     = ct2d[i]
        ct_bg  = ct_bg2d[i]
        anom   = ct_anom2d[i]

        # detect peaks in residual
        peaks_mask = find_step_peaks(anom,
                                     min_prominence=min_prominence,
                                     mode=peak_mode)
        idxs = np.flatnonzero(peaks_mask)
        if idxs.size == 0:
            segments[i] = []
            continue

        # manual central‐difference gradients
        ct_vals    = ct.filled(np.nan)
        ct_bg_vals = ct_bg.filled(np.nan)
        n = p.size
        grad_raw = np.full(n, np.nan)
        grad_bg  = np.full(n, np.nan)
        for j in range(1, n-1):
            dp = p[j+1] - p[j-1]
            if dp == 0:
                continue
            grad_raw[j] = (ct_vals[j+1]    - ct_vals[j-1])    / dp
            grad_bg[j]  = (ct_bg_vals[j+1] - ct_bg_vals[j-1]) / dp

        eps   = 1e-8
        ratio = np.abs(grad_raw) / np.maximum(np.abs(grad_bg), eps)
        ratio2d[i] = ratio

        # group interface‐type peaks into staircase segments
        is_iface   = ratio[idxs] >  thr_iface
        iface_idxs = idxs[is_iface]
        splits     = np.where(np.diff(p[iface_idxs]) > max_sep)[0] + 1
        groups     = np.split(iface_idxs, splits)

        segs = []
        for grp in groups:
            if grp.size < min_layers:
                continue
            types = np.where(ratio[grp] > thr_iface, 'I', 'M')
            if np.all(types[:-1] != types[1:]):
                segs.append((grp[0], grp[-1]))
        segments[i] = segs

        # Stage 1: classify by residual‐sign only
        resid = anom.filled(np.nan)
        ml_cands  = []
        int_cands = []
        for start_idx, end_idx in segs:
            if resid[start_idx] > 0 and resid[end_idx] < 0:
                ml_cands.append((start_idx, end_idx))
            elif resid[start_idx] < 0 and resid[end_idx] > 0:
                int_cands.append((start_idx, end_idx))

        # Stage 2: gradient‐ratio filter
        for start_idx, end_idx in ml_cands:
            if np.nanmin(ratio[start_idx:end_idx+1]) < thr_ml:
                mask_ml[i, start_idx:end_idx+1]    = True
                mask_stair[i, start_idx:end_idx+1] = True

        for start_idx, end_idx in int_cands:
            if np.nanmax(ratio[start_idx:end_idx+1]) > thr_iface:
                mask_int[i, start_idx:end_idx+1]   = True
                mask_stair[i, start_idx:end_idx+1] = True

    return mask_int, mask_ml, mask_stair, segments, ratio2d, ct_bg2d, ct_anom2d, bg_only2d