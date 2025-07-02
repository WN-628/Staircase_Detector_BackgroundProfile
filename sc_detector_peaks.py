import numpy as np
from smooth_temp import smooth_background_by_depth
from peak_prominence import find_step_peaks

def detect_staircase_peaks(
    p2d,
    ct2d,
    dz,
    depth_threshold=310.0,
    asg_kwargs=None,
    fixed_kwargs=None,
    min_prominence=0.003,
    margin=25.0,
    min_layers=3
):
    """
    Detect double-diffusive staircases based on zero-crossing peak method,
    restricted to ±margin meters around the coldest (minimum CT) depth.

    Parameters
    ----------
    p2d : ndarray, shape (n_profiles, n_levels)
        Depth or pressure [m or dbar] arrays.
    ct2d : numpy.ma.MaskedArray, shape (n_profiles, n_levels)
        Conservative Temperature profiles (masked where invalid).
    dz : float
        Vertical resolution in meters (unused here but kept for signature).
    depth_threshold : float, optional
        Threshold for choosing smoothing method in smooth_background_by_depth.
    asg_kwargs : dict, optional
        Passed to smooth_background_by_depth for adaptive smoothing.
    fixed_kwargs : dict, optional
        Passed to smooth_background_by_depth for fixed smoothing.
    min_prominence : float, optional
        Minimum prominence (|°C|) to accept a peak/trough in CT anomaly.
    margin : float, optional
        Vertical window [m] around the coldest-CT depth to restrict detections.

    Returns
    -------
    mask_int : ndarray(bool)
        True where interface (negative→positive anomaly) detected.
    mask_ml : ndarray(bool)
        True where mixed layer (positive→negative anomaly) detected.
    mask_stair : ndarray(bool)
        True where either interface or mixed-layer segments occur.
    segments : list of lists
        Per-profile list of (start_idx, end_idx, 'int'|'ml') tuples.
    peaks2d : ndarray(bool)
        True at detected peaks/troughs in CT anomaly.
    ct_bg2d : numpy.ma.MaskedArray
        Smoothed CT background profiles.
    ct_anom2d : numpy.ma.MaskedArray
        CT anomalies (ct2d - ct_bg2d).
    background_only : numpy.ma.MaskedArray
        Background masked where anomalies exceed detection threshold.
    max_p_bg : ndarray(float)
        Depths of maximum CT in background per profile.
    min_p_bg : ndarray(float)
        Depths of minimum CT in background per profile (coldest layer).
    """
    # 1) Smooth background and compute anomalies
    ct_bg2d, ct_anom2d, background_only = \
        smooth_background_by_depth(ct2d, p2d, dz,
                                    depth_threshold=depth_threshold,
                                    asg_kwargs=asg_kwargs,
                                    fixed_kwargs=fixed_kwargs)

    # 2) Prepare raw CT array
    ct_raw = np.where(np.ma.getmaskarray(ct2d), np.nan, ct2d)

    N, L = ct2d.shape
    mask_int   = np.zeros((N, L), dtype=bool)
    mask_ml    = np.zeros((N, L), dtype=bool)
    mask_stair = np.zeros((N, L), dtype=bool)
    peaks2d    = np.zeros((N, L), dtype=bool)
    segments = [[] for _ in range(N)]

    # 3) Compute background gradient
    grad_bg = np.full((N, L), np.nan)
    for i in range(N):
        p_i  = p2d[i]
        bg_i = ct_bg2d[i].filled(np.nan)
        grad_bg[i] = np.gradient(bg_i, p_i, edge_order=2)

    # 4) Find extrema in raw and background CT
    max_p_raw = np.full(N, np.nan)
    min_p_raw = np.full(N, np.nan)
    max_ct_bg  = np.full(N, np.nan)
    max_p_bg   = np.full(N, np.nan)
    min_ct_bg  = np.full(N, np.nan)
    min_p_bg   = np.full(N, np.nan)

    # 5) Find extrema per profile
    for i in range(N):
        bg = ct_bg2d[i].filled(np.nan)
        raw = ct_raw[i]
        p_i = p2d[i]
        # raw CT extremes
        if not np.all(np.isnan(raw)):
            idx_max_raw = np.nanargmax(raw)
            idx_min_raw = np.nanargmin(raw)
            max_p_raw[i] = p_i[idx_max_raw]
            min_p_raw[i] = p_i[idx_min_raw]
        # smoothed background max
        if not np.all(np.isnan(bg)):
            idx_max_bg = np.nanargmax(bg)
            max_ct_bg[i] = bg[idx_max_bg]
            max_p_bg[i]  = p_i[idx_max_bg]
            # search upward (shallower: decreasing index) for grad_bg sign change neg->pos
            found_sign_change = False
            for j in range(idx_max_bg-1, 0, -1):
                g_down = grad_bg[i, j+1]
                g_up   = grad_bg[i, j]
                if np.isnan(g_down) or np.isnan(g_up):
                    continue
                # detect crossing: gradient negative below and positive above,
                # but only in upper 200 m
                if g_down > 0 and g_up < 0 and p_i[j] <= 250.0:
                    min_ct_bg[i] = bg[j]
                    min_p_bg[i]  = p_i[j]
                    found_sign_change = True
                    break
            # fallback: if no sign change, use raw CT minimum depth
            if not found_sign_change:
                raw_vals = raw
                # ensure there is valid raw data
                if not np.all(np.isnan(raw_vals)):
                    idx_min_raw = np.nanargmin(raw_vals)
                    min_ct_bg[i] = raw_vals[idx_min_raw]
                    min_p_bg[i]  = p_i[idx_min_raw]

    # 6) Detect peaks/troughs in each CT anomaly
    for i in range(N):
        anom = ct_anom2d[i].filled(np.nan)
        peaks2d[i] = find_step_peaks(anom,
                                        min_prominence=min_prominence,
                                        mode='zero')
    
    # 6) Build segments from adjacent peaks only
    segments = [[] for _ in range(N)]
    for i in range(N):
        # if you want to restrict to ±margin, compute region here:
        region = (p2d[i] >= min_p_bg[i] + margin) & (p2d[i] <= max_p_bg[i] - margin)
        idxs = np.where(peaks2d[i] & region)[0]

        for start, end in zip(idxs[:-1], idxs[1:]):
            v0 = ct_anom2d[i].filled(np.nan)[start]
            v1 = ct_anom2d[i].filled(np.nan)[end]
            if np.isnan(v0) or np.isnan(v1):
                continue
            if v0 < 0 < v1:
                segments[i].append((start, end, 'int'))
            elif v0 > 0 > v1:
                segments[i].append((start, end, 'ml'))

    # 7) Paint masks strictly from those segments
    mask_int[:] = False
    mask_ml[:]  = False
    for i, segs in enumerate(segments):
        for start, end, kind in segs:
            if kind == 'int':
                mask_int[i, start:end+1] = True
            else:
                mask_ml[i, start:end+1] = True

    mask_stair = mask_int | mask_ml

    return (
        mask_int,
        mask_ml,
        mask_stair,
        segments,
        peaks2d,
        ct_bg2d,
        ct_anom2d,
        background_only,
        max_p_bg,
        min_p_bg
    )
