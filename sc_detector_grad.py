import numpy as np
from smooth_temp import smooth_background_fixed
from config import FIXED_RESOLUTION_METER

'''
Detect staircase gradients in CTD profiles using a gradient ratio method.
'''

def continuity(arr, num_one, num_three):
    '''
    Enforce continuity in a 1D staircase‐mask array.

    Parameters
    ----------
    arr : sequence of int
        Original detection flags per level:
            0 = no detection,
            1 = mixed‐layer detection,
            2 = interface detection.
    num_one : int
        Minimum run‐length (in grid points) for mixed‐layer runs (value==1).
        Any shorter runs of 1’s are zeroed out.
    num_three : int
        Maximum gap (in grid points) to fill between opposing detections
        (value==0 between a run of 1’s and a run of 2’s, or vice versa).
        Zero‐runs ≤ num_three that bridge 1→2 or 2→1 get relabeled as “continuity” (3).

    Returns
    -------
    numpy.ndarray of int
        Cleaned array with values:
            0 = removed/no detection,
            1 = retained mixed‐layer detection,
            2 = retained interface detection,
            3 = filled continuity gap between opposing runs.
    
    Notes:
    - The input array is expected to contain only 0, 1, 2, or 3.
    - The output array will contain 0, 1, 2, or 3.
    '''
    out = list(arr)
    n = len(out)

    # Stage 1
    for val, thresh in ((1, num_one), (3, num_three)):
        i = 0
        while i < n:
            if out[i] == val:
                start = i
                while i < n and out[i] == val:
                    i += 1
                if (i - start) < thresh:
                    for j in range(start, i):
                        out[j] = 0
            else:
                i += 1

    # Stage 2
    i = 0
    while i < n:
        if out[i] != 0:
            i += 1
            continue
        start = i
        while i < n and out[i] == 0:
            i += 1
        end = i
        run_len = end - start

        if (start > 0 and end < n
            and out[start-1] != 0
            and out[end] != 0
            and out[start-1] != out[end]
            and run_len <= num_three):
            for j in range(start, end):
                out[j] = 3

    # Stage 3
    i = 0
    while i < n:
        if out[i] == 0:
            i += 1
            continue
        rs = i
        while i < n and out[i] != 0:
            i += 1
        re = i

        runs = []
        j = rs
        while j < re:
            if out[j] in (1, 2):
                v = out[j]
                while j < re and out[j] == v:
                    j += 1
                runs.append(v)
            else:
                j += 1

        ok = False
        for k in range(len(runs)):
            cnt, last = 1, runs[k]
            for l in range(k+1, len(runs)):
                if runs[l] != last:
                    cnt += 1
                    last = runs[l]
                else:
                    break
            if cnt >= 3:
                ok = True
                break

        if not ok:
            for j in range(rs, re):
                out[j] = 0

    return np.array(out)

def detect_staircase_gradient_ratio(
    p2d,
    ct2d,
    dz,
    Theta=40.0,
    theta_anom=0.06,
    thr_iface=1.8,
    thr_ml=0.4,
    min_layers=3,
    max_sep=3.0
):
    '''
    Detect double-diffusive staircases based on the gradient ratio method,
    restricted to a vertical window defined by raw CT extremes (min/max) with margins.

    After finding the smoothed background temperature maximum, locate the first depth
    above that (shallower) where the background gradient (grad_bg) changes sign from
    negative to positive, and label that as the local minimum.

    Returns:
      mask_int, mask_ml, mask_stair, segments,
      ratio2d, ct_bg2d, ct_anom2d, background_only,
      max_ct_bg, max_p_bg, min_ct_bg, min_p_bg,
      max_p_raw, min_p_raw
    '''
    N, L = p2d.shape

    # 1) Smooth background and anomalies
    ct_bg2d, ct_anom2d, background_only = smooth_background_fixed(
        ct2d, dz, Theta=Theta, theta=theta_anom
    )

    # 2) Prepare raw CT (masked -> nan)
    ct_raw = np.where(np.ma.getmaskarray(ct2d), np.nan, ct2d)

    # 3) Compute raw and background gradients
    grad_raw = np.full((N, L), np.nan)
    grad_bg  = np.full((N, L), np.nan)
    for i in range(N):
        p_i = p2d[i]
        # raw
        raw_i = ct_raw[i]
        grad_raw[i] = np.gradient(raw_i, p_i, edge_order=2)
        # background
        bg_i = ct_bg2d[i].filled(np.nan)
        grad_bg[i]  = np.gradient(bg_i, p_i, edge_order=2)

    # 4) Initialize extrema arrays
    max_ct_bg = np.full(N, np.nan)
    max_p_bg  = np.full(N, np.nan)
    min_ct_bg = np.full(N, np.nan)
    min_p_bg  = np.full(N, np.nan)
    max_p_raw = np.full(N, np.nan)
    min_p_raw = np.full(N, np.nan)

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
                if g_down > 0 and g_up < 0 and p_i[j] <= 200.0:
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

    # 6) Compute gradient ratio
    eps = 1e-8
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio2d = np.abs(grad_raw) / np.maximum(np.abs(grad_bg), eps)
    # handle inf
    is_inf = np.isposinf(ratio2d)
    valid = ~is_inf & np.isfinite(ratio2d)
    max_val = np.max(ratio2d[valid]) if np.any(valid) else thr_iface*2
    ratio2d[is_inf] = max(max_val, thr_iface*2)
    ratio2d = np.nan_to_num(ratio2d, nan=0.0, neginf=0.0)

    # 7) Interface / mixed-layer masks
    mask_int = ratio2d > thr_iface
    mask_ml  = ratio2d < thr_ml

    # 8) Define detection window from raw extremes
    depth_margin = 25.0  # margin in meters
    lower = min_p_bg + depth_margin  # deeper than coldest raw
    upper = max_p_bg - depth_margin  # shallower than warmest raw

    # 9) Detect staircases within window
    mask_stair = np.zeros_like(mask_int, dtype=bool)
    segments   = [[] for _ in range(N)]
    for i in range(N):
        p_i = p2d[i]
        lo, hi = lower[i], upper[i]
        if np.isnan(lo) or np.isnan(hi) or lo > hi:
            continue
        region = (p_i >= lo) & (p_i <= hi)
        idxs = np.nonzero((mask_int[i] | mask_ml[i]) & region)[0]
        if idxs.size < min_layers:
            continue
        splits = np.where(np.diff(p_i[idxs]) > max_sep)[0] + 1
        groups = np.split(idxs, splits)
        for grp in groups:
            if grp.size < min_layers:
                continue
            # look for I-M-I or M-I-M
            if any(
                (mask_int[i,k] and mask_ml[i,k1] and mask_int[i,k2]) or
                (mask_ml[i,k] and mask_int[i,k1] and mask_ml[i,k2])
                for k,k1,k2 in zip(grp, grp[1:], grp[2:])
            ):
                s, e = grp[0], grp[-1]
                segments[i].append((s,e))
                mask_stair[i,s:e+1] = True

    # 10) Prune
    mask_int &= mask_stair
    mask_ml  &= mask_stair

    return (
        mask_int, mask_ml, mask_stair, segments,
        ratio2d, ct_bg2d, ct_anom2d, background_only,
        max_p_bg, min_p_bg
    )

def filter_staircase_masks_local(p2d, ct2d, dz,
                                 mask_int, mask_ml,
                                 Theta=40.0, theta_anom=0.06,
                                 thr_iface=1.8, thr_ml=0.4,
                                 ml_min_depth=1.0,
                                 cl_points=3,
                                 resolution=FIXED_RESOLUTION_METER):
    """
    1) Compute background-smoothed CT for profiles.
    2) For each detected interface or mixed-layer region,
       calculate the average gradient across that entire layer:
         raw_grad = ΔCT / Δz over region boundaries,
         bg_grad  = ΔCT_bg / Δz over same.
    3) Keep only regions whose average ratio satisfies:
         interfaces: ratio > thr_iface
         mixed-layers: ratio < thr_ml
    4) Enforce continuity via the `continuity` function
       to prune/fill small gaps and remove isolated segments.

    Returns cleaned boolean masks: (mask_int_f, mask_ml_f, mask_sc_f)
    """
    N, L = ct2d.shape

    # infer grid resolution if not provided
    if resolution is None:
        resolution = np.median(np.diff(dz))
    ml_min_grid = int(np.ceil(ml_min_depth / resolution))

    # smooth background once
    ct_bg2d, _, _ = smooth_background_fixed(ct2d, dz,
                                            Theta=Theta,
                                            theta=theta_anom)

    # prepare filtered masks
    mask_int_f = np.zeros_like(mask_int, dtype=bool)
    mask_ml_f  = np.zeros_like(mask_ml,  dtype=bool)

    eps = 1e-8
    for i in range(N):
        # process interface regions
        idxs = np.nonzero(mask_int[i])[0]
        if idxs.size > 0:
            # split contiguous runs
            breaks = np.where(np.diff(idxs) > 1)[0] + 1
            runs = np.split(idxs, breaks)
            for run in runs:
                j0, j1 = run[0], run[-1]
                # must have neighbors for boundary diff
                if j0 < 1 or j1 > L-2:
                    continue
                z_lo, z_hi = p2d[i, j0-1], p2d[i, j1+1]
                raw = ct2d[i]
                grad_raw = (raw[j1+1] - raw[j0-1]) / (z_hi - z_lo)
                bg  = ct_bg2d[i]
                grad_bg  = (bg[j1+1] - bg[j0-1]) / (z_hi - z_lo)
                ratio = abs(grad_raw) / max(abs(grad_bg), eps)
                if ratio > thr_iface:
                    mask_int_f[i, j0:j1+1] = True

        # process mixed-layer regions similarly
        idxs = np.nonzero(mask_ml[i])[0]
        if idxs.size > 0:
            breaks = np.where(np.diff(idxs) > 1)[0] + 1
            runs = np.split(idxs, breaks)
            for run in runs:
                j0, j1 = run[0], run[-1]
                if j0 < 1 or j1 > L-2:
                    continue
                z_lo, z_hi = p2d[i, j0-1], p2d[i, j1+1]
                raw = ct2d[i]
                grad_raw = (raw[j1+1] - raw[j0-1]) / (z_hi - z_lo)
                bg  = ct_bg2d[i]
                grad_bg  = (bg[j1+1] - bg[j0-1]) / (z_hi - z_lo)
                ratio = abs(grad_raw) / max(abs(grad_bg), eps)
                if ratio < thr_ml:
                    mask_ml_f[i, j0:j1+1] = True

    # continuity cleaning
    clean_int = np.zeros_like(mask_int, dtype=bool)
    clean_ml  = np.zeros_like(mask_ml,  dtype=bool)
    mask_sc   = np.zeros_like(mask_int, dtype=bool)

    for i in range(N):
        arr = np.zeros(L, dtype=int)
        arr[mask_ml_f[i]]  = 1
        arr[mask_int_f[i]] = 2

        cleaned = continuity(arr, ml_min_grid, cl_points)

        clean_ml[i]  = (cleaned == 1)
        clean_int[i] = (cleaned == 2)
        mask_sc[i]   = (cleaned > 0)

    return clean_int, clean_ml, mask_sc
