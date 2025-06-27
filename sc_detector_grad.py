import numpy as np
from smooth_temp import smooth_background_fixed

# def detect_staircase_gradient_ratio(
#     p2d,
#     ct2d,
#     dz,
#     Theta=30.0,
#     theta_anom=0.04,
#     thr_iface=1.8,
#     thr_ml=0.4,
#     min_layers=3,
#     max_sep=2.0,
#     depth_margin=25.0,
# ):
#     """
#     Detect double-diffusive staircases based on the gradient ratio method, restricted to the vertical window
#     that spans from 25 m deeper than the first local minimum (blue line) down to 25 m shallower than the
#     maximum background temperature depth (green line).

#     Returns:
#         mask_int (bool array): interface points within detected staircases
#         mask_ml (bool array): mixed-layer points within detected staircases
#         mask_stair (bool array): all staircase points
#         segments (list): list of (start, end) index tuples for each staircase segment
#         ratio2d (float array): gradient ratio array
#         ct_bg2d (masked array): smoothed background temperature
#         ct_anom2d (masked array): temperature anomalies
#         background_only (masked array): background where anomalies below threshold
#         max_ct_bg (float array): maximum background temperature per profile
#         max_p (float array): pressure at maximum background temperature per profile (green line)
#         min_ct_bg (float array): first local minimum of background temperature above the maximum
#         min_p (float array): pressure at the first local minimum per profile (blue line)
#     """
#     N, L = p2d.shape

#     # 1) Smooth and compute background & anomalies
#     ct_bg2d, ct_anom2d, background_only = smooth_background(
#         ct2d, dz, Theta=Theta, theta=theta_anom
#     )

#     # 2) Compute raw and background gradients
#     grad_raw = np.full((N, L), np.nan)
#     grad_bg = np.full((N, L), np.nan)
#     for i in range(N):
#         p_i = p2d[i]
#         ct_i = ct2d[i].filled(np.nan)
#         bg_i = ct_bg2d[i].filled(np.nan)
#         grad_raw[i] = np.gradient(ct_i, p_i, edge_order=2)
#         grad_bg[i] = np.gradient(bg_i, p_i, edge_order=2)

#     # 3) Gradient ratio calculation
#     eps = 1e-8
#     with np.errstate(divide='ignore', invalid='ignore'):
#         ratio2d = np.abs(grad_raw) / np.maximum(np.abs(grad_bg), eps)
#     is_posinf = np.isposinf(ratio2d)
#     finite = ~is_posinf & np.isfinite(ratio2d)
#     max_val = np.max(ratio2d[finite]) if np.any(finite) else thr_iface * 2
#     ratio2d[is_posinf] = max(max_val, thr_iface * 2)
#     ratio2d = np.nan_to_num(ratio2d, nan=0.0, neginf=0.0)

#     # 4) Initial masks
#     mask_int = ratio2d > thr_iface
#     mask_ml = ratio2d < thr_ml

#     # 5) Define detection region: between (min_p + depth_margin) and (max_p - depth_margin)
#     # blue line (min_p) + margin: deeper bound
#     # green line (max_p) - margin: shallower bound
#     region_lower = min_p + depth_margin
#     region_upper = max_p - depth_margin

#     # 6) Staircase detection within region
#     mask_stair = np.zeros_like(mask_int, dtype=bool)
#     segments = [[] for _ in range(N)]
#     for i in range(N):
#         p_i = p2d[i]
#         rl, ru = region_lower[i], region_upper[i]
#         if np.isnan(rl) or np.isnan(ru):
#             continue
#         # region mask: levels with p between rl and ru
#         region_mask = (p_i >= rl) & (p_i <= ru)

#         active = (mask_int[i] | mask_ml[i]) & region_mask
#         idxs = np.nonzero(active)[0]
#         if idxs.size < min_layers:
#             continue
#         splits = np.where(np.diff(p_i[idxs]) > max_sep)[0] + 1
#         groups = np.split(idxs, splits)
#         for grp in groups:
#             if grp.size < min_layers:
#                 continue
#             valid = any(
#                 (mask_int[i, grp[k]] and mask_ml[i, grp[k+1]] and mask_int[i, grp[k+2]]) or
#                 (mask_ml[i, grp[k]] and mask_int[i, grp[k+1]] and mask_ml[i, grp[k+2]])
#                 for k in range(grp.size - 2)
#             )
#             if not valid:
#                 continue
#             start, end = grp[0], grp[-1]
#             segments[i].append((start, end))
#             mask_stair[i, start:end+1] = True

#     # 7) Prune masks outside detected staircases
#     mask_int &= mask_stair
#     mask_ml &= mask_stair

#     return (
#         mask_int, mask_ml, mask_stair, segments,
#         ratio2d, ct_bg2d, ct_anom2d, background_only,
#         max_p, min_p
#     )

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
    """
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
    """
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


