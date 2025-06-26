import numpy as np
from smooth_temp import smooth_background

def detect_staircase_gradient_ratio(
    p2d,
    ct2d,
    dz,
    Theta=6.0,
    theta_anom=0.04,
    thr_iface=1.9,
    thr_ml=0.4,
    min_layers=3,
    max_sep=15.0,
):
    """
    (…docstring unchanged…)
    """
    N, L = p2d.shape

    # 1) Smooth and compute background & anomalies
    ct_bg2d, ct_anom2d, background_only = smooth_background(
        ct2d, dz, Theta=Theta, theta=theta_anom
    )

    # 2) Compute gradients per profile
    grad_raw = np.full((N, L), np.nan)
    grad_bg  = np.full((N, L), np.nan)
    for i in range(N):
        p_i    = p2d[i]
        ct_i   = ct2d[i].filled(np.nan)
        bg_i   = ct_bg2d[i].filled(np.nan)
        grad_raw[i] = np.gradient(ct_i, p_i, edge_order=2)
        grad_bg[i]  = np.gradient(bg_i, p_i, edge_order=2)

    # 3) Compute ratio, suppress warnings, and handle infinities
    eps = 1e-8
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio2d = np.abs(grad_raw) / np.maximum(np.abs(grad_bg), eps)

    # 3a) replace +∞ with a large but finite value (at least 2× interface threshold)
    is_posinf = np.isposinf(ratio2d)
    finite     = ~is_posinf & np.isfinite(ratio2d)
    if np.any(finite):
        max_fin = np.max(ratio2d[finite])
    else:
        max_fin = thr_iface * 2
    ratio2d[is_posinf] = max(max_fin, thr_iface * 2)

    # 3b) zero out NaNs and −∞ only
    ratio2d = np.nan_to_num(ratio2d, nan=0.0, neginf=0.0)

    # 4) Build interface and mixed‐layer masks
    mask_int = ratio2d > thr_iface
    mask_ml  = ratio2d < thr_ml

    # 5) Detect runs of alternating layers
    mask_stair = np.zeros_like(mask_int, dtype=bool)
    segments   = [[] for _ in range(N)]

    for i in range(N):
        pi       = p2d[i]
        mask_any = mask_int[i] | mask_ml[i]
        idxs_any = np.nonzero(mask_any)[0]
        if idxs_any.size < min_layers:
            continue

        # split when depth‐gap > max_sep
        splits = np.where(np.diff(pi[idxs_any]) > max_sep)[0] + 1
        groups = np.split(idxs_any, splits)

        for grp in groups:
            if grp.size < min_layers:
                continue
            # look for any 3‐point alternating pattern
            found = False
            for k in range(grp.size - min_layers + 1):
                j0, j1, j2 = grp[k], grp[k+1], grp[k+2]
                if ((mask_int[i, j0] and mask_ml[i, j1] and mask_int[i, j2]) or
                    (mask_ml[i, j0] and mask_int[i, j1] and mask_ml[i, j2])):
                    found = True
                    break
            if not found:
                continue

            start, end = grp[0], grp[-1]
            segments[i].append((start, end))
            mask_stair[i, start:end+1] = True

    return mask_int, mask_ml, mask_stair, segments, ratio2d, ct_bg2d, ct_anom2d, background_only
