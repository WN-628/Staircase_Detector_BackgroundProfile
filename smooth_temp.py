import numpy as np

from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_coeffs
from scipy.interpolate import UnivariateSpline

'''
smoothing_temp.py
Apply a running-mean smoother to CTD temperature data (background profile) and optionally flag small-scale anomalies.
'''

def smooth_background_fixed(ct, dz, Theta=8.0, theta=0.04):
    """
    Compute a smoothed (background) temperature profile and anomalies.

    Parameters
    ----------
    ct : numpy.ma.MaskedArray
        2D array of Conservative Temperature with shape (n_profiles, n_levels).
    dz : float
        Vertical resolution in meters (e.g., 1.0 m).
    Theta : float, optional
        Smoothing window width in meters (default 6.0).
    theta : float, optional
        Anomaly threshold in degrees Celsius (default 0.04).

    Returns
    -------
    ct_bg : numpy.ma.MaskedArray
        Smoothed background temperature profiles.
    ct_anom : numpy.ma.MaskedArray
        Small-scale temperature anomalies (ct - ct_bg).
    background_only : numpy.ma.MaskedArray
        Background profile masked where anomalies exceed threshold.
    """
    # number of points for the boxcar window
    window_pts = int(round(Theta / dz))
    if window_pts % 2 == 0:
        window_pts += 1  # ensure symmetry

    # prepare output arrays
    ct_bg = np.full_like(ct, np.nan)

    # smooth each profile
    for i in range(ct.shape[0]):
        valid = ~ct.mask[i]
        if not valid.any():
            continue

        raw = ct.data[i, valid]
        smoothed = uniform_filter1d(raw, size=window_pts, mode='nearest')
        ct_bg[i, valid] = smoothed

    # reapply mask
    ct_bg = np.ma.masked_where(ct.mask, ct_bg)

    # compute anomalies
    ct_anom = ct - ct_bg

    # mask background where anomaly > threshold
    bg_mask = np.abs(ct_anom) < theta
    background_only = np.ma.masked_where(~bg_mask, ct_bg)

    return ct_bg, ct_anom, background_only

def smooth_background_asg(ct, dz,
                        Theta=6.0, theta=0.04,
                        M_min=1, polyorder=2,
                        g_pctile=90,
                        Mf_smooth_pts=6,
                        large_grad_factor=1.5):
    """
    Adaptive Savitzky–Golay smoothing plus half-width smoothing, with override to smooth large steps.

    Params
    ------
    large_grad_factor : float
        Multiply g0 by this factor to decide “this is a *large* step.”  E.g. 2.0 means
        any |dT/dp| > 2*g0 will be forced to a full window.
    """

    # 1) compute M_max from Theta/dz
    window_pts = int(round(Theta / dz))
    if window_pts % 2 == 0:
        window_pts += 1
    M_max = window_pts // 2

    n_prof, n_lev = ct.shape
    ct_bg = np.full_like(ct, np.nan)

    for i in range(n_prof):
        valid = ~ct.mask[i]
        if not valid.any(): continue

        raw = ct.data[i, valid]
        N   = raw.size
        p   = np.arange(N) * dz

        # 2) local gradient and characteristic scale
        g  = np.gradient(raw, p)
        g0 = np.percentile(np.abs(g), g_pctile)

        # 3) continuous Mf and smooth it
        Mf = M_min + (M_max - M_min)*np.exp(-(np.abs(g)/g0)**2)
        Mf_s = uniform_filter1d(Mf, size=Mf_smooth_pts, mode='nearest')

        # 4) round to integer half-widths
        Mi = np.clip(np.round(Mf_s).astype(int), M_min, M_max)

        # ─── *** NEW OVERRIDE *** ───
        # force full smoothing over *really* large jumps
        large = np.abs(g) > (large_grad_factor * g0)
        Mi[large] = M_max

        # 5) keep windows from overrunning the ends
        dist_start = np.arange(N)
        dist_end   = dist_start[::-1]
        Mi = np.minimum(Mi, np.minimum(dist_start, dist_end))

        # 6) do the SG fit point-by-point
        sm = np.empty_like(raw)
        for j in range(N):
            m  = Mi[j]
            wl = 2*m + 1
            if wl <= polyorder:
                sm[j] = raw[j]
            else:
                c    = savgol_coeffs(wl, polyorder, deriv=0)
                idx  = np.arange(j-m, j+m+1)
                sm[j] = np.dot(c, raw[idx])

        ct_bg[i, valid] = sm

    # 7) mask & compute anomalies exactly as before
    ct_bg         = np.ma.masked_where(ct.mask, ct_bg)
    ct_anom       = ct - ct_bg
    bg_mask       = np.abs(ct_anom) < theta
    background_only = np.ma.masked_where(~bg_mask, ct_bg)

    return ct_bg, ct_anom, background_only


# ----------------------------------------------------------------------------# smooth_background_by_depth
# ----------------------------------------------------------------------------

def smooth_background_by_depth(ct, depth, dz, depth_threshold=310,
                                asg_kwargs=None, fixed_kwargs=None):
    """
    Choose smoothing method based on maximum profile depth.

    Parameters
    ----------
    ct : numpy.ma.MaskedArray, shape (n_profiles, n_levels)
        Conservative Temperature profiles (masked where invalid).
    depth : ndarray, shape (n_profiles, n_levels)
        Depth or pressure corresponding to ct.
    dz : float
        Vertical resolution (m).
    depth_threshold : float, optional
        If max(depth[i]) <= this, use ASG smoothing; otherwise use fixed smoothing.
    asg_kwargs : dict, optional
        Keyword args passed to smooth_background_asg (adaptive SG).
    fixed_kwargs : dict, optional
        Keyword args passed to smooth_background_fixed (boxcar or fixed smoother).

    Returns
    -------
    ct_bg : numpy.ma.MaskedArray
        Background profiles.
    ct_anom : numpy.ma.MaskedArray
        Anomalies (ct - ct_bg).
    background_only : numpy.ma.MaskedArray
        Background masked where anomalies exceed theta.
    """
    # default parameter dicts
    asg_kwargs   = asg_kwargs or {}
    fixed_kwargs = fixed_kwargs or {}

    n_prof, n_lev = ct.shape
    ct_bg         = np.ma.masked_all_like(ct)
    ct_anom       = np.ma.masked_all_like(ct)
    background_only = np.ma.masked_all_like(ct)

    for i in range(n_prof):
        p_i = depth[i]
        ct_i = ct[i:i+1]         # preserve 2D shape for the called funcs
        if np.nanmax(p_i) <= depth_threshold:
            bg, anom, bg_only = smooth_background_asg(ct_i, dz, **asg_kwargs)
        else:
            bg, anom, bg_only = smooth_background_fixed(ct_i, dz, **fixed_kwargs)

        # assign results back
        ct_bg[i]         = bg[0]
        ct_anom[i]       = anom[0]
        background_only[i] = bg_only[0]

    return ct_bg, ct_anom, background_only