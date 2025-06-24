import numpy as np

from scipy.ndimage import uniform_filter1d

# ----------------------------------------------------------------------------
# smoothing_temp.py
# Apply a running-mean smoother to CTD temperature data (background profile)
# and optionally flag small-scale anomalies.
# ----------------------------------------------------------------------------

def smooth_background(ct, dz, Theta=6.0, theta=0.04):
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