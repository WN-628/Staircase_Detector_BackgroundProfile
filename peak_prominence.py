import xarray as xr
import numpy as np
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt

'''
This module provides a function to detect peaks or troughs in a CT anomaly profile using two methods:
1. Zero-crossing maximum absolute value method.
2. Prominence-based local peak detection.

It also provides an example usage commented out at the end.
'''

def find_step_peaks(ct_anom, min_prominence=0.005, mode='prom'):
    """
    Identify peaks/troughs per zero-crossing segment in a CT anomaly profile.

    Parameters:
      ct_anom (array-like): 1D array of CT anomalies.
      min_prominence (float): Minimum threshold for either absolute value or prominence.
      mode (str): 'zero' for zero-crossing max-|ΔT| method, 'prom' for prominence-based method.

    Returns:
      peaks_mask (np.ndarray): Boolean mask of same shape as ct_anom; True at detected peaks.
    """
    ct_anom = np.asarray(ct_anom, dtype=float)
    # Determine zero-crossing boundaries
    signs = np.sign(ct_anom)
    zc = np.where(np.diff(signs) != 0)[0] + 1
    bounds = np.concatenate(([0], zc, [len(ct_anom)]))

    peaks_mask = np.zeros_like(ct_anom, dtype=bool)
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg = ct_anom[start:end]
        if seg.size < 1:
            continue
        if mode == 'zero':
            # pick the maximum absolute value in segment
            rel_idx = np.argmax(np.abs(seg))
            if np.abs(seg[rel_idx]) >= min_prominence:
                peaks_mask[start + rel_idx] = True
        elif mode == 'prom':
            if seg.size < 3:
                continue
            # detect local peaks/troughs based on segment polarity
            if seg.mean() >= 0:
                local_peaks, _ = find_peaks(seg)
                if local_peaks.size == 0:
                    continue
                prom, _, _ = peak_prominences(seg, local_peaks)
            else:
                local_peaks, _ = find_peaks(-seg)
                if local_peaks.size == 0:
                    continue
                prom, _, _ = peak_prominences(-seg, local_peaks)
            # choose the most prominent
            i_max = np.argmax(prom)
            if prom[i_max] >= min_prominence:
                peaks_mask[start + local_peaks[i_max]] = True
        else:
            raise ValueError("mode must be 'zero' or 'prom'")
    return peaks_mask

# # === Example usage ===
# # Load CT profile #1
# ds = xr.open_dataset('prod_files/itp65cormat.nc')
# float_ids = ds.FloatID.values
# idxs = np.where(float_ids == 6)[0]
# if idxs.size == 0:
#     raise RuntimeError("FloatID 1 not found.")
# idx = int(idxs[0])
# dim0 = list(ds.dims)[0]
# ds1 = ds.isel({dim0: idx})

# # Extract CT anomaly, CT and pressure
# temp_anom = np.asarray(ds1.ct_anom.values, dtype=float)
# ct = np.asarray(ds1.ct.values, dtype=float)
# pressure = np.asarray(ds1.pressure.values, dtype=float)
# ct_bg = np.asarray(
#     ds1.ct_bg.values if 'ct_bg' in ds1 else ds1.ct_bg_only.values,
#     dtype=float
# )

# # Detect peaks using both methods
# gmin = 0.005  # example threshold
# mask_zero = find_step_peaks(temp_anom, min_prominence=gmin, mode='zero')
# # mask_prom = find_step_peaks(temp_anom, min_prominence=gmin, mode='prom')s

# # Plot both sets of peaks in one figure
# grid = plt.figure(figsize=(12, 7))
# ax0 = grid.add_subplot(121)
# ax1 = grid.add_subplot(122, sharey=ax0)

# # Left: Original vs Background CT with both peak sets
# ax0.plot(ct,    pressure,    label='Original CT')
# ax0.plot(ct_bg, pressure,    label='Background CT')
# ax0.scatter(ct[mask_zero], pressure[mask_zero], marker='x', color='blue', label='Zero method')
# # ax0.scatter(ct[mask_prom], pressure[mask_prom], marker='o', color='red', label='Prom method')
# ax0.invert_yaxis()
# ax0.set_xlabel('Conservative Temperature (°C)')
# ax0.set_ylabel('Pressure (dbar)')
# ax0.set_title('CT vs Background CT')
# ax0.grid(True)
# ax0.legend()

# # Right: CT Anomaly with both peak sets
# ax1.plot(temp_anom, pressure, label='CT Anomaly')
# ax1.axvline(0, color='gray')
# ax1.scatter(temp_anom[mask_zero], pressure[mask_zero], marker='x', color='blue', label='Zero method')
# # ax1.scatter(temp_anom[mask_prom], pressure[mask_prom], marker='o', color='red', label='Prom method')
# for i in np.where(mask_zero)[0]:
#     ax1.text(temp_anom[i], pressure[i], f'{temp_anom[i]:+.3f}', va='center', fontsize=7, color='blue')
# # for i in np.where(mask_prom)[0]:
# #     ax1.text(temp_anom[i], pressure[i], f'{temp_anom[i]:+.3f}', va='bottom', fontsize=7, color='red')
# # ax1.invert_yaxis()
# ax1.set_xlabel('CT Anomaly (°C)')
# ax1.set_title('CT Anomaly with Detected Peaks')
# ax1.grid(True)
# ax1.legend()

# # plt.tight_layout()
# # plt.show()
