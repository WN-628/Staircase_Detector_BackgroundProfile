import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

'''
This file reads a NetCDF file containing temperature profiles and plots the original temperature profile, background-only CT, and CT anomaly for a specified profile. The heatmap shows the ratio of the raw temperature gradient to the smoothed gradient, with a color scale from 0 to 2.
'''

# Read the NetCDF file
nc_path = os.path.join('prod_files', 'itp65cormat.nc')
if not os.path.isfile(nc_path):
    raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

# Open dataset
ds = nc.Dataset(nc_path, 'r')

prof_no = 1

# Find index of profile with FloatID == 1
ids = ds.variables['FloatID'][:]
idxs = np.where(ids == prof_no)[0]
if idxs.size == 0:
    available = sorted(ids.tolist())
    ds.close()
    raise ValueError(f"Profile with FloatID={prof_no} not found. Available FloatIDs: {available}")
idx = idxs[0]

# Extract data for this profile
pressure   = ds.variables['pressure'][idx]
ct_raw     = ds.variables['ct'][idx]
ct_smooth  = ds.variables['ct_bg_only'][idx]
ct_anom    = ds.variables['ct_anom'][idx]

ds.close()

# ── Compute vertical gradients using central differences ──────────────
n = len(pressure)
grad_raw = np.full(n, np.nan)
grad_smooth = np.full(n, np.nan)
for j in range(1, n-1):
    dp = pressure[j+1] - pressure[j-1]
    if dp == 0:
        continue
    grad_raw[j] = (ct_raw[j+1] - ct_raw[j-1]) / dp
    grad_smooth[j] = (ct_smooth[j+1] - ct_smooth[j-1]) / dp

colormap = 'plasma' 

# ── Create side-by-side figure with linked y-axis zoom ────────────────
fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    figsize=(12, 8),
    sharey=True    # share y-axis so depth zooms together
)

# Add a big title above both graphs
basename = os.path.splitext(os.path.basename(nc_path))[0]
fig.suptitle(f"Temperature profile for {basename} profile number {prof_no}", fontsize=16)

# ── Form the point‐wise ratio, mask zeros ─────────────────────────────
grad_ratio = np.full_like(grad_raw, np.nan)
mask       = grad_smooth != 0
grad_ratio[mask] = grad_raw[mask] / grad_smooth[mask]

# Panel 1 (left): raw CT coloured by gradient ratio + smoothed CT line
# Set color bounds between 0 and 1
norm = Normalize(vmin=0, vmax=2)
dots = ax1.scatter(
    ct_raw, pressure,
    c=grad_ratio,
    cmap=colormap,
    norm = norm,
    s=12,
    edgecolors='none'
)
ax1.plot(ct_smooth, pressure, 'k-', linewidth=2, label='Smoothed CT')
ax1.invert_yaxis()
ax1.set_xlabel('Conservative Temperature (°C)')
ax1.set_ylabel('Pressure (dbar)')
ax1.set_title('Temperature-Difference Heatmap')

# Colourbar for gradient ratio on panel 1
cbar1 = fig.colorbar(dots, ax=ax1, pad=0.02)
# cbar1.set_label('$(dCT/dz)_{raw} - (dCT/dz)_{smooth}$')

# Panel 2 (right): CT anomaly vs. pressure
ax2.plot(ct_anom, pressure, 'C0-', linewidth=2)
ax2.axvline(0, color='k', linewidth=2)
# ax2.invert_yaxis()
ax2.set_xlabel('CT Anomaly (°C)')
ax2.set_title('CT Anomaly')

plt.tight_layout()
plt.show()