import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

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
ct          = ds.variables['ct'][idx]
ct_bg_only  = ds.variables['ct_bg_only'][idx]
ct_anom     = ds.variables['ct_anom'][idx]

ds.close()

# Create side-by-side subplots sharing the y-axis
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

# Add a big title above both graphs
basename = os.path.splitext(os.path.basename(nc_path))[0]
fig.suptitle(f"Temperature profile for {basename} profile number {prof_no}", fontsize=16)

# Left panel: CT vs background-only CT
axes[0].plot(ct, pressure, label='Original CT')
axes[0].plot(ct_bg_only, pressure, label='Background-only CT')
axes[0].invert_yaxis()
axes[0].grid(True)
axes[0].set_xlabel('Conservative Temperature (°C)')
axes[0].set_ylabel('Pressure (dbar)')
axes[0].set_title('CT vs Background-only CT')
axes[0].legend()

# Right panel: CT anomaly
axes[1].plot(ct_anom, pressure, label='CT Anomaly')
axes[1].axvline(0, color='k', linewidth=2)
# axes[1].invert_yaxis()
axes[1].grid(True)
axes[1].set_xlabel('CT Anomaly (°C)')
axes[1].set_title('CT Anomaly Profile (Residual Graph)')
axes[1].legend()

# Adjust layout to make room for the suptitle
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
