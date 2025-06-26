import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# USER CONFIG ─────────────────────────────────────────
nc_path   = 'prod_files/itp65cormat.nc'
target_id = 1  # change to your FloatID

# 1) Open the file and find the profile index
ds        = Dataset(nc_path, 'r')
float_ids = ds.variables['FloatID'][:]       # shape (Nobs,)
inds      = np.where(float_ids == target_id)[0]
if inds.size == 0:
    raise ValueError(f"FloatID {target_id} not found in following float IDs: {float_ids}")
prof      = int(inds[0])

# # 2) Read masks
# dmask_sc  = ds.variables['mask_sc'][:]  # staircase mask
# mask_int = ds.variables['mask_int'][:] # interface mask
# mask_ml  = ds.variables['mask_ml'][:] # mixed-layer mask

# Helper to grab variables for this profile
def grab(varname):
    arr = ds.variables[varname][prof, ...]
    if hasattr(arr, 'mask'):
        return np.ma.array(arr)
    return np.array(arr)

# 3) Grab pressure, CT, and masks for the profile
pressure  = grab('pressure')
ct        = grab('ct')
ct_bg     = grab('ct_bg')
mask_sc   = grab('mask_sc').astype(bool)
mask_int  = grab('mask_int').astype(bool)
mask_ml   = grab('mask_ml').astype(bool)
# New: read depths of max and min temperature
depth_max_T   = grab('depth_max_T')
depth_min_T   = grab('depth_min_T')

# 4) Plot profile & detected features
plt.figure(figsize=(6, 8))
plt.plot(ct, pressure, '-', label='CT profile', linewidth=1.5)
plt.plot(ct_bg, pressure, '--', label='Smoothed CT profile', color='black')

# interface points
if mask_int.any():
    plt.scatter(ct[mask_int], pressure[mask_int],
                marker='o', s=30,
                label='Interface points', color='orange')

# mixed-layer points
if mask_ml.any():
    plt.scatter(ct[mask_ml], pressure[mask_ml],
                marker='s', s=30,
                label='Mixed-layer points', color='green')

# optional: highlight full staircase region
if mask_sc.any():
    plt.scatter(ct[mask_sc], pressure[mask_sc],
                marker='o', s=10, alpha=0.3,
                label='Staircase region', color='red')
    
# Plot horizontal lines for depth_max_T and depth_min_T
plt.axhline(depth_max_T, color='green', linestyle='--', label='Depth at max T')
plt.axhline(depth_min_T, color='blue',  linestyle='--', label='Depth at min T')

plt.gca().invert_yaxis()
plt.xlabel('Conservative Temperature (°C)')
plt.ylabel('Pressure (dbar)')
plt.legend()
plt.title(f'Float {target_id} CT Profile and Staircase Detection')
plt.tight_layout()
plt.show()
