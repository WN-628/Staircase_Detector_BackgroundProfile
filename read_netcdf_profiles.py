import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os

'''
This script reads temperature profiles from a NetCDF file, plots them with mixed-layer and interface highlights,
and labels each profile with its FloatID.
'''

# ── USER CONFIG ─────────────────────────────────────────
nc_path = 'prod_files/itp65cormat.nc'  # path to your NetCDF file
# ────────────────────────────────────────────────────────

# 1) Open dataset and extract variables
ds = nc.Dataset(nc_path, 'r')
pressure_var = ds.variables['pressure']
ct_var = ds.variables['ct']
mask_ml_var = ds.variables['mask_ml']
mask_int_var = ds.variables['mask_int']
floatid_var = ds.variables.get('FloatID')

# Number of profiles\ 
n_profiles = len(pressure_var)

# Load raw profiles and IDs
profiles = [np.array(ct_var[i]) for i in range(n_profiles)]
pressures = [np.array(pressure_var[i]) for i in range(n_profiles)]
masks_ml = [np.array(mask_ml_var[i], dtype=bool) for i in range(n_profiles)]
masks_int = [np.array(mask_int_var[i], dtype=bool) for i in range(n_profiles)]

# Handle FloatID sorting if available
if floatid_var is not None:
    floatids = np.array([int(floatid_var[i]) for i in range(n_profiles)])
    sort_idx = np.argsort(floatids)
    profiles = [profiles[i] for i in sort_idx]
    pressures = [pressures[i] for i in sort_idx]
    masks_ml = [masks_ml[i] for i in sort_idx]
    masks_int = [masks_int[i] for i in sort_idx]
    floatids = floatids[sort_idx]
else:
    floatids = np.arange(n_profiles) + 1

# Determine horizontal separation based on CT range
all_ct = np.concatenate(profiles)
ct_min, ct_max = np.nanmin(all_ct), np.nanmax(all_ct)
separation = (ct_max - ct_min) * 1.2 if ct_max > ct_min else 1.0

# Plot setup
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each profile with mixed-layer and interface highlights
for idx, (t, p, m_ml, m_int) in enumerate(zip(profiles, pressures, masks_ml, masks_int)):
    offset = idx * separation
    t_shifted = t + offset

    # Draw profile line
    ax.plot(t_shifted, p, '-', color='gray', linewidth=1)

    # Mixed-layer markers
    if m_ml.any():
        ax.scatter(t_shifted[m_ml], p[m_ml], marker='s', s=20,
                   color='green', label='Mixed Layer' if idx == 0 else None)

    # Interface markers
    if m_int.any():
        ax.scatter(t_shifted[m_int], p[m_int], marker='o', s=20,
                   color='red', label='Interface' if idx == 0 else None)

    # Label each profile
    profile_label = f'ID {floatids[idx]}'
    y_top = np.nanmin(p)
    x_label = np.nanmean(t_shifted)
    depth_range = np.nanmax(p) - y_top
    y_text = y_top - 0.02 * depth_range
    ax.text(x_label, y_text, profile_label,
            ha='center', va='bottom', fontsize=8)

# Finalize plot
ax.invert_yaxis()
ax.set_xlabel('CT (°C) + offset')
ax.set_ylabel('Pressure (dbar)')
ax.set_title(f'Temperature Profiles with Mixed Layers & Interfaces')
ax.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()
