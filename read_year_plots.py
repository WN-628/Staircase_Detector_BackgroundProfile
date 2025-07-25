import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

''' This script reads temperature profiles from netCDF files for a specified year and plots them with mixed-layer and interface markers in waterfall plots. '''

# ── CONFIG ─────────────────────────────────────────
folder = 'prod_files'
target_year = 2018
# ───────────────────────────────────────────────────

def load_profiles_year(nc_file, target_year):
    ds = nc.Dataset(nc_file, 'r')
    dates = ds.variables['dates'][:]
    years = np.array([
        datetime.utcfromtimestamp(ts).year if not np.isnan(ts) else 0
        for ts in dates
    ])
    indices = np.where(years == target_year)[0]

    if indices.size == 0:
        ds.close()
        return []

    pressure = ds.variables['pressure']
    ct = ds.variables['ct']
    mask_ml = ds.variables['mask_ml']
    mask_int = ds.variables['mask_int']
    floatids = ds.variables.get('FloatID')

    basename = os.path.basename(nc_file)
    itp_tag = basename.split('cormat')[0].upper()

    profiles = []
    for i in indices:
        try:
            t = np.array(ct[i])
            p = np.array(pressure[i])
            ml = np.array(mask_ml[i], dtype=bool)
            inte = np.array(mask_int[i], dtype=bool)
            fid = int(floatids[i]) if floatids is not None else i + 1
            ts = dates[i]
            dt = datetime.utcfromtimestamp(ts) if not np.isnan(ts) else None
            profiles.append((t, p, ml, inte, fid, itp_tag, dt))
        except Exception as e:
            print(f"⚠️ Skipping corrupt profile {i} in {basename}: {e}")
            continue

    ds.close()
    return profiles

# ── Load all valid profiles ─────────────────────────────
all_profiles = []
for filename in os.listdir(folder):
    if filename.endswith('.nc'):
        path = os.path.join(folder, filename)
        profiles = load_profiles_year(path, target_year)
        all_profiles.extend([p for p in profiles if len(p) == 7])

# Sort profiles by datetime
all_profiles = sorted(all_profiles, key=lambda x: x[6] or datetime(1900, 1, 1))

# ✅ Count and print number of profiles
n_profiles = len(all_profiles)
print(f"✅ Found {n_profiles} profiles for year {target_year}")

if n_profiles == 0:
    print(f"❌ No profiles found for year {target_year}.")
    exit()

# ── Determine CT range for horizontal offset ─────────────────
ct_all = np.concatenate([t for t, *_ in all_profiles])
ct_min, ct_max = np.nanmin(ct_all), np.nanmax(ct_all)
separation = (ct_max - ct_min) * 1.2 if ct_max > ct_min else 1.0

# ── Plot Setup ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

for idx, (t, p, m_ml, m_int, fid, itp_tag, dt) in enumerate(all_profiles):
    offset = idx * separation
    t_shifted = t + offset

    ax.plot(t_shifted, p, '-', color='gray', linewidth=1)

    if m_ml.any():
        ax.scatter(t_shifted[m_ml], p[m_ml], marker='s', s=20,
                   color='green', label='Mixed Layer' if idx == 0 else None)

    if m_int.any():
        ax.scatter(t_shifted[m_int], p[m_int], marker='o', s=20,
                   color='red', label='Interface' if idx == 0 else None)

    # Add label with ITP, FloatID, and Date
    y_top = np.nanmin(p)
    x_label = np.nanmean(t_shifted)
    depth_range = np.nanmax(p) - y_top
    y_text = y_top - 0.02 * depth_range
    date_str = dt.strftime('%Y-%m-%d') if dt else 'Unknown'
    ax.text(x_label, y_text, f'{itp_tag} / ID {fid}\n{date_str}',
            ha='center', va='bottom', fontsize=8)

# ── Finalize Plot ────────────────────────────────────────────
ax.invert_yaxis()
ax.set_xlabel('CT (°C) + offset')
ax.set_ylabel('Pressure (dbar)')
ax.set_title(f'Temperature Profiles with Mixed Layers & Interfaces (Year {target_year})')
ax.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()
