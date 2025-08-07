#!/usr/bin/env python3
"""
This script reads all NetCDF (.nc) files in the `prod_files` directory and, for each profile,
computes the number of contiguous mixed-layer segments. It then creates a scatter plot with:

  - x-axis: Year of profile
  - y-axis: Month of profile (1–12)
  - point color: number of mixed-layer segments in that profile

Configure the `FOLDER` variable below if needed.
"""
import os
from datetime import datetime
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────
FOLDER = 'prod_files'  # directory containing your .nc files
# ───────────────────────────────────────────────────

def count_runs(mask):
    """
    Count contiguous True runs in a boolean array.
    """
    runs = 0
    in_run = False
    for val in mask:
        if val and not in_run:
            runs += 1
            in_run = True
        elif not val:
            in_run = False
    return runs

# Arrays to store per-profile data
years = []
months = []
ml_counts = []  # mixed layer counts per profile

# Loop over files and profiles
for fname in sorted(os.listdir(FOLDER)):
    if not fname.endswith('.nc'):
        continue
    path = os.path.join(FOLDER, fname)
    try:
        ds = nc.Dataset(path, 'r')
        dates   = ds.variables['dates'][:]
        mask_ml = ds.variables['mask_ml'][:]  # mixed-layer mask: True where mixed layer detected
    except Exception as e:
        print(f"⚠️ Skipping {fname}: {e}")
        continue

    for i, ts in enumerate(dates):
        try:
            dt = datetime.utcfromtimestamp(ts)
        except Exception:
            continue
        # count mixed-layer runs in this profile
        mask = mask_ml[i].astype(bool)
        n_ml = count_runs(mask)
        years.append(dt.year)
        months.append(dt.month)
        ml_counts.append(n_ml)
    ds.close()

# Convert to numpy arrays for plotting
years = np.array(years)
months = np.array(months)
ml_counts = np.array(ml_counts)

# Create scatter plot
plt.figure(figsize=(12, 6))
sc = plt.scatter(
    years,
    months,
    c=ml_counts,
    cmap='viridis',
    s=30,
    alpha=0.7,
    edgecolors='k',
    linewidths=0.3
)
plt.colorbar(sc, label='Number of Mixed-Layer Segments')

# Axis formatting
years_unique = np.unique(years)
plt.xticks(years_unique, rotation=45)
plt.yticks(range(1, 13))
plt.ylim(0.5, 12.5)
plt.xlabel('Year')
plt.ylabel('Month')
plt.title('Mixed-Layer Counts per Profile by Year and Month')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
