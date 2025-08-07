#!/usr/bin/env python3
"""
This script reads all NetCDF (.nc) files in a directory and computes the percentage of profiles
with at least three mixed layers for specified months of each year. It then plots this percentage
over the years in a scatter plot, analogous to read_percent_years.py.

Configure START_MONTH and END_MONTH below.
"""
import os
from collections import defaultdict
from datetime import datetime

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, t
import math

# ── CONFIG ─────────────────────────────────────────
FOLDER = 'prod_files'
START_MONTH = 'January'  # e.g., 'January' or '1'
END_MONTH   = 'June'     # e.g., 'June'   or '6'
MIN_ML_COUNT = 3         # minimum number of mixed layers to consider a profile valid
# ───────────────────────────────────────────────────

# Mapping month names to numbers
MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}

def parse_month(m):
    """Convert month name or number to integer (1-12)."""
    m_lower = str(m).strip().lower()
    if m_lower.isdigit():
        num = int(m_lower)
        if 1 <= num <= 12:
            return num
    elif m_lower in MONTH_MAP:
        return MONTH_MAP[m_lower]
    raise ValueError(f"Invalid month: {m}")


def get_profiles_ml_for_months(nc_path, start_m, end_m):
    """
    Return list of (year, has_valid_ml) for profiles whose month
    lies between start_m and end_m inclusive, and have at least MIN_ML_COUNT mixed layers.
    """
    try:
        ds = nc.Dataset(nc_path, 'r')
        dates = ds.variables['dates'][:]
        mask_ml = ds.variables['mask_ml'][:]
    except Exception as e:
        print(f"⚠️ Skipping {nc_path}: {e}")
        return []

    stats = []
    if start_m <= end_m:
        allowed = set(range(start_m, end_m + 1))
    else:
        allowed = set(range(start_m, 13)) | set(range(1, end_m + 1))

    for i, ts in enumerate(dates):
        try:
            dt = datetime.utcfromtimestamp(ts)
            if dt.month not in allowed:
                continue
            count_ml = np.sum(mask_ml[i])
            has_valid_ml = count_ml >= MIN_ML_COUNT
            stats.append((dt.year, has_valid_ml))
        except Exception:
            continue

    ds.close()
    return stats

# Convert configured months to numbers
start_m = parse_month(START_MONTH)
end_m   = parse_month(END_MONTH)

profile_counts  = defaultdict(int)
profile_with_ml = defaultdict(int)

# Process each NetCDF file
for fname in os.listdir(FOLDER):
    if not fname.endswith('.nc'):
        continue
    path = os.path.join(FOLDER, fname)
    for year, has_valid in get_profiles_ml_for_months(path, start_m, end_m):
        profile_counts[year] += 1
        if has_valid:
            profile_with_ml[year] += 1

# Sort years and compute percentages
years_sorted = sorted(profile_counts)
percentages  = [100 * profile_with_ml[y] / profile_counts[y] if profile_counts[y] else 0
                for y in years_sorted]
counts        = [profile_counts[y] for y in years_sorted]

# Determine y-axis lower bound based on minimum percentage
min_pct = min(percentages)
# floor to nearest lower tens
y_lower = math.floor(min_pct / 10) * 10
# ensure y_lower is at least 0
y_lower = max(0, y_lower)

# Linear regression fit with 95% CI for slope
n = len(years_sorted)
df = n - 2
slope, intercept, r_value, p_value, stderr = linregress(years_sorted, percentages)
t_crit = t.ppf(0.975, df)
ci_slope = t_crit * stderr
slope_lower = slope - ci_slope
slope_upper = slope + ci_slope

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(years_sorted, percentages)
plt.plot(years_sorted, percentages, linestyle='--', alpha=0.5)

# Annotate counts
for x, pct, cnt in zip(years_sorted, percentages, counts):
    plt.text(x, pct + 1.5, str(cnt), ha='center', va='bottom', fontsize=9)

# Plot regression line with CI in legend
x_fit = np.array(years_sorted)
y_fit = intercept + slope * x_fit
label_ci = (
    f'Slope: {slope:.2f}%/yr '
    f'(95% CI: [{slope_lower:.2f}, {slope_upper:.2f}])'
)
plt.plot(x_fit, y_fit, 'r--', label=label_ci)

# Final styling
plt.xticks(years_sorted, rotation=45)
plt.ylim(y_lower, 105)
plt.xlabel(
    f'Year'
)
plt.ylabel(
    f'Percentage of Profiles with ≥{MIN_ML_COUNT} Mixed Layers '
    f'({START_MONTH} to {END_MONTH})'
)
plt.title(
    f'Profiles with ≥{MIN_ML_COUNT} Mixed Layers: '
    f'{START_MONTH} to {END_MONTH}'
)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
