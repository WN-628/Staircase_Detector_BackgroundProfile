import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

''' This script reads all netCDF files in a directory and computes the percentage of profiles with at least one mixed layer per year. It then plots this percentage over the years in a scatter plot. '''

# ── CONFIG ─────────────────────────────────────────
folder = 'prod_files'
# ───────────────────────────────────────────────────

def get_profiles_ml_per_year(nc_path):
    try:
        ds = nc.Dataset(nc_path, 'r')
        dates = ds.variables['dates'][:]
        mask_ml = ds.variables['mask_ml'][:]
    except Exception as e:
        print(f"⚠️ Skipping {nc_path}: {e}")
        return []

    year_stats = []
    for i in range(len(dates)):
        try:
            ts = dates[i]
            year = datetime.utcfromtimestamp(ts).year
            has_ml = np.sum(mask_ml[i]) > 0
            year_stats.append((year, has_ml))
        except:
            continue

    ds.close()
    return year_stats

# ── Process all profiles across all files ─────────────────────
profile_counts = defaultdict(int)
profile_with_ml = defaultdict(int)

for fname in os.listdir(folder):
    if fname.endswith('.nc'):
        path = os.path.join(folder, fname)
        results = get_profiles_ml_per_year(path)
        for year, has_ml in results:
            profile_counts[year] += 1
            if has_ml:
                profile_with_ml[year] += 1

# ── Compute percentages ───────────────────────────────────────
years_sorted = sorted(profile_counts.keys())
percentages = [
    100 * profile_with_ml[y] / profile_counts[y] if profile_counts[y] > 0 else 0
    for y in years_sorted
]

# Prepare count list
counts = [profile_counts[y] for y in years_sorted]

# Plot percentage
plt.figure(figsize=(10, 5))
plt.scatter(years_sorted, percentages, color='blue')
plt.plot(years_sorted, percentages, color='gray', linestyle='--', alpha=0.5)

# Annotate each point with the count
for x, pct, cnt in zip(years_sorted, percentages, counts):
    plt.text(x, pct + 1.5,      # offset a bit above the dot
             str(cnt), 
             ha='center', 
             va='bottom', 
             fontsize=9,
             color='darkred')

# Finish styling
plt.xticks(years_sorted, rotation=45)
plt.ylim(0, 105)
plt.xlabel("Year")
plt.ylabel("Percentage of Profiles with Mixed Layer (%)")
plt.title("Percentage of Profiles with ≥1 Mixed Layer per Year\n(Labels show total profiles)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()