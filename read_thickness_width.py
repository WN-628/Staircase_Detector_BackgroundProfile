import os
import glob
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress

'''
This script computes yearly averages of mixed-layer thickness and interface temperature differences
from NetCDF files, and plots the results with error bars and linear regression fits.
It expects the following functions to be defined:
- compute_avg_ml_by_year: computes average mixed-layer thickness by year
- compute_avg_interface_temp_by_year: computes average interface temperature difference by year
- plot_yearly_with_error_and_fit: plots yearly averages with error bars and fits
'''

# ── USER SETTINGS ──────────────────────────────────────────────────────────
NC_DIR      = 'prod_files'   # directory with your .nc files
DEPTH_MIN   = 250.0          # lower bound of the depth window [m or dbar]
DEPTH_MAX   = 500.0          # upper bound of the depth window [m or dbar]
# ── END USER SETTINGS ──────────────────────────────────────────────────────

def compute_avg_ml_by_year(nc_dir, z_min, z_max,
                           error_type='sem',
                           percentiles=(25,75)):
    """
    Returns:
      years[:]      sorted list of years
      means[:]      mean mixed-layer thickness per year
      errors[:]     symmetric errors: SEM if error_type='sem',
                    1*std if error_type='std',
                    or asymmetric percentiles if error_type='percentile'
    """
    data = {}
    for fn in glob.glob(os.path.join(nc_dir, '*.nc')):
        with Dataset(fn, 'r') as ds:
            dates = ds.variables['dates'][:]
            p_all = ds.variables['pressure'][:]
            ml_all = ds.variables['mask_ml'][:]
            for i, ts in enumerate(dates):
                if np.isnan(ts): continue
                year = datetime.utcfromtimestamp(float(ts)).year
                p  = np.array(p_all[i], dtype=float)
                ml = np.array(ml_all[i], dtype=bool)
                # find mixed-layer segments overlapping the window
                starts, ends, in_seg, prev_p = [], [], False, None
                for depth, flag in zip(p, ml):
                    if flag and not in_seg:
                        in_seg, seg_start = True, depth
                    elif not flag and in_seg:
                        in_seg = False
                        starts.append(seg_start)
                        ends.append(prev_p)
                    prev_p = depth
                if in_seg:
                    starts.append(seg_start); ends.append(p[-1])
                for zs, ze in zip(starts, ends):
                    if zs <= z_max and ze >= z_min:
                        data.setdefault(year, []).append(ze - zs)
    years = sorted(data)
    means, errors = [], []
    errs_low, errs_high = [], []
    for y in years:
        arr = np.array(data[y])
        m   = arr.mean()
        means.append(m)
        if error_type == 'sem':
            errors.append(arr.std(ddof=1) / np.sqrt(arr.size))
        elif error_type == 'std':
            errors.append(arr.std(ddof=1))
        else:
            p_low, p_high = np.percentile(arr, percentiles)
            errs_low.append(m - p_low)
            errs_high.append(p_high - m)
    if error_type in ('sem', 'std'):
        return years, means, errors
    else:
        return years, means, (errs_low, errs_high)


def compute_avg_interface_temp_by_year(nc_dir, z_min, z_max,
                                       error_type='sem',
                                       percentiles=(25,75)):
    data = {}
    for fn in glob.glob(os.path.join(nc_dir, '*.nc')):
        with Dataset(fn, 'r') as ds:
            dates = ds.variables['dates'][:]
            p_all = ds.variables['pressure'][:]
            int_all = ds.variables['mask_int'][:]
            ct_all = ds.variables['ct'][:]
            for i, ts in enumerate(dates):
                if np.isnan(ts): continue
                year = datetime.utcfromtimestamp(float(ts)).year
                p    = np.array(p_all[i], dtype=float)
                ints = np.array(int_all[i], dtype=bool)
                ct   = np.array(ct_all[i], dtype=float)
                starts, ends, in_seg, prev_i = [], [], False, None
                for idx, flag in enumerate(ints):
                    if flag and not in_seg:
                        in_seg, start_i = True, idx
                    elif not flag and in_seg:
                        in_seg = False
                        starts.append(start_i); ends.append(prev_i)
                    prev_i = idx
                if in_seg:
                    starts.append(start_i); ends.append(len(ints)-1)
                for si, ei in zip(starts, ends):
                    zs, ze = p[si], p[ei]
                    if zs <= z_max and ze >= z_min:
                        data.setdefault(year, []).append(abs(ct[ei] - ct[si]))
    years = sorted(data)
    means, errors = [], []
    errs_low, errs_high = [], []
    for y in years:
        arr = np.array(data[y])
        m   = arr.mean()
        means.append(m)
        if error_type == 'sem':
            errors.append(arr.std(ddof=1) / np.sqrt(arr.size))
        elif error_type == 'std':
            errors.append(arr.std(ddof=1))
        else:
            p_low, p_high = np.percentile(arr, percentiles)
            errs_low.append(m - p_low)
            errs_high.append(p_high - m)
    if error_type in ('sem', 'std'):
        return years, means, errors
    else:
        return years, means, (errs_low, errs_high)


def plot_yearly_with_error_and_fit(years, means, errors,
                                   ylabel, title):
    fig, ax = plt.subplots(figsize=(8,5))
    # plot error bars
    if isinstance(errors, tuple):
        low, high = errors
        ax.errorbar(years, means, yerr=(low, high), fmt='o', markersize=10, capsize=5,
                    markeredgecolor='k', alpha=0.8)
    else:
        ax.errorbar(years, means, yerr=errors, fmt='o', markersize=10, capsize=5,
                    markeredgecolor='k', alpha=0.8)
    # fit linear regression
    slope, intercept, r_value, p_value, slope_stderr = linregress(years, means)
    line = intercept + slope * np.array(years)
    ax.plot(years, line, 'r--')
    print(f'Slope = {slope:.5f} ± {1.96*slope_stderr:.5f} (95% CI)')
    # labels and grid
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Mixed-layer with ±1 STD and fit
    print(f'Computing yearly average mixed-layer thickness in [{DEPTH_MIN}, {DEPTH_MAX}] m')
    yrs_ml, avg_ml, err_ml = compute_avg_ml_by_year(
        NC_DIR, DEPTH_MIN, DEPTH_MAX, error_type='std'
    )
    plot_yearly_with_error_and_fit(
        yrs_ml, avg_ml, err_ml,
        ylabel='Yearly Average Mean Mixed-Layer Thickness (m)',
        title=f'ML Thickness (±1 STD) & Linear Fit'
    )

    # Interface ΔT with ±1 STD and fit
    print(f'Computing yearly average interface ΔT in [{DEPTH_MIN}, {DEPTH_MAX}] m')
    yrs_int, avg_int, err_int = compute_avg_interface_temp_by_year(
        NC_DIR, DEPTH_MIN, DEPTH_MAX, error_type='std'
    )
    plot_yearly_with_error_and_fit(
        yrs_int, avg_int, err_int,
        ylabel='Yearly Average Mean Interface ΔT (°C)',
        title=f'Interface ΔT (±1 STD) & Linear Fit'
    )
