import os
import glob
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt

'''
This script computes and plots the average mixed-layer thickness and interface temperature width in scatter plots with either standard error of the mean (SEM) or 25th–75th percentile error bars.
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
      errors[:]     if error_type='sem': 1D list of SEMs
                    if error_type='percentile': tuple([low_errs], [high_errs])
    """
    data = {}
    for fn in glob.glob(os.path.join(nc_dir, '*.nc')):
        with Dataset(fn, 'r') as ds:
            dates, p_all, ml_all = ds.variables['dates'][:], ds.variables['pressure'][:], ds.variables['mask_ml'][:]
            for i, ts in enumerate(dates):
                if np.isnan(ts): continue
                year = datetime.utcfromtimestamp(float(ts)).year
                p  = np.array(p_all[i], dtype=float)
                ml = np.array(ml_all[i], dtype=bool)

                # find contiguous mixed-layer segments
                starts, ends, in_seg, prev_p = [], [], False, None
                for depth, flag in zip(p, ml):
                    if flag and not in_seg:
                        in_seg, seg_start = True, depth
                    elif not flag and in_seg:
                        in_seg = False
                        starts.append(seg_start); ends.append(prev_p)
                    prev_p = depth
                if in_seg:
                    starts.append(seg_start); ends.append(p[-1])

                # accumulate those overlapping window
                for zs, ze in zip(starts, ends):
                    if zs <= z_max and ze >= z_min:
                        data.setdefault(year, []).append(ze - zs)

    years = sorted(data)
    means, errs_low, errs_high, sems = [], [], [], []
    for y in years:
        arr = np.array(data[y])
        m   = arr.mean()
        means.append(m)
        if error_type == 'sem':
            sems.append(arr.std(ddof=1) / np.sqrt(arr.size))
        else:
            # compute chosen percentiles
            p_low, p_high = np.percentile(arr, percentiles)
            errs_low.append(m - p_low)
            errs_high.append(p_high - m)

    if error_type == 'sem':
        return years, means, sems
    else:
        return years, means, (errs_low, errs_high)


def compute_avg_interface_temp_by_year(nc_dir, z_min, z_max,
                                       error_type='sem',
                                       percentiles=(25,75)):
    """
    Same signature as compute_avg_ml_by_year, but for interface ΔT.
    """
    data = {}
    for fn in glob.glob(os.path.join(nc_dir, '*.nc')):
        with Dataset(fn, 'r') as ds:
            dates   = ds.variables['dates'][:]
            p_all   = ds.variables['pressure'][:]
            int_all = ds.variables['mask_int'][:]
            ct_all  = ds.variables['ct'][:]
            for i, ts in enumerate(dates):
                if np.isnan(ts): continue
                year = datetime.utcfromtimestamp(float(ts)).year

                p    = np.array(p_all[i], dtype=float)
                ints = np.array(int_all[i], dtype=bool)
                ct   = np.array(ct_all[i], dtype=float)

                # detect contiguous interface runs
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
    means, errs_low, errs_high, sems = [], [], [], []
    for y in years:
        arr = np.array(data[y])
        m   = arr.mean()
        means.append(m)
        if error_type == 'sem':
            sems.append(arr.std(ddof=1) / np.sqrt(arr.size))
        else:
            p_low, p_high = np.percentile(arr, percentiles)
            errs_low.append(m - p_low)
            errs_high.append(p_high - m)

    if error_type == 'sem':
        return years, means, sems
    else:
        return years, means, (errs_low, errs_high)


def plot_yearly_with_error(years, means, errors,
                           ylabel, title):
    """
    Plot with either symmetric SEM or asymmetric percentiles.
    """
    fig, ax = plt.subplots(figsize=(8,5))
    if isinstance(errors, tuple):
        low, high = errors
        ax.errorbar(years, means, yerr=(low, high),
                    fmt='o', capsize=5, markeredgecolor='k', alpha=0.8)
    else:
        ax.errorbar(years, means, yerr=errors,
                    fmt='o', capsize=5, markeredgecolor='k', alpha=0.8)

    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     # example: use SEM
#     yrs_ml, avg_ml, err_ml = compute_avg_ml_by_year(
#         NC_DIR, DEPTH_MIN, DEPTH_MAX,
#         error_type='sem',
#     )
#     plot_yearly_with_error(
#         yrs_ml, avg_ml, err_ml,
#         ylabel='Mean Mixed-Layer Thickness (m)',
#         title=f'Mixed-Layer Thickness (sem): {DEPTH_MIN:.0f}–{DEPTH_MAX:.0f} m'
#     )

#     yrs_int, avg_int, err_int = compute_avg_interface_temp_by_year(
#         NC_DIR, DEPTH_MIN, DEPTH_MAX,
#         error_type='sem',
#     )
#     plot_yearly_with_error(
#         yrs_int, avg_int, err_int,
#         ylabel='Mean Interface ΔT (°C)',
#         title=f'Interface ΔT (sem): {DEPTH_MIN:.0f}–{DEPTH_MAX:.0f} m'
#     )
    
#     # example: use 1st & 3rd quartiles
#     yrs_ml, avg_ml, err_ml = compute_avg_ml_by_year(
#         NC_DIR, DEPTH_MIN, DEPTH_MAX,
#         error_type='percentile',
#         percentiles=(25, 75),
#     )
#     plot_yearly_with_error(
#         yrs_ml, avg_ml, err_ml,
#         ylabel='Mean Mixed-Layer Thickness (m)',
#         title=f'Mixed-Layer Thickness (25th–75th pct): {DEPTH_MIN:.0f}–{DEPTH_MAX:.0f} m'
#     )
    
#     yrs_int, avg_int, err_int = compute_avg_interface_temp_by_year(
#         NC_DIR, DEPTH_MIN, DEPTH_MAX,
#         error_type='percentile',
#         percentiles=(25, 75),
#     )
#     plot_yearly_with_error(
#         yrs_int, avg_int, err_int,
#         ylabel='Mean Interface ΔT (°C)',
#         title=f'Interface ΔT (25th–75th pct): {DEPTH_MIN:.0f}–{DEPTH_MAX:.0f} m'
#     )

def plot_yearly_two_errors(years, means,
                           sems, pct_errs,
                           ylabel, title):
    """
    Overlay SEM (blue) and 25–75th pct band (red) on the same scatter.
    
    pct_errs: tuple of (low_errors, high_errors)
    """
    low_pct, high_pct = pct_errs

    fig, ax = plt.subplots(figsize=(8,5))

    # 1) SEM in blue
    ax.errorbar(
        years, means,
        yerr=sems,
        fmt='o',
        color='C0',
        ecolor='C0',
        capsize=4,
        label='Standard Error of the Mean (SEM)',
        alpha=0.8
    )

    # 2) 25–75th percentile in red
    ax.errorbar(
        years, means,
        yerr=(low_pct, high_pct),
        fmt='o',
        color='C3',
        ecolor='C3',
        capsize=4,
        label='25–75 percentile',
        alpha=0.6
    )

    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- Mixed-layer thickness ---
    # SEM
    yrs_ml, ml_means, ml_sems = compute_avg_ml_by_year(
        NC_DIR, DEPTH_MIN, DEPTH_MAX,
        error_type='sem'
    )
    # 25th–75th percentile
    _,    _,        ml_pct_errs = compute_avg_ml_by_year(
        NC_DIR, DEPTH_MIN, DEPTH_MAX,
        error_type='percentile',
        percentiles=(25,75)
    )
    plot_yearly_two_errors(
        yrs_ml,
        ml_means,
        ml_sems,
        ml_pct_errs,
        ylabel='Mean Mixed-Layer Thickness (m)',
        title=f'Mixed-Layer Thickness (250–500 m)'
    )

    # --- Interface ΔT ---
    # SEM
    yrs_int, int_means, int_sems = compute_avg_interface_temp_by_year(
        NC_DIR, DEPTH_MIN, DEPTH_MAX,
        error_type='sem'
    )
    # 25th–75th percentile
    _,        _,           int_pct_errs = compute_avg_interface_temp_by_year(
        NC_DIR, DEPTH_MIN, DEPTH_MAX,
        error_type='percentile',
        percentiles=(25,75)
    )
    plot_yearly_two_errors(
        yrs_int,
        int_means,
        int_sems,
        int_pct_errs,
        ylabel='Mean Interface ΔT (°C)',
        title=f'Interface ΔT (250–500 m)'
    )