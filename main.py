import os
import shutil
import zipfile
import warnings

import numpy as np
import netCDF4 as netcdf

from data_preparation import load_data_csv_zip
from create_netcdf import create_netcdf
from staircase_detector import get_mixed_layers
from config import FIXED_RESOLUTION_METER
from after_detector import *

"""
Script to detect thermohaline staircases in Ice Tethered Profiler data.
Each profile maintains its own true min→max depth grid without artificial zero-padding.
"""

print('Ice tethered profiles')
zip_files = [f for f in os.listdir() if f.endswith('.zip')]

# Detection thresholds
thres_ml_upper  = 0.002  # mixed layer gradient threshold
thres_int_lower = 0.005  # interface gradient threshold
ml_min_length   = 0.75   # mixed layer min depth length (m)
int_min_temp    = 0.01   # interface min temperature width (°C)
cl_length       = 1.0    # connecting layer max length (m)
smooth_length   = 7      # smoothing window (grid points)

for zip_name in zip_files:
    base = os.path.splitext(zip_name)[0]
    ncfile = f"{base}.nc"
    print(f"📦 Processing {zip_name} → will save to {ncfile}")

    # 1) Extract CSVs
    tmp_dir = 'tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    with zipfile.ZipFile(zip_name, 'r') as z:
        z.extractall(tmp_dir)
    os.remove(zip_name)

    # 2) Gather profile files
    profiles = []
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith('.csv') and not f.startswith('._'):
                profiles.append(os.path.join(root, f))
            # else:
                # warnings.warn(f"File {f} is not a CSV file and will be ignored.")

    # 3) Load raw profiles (no interpolation) using absolute paths
    prof_no, p_raw, lat, lon, ct_raw, sa_raw, dates = load_data_csv_zip(
        '', profiles, interp=False,
        resolution=FIXED_RESOLUTION_METER
    )
    N = len(prof_no)
    if N == 0:
        print(f"No valid profiles in {zip_name}")
        shutil.rmtree(tmp_dir)
        continue

    # 4) Determine the maximum true profile length
    valid_mask = ~np.ma.getmaskarray(p_raw)
    lengths = valid_mask.sum(axis=1)
    max_len = int(np.max(lengths))

    # 5) Allocate per-profile grids preserving true min→max depths
    p   = np.ma.masked_all((N, max_len))
    ct  = np.ma.masked_all((N, max_len))
    sa  = np.ma.masked_all((N, max_len))

    # 6) Populate arrays
    for i in range(N):
        valid = ~np.ma.getmaskarray(p_raw[i])
        L = valid.sum()
        p[i, :L]  = p_raw[i, valid]
        ct[i, :L] = ct_raw[i, valid]
        sa[i, :L] = sa_raw[i, valid]

    # Clean up temporary files
    shutil.rmtree(tmp_dir)

    # 7) Create NetCDF with nlevels = max_len
    fh = create_netcdf(ncfile, max_len)

    # 8) Run detection on the per-profile grids
    masks, depth_min_T, depth_max_T = get_mixed_layers(
        np.ma.copy(p), np.ma.copy(ct),
        thres_ml_upper, thres_int_lower,
        ml_min_length, int_min_temp,
        cl_length, smooth_length
    )

    # 9) Write to NetCDF
    t0, t1 = 0, N
    fh.variables['lat'][t0:t1]          = lat
    fh.variables['lon'][t0:t1]          = lon
    fh.variables['prof'][t0:t1]         = np.arange(N, dtype=np.int32)
    fh.variables['dates'][t0:t1]        = dates
    fh.variables['FloatID'][t0:t1]      = prof_no

    fh.variables['pressure'][t0:t1, :]  = p.filled(np.nan)
    fh.variables['ct'][t0:t1, :]        = ct.filled(np.nan)
    fh.variables['sa'][t0:t1, :]        = sa.filled(np.nan)

    fh.variables['mask_ml'][t0:t1, :]   = masks.ml
    fh.variables['mask_int'][t0:t1, :]  = masks.int
    fh.variables['mask_cl'][t0:t1, :]   = masks.cl
    fh.variables['mask_sc'][t0:t1, :]   = masks.sc

    fh.variables['depth_max_T'][t0:t1]  = depth_max_T
    fh.variables['depth_min_T'][t0:t1]  = depth_min_T

    fh.close()
