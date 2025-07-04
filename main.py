import os
import shutil
import zipfile

import numpy as np

from data_preparation import load_data_csv_zip
from create_netcdf import create_netcdf
from config import FIXED_RESOLUTION_METER
from sc_detector_grad import filter_staircase_masks_local
from smooth_temp import *
from sc_detector_peaks import detect_staircase_peaks

'''
Script to process CTD data: smooth temperature profiles and save to NetCDF.
Expects:
    - INPUT_DIR: directory with .zip files of CTD CSVs
    - OUTPUT_DIR: empty or new directory for .nc outputs
'''

# --- Configuration: input and output folders ---
INPUT_DIR = 'gridData_zip'
OUTPUT_DIR = 'prod_files'

# Ensure input exists
if not os.path.isdir(INPUT_DIR):
    print(f"❌ Input directory '{INPUT_DIR}' does not exist.")
    exit(1)

# Clean or create output directory
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Ice tethered profiles')

# Find all zip files in the input directory
zip_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.zip')]
if not zip_files:
    print(f"⚠️ No .zip files found in '{INPUT_DIR}'")
    exit(0)


count = 0

# Loop over zip files containing CTD CSVs
for src_zip in zip_files:
    zip_path = os.path.join(INPUT_DIR, src_zip)
    tmp_dir = os.path.splitext(src_zip)[0]

    # Prepare temporary extraction folder
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # Unzip profiles
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp_dir)

    # Gather profile CSV files
    profiles = []
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith('.csv') and not f.startswith('._'):
                profiles.append(os.path.join(root, f))

    # Load raw profiles (no interpolation)
    prof_no, p_raw, lat, lon, ct_raw, sa_raw, dates = load_data_csv_zip(
        '', profiles,
        interp=False,
        resolution=FIXED_RESOLUTION_METER
    )
    N = len(prof_no)
    if N == 0:
        print(f"⚠️ No valid profiles in '{src_zip}'")
        shutil.rmtree(tmp_dir)
        continue

    # Determine maximum profile length
    valid_mask = ~np.ma.getmaskarray(p_raw)
    max_len = int(valid_mask.sum(axis=1).max())

    # Allocate arrays
    p   = np.ma.masked_all((N, max_len))
    ct  = np.ma.masked_all((N, max_len))
    sa  = np.ma.masked_all((N, max_len))

    for i in range(N):
        vm = valid_mask[i]
        L = vm.sum()
        p[i, :L]  = p_raw[i, vm]
        ct[i, :L] = ct_raw[i, vm]
        sa[i, :L] = sa_raw[i, vm]

    # Clean up temporary files
    shutil.rmtree(tmp_dir)

    # Apply peak finding method for staircase detection
    mask_int, mask_ml, mask_sc, segments, ratio2d, ct_bg, ct_anom, background_only, max_p, min_p = detect_staircase_peaks(p, ct, FIXED_RESOLUTION_METER)
    
    # Filter staircase masks locally based on gradient ratio
    gradient_kwargs = {'Theta':40.0, 'theta_anom':0.06,
                        'thr_iface':1.1, 'thr_ml':0.7}
    
    mask_int, mask_ml, mask_sc = filter_staircase_masks_local(
        p, ct, FIXED_RESOLUTION_METER,
        mask_int, mask_ml,
        **gradient_kwargs
    )

    # Define output NetCDF path
    out_ncfile = os.path.join(OUTPUT_DIR, os.path.splitext(src_zip)[0] + '.nc')

    # Create NetCDF and write results
    fh = create_netcdf(out_ncfile, _nlevels_unused=None)

    # Metadata
    fh.variables['lat'][:]      = lat
    fh.variables['lon'][:]      = lon
    fh.variables['dates'][:]    = dates
    fh.variables['FloatID'][:]  = prof_no

    # Profile data variables
    p_var       = fh.variables['pressure']
    ct_var      = fh.variables['ct']
    sa_var      = fh.variables['sa']
    ct_bg_var   = fh.variables['ct_bg']
    ct_anom_var = fh.variables['ct_anom']
    bg_only_var = fh.variables['ct_bg_only']
    ml_var      = fh.variables['mask_ml']
    int_var     = fh.variables['mask_int']
    sc_var      = fh.variables['mask_sc']
    depth_max_T = fh.variables['depth_max_T']
    depth_min_T = fh.variables['depth_min_T']

    for i in range(N):
        vm = valid_mask[i]
        p_var[i]       = p[i, vm].data.astype(np.float32)
        ct_var[i]      = ct[i, vm].data
        sa_var[i]      = sa[i, vm].data
        ct_bg_var[i]   = ct_bg[i, vm].data
        ct_anom_var[i] = ct_anom[i, vm].data
        bg_only_var[i] = background_only[i, vm].data
        ml_var[i]      = mask_ml[i, vm]
        int_var[i]     = mask_int[i, vm]
        sc_var[i]      = mask_sc[i, vm]
        depth_max_T[i] = max_p[i] 
        depth_min_T[i] = min_p[i]

    count += 1

    fh.close()
    print(f"✅ Written staircase data to '{out_ncfile}'")

print(f"Processed {count} .nc files. Output saved to '{OUTPUT_DIR}'")
