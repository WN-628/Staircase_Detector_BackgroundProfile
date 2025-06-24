# Staircase_Detector_BackgroundProfile

## Description

- This code aims to detect staircase structures in ITP profiles using the method by comparing with the background profile as mentioned in [Sommer et al.](https://doi.org/10.1175/JTECH-D-12-00272.1)  

------

## Features

- **Flexible CSV ingestion**
   • Reads any number of zipped CSV files from folder **gridData_zip**
   • Automatically skips non‐CSV files and lists skipped files.
- **Physical preprocessing**
   • Converts Practical Salinity → Absolute Salinity → Conservative Temperature (via [GSW](https://teos-10.github.io/GSW-Python/)).
   • Optionally interpolates to a fixed vertical grid (default: 0.25 m).
- **Background smoothing & anomaly detection**
   • Applies a running‐mean smoother over a configurable window (default: 6 m).
   • Computes small‐scale temperature anomalies and masks “background‐only” regions.
- **NetCDF4 output with variable–length arrays**
   • Stores each profile’s raw data (`pressure`, `ct`, `sa`) alongside `ct_bg`, `ct_anom`, and `ct_bg_only`.
   • Includes profile metadata (`lat`, `lon`, `dates`, `FloatID`), extrema (`depth_max_T`, `depth_min_T`), and layer masks (`mask_ml`, `mask_int`, `mask_cl`, `mask_sc`, `cl_mushy`, `cl_supermushy`) as VL-arrays.
- **Quick‐look plotting utility**
   • `read_background.py` reads a given NetCDF and produces side-by-side plots of raw vs. background temperature and the anomaly profile.

------

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `netCDF4`
- `gsw` (TEOS‑10)
- `scipy`
- `matplotlib`

Install via pip:

```bash
pip install numpy pandas netCDF4 gsw scipy matplotlib
```

------

## Installation & Directory Layout

```text
├── config.py               # Global settings (e.g. vertical resolution)
├── smooth_temp.py          # Background smoothing + anomaly masking
├── data_preparation.py     # CSV → pandas → CTD arrays
├── create_netcdf.py        # NetCDF file skeleton with VL‐arrays & metadata
├── read_background.py      # Plotting utility (raw vs. background CT)
├── main.py                 # Driver: unzip → load → smooth → write .nc
└── gridData_zip/           # INPUT_DIR: place your .zip archives here
```

------

## Configuration

Edit the constants in **config.py**:

```python
FIXED_RESOLUTION_METER = 0.25   # Vertical grid spacing [m]
```

Other parameters (smoothing window¹, anomaly threshold²) can be tuned in `smooth_temp.py`:

- **Theta**: smoothing window width [m] (default 6.0)
- **theta**: anomaly threshold [°C] (default 0.04)

------

## Usage

1. **Populate** `gridData_zip/` with one or more `.zip` files.
    Each archive should contain CTD-style CSVs (with columns like `depth, temperature, salinity, latitude, longitude, startdate`).

2. **Run** the pipeline:

   ```bash
   python main.py
   ```

   - Processes each `.zip` → extracts CSVs → loads & (optionally) interpolates → smooths & masks → writes `prod_files/<archive>.nc`.

3. **Inspect** outputs in `prod_files/`.
    Each NetCDF contains:

   - `pressure` (vlen float32)
   - `ct`, `sa`, `ct_bg`, `ct_anom`, `ct_bg_only` (vlen float64)
   - Metadata: `lat`, `lon`, `dates`, `FloatID`, `depth_max_T`, `depth_min_T`
   - Masks: `mask_ml`, `mask_int`, `mask_cl`, `mask_sc`, `cl_mushy`, `cl_supermushy` (vlen int8)

4. **Plot** a profile:

   ```bash
   python read_background.py
   ```

   - Modify `prof_no` in the script to pick a `FloatID`.
   - Generates side-by-side plots of original CT vs. background CT and the CT anomaly.

------

## File Summaries

- **config.py**
   Centralizes vertical-grid resolution.
- **data_preparation.py**
  - `load_data_csv_zip(path, profiles, interp, resolution)`
  - Reads CSV → interpolates → applies TEOS-10 conversions → returns masked arrays.
- **smooth_temp.py**
  - `smooth_background(ct, dz, Theta=6.0, theta=0.04)`
  - Returns `(ct_bg, ct_anom, ct_bg_only)`.
- **create_netcdf.py**
  - `create_netcdf(filename)`
  - Defines dimensions, VL types, variables, attributes.
- **main.py**
   Drives the end-to-end workflow:
  1. Unzip archives
  2. Load & preprocess CSVs
  3. Smooth & compute anomalies
  4. Build & write NetCDFs
- **read_background.py**
   Quick plot of CTD vs. smoothed background and anomalies for any profile.

------

