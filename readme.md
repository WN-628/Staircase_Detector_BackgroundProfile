# Staircase_Detector_BackgroundProfile

## Description:

- This code aims to detect staircase structures in ITP profiles using the method by comparing with the background profile as mentioned in [Sommer et al.](https://doi.org/10.1175/JTECH-D-12-00272.1)  

------

## Algorithm Design:

- **Flexible CSV ingestion**
   - Reads any number of zipped CSV files from folder **gridData_zip**
   - Automatically skips and lists skipped non-CSV files in the .zip file.

- **Physical preprocessing**: in  `data_prepration.py`
   - Converts Practical Salinity → Absolute Salinity → Conservative Temperature (via [GSW](https://teos-10.github.io/GSW-Python/)).
   - Optionally interpolates to a fixed vertical grid (default: 0.25 m), which is turned off by default since we are using data that are alreayd interpolated.

- **Background smoothing & anomaly detection**: 
   - Applies a running‐mean smoother over a configurable window (default: 6 m) in `smooth_temp.py`
   - Find staircase structure based purely on peaks finding method, where peaks are found between two zero-residual points, with the residual value being at least 0.003. `sc_detector_peaks.py` which calls `peak_prominence.py`
   - Produces: `mask_int`, `mask_ml`, `mask_sc`, `ct_bg`, `ct_anom`, `max_p`, `min_p`

- **Temperature gradient ratio criteria**:
   - By calling function `filter_staircase_masks_local` from `sc_detector_grad`, calculate the temperature gradient ratio between the original temperature profile and the smooted temperature profiled smoothed by Gaussian distribution with the function `smooth_background_gaussian` from `smooth_temp.py`. 
   - Temperature gradient ratio threshold: **interface**: 1.5, **mixed_layer**: 0.5

- **NetCDF4 output with variable–length arrays**
   - Stores each profile’s raw data (`pressure`, `ct`, `sa`) alongside `ct_bg`, `ct_anom`, and `ct_bg_only`.
   - Includes profile metadata (`lat`, `lon`, `dates`, `FloatID`), extrema (`depth_max_T`, `depth_min_T`), and detect layer masks (`mask_ml`, `mask_int`, `mask_cl`, `mask_sc`, `cl_mushy`, `cl_supermushy`) as VL-arrays with the same size of the original pressure.

- **Quick‐look plotting utility**
   - `read_background_heatmap.py` reads a given NetCDF and produces a heatmap on the depth vs temperature to show the value of temperature gradient ratio between the original temperature profile and the smoothed temperature profile. 
   - `read_graph_profile.py` graphs the temperature profile for a specific file in a specific itp number which shows the detected interface and mixed layer
   - `read_histogram.py` reads all the .nc files in the folder **prod_files** and plots the temperature gradient ratio for the detected mixed layers and interfaces. 
   - `read_plot.profiles.py` plots all the temperature profile in one ITP side by side with the staircase structures labelled.  


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

