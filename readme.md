# ITP Staircase Detection Pipeline

A Python toolkit for detecting double-diffusive “staircase” structures in conservative temperature (CT) profiles from ice-tethered profilers (ITP). Profiles are loaded from CSVs (optionally zipped), smoothed, and analyzed using two complementary methods:

1. **Zero-crossing peak detection** (Prominence-based method)
2. **Gradient-ratio filtering** (Local gradient ratio method)

Results (mixed layers, interfaces, connecting layers, masks, background fields) are written into self-describing NetCDF4 files with variable-length (vlen) arrays.

## Folders:

- **Staircase_Detector_Background**:
  - .py files, `readme.md`
    - All codes are in the main folder, not in the sub folder
  - **Relavant_Paper**: storing the paper inspired the code
  - **gridData_zip**: storing zipped .csv files which are the original data to be executed by the code (**Input Data**)
  - **prod_files**: the generated .nc (netcdf) files would be produced in this folder (**Produced Results**)
    - Note: Each produced .nc file would have the same name as the corresponding imported .zip file

---

## General Algorithm Design 

1. Load raw ITP profiles (depth, temperature, salinity) from CSV or zipped CSV bundles

2. Ability to interpolate the data to a fixed vertical resolution, configurable in `config.py` (default is not interpolating since we have data already interpolated to 0.25m) 

3. Compute Absolute Salinity and Conservative Temperature via [GSW-Oceanographic toolbox](https://teos-10.github.io/GSW-Python/)

4. Smooth background temperature with Gaussian, boxcar, or adaptive Savitzky–Golay filters

5. Detect CT anomaly peaks using zero-crossing or prominence methods

6. Compute local gradient ratios to refine interface & mixed-layer masks

7. Enforce continuity and prune spurious detections

8. **Output** NetCDF4 files containing:

   - Profile metadata (`lat`, `lon`, `dates`, `FloatID`)

   - Raw & background CT (smoothed by Gaussian distribution) & anomalies

   - Boolean masks (`mask_ml`, `mask_int`, `mask_sc`, `mask_cl`, …)

   - Extrema depths (`depth_max_T`, `depth_min_T`)

## Algorithm Design 

1. **Loading Data \& Create netcdf files**: 

   - Used code: `data_preparation.py`, `create_netcdf.py`, `config.py`(possible) 

   1. Load raw ITP profiles (depth, temperature, salinity) from CSV or zipped CSV bundles
   2. Ability to interpolate the data to a fixed vertical resolution, configurable in `config.py` (default is not interpolating since we have data already interpolated to 0.25m) 
   3. Compute Absolute Salinity and Conservative Temperature via [GSW-Oceanographic toolbox](https://teos-10.github.io/GSW-Python/)

2. **Peak detection**

   - Used code: `sc_detector_peaks` which calls `peak_prominance.py` & `smooth_temp.py`

   1. We smooth the data by depth:
      - $$0-300m$$: `smooth_background_asg`  (adaptive SG)
      - $$300m - deeper$$: `smooth_background_fixed` (moving mean)
   2. Detect CT anomaly peaks using zero-crossing or prominence methods
      - Note: we are using zero-crossing method for now, can be changed to prominence method if the user desire 
   3. A segment between 2 peaks from negative residual to positive is labeled as `mask_int`, from positive residual to negative is labeled as `mask_ml`

3. **Gradient ratio filtering**

   - Used code: `sc_detector_grad.py` which calls `smooth_temp.py`

   1. We smooth data again by Gaussian distribution to get the background temperature profile
   2. Compute ∂CT/∂z for raw and background temperature for each labelled segments, `mask_ml` \& `mask_int` from step 2
   3. Store segments as interface if $$M = \left(\frac{\partial ct}{\partial z}\right)_\text{raw} \large/ \left(\frac{\partial ct}{\partial z}\right)_\text{background}$$ > `thr_iface`=1.8, mixed layer if $$M$$< `thr_ml`=0.6.
      - Note: labelled segments from step 2 which do not satisfy the gradient threshold are unlabelled in this step, meaning we are not labelling new segments 
   4. Enforce minimum run lengths (at least 3) for each staircase structure, store them in `mask_sc`= `mask_ml` + `mask_int` 

4. **Mask assembly & output**

   - Store masks into final staircase mask.
   - Record metadata and extrema depths.
   - Export to NetCDF4.

---

## Library Requirements

- Python ≥ 3.8: testing environment is python 3.13.5
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [xarray](http://xarray.pydata.org/)
- [netCDF4](https://unidata.github.io/netcdf4-python/)
- [GSW-Oceanographic-Toolbox (GSW)](https://teos-10.github.io/GSW-Python/)

Install via (pip can also work):

```bash
conda install numpy scipy pandas xarray netCDF4 
conda install -c conda-forge gsw
```

## Configuration

Edit `config.py`:

```python
# Vertical resolution in meters for interpolation
FIXED_RESOLUTION_METER = 0.25
```
Note: We are not doing interpolation by default since we are using data already interpolated by Jiaming Chang.

------

## Usage

1. **Prepare your data**

   - Place one or more `.zip` archives under `gridData_zip`.
   - Each archive should contain one or more CTD CSV files with columns:
     - `depth` (or first column)
     - `temperature` (or second column)
     - `salinity` (or third column)
     - Optional metadata columns: `latitude`, `longitude`, `startdate`

2. **Run the main script**:

   ```bash
   python main.py
   ```

   - Reads each `.zip`, extracts CSVs, loads profiles, smooths, detects staircase segments, filters them, and writes one `.nc` per zip into `prod_files/`.
   - Prints summary of processed profiles and output filenames.

3. **Inspect output**
    Each NetCDF file includes:

   - **Dimensions**
     - `Nobs` (unlimited, number of profiles)
   - **Variables**
     - **Metadata**:
       - `lat`, `lon`, `dates` (YYYYMMDD as `int32`), `FloatID`
     - **Profile data** (`vlen` arrays):
       - `pressure` (dbar), `ct` (°C), `sa` (g/kg)
       - `ct_bg`, `ct_anom`, `ct_bg_only`
     - **Masks**:
       - `mask_ml`, `mask_int`, `mask_cl`, `mask_sc`, `cl_mushy`, `cl_supermushy`
     - **Extrema depths**:
       - `depth_max_T`, `depth_min_T`

------

## Algorithm Module Breakdown

- **`data_preparation.py`**
   Load and (optionally) interpolate CSV profiles → returns masked NumPy arrays.
- **`smooth_temp.py`**
   Background smoothing (fixed window, adaptive SG, Gaussian) → compute CT anomalies & background-only fields.
- **`peak_prominence.py`**
   Zero-crossing & prominence-based peak detection in CT anomaly segments.
- **`sc_detector_peaks.py`**
   Full pipeline: smooth, detect peaks, build interface/mixed-layer masks.
- **`sc_detector_grad.py`**
   Compute raw vs. background gradient ratio → refine masks & enforce continuity.
- **`create_netcdf.py`**
   Define NetCDF4 schema with vlen types and write all profiles, fields, masks.
- **`main.py`**
   Orchestrates data extraction → loading → detection → NetCDF writing.

------

## Utilities for Result Visualization

- **`read_background.py`**
   Reads a NetCDF file and plots original CT vs. background-only CT (left) and CT anomaly (right) for a specified profile. 
- **`read_background_heatmap.py`**
   Reads a NetCDF file and plots the raw CT colored by the raw-to-smoothed gradient ratio heatmap alongside the CT anomaly plot.
- **`read_single_profile.py`**
   Reads a NetCDF file, retrieves CT profiles and masks, applies peak prominence detection, and visualizes CT raw, CT smooth, interface and mixed-layer points for a profile specified by float-id. 
- **`read_histogram.py`**
   Computes and plots histograms of gradient ratios in mixed-layer and interface segments, marks thresholds, and identifies profiles exceeding a threshold. 
   - Note: We are resmoothing the temperature profile and find mixed-layer and interface with only peak-finding algorithm in this program. 
   
- **`read_percent_years.py`**
   Calculates the annual percentage of profiles with at least one mixed layer, fits a linear trend, and plots the time series. 
- **`read_netcdf_profiles.py`**
   Plots multiple CT profiles side-by-side, highlighting mixed-layer and interface markers, labeling each with FloatID. 
- **`read_thickness_width.py`**
   Computes yearly averages of mixed-layer thickness and interface temperature width (with errors), plots error bars, and fits regressions. 
- **`read_year_waterfall.py`**
   Generates waterfall plots of CT profiles for a specified year, showing mixed-layer and interface points with annotations. 

------

## Reference

Sommer, T. et al. (2013). Revisiting Sensor Responses with Implications for Double-Diffusive Fluxes. *Journal of Physical Oceanography*, Appendix A.

------

## License

MIT © 2025
