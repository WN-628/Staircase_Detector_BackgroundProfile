import gsw
import numpy as np
import pandas as pd
import os
import re 
from datetime import datetime

import h5py #only used for .mat files

from config import FIXED_RESOLUTION_METER

# List to record files skipped due to depth threshold
SKIPPED_DEPTH_FILES = []

def load_data_mat_zip(path, profiles, interp=True, resolution=FIXED_RESOLUTION_METER):
    valid_profiles = []
    target_levels = np.arange(0, 2000, resolution)
    # print("Files in tmp/:", profiles)

    for fname in profiles:
        if not fname.endswith(".mat"):
            continue
        try:
            with h5py.File(os.path.join(path, fname), 'r') as f:
                pressure = f['pr_filt'][:].flatten()
                temperature = f['te_cor'][:].flatten()
                salinity = f['sa_cor'][:].flatten()
                lat = float(f['latitude'][0][0])
                lon = float(f['longitude'][0][0])
        except Exception as e:
            # print(f"Skipping {fname}: {e}")
            continue

        # print(f"{fname}: pressure min = {pressure.min()}, max = {pressure.max()}, steps = {np.mean(np.diff(pressure))}")

        if pressure.min() > 0 and pressure.max() > 500:
            p_interp = target_levels
            try:
                temp_interp = np.interp(p_interp, pressure, temperature)
                salt_interp = np.interp(p_interp, pressure, salinity)
            except Exception as e:
                # print(f"Interpolation failed for {fname}: {e}")
                continue

            sa = gsw.SA_from_SP(salt_interp, p_interp, lon, lat)
            ct = gsw.CT_from_t(sa, temp_interp, p_interp)

            profile = {
                "p": p_interp,
                "ct": ct,
                "sa": sa,
                "lat": lat,
                "lon": lon,
                "juld": 0,  # default value (not found in .mat)
                "prof_no": int(fname[-8:-4]) if fname[-8:-4].isdigit() else 0
            }
            valid_profiles.append(profile)

    N = len(valid_profiles)
    array_shape = (N, len(target_levels))

    p = np.ma.masked_all(array_shape)
    ct = np.ma.masked_all(array_shape)
    sa = np.ma.masked_all(array_shape)
    lat = np.ma.masked_all(N)
    lon = np.ma.masked_all(N)
    juld = np.ma.masked_all(N)
    prof_no = np.zeros(N, dtype=int)

    for i, prof in enumerate(valid_profiles):
        p[i, :] = prof['p']
        ct[i, :] = prof['ct']
        sa[i, :] = prof['sa']
        lat[i] = prof['lat']
        lon[i] = prof['lon']
        juld[i] = prof['juld']
        prof_no[i] = prof['prof_no']
        
    # print(f"✅ Loaded {len(valid_profiles)} valid profile(s): {[p['prof_no'] for p in valid_profiles]}")

    return prof_no, p, lat, lon, ct, sa, juld

def load_data_csv_zip(path, profiles, interp=True, resolution=FIXED_RESOLUTION_METER):
    valid_profiles = []

    for fname in profiles:
        if not fname.endswith(".csv") or os.path.basename(fname).startswith("._"):
            continue

        full_path = os.path.join(path, fname)
        try:
            df = pd.read_csv(full_path)
            df.columns = df.columns.str.strip().str.lower()
            
            # Loading p, ct, sa 
            pressure = df['depth'].to_numpy().flatten() if 'depth' in df.columns else df.iloc[:, 0].to_numpy().flatten()
            temperature = df['temperature'].to_numpy().flatten() if 'temperature' in df.columns else df.iloc[:, 1].to_numpy().flatten()
            salinity = df['salinity'].to_numpy().flatten() if 'salinity' in df.columns else df.iloc[:, 2].to_numpy().flatten()
            
            # --- Metadata etraction from first line ---
            # Latitude and longtitude: one value per profile
            lat = float(df['latitude'].iloc[1]) if 'latitude' in df.columns else 0.0  # Edited: read from first row
            lon = float(df['longitude'].iloc[1]) if 'longitude' in df.columns else 0.0  # Edited: read from first row
            # parse date, convert to seconds since epoch
            if 'startdate' in df.columns:
                raw = df['startdate'].iloc[0]
                try:
                    dt = datetime.fromisoformat(str(raw))
                except ValueError:
                    dt = pd.to_datetime(raw).to_pydatetime()
                date_sec = dt.timestamp()
            else:
                date_sec = np.nan

            
        except Exception as e:
            print(f"❌ Failed to read {fname}: {e}")
            continue
        
        # if pressure.size == 0 or pressure.max() <= depth_thres:
        #     count += 1
        #     SKIPPED_DEPTH_FILES.append(fname)
        #     # print(f"⛔ Skipping {fname}: invalid pressure range")
        #     continue
        
        try:
            if interp:
                min_p = pressure.min()
                max_p = pressure.max()
                p_interp = np.arange(min_p, max_p + resolution, resolution)
                temp_interp = np.interp(p_interp, pressure, temperature)
                salt_interp = np.interp(p_interp, pressure, salinity)

                sa = gsw.SA_from_SP(salt_interp, p_interp, lon, lat)
                ct = gsw.CT_from_t(sa, temp_interp, p_interp)

                profile = {
                    "p": p_interp,
                    "ct": ct,
                    "sa": sa,
                }
            else:
                sa = gsw.SA_from_SP(salinity, pressure, lon, lat)
                ct = gsw.CT_from_t(sa, temperature, pressure)
                profile = {
                    "p": pressure,
                    "ct": ct,
                    "sa": sa,
                }

            # Extract profile number from filename (now matches 'cor0001.csv')
            match = re.search(r"cor(\d{4})\.csv$", os.path.basename(fname))
            prof_number = int(match.group(1)) if match else 0
            
            profile.update({
                "lat": lat,
                "lon": lon,
                "dates": date_sec,
                "prof_no": prof_number
            })
            valid_profiles.append(profile)
        except Exception as e:
            print(f"⚠️ GSW conversion failed for {fname}: {e}")
            continue

    N = len(valid_profiles)
    if N == 0:
        print("⛔ No valid profiles found.")
        return [], None, None, None, None, None, None
    
    N = len(valid_profiles)
    max_len = max(len(prof["p"]) for prof in valid_profiles)
    array_shape = (N, max_len)

    p = np.ma.masked_all(array_shape)
    ct = np.ma.masked_all(array_shape)
    sa = np.ma.masked_all(array_shape)
    lat = np.ma.masked_all(N)
    lon = np.ma.masked_all(N)
    dates = np.ma.masked_all(N, dtype='f8')
    prof_no = np.zeros(N, dtype=int)

    for i, prof in enumerate(valid_profiles):
        L = len(prof["p"])
        p[i, :L] = prof["p"]
        ct[i, :L] = prof["ct"]
        sa[i, :L] = prof["sa"]
        lat[i] = prof["lat"]
        lon[i] = prof["lon"]
        dates[i] = prof["dates"]
        prof_no[i] = prof["prof_no"]
    
    print(f"✅ Loaded {N} valid profile(s)")
    print("📊 Final variable shapes:")
    print(f"  p.shape     = {p.shape}")
    print(f"  ct.shape    = {ct.shape}")
    print(f"  sa.shape    = {sa.shape}")
    print(f"  lat.shape   = {lat.shape}")
    print(f"  lon.shape   = {lon.shape}")
    print(f"  dates.shape  = {dates.shape}")
    print(f"  prof_no.shape = {prof_no.shape}")
    return prof_no, p, lat, lon, ct, sa, dates
