import netCDF4 as nc
import numpy as np
import pandas as pd

# 1. Open your NetCDF
ds = nc.Dataset('itp65cormat.nc', 'r')

# 2. Read variables
ct_vals     = ds.variables['ct'][:]       # shape: (n_profiles, n_levels)
profile_ids = ds.variables['FloatID'][:]  # shape: (n_profiles,)
pressure    = ds.variables['pressure'][:] # shape: (n_levels,)

# 3. Choose your profile ID
profile = 492  # ← change to the FloatID you want

# 4. Find its index
idx = np.where(profile_ids == profile)[0]
if idx.size == 0:
    raise ValueError(f"Profile {profile} not found in FloatID")
i = idx[0]

# 5. Extract that row of CT
ct_profile = ct_vals[i, :]

# 6. Build a DataFrame of pressure vs. ct
df = pd.DataFrame({
    'pressure_m': pressure,
    'ct_degC':    ct_profile
})

# 7. Export
csv_name = f'ct_profile_{profile}.csv'
df.to_csv(csv_name, index=False)
print(f"Wrote {len(df)} levels for profile {profile} → {csv_name}")
