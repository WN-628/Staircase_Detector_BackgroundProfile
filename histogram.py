import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Directory containing your NetCDF files
nc_dir = 'prod_files'
nc_files = glob.glob(f'{nc_dir}/*.nc')

ratio_ml = []
ratio_int = []

for fn in nc_files:
    ds = xr.open_dataset(fn)
    Nobs = ds.sizes['Nobs']
    
    for i in range(Nobs):
        # 1) read pressure, raw & background CT, and masks
        p     = ds['pressure'].isel(Nobs=i).values
        ct    = ds['ct'].isel(Nobs=i).values
        ct_bg = ds['ct_bg'].isel(Nobs=i).values
        ml    = ds['mask_ml'].isel(Nobs=i).values.astype(bool)
        ii    = ds['mask_int'].isel(Nobs=i).values.astype(bool)
        
        # 2) compute raw & background gradients
        grad_raw = np.gradient(ct, p, edge_order=2)
        grad_bg  = np.gradient(ct_bg, p, edge_order=2)
        
        # 3) compute ratio with small‐number safeguard
        eps   = 1e-8
        ratio = np.abs(grad_raw) / np.maximum(np.abs(grad_bg), eps)
        
        # 4) collect only the values in each mask
        ratio_ml.extend(ratio[ml])
        ratio_int.extend(ratio[ii])
    
    ds.close()

# 5) choose binning (e.g. 0.1 per bin up to ratio=5)
bins = np.arange(0, 5.0 + 0.1, 0.1)

# Mixed-layer ratio histogram
plt.figure(figsize=(8,4))
plt.hist(ratio_ml, bins=bins, log=True)
plt.axvline(0.4, color='red', linestyle='--', linewidth=1.5,
            label='Threshold 0.4')
plt.xlabel('Gradient ratio in mixed-layer (|dT/dz|/|dT_bg/dz|)')
plt.ylabel('Count (log scale)')
plt.title('Histogram of Gradient Ratio in Mixed-Layer Segments')
plt.xlim(0, 5)
plt.legend()
plt.tight_layout()
plt.show()

# Interface ratio histogram
plt.figure(figsize=(8,4))
plt.hist(ratio_int, bins=bins, log=True)
plt.axvline(1.8, color='red', linestyle='--', linewidth=1.5,
            label='Threshold 1.8')
plt.xlabel('Gradient ratio in interface (|dT/dz|/|dT_bg/dz|)')
plt.ylabel('Count (log scale)')
plt.title('Histogram of Gradient Ratio in Interface Segments')
plt.xlim(0, 5)
plt.legend()
plt.tight_layout()
plt.show()

# mixed_thicknesses = []
# interface_widths = []

# # Read and collect data
# for file in nc_files:
#     ds = xr.open_dataset(file)
#     Nobs = ds.sizes['Nobs']
#     for i in range(Nobs):
#         p = ds['pressure'].isel(Nobs=i).values
#         ct = ds['ct'].isel(Nobs=i).values
#         mask_ml = ds['mask_ml'].isel(Nobs=i).values.astype(bool)
#         mask_int = ds['mask_int'].isel(Nobs=i).values.astype(bool)
        
#         # Mixed-layer thickness
#         ml_idx = np.where(mask_ml)[0]
#         if ml_idx.size:
#             splits = np.where(np.diff(ml_idx) > 1)[0] + 1
#             for run in np.split(ml_idx, splits):
#                 if run.size:
#                     mixed_thicknesses.append(p[run[-1]] - p[run[0]])
        
#         # Interface temperature width
#         int_idx = np.where(mask_int)[0]
#         if int_idx.size:
#             splits = np.where(np.diff(int_idx) > 1)[0] + 1
#             for run in np.split(int_idx, splits):
#                 if run.size:
#                     interface_widths.append(abs(ct[run[-1]] - ct[run[0]]))
#     ds.close()

# # Determine bin edges: 5 m per bin for thickness, 0.1 °C per bin for interface width
# max_thick = max(mixed_thicknesses) if mixed_thicknesses else 0
# max_intw = max(interface_widths) if interface_widths else 0

# bins_thick = np.arange(0, max_thick + 5, 5)
# bins_intw = np.arange(0, max_intw + 0.01, 0.01)

# # Plot histogram of mixed-layer thickness
# plt.figure(figsize=(8, 4))
# plt.hist(mixed_thicknesses, bins=bins_thick)
# plt.xlabel('Mixed-layer thickness (m)')
# plt.ylabel('Count')
# plt.title('Histogram of Mixed-layer Thickness (5 m bins)')
# plt.tight_layout()
# plt.xlim(0, 50)
# plt.show()

# # Plot histogram of interface temperature widths
# plt.figure(figsize=(8, 4))
# plt.hist(interface_widths, bins=bins_intw)
# plt.xlabel('Interface width in Temperature (°C)')
# plt.ylabel('Count')
# plt.title('Histogram of Interface Temperature Width (0.1 °C bins)')
# plt.tight_layout()
# plt.xlim(0, 0.25)
# plt.show()

# grad_ml = []
# grad_int = []

# # Read and compute gradients
# for file in nc_files:
#     ds = xr.open_dataset(file)
#     Nobs = ds.sizes['Nobs']
#     for i in range(Nobs):
#         p = ds['pressure'].isel(Nobs=i).values
#         ct = ds['ct'].isel(Nobs=i).values
#         mask_ml = ds['mask_ml'].isel(Nobs=i).values.astype(bool)
#         mask_int = ds['mask_int'].isel(Nobs=i).values.astype(bool)
        
#         # Mixed-layer segments
#         ml_idx = np.where(mask_ml)[0]
#         if ml_idx.size:
#             splits = np.where(np.diff(ml_idx) > 1)[0] + 1
#             for run in np.split(ml_idx, splits):
#                 if run.size > 1:
#                     g_ml = np.gradient(ct[run], p[run])
#                     grad_ml.extend(g_ml)
        
#         # Interface segments
#         int_idx = np.where(mask_int)[0]
#         if int_idx.size:
#             splits = np.where(np.diff(int_idx) > 1)[0] + 1
#             for run in np.split(int_idx, splits):
#                 if run.size > 1:
#                     g_int = np.gradient(ct[run], p[run])
#                     grad_int.extend(g_int)
#     ds.close()

# # Plot histogram of gradients in mixed-layer
# plt.figure(figsize=(8,4))
# plt.hist(grad_ml, bins=50)
# plt.xlabel('Temperature gradient (°C/m)')
# plt.ylabel('Count')
# plt.title('Histogram of Temperature Gradients in Mixed-layer Segments')
# plt.tight_layout()
# plt.show()

# # Plot histogram of gradients in interface
# plt.figure(figsize=(8,4))
# plt.hist(grad_int, bins=50)
# plt.xlabel('Temperature gradient (°C/m)')
# plt.ylabel('Count')
# plt.title('Histogram of Temperature Gradients in Interface Segments')
# plt.tight_layout()
# plt.show()