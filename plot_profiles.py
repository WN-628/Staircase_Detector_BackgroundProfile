"""
plot_profiles.py: Read a NetCDF file and plot all temperature profiles side by side,
highlighting staircase points (mask_sc).
"""
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_profiles(nc_file):
    """
    Read the NetCDF file and plot all temperature profiles on a single graph,
    offsetting each profile horizontally, marking staircase points,
    marking specified max/min CT depths, and placing profile labels at the top of each shifted profile,
    sorted by FloatID if available.
    """
    # Open dataset and extract variables
    ds = nc.Dataset(nc_file, 'r')
    pressure_var = ds.variables['pressure']
    ct_var = ds.variables['ct']
    mask_sc_var = ds.variables['mask_sc']
    depth_max_var = ds.variables.get('depth_max_T')
    depth_min_var = ds.variables.get('depth_min_T')
    floatid_var = ds.variables.get('FloatID')

    # Number of profiles
    num_profiles = len(pressure_var)

    # Load raw profiles and IDs
    profiles = [np.array(ct_var[i]) for i in range(num_profiles)]
    pressures = [np.array(pressure_var[i]) for i in range(num_profiles)]
    masks = [np.array(mask_sc_var[i], dtype=bool) for i in range(num_profiles)]

    # Load max/min depth arrays if present
    if depth_max_var is not None and depth_min_var is not None:
        depth_max = np.array(depth_max_var[:])
        depth_min = np.array(depth_min_var[:])
    else:
        depth_max = depth_min = None

    # Handle FloatID sorting if available
    if floatid_var is not None:
        floatids = np.array([int(floatid_var[i]) for i in range(num_profiles)])
        sort_idx = np.argsort(floatids)
        profiles = [profiles[i] for i in sort_idx]
        pressures = [pressures[i] for i in sort_idx]
        masks = [masks[i] for i in sort_idx]
        if depth_max is not None:
            depth_max = depth_max[sort_idx]
            depth_min = depth_min[sort_idx]
        floatids = floatids[sort_idx]
    else:
        floatids = None

    # Determine horizontal separation based on CT range
    all_ct = np.concatenate(profiles)
    ct_min, ct_max = np.nanmin(all_ct), np.nanmax(all_ct)
    separation = (ct_max - ct_min) * 1.2 if ct_max > ct_min else 1.0

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 6))
    staircase_plotted = False
    maxdepth_plotted = False
    mindepth_plotted = False

    # Plot each profile
    for idx, (t, p, m) in enumerate(zip(profiles, pressures, masks)):
        offset = idx * separation
        t_shifted = t + offset

        # Draw profile line
        ax.plot(t_shifted, p, '-', label=None)

        # Staircase markers
        if m.any():
            if not staircase_plotted:
                ax.scatter(t_shifted[m], p[m], marker='x', s=50, label='Staircase')
                staircase_plotted = True
            else:
                ax.scatter(t_shifted[m], p[m], marker='x', s=50)

        # Mark specified min/max CT depths
        if depth_min is not None and depth_max is not None:
            d_min = depth_min[idx]
            d_max = depth_max[idx]
            # Interpolate CT at those depths
            ct_min_val = np.interp(d_min, p, t)
            ct_max_val = np.interp(d_max, p, t)
            # Min depth marker
            if not mindepth_plotted:
                ax.scatter(ct_min_val + offset, d_min, marker='v', s=50, label='Depth Min CT')
                mindepth_plotted = True
            else:
                ax.scatter(ct_min_val + offset, d_min, marker='v', s=50)
            # Max depth marker
            if not maxdepth_plotted:
                ax.scatter(ct_max_val + offset, d_max, marker='^', s=50, label='Depth Max CT')
                maxdepth_plotted = True
            else:
                ax.scatter(ct_max_val + offset, d_max, marker='^', s=50)

        # Label text for profile
        if floatids is not None:
            profile_label = f'ID {floatids[idx]}'
        else:
            profile_label = f'Profile {idx+1}'
        y_top = np.nanmin(p)
        x_label = np.nanmean(t_shifted)
        depth_range = np.nanmax(p) - y_top
        y_text = y_top - 0.02 * depth_range
        ax.text(x_label, y_text, profile_label,
                ha='center', va='bottom', fontsize=9)

    # Finalize
    basename = os.path.basename(nc_file)
    ax.invert_yaxis()
    ax.set_xlabel('CT (°C) + offset')
    ax.set_ylabel('Pressure (dbar)')
    ax.set_title(f'Temperature Profiles with Staircase and Depth Min/Max CT for {basename}')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


plot_profiles('prod_files/itp65cormat.nc')  # Replace with your NetCDF file path