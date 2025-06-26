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
    and placing profile labels at the top of each shifted profile,
    sorted by FloatID if available.
    """
    # Open dataset and extract variables
    ds = nc.Dataset(nc_file, 'r')
    pressure_var = ds.variables['pressure']
    ct_var = ds.variables['ct']
    mask_sc_var = ds.variables['mask_sc']
    floatid_var = ds.variables.get('FloatID')

    # Number of profiles
    num_profiles = len(pressure_var)

    # Load raw profiles and IDs
    profiles = [np.array(ct_var[i]) for i in range(num_profiles)]
    pressures = [np.array(pressure_var[i]) for i in range(num_profiles)]
    masks = [np.array(mask_sc_var[i], dtype=bool) for i in range(num_profiles)]
    if floatid_var is not None:
        floatids = np.array([int(floatid_var[i]) for i in range(num_profiles)])
        # Determine sort order by FloatID
        sort_idx = np.argsort(floatids)
        profiles = [profiles[i] for i in sort_idx]
        pressures = [pressures[i] for i in sort_idx]
        masks = [masks[i] for i in sort_idx]
        floatids = floatids[sort_idx]
    else:
        floatids = None

    # Determine horizontal separation based on CT range
    all_ct = np.concatenate(profiles)
    ct_min, ct_max = np.nanmin(all_ct), np.nanmax(all_ct)
    separation = (ct_max - ct_min) * 1.2 if ct_max > ct_min else 1.0

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    staircase_plotted = False

    # Plot each sorted profile and annotate
    for idx, (t, p, m) in enumerate(zip(profiles, pressures, masks)):
        offset = idx * separation
        t_shifted = t + offset

        # Draw profile line
        ax.plot(t_shifted, p, '-')

        # Staircase markers (single legend entry)
        if m.any():
            if not staircase_plotted:
                ax.scatter(t_shifted[m], p[m], marker='x', s=50, label='Staircase')
                staircase_plotted = True
            else:
                ax.scatter(t_shifted[m], p[m], marker='x', s=50)

        # Label text based on sorted order
        if floatids is not None:
            profile_label = f'ID {floatids[idx]}'
        else:
            profile_label = f'Profile {idx+1}'

        # Annotate at top of the curve
        y_top = np.nanmin(p)
        x_label = np.nanmean(t_shifted)
        depth_range = np.nanmax(p) - y_top
        y_text = y_top - 0.02 * depth_range
        ax.text(x_label, y_text, profile_label,
                horizontalalignment='center', verticalalignment='bottom', fontsize=9)

    
    basename = os.path.basename(nc_file)
    
    # Finalize plot
    ax.invert_yaxis()  # Depth increases downward
    ax.set_xlabel('CT (°C) + offset')
    ax.set_ylabel('Pressure (dbar)')
    ax.set_title(f'Temperature Profiles with Staircase Mask for {basename} (offset)')
    ax.legend(loc='best')  # Only staircase entry
    plt.tight_layout()
    plt.show()

plot_profiles('prod_files/itp65cormat.nc')  # Replace with your NetCDF file path