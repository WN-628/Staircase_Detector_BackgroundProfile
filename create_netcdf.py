import netCDF4 as netcdf
import numpy as np

from config import FIXED_RESOLUTION_METER

def create_netcdf(filename, _nlevels_unused=None):
    """
    Create a NetCDF4 file with variable-length (vlen) arrays for profiles and masks.
    Masks are stored as int8 (0/1) since boolean vlen is unsupported.
    Adds background temperature fields: smoothed background, anomaly, and masked background.
    """
    fh = netcdf.Dataset(filename, 'w', format='NETCDF4')

    # Unlimited profile dimension
    fh.createDimension('Nobs', None)

    # Define vlen types
    vl32   = fh.createVLType(np.float32, 'vl_float32')
    vl64   = fh.createVLType(np.float64, 'vl_float64')
    vlint8 = fh.createVLType(np.int8,    'vl_int8')

    # Profile metadata
    lat_var     = fh.createVariable('lat',    np.float64, ('Nobs',))
    lat_var.long_name = 'Latitude'
    lon_var     = fh.createVariable('lon',    np.float64, ('Nobs',))
    lon_var.long_name = 'Longitude'
    dates_var   = fh.createVariable('dates',  np.int32,   ('Nobs',))
    dates_var.long_name = 'Profile date as YYYYMMDD'
    prof_id     = fh.createVariable('FloatID', np.int32,  ('Nobs',))
    prof_id.long_name = 'Profile identifier number'

    depth_max_T_var = fh.createVariable('depth_max_T', np.float32, ('Nobs',))
    depth_max_T_var.long_name = 'Depth at max temperature'
    depth_min_T_var = fh.createVariable('depth_min_T', np.float32, ('Nobs',))
    depth_min_T_var.long_name = 'Depth at min temperature'

    # Profile data
    p_var  = fh.createVariable('pressure', vl32,  ('Nobs',))
    p_var.long_name = 'Pressure profile'
    p_var.units     = 'dbar'

    ct_var = fh.createVariable('ct', vl64,   ('Nobs',))
    ct_var.long_name = 'Conservative Temperature profile'
    ct_var.units     = 'degC'

    sa_var = fh.createVariable('sa', vl64,   ('Nobs',))
    sa_var.long_name = 'Absolute Salinity profile'
    sa_var.units     = 'g/kg'

    # New background temperature fields
    ct_bg_var = fh.createVariable('ct_bg', vl64, ('Nobs',))
    ct_bg_var.long_name = 'Background smoothed Conservative Temperature'
    ct_bg_var.units     = 'degC'

    ct_anom_var = fh.createVariable('ct_anom', vl64, ('Nobs',))
    ct_anom_var.long_name = 'Temperature anomaly (ct - ct_bg)'
    ct_anom_var.units     = 'degC'

    bg_only_var = fh.createVariable('ct_bg_only', vl64, ('Nobs',))
    bg_only_var.long_name = 'Background-only profile masked where anomalies exceed threshold'
    bg_only_var.units     = 'degC'

    # Variable-length mask arrays (0/1 int8)
    ml_var = fh.createVariable('mask_ml',    vlint8, ('Nobs',))
    ml_var.long_name = 'Mixed-layer mask (1 where in mixed layer)'

    int_var = fh.createVariable('mask_int',  vlint8, ('Nobs',))
    int_var.long_name = 'Interface mask (1 at interface)'

    cl_var = fh.createVariable('mask_cl',   vlint8, ('Nobs',))
    cl_var.long_name = 'Connecting-layer mask (1 in connecting layers)'

    sc_var = fh.createVariable('mask_sc',   vlint8, ('Nobs',))
    sc_var.long_name = 'Staircase mask (1 in staircase)'
    
    cl_mushy_var = fh.createVariable('cl_mushy',     vlint8, ('Nobs',))
    cl_mushy_var.long_name = 'Mushy-layer indicator'

    cl_supermushy_var = fh.createVariable('cl_supermushy', vlint8, ('Nobs',))
    cl_supermushy_var.long_name = 'Supermushy-layer indicator'

    return fh
