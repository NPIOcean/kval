'''
## OceanograPy.data.ship_ctd.tools

Various functions for making modifications to CTD dataframes in the format produced by OceanograPy.io.cnv:

- Presssure binning
- Chopping
'''

from oceanograpy.io import cnv, _variable_defs as vardef
import numpy as np
import xarray as xr

def bin_to_pressure(ds, dp = 1):
    '''
    Apply pressure binning into bins of *dp* dbar.

    Reproducing the SBE algorithm as documented in:
    https://www.seabird.com/cms-portals/seabird_com/
    cms/documents/training/Module13_AdvancedDataProcessing.pdf


    # Provides not a bin *average* but a*linear estimate of variable at bin
    pressure* (in practice a small but noteiceable difference)


    (See page 13 for the formula used)

    No surface bin included.

    Equivalent to this in SBE terms (I think)
    # binavg_bintype = decibars
    # binavg_binsize = *dp*
    # binavg_excl_bad_scans = yes
    # binavg_skipover = 0
    # binavg_omit = 0
    # binavg_min_scans_bin = 1
    # binavg_max_scans_bin = 2147483647
    # binavg_surface_bin = no, min = 0.000, max = 0.000, value = 0.000
    '''

    # Tell xarray to conserve attributes across operations
    # (we will set this value back to whatever it was after the calculation)
    _keep_attr_value = xr.get_options()['keep_attrs']
    xr.set_options(keep_attrs=True)

    # Define the bins over which to average
    pmax = float(ds.PRES.max())
    pmax_bound = np.floor(pmax-dp/2)+dp/2

    pmin = float(ds.PRES.min())
    pmin_bound = np.floor(pmin+dp/2)-dp/2

    p_bounds = np.arange(pmin_bound, pmax_bound+1e-9, dp) 
    p_centre = np.arange(pmin_bound, pmax_bound, dp)+dp/2

    # Pressure averaged 
    ds_pavg = ds.groupby_bins('PRES', bins = p_bounds).mean()
        
    # Get pressure *binned* according to formula on page 13 in SBEs module 13 document

    ds_curr = ds_pavg.isel({'PRES_bins':slice(1, None)})
    ds_prev = ds_pavg.isel({'PRES_bins':slice(None, -1)})
    # Must assign the same coordinates in order to be able to matrix multiply
    ds_prev.coords['PRES_bins'] =  ds_curr.PRES_bins

    p_target = p_centre[slice(1, None)]
    _numerator = ds_pavg.diff('PRES_bins')*(p_target - ds_prev.PRES)
    _denominator = ds_pavg.PRES.diff('PRES_bins')

    ds_binned = _numerator/_denominator + ds_prev


    # Replace the PRES_bins coordinate and dimension
    # with PRES
    ds_binned = (ds_binned
        .swap_dims({'PRES_bins':'PRES'})
        .drop_vars('PRES_bins'))

    # Set xarray option "keep_attrs" back to whatever it was
    xr.set_options(keep_attrs=_keep_attr_value)

    return ds_binned


def combine_binned():
    '''
    Read netCDF of CTD data read from cnv and binned using oceanograpy.io.cnv,
    and combine into on single file. 
    '''
    pass
