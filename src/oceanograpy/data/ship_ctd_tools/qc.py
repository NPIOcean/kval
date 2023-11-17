'''
## OceanograPy.data.ship_ctd.qc

Various functions for quality control and editing of CTD dataframes in the
format produced by OceanograPy.io.cnv:

Profile editing:

- Quicklook visualizations and statistics 
- Threshold editing
- Outlier editing (filter approach)
- Outlier editing (manual/visual approach)

Dataset editing:

- Quicklook visualizations and statistics 
- Applying offsets (from calibrations etc)
- Metadata handling

'''

def edit_thr(ds, var, min = None, max = None):
    '''
    - Apply mask based on levels.
    - Add history attributes on variable level.
    - Preserve in processing history string.
    '''

    pass

def edit_outliers():
    '''
    Shoudl at the very least do what Pauls mat-functions do.
    '''
    pass