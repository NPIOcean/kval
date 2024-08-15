'''
## kval.data.ship_ctd.qc_salt

Various functions for comparing CTD and salinometer salintiy

NOTE: The input formats are not standardized, so this is more or
less hardcoded to work with the FS2014 data (may work with other FS data?)

Not extensively tested either - will put more work into this module
once we have a more standardized format.
'''
from kval.data import ctd
import xarray as xr
import mplcursors
from seawater import eos80
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display


def setup_sal_qc(salts_excel_sheet, log_excel_sheet, btl_dir,
                 bath_temp = None,
                 sample_column ='Sample:',
                 salt_column = 'Median -mean offset',):
    """
    Setup a joint file to be used in salinity quality control (QC) by
    combining salinometer readings with CTD data.

    Parameters:
    - salts_excel_sheet (str):
        Path to the Excel sheet containing salinometer readings.
    - log_excel_sheet (str):
        Path to the Excel sheet containing sample numbers and station/Niskin
         information.
    - btl_dir (str):
        Path to the directory containing CTD .btl files.
    - bath_temp (float):
        Bath temperature used for salinity conversion (when PSAL<2). If None,
        we will attempt to read it from the salts sheet.
    - sample_column:
        Header of the column of salts_excel_sheet that contains sample numbers.
    - salt_column:
        Header of the column of salts_excel_sheet that contains salinity values.

    Returns:
    xr.Dataset: Combined dataset with salinometer readings and CTD data.

    This function reads salinometer readings, sample information, and CTD data, then combines them into
    a single dataset. It also performs necessary conversions and assigns metadata to the resulting dataset.

    Example usage:
    ds_combined = setup_sal_qc('path/to/salts_sheet.xlsx',
                               'path/to/log_sheet.xlsx',
                               'path/to/btl_directory',
                                20.0)
    """

    # Read sample numbers and salinometer readings from log and salts sheets
    print('Reading log sheet..', )
    ds_log = read_log_sheet(log_excel_sheet)
    print('Reading salts sheet..',)
    df_salt = read_salts_sheet(salts_excel_sheet, bath_temp)


    # Read CTD data from .btl files
    print('Reading .btl data:',)
    ds_btl = ctd.dataset_from_btl_dir(btl_dir)

    # Fill salinometer values into ds_log dataset

    ds_log['PSAL_SALINOMETER'] = ds_log['SAMPLE_NUMBER'].copy() * np.nan

    for sample_num, psal_salinometer in zip(df_salt.SAMPLE_NUM, df_salt.PSAL):
        index_sample = np.argwhere(ds_log['SAMPLE_NUMBER'].values == sample_num)
        try:
            ds_log['PSAL_SALINOMETER'][index_sample[0][0], index_sample[0][1]] = psal_salinometer
        except:
            pass

    # Add metadata to the salinometer variable
    ds_log['PSAL_SALINOMETER'].attrs = {
        'units': 1,
        'long_name': 'Practical salinity measured by salinometer',
        'source_file': salts_excel_sheet,
        'note': ('For salinities below 2, salinity was calculated from '
                 'conductivity ratio using the function seawater.eos80.salt().'),
        'bath_temperature': bath_temp
    }

    # Swap dimensions/coordinates of ds_btl from (TIME, NISKIN_NUMBER) to (STATION, NISKIN_NUMBER)
    ds_btl_swapped = ds_btl.swap_dims({'TIME': 'STATION'}).reset_coords('TIME', drop=True)

    # Combine salinometer and CTD information
    ds_combined = ds_log.combine_first(ds_btl_swapped)

    # Add metadata comments
    comment = f'''
    File used for QCing of CTD salinity measurements by comparing
    with salinometer readings based on bottle samples.

    Produced using:
     oceanography.data.ship_ctd.qc_salt.setup_sal_qc()

    CONTENTS:
    - Salinometer salinity PSAL_SALINOMETER
        - Read from the file {salts_excel_sheet}
        - Assigned STATION, NISKIN_NUMBER coordinates based on
          {log_excel_sheet}.
    - Other values obtained from .btl files output from the CTD.
        - Show CTD sensor mean values around bottle closing.
        - Obtained from .btl files - one per profile.
          ({ds_btl.source_files.lower()}).
    '''
    ds_combined.attrs['comment'] = comment

    if bath_temp:
        ds_combined.attrs['salinometer_bath_temperature'] = (
            f'{float(bath_temp)} C (fixed)')
    else:
        bath_temp_min = df_salt['BATH_TEMP'].min()
        bath_temp_max = df_salt['BATH_TEMP'].max()
        if bath_temp_min==bath_temp_max:
            ds_combined.attrs['salinometer_bath_temperature'] = (
                f'Constant {float(bath_temp_min)} C (read from salts sheet)')
        else:
            ds_combined.attrs['salinometer_bath_temperature'] = (
                f'Variable: {float(bath_temp_min)} to {float(bath_temp_max)} C'
                ' (read from salts sheet)')

    ds_combined['SAMPLE_NUMBER'] = ds_combined['SAMPLE_NUMBER'].transpose()
    ds_combined['PSAL_SALINOMETER'] = ds_combined['PSAL_SALINOMETER'].transpose()

    return ds_combined


def read_log_sheet(log_excel_sheet):
    '''
    Read a cruise log sheet linking sample numbers to station/Niskin bottle
    '''
    ##### Read sample numbers from log sheets
    df_log = pd.read_excel(log_excel_sheet)

    # Column # where the station data starts
    column_start_index = np.where(df_log.columns == 'STN')[0][0]+1

    stations_int = df_log.columns[column_start_index:].to_list()
    stations_str = ['%03i'%station_int for station_int in stations_int]

    # Find the row where salinity starts
    # (column varies from file to file, so we look through columns
    # until we find the rigth one)
    salinity_start_row, try_column = None, 0

    while salinity_start_row==None:
        try:
            salinity_start_row = df_log.index[
                df_log.iloc[:, try_column] == 'Salinity'][0]
        except:
            try_column += 1
            if try_column>10: # Avoid infinite loop if we don't find anything..
                raise Exception('Failed to find "Salinity" within the first'
                    ' 30 rows of the salts excel sheet..')

    # Loop through salinity rows: Grab row nr and bottle number
    salinity_niskins = []
    salinity_rows = []
    ii = salinity_start_row

    still_salt = True
    while still_salt:
        salinity_niskins += [int(df_log.STN[ii])]
        salinity_rows += [ii]

        if isinstance(df_log.STN[ii+1], (int, float)):
            if not np.isnan(df_log.STN[ii+1]):
                ii += 1
            else:
                still_salt = False # Stop loop after the final niskin
        else:
            still_salt = False # Stop loop after the final niskin

        if ii>1000: # Avoid infinite loop if we don't find anything..
            raise Exception('Failed to find end of salinity entries within the first'
                ' 1000 rows of the salts excel sheet..')
    # Put together in a data .xr
    ds_log = xr.Dataset(coords = {'STATION':stations_str,
                             'NISKIN_NUMBER':salinity_niskins},
                data_vars = {'SAMPLE_NUMBER':(('NISKIN_NUMBER', 'STATION', ),
                            df_log.iloc[salinity_rows, column_start_index:])})

    return ds_log

def read_salts_sheet(salts_excel_sheet, bath_temp=False,
                     sample_column ='Sample:',
                     salt_column = 'Median -mean offset',
                     bath_temp_column = 'Bath Temp',):

    #### Read salts file
    df = pd.read_excel(salts_excel_sheet)
    df_salt = pd.DataFrame({'SAMPLE_NUM' : df[sample_column],
                            'PSAL' : df[salt_column]})

    # Use bath temp if specified
    if bath_temp:
        df_salt['BATH_TEMP'] = np.ones(df_salt['PSAL'].shape) * bath_temp
    # If not, try reading from the file
    else:
        try:
            df_salt['BATH_TEMP'] = df[bath_temp_column]
        except:
            raise Exception('Could not find bath temperature '
                f'("{bath_temp_column}") in {salts_excel_sheet}. '
                'Consider specifying a fixed bath temperature using the'
                ' "bath_temp" input parameter.')

    ## Convert C ratio to SP
    for ii, psal in enumerate(df_salt.PSAL):
        if psal<2:
            df_salt.loc[ii, 'PSAL'] = eos80.salt(psal, df_salt.loc[ii, 'BATH_TEMP'], 0)

    return df_salt



def plot_histograms(ds,  min_pres=500,  salinometer_var = 'PSAL_SALINOMETER',
                    psal_var=None, N=20, figsize=(7, 3.5)):
    """
    Generate histograms for the difference between a salinity variable and PSAL_SALINOMETER
    for samples taken at depths greater than a specified minimum pressure.

    Parameters:
    - ds (xr.Dataset): Input dataset containing salinity variables.
    - min_pres (int): Minimum pressure threshold for samples. Defaults to 500.
    - psal_var (str): Name of the salinity variable to compare with PSAL_SALINOMETER.
                      Defaults to 'PSAL' or 'PSAL1' .
    - N (int): Number of bins in the histogram. Defaults to 20.
    - figsize (tuple): X/y-size of the histogram figure (inches). Defaults to (7, 3.5).

    Returns:
    None

    Plots and displays histograms for the specified salinity variable's difference from
    PSAL_SALINOMETER, with mean and median lines.

    Clicking the 'Close' button will close the plot and the button.

    Example usage:
    histograms(ds,  min_pres=500, psal_var='PSAL1', N=20)
    """

    if psal_var==None:
        if 'PSAL' in ds.keys():
            psal_var = 'PSAL'
        elif 'PSAL1' in ds.keys():
            psal_var = 'PSAL1'
        else:
            raise Exception('Could not find PSAL or PSAL1 in dataset. '
                'Please specify which variable *psal_var* contains PSAL')

    # Calculate the difference between the specified salinity variable and PSAL_SALINOMETER
    SAL_diff = ds[psal_var] - ds[salinometer_var]

    # Select samples taken at depths greater than the minimum pressure
    SAL_diff_deep = SAL_diff.where(ds.PRES > min_pres).astype(float)

    # Calculate mean and median values
    SAL_diff_deep_mean = SAL_diff_deep.mean().values
    SAL_diff_deep_median = SAL_diff_deep.median().values

    # Calculate the count of valid values
    N_count = SAL_diff_deep.count().values

    # Create a figure and axis
    fig, ax = plt.subplots(figsize = figsize)
    fig.canvas.header_visible = False  # Hide the figure header

    # Plot the histogram
    ax.hist(SAL_diff_deep.values.flatten(), N, density=False, alpha=0.7)

    # Add vertical lines for mean and median
    ax.axvline(0, color='k', ls='--', lw=2)
    ax.axvline(SAL_diff_deep_mean, color='tab:red', dashes=(5, 3), lw=1,
               label=f'Mean = {SAL_diff_deep_mean:.2e}')
    ax.axvline(SAL_diff_deep_median, color='tab:red', ls=':', lw=1.5,
               label=f'Median = {SAL_diff_deep_median:.2e}')

    # Set axis labels and title
    ax.set_xlabel(f'{psal_var} (CTD)- PSAL_SALINOMETER')
    ax.set_ylabel('Frequency')
    leg = ax.legend()
    ax.set_title(f'Salinity comparison for samples taken at >{min_pres}'
                 f' dbar (n = {N_count})')
    ax.grid()
    plt.tight_layout()

    # Define a function to close the figure and widgets
    def close_everything(_):
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        button_exit.close()
        plt.close(fig)

    # Create and display the 'Close' button
    button_exit = widgets.Button(description=f"Close")
    button_exit.on_click(close_everything)
    button_exit.layout.width = '200px'
    display(button_exit)



def plot_by_sample(ds, psal_var='PSAL1', salinometer_var = 'PSAL_SALINOMETER',
                    sample_number_var = 'SAMPLE_NUMBER', min_pres=500):
    """
    Plot salinity comparison for samples taken at depths greater than a specified minimum pressure,
    organized by sample number.

    Parameters:
    - ds (xr.Dataset): Input dataset containing salinity variables.
    - psal_var (str): Name of the salinity variable to compare with PSAL_SALINOMETER.
                      Defaults to 'PSAL1'.
    - min_pres (int): Minimum pressure threshold for samples. Defaults to 500.

    Returns:
    None

    Plots and displays comparisons between the specified salinity variable and PSAL_SALINOMETER
    organized by station. The 'Close' button allows you to close the plot and the button itself.

    Example usage:
    plot_by_sample(ds, psal_var='PSAL1', min_pres=500)
    """

    # Create a new dataset 'b' with necessary coordinates and variables
    b = xr.Dataset(coords={'NISKIN_NUMBER': ds.NISKIN_NUMBER, 'STATION': ds.STATION})
    b[psal_var] = ds[psal_var].where(ds.PRES.values > float(min_pres))
    b['PSAL_SALINOMETER'] = ds[salinometer_var].where(ds.PRES.values > float(min_pres))
    b['SAMPLE_NUMBER'] = ds[sample_number_var].where(ds.PRES.values > float(min_pres))
    b['PRES'] = ds.PRES.where(ds.PRES.values > float(min_pres))

    # Calculate the salinity difference
    SAL_diff = (b[psal_var] - b.PSAL_SALINOMETER).values.flatten().astype(float)

    # Count number of samples
    N_count = (b[psal_var] - b.PSAL_SALINOMETER).count().values

    # Calculate mean of the salinity difference
    Sdiff_mean = np.nanmean(SAL_diff).astype(float)

    # Sort samples by SAMPLE_NUMBER
    sample_num_sortind = np.argsort(b.SAMPLE_NUMBER.values.astype('float').flatten())
    sample_num_sorted = b.SAMPLE_NUMBER.values.flatten()[sample_num_sortind].astype(float)
    Sdiff_num_sorted = SAL_diff[sample_num_sortind].astype(float)
    point_labels = [f"Sample #{sample_num:.0f} ({pres:.0f} dbar)" for sample_num, pres in zip(
        b['SAMPLE_NUMBER'].values.flatten()[sample_num_sortind], b['PRES'].values.flatten()[sample_num_sortind])]

    # Create a figure and subplots
    fig = plt.figure(figsize=(10, 6))
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((2, 4), (1, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 4), (1, 3), colspan=1)

    fig.canvas.header_visible = False  # Hide the figure header

    # Plotting on ax0
    ax0.plot(b.SAMPLE_NUMBER.values.flatten(),
                b[psal_var].values.flatten(),
                '.', color='tab:blue',lw=0.2, alpha=0.6,
                label=f'Bottle file {psal_var}', zorder=2,)
    ax0.plot(b.SAMPLE_NUMBER.values.flatten(),
             b.PSAL_SALINOMETER.values.flatten(),
             '.', zorder=2, color='tab:orange', lw=0.2,
             alpha=0.6, label='Salinometer')

    ax0.set_ylabel('Practical salinity')
    ax0.set_xlabel(f'SAMPLE NUMBER')

    # Add cursor hover annotations
    mplcursors.cursor(hover=True).connect("add",
                lambda sel: sel.annotation.set_text(point_labels[sel.target.index]))

    # Plotting on ax1
    #return sample_num_sorted, Sdiff_num_sorted

    ax1.fill_between(sample_num_sorted, Sdiff_num_sorted, zorder=2,
            color='k', lw=0.2, alpha=0.3, label='Bottle file')
    ax1.plot(sample_num_sorted, Sdiff_num_sorted, '.', zorder=2,
            color='tab:red', lw=0.2, alpha=0.8, label='Salinometer')
    ax1.axhline(Sdiff_mean, color='tab:blue', lw=1.6, zorder=2,
            label=f'Mean = {Sdiff_mean:.2e}', alpha=0.75, ls=':')
    ax1.set_xlabel(f'SAMPLE NUMBER')

    ax1.set_ylabel(f'{psal_var} $-$ salinometer S')

    # Plotting on ax2
    ax2.hist(SAL_diff, 20, orientation="horizontal", alpha=0.7,
             color='tab:red')
    ax2.set_ylim(ax1.get_ylim())
    ax2.grid()
    ax2.axhline(0, color='k',ls='--')
    ax2.set_xlabel(f'FREQUENCY')
    ax2.set_ylabel(f'{psal_var} $-$ salinometer S')
    ax2.axhline(Sdiff_mean, color='tab:blue', lw=1.6, alpha=0.75, ls=':',
               label=f'Mean = {Sdiff_mean:.2e}')
    ax0.grid()
    ax1.grid()
    leg0 = ax0.legend()
    leg1 = ax1.legend()
    leg1.set_zorder(0)

    fig.suptitle(f'Salinity comparison for samples taken at >{min_pres}'
                 f' dbar (n = {N_count})')

    plt.tight_layout()

    # Define a function to close the figure and widgets
    def close_everything(_):
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        button_exit.close()
        plt.close(fig)

    button_exit = widgets.Button(description=f"Close")
    button_exit.on_click(close_everything)
    button_exit.layout.width = '200px'

    display(button_exit)




def _plot_by_station(ds, psal_var='PSAL1', salinometer_var = 'PSAL_SALINOMETER',
                    sample_number_var = 'SAMPLE_NUMBER', min_pres=500):
    """

    DOES NOT WORK AT THE MOMENT!!

    Plot salinity comparison for samples taken at depths greater than a specified minimum pressure,
    organized by station.

    Parameters:
    - ds (xr.Dataset): Input dataset containing salinity variables.
    - psal_var (str): Name of the salinity variable to compare with PSAL_SALINOMETER.
                      Defaults to 'PSAL1'.
    - min_pres (int): Minimum pressure threshold for samples. Defaults to 500.

    Returns:
    None

    Plots and displays comparisons between the specified salinity variable and PSAL_SALINOMETER
    organized by station. The 'Close' button allows you to close the plot and the button itself.

    Example usage:
    plot_by_station(ds, psal_var='PSAL1', min_pres=500)
    """

    # Create a new dataset 'b' with necessary coordinates and variables
    b = xr.Dataset(coords={'NISKIN_NUMBER': ds.NISKIN_NUMBER, 'STATION': ds.STATION})
    b[psal_var] = ds[psal_var].where(ds.PRES.values > float(min_pres))
    b['PSAL_SALINOMETER'] = ds[salinometer_var].where(ds.PRES.values > float(min_pres))
    b['SAMPLE_NUMBER'] = ds[sample_number_var].where(ds.PRES.values > float(min_pres))
    b['PRES'] = ds.PRES.where(ds.PRES.values > float(min_pres))

    # Calculate the salinity difference
    SAL_diff = (b[psal_var] - b.PSAL_SALINOMETER).values.astype(float)

    # Count number of samples
    N_count = (b[psal_var] - b.PSAL_SALINOMETER).count().values

    # Calculate mean of the salinity difference
    Sdiff_mean = np.nanmean(SAL_diff).astype(float)

    # Sort samples by SAMPLE_NUMBER
    station_sortind = np.argsort(b.SAMPLE_NUMBER.values.astype('float').flatten())
    sample_num_sorted = b.SAMPLE_NUMBER.values.flatten()[station_sortind].astype(float)

    Sdiff_num_sorted = SAL_diff[station_sortind].astype(float)
    point_labels = [f"Sample #{sample_num:.0f} ({pres:.0f} dbar)" for sample_num, pres in zip(
        b['SAMPLE_NUMBER'].values.flatten()[station_sortind], b['PRES'].values.flatten()[station_sortind])]

    # Create a figure and subplots
    fig = plt.figure(figsize=(10, 6))
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((2, 4), (1, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 4), (1, 3), colspan=1)

    fig.canvas.header_visible = False  # Hide the figure header

    # Plotting on ax0
    ax0.plot(b.SAMPLE_NUMBER.values.flatten(),
                b[psal_var].values.flatten(),
                '.', color='tab:blue',lw=0.2, alpha=0.6,
                label=f'Bottle file {psal_var}', zorder=2,)
    ax0.plot(b.SAMPLE_NUMBER.values.flatten(),
             b.PSAL_SALINOMETER.values.flatten(),
             '.', zorder=2, color='tab:orange', lw=0.2,
             alpha=0.6, label='Salinometer')

    ax0.set_ylabel('Practical salinity')
    ax0.set_xlabel(f'SAMPLE NUMBER')

    # Add cursor hover annotations
    mplcursors.cursor(hover=True).connect("add",
                lambda sel: sel.annotation.set_text(point_labels[sel.target.index]))

    # Plotting on ax1
    #return sample_num_sorted, Sdiff_num_sorted

    ax1.fill_between(b.STATION, SAL_diff, zorder=2,
            color='k', lw=0.2, alpha=0.3, label='Bottle file')
    ax1.plot(b.STATION, SAL_diff, '.', zorder=2,
            color='tab:red', lw=0.2, alpha=0.8, label='Salinometer')
    ax1.axhline(Sdiff_mean, color='tab:blue', lw=1.6, zorder=2,
            label=f'Mean = {Sdiff_mean:.2e}', alpha=0.75, ls=':')
    ax1.set_xlabel(f'SAMPLE NUMBER')

    ax1.set_ylabel(f'{psal_var} $-$ salinometer S')

    # Plotting on ax2
    ax2.hist(SAL_diff, 20, orientation="horizontal", alpha=0.7,
             color='tab:red')
    ax2.set_ylim(ax1.get_ylim())
    ax2.grid()
    ax2.axhline(0, color='k',ls='--')
    ax2.set_xlabel(f'FREQUENCY')
    ax2.set_ylabel(f'{psal_var} $-$ salinometer S')
    ax2.axhline(Sdiff_mean, color='tab:blue', lw=1.6, alpha=0.75, ls=':',
               label=f'Mean = {Sdiff_mean:.2e}')
    ax0.grid()
    ax1.grid()
    leg0 = ax0.legend()
    leg1 = ax1.legend()
    leg1.set_zorder(0)

    fig.suptitle(f'Salinity comparison for samples taken at >{min_pres}'
                 f' dbar (n = {N_count})')

    plt.tight_layout()

    # Define a function to close the figure and widgets
    def close_everything(_):
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        button_exit.close()
        plt.close(fig)

    button_exit = widgets.Button(description=f"Close")
    button_exit.on_click(close_everything)
    button_exit.layout.width = '200px'

    display(button_exit)

