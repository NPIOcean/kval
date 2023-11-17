from compliance_checker.runner import ComplianceChecker, CheckSuite
import os
import xarray as xr

def check_file(file):
    '''
    Use the IOOS compliance checker 
    (https://github.com/ioos/compliance-checker-web)
    to chek an nc file (CF and ACDD conventions).

    Can take a file path or an xr.Dataset as input
    '''
    # Load all available checker classes
    temp = False
    if isinstance(file, xr.Dataset):
        # Store a temp copy for checking
        temp_file = './temp.nc'
        file.to_netcdf(temp_file)
        D = xr.open_dataset(temp_file)
        file = temp_file
        temp = True

    check_suite = CheckSuite()
    check_suite.load_all_available_checkers()
    
    # Run cf and adcc checks
    path = file
    checker_names = ['cf', 'acdd']
    verbose = 0
    criteria = 'normal'

    return_value, errors = ComplianceChecker.run_checker(path,
                                                         checker_names,
                                                         verbose,
                                                         criteria,)
    
    if temp:
        os.remove(temp_file)

