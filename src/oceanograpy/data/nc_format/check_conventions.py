from compliance_checker.runner import ComplianceChecker, CheckSuite


def check_file(file):
    '''
    Use the IOOS compliance checker 
    (https://github.com/ioos/compliance-checker-web)
    to chek an nc file (CF and ACDD conventions)
    '''
    # Load all available checker classes
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