'''
EMPIRICAL.PY

Collection of empirical formulas used in oceanography.

(nothing for now)
'''

if False:
    def windstress(wsp, z = 10, rho_air = 1.2):
        '''
        Calculate wind stress from wind speed using the bulk formlula of Large and Pond (1981).

        z: height of measurement in m
        rho_air: Air density in kg m-3
        '''

        wsp_10m = np.log()
        c_d = _c_d_large_pond(wsp_10m)


    def _c_d_large_pond(wsp_10m):
        '''
        Calculate
        '''

        if wsp_10m < 11:
            cd = 1.2
        elif wsp_10m >= 11:
            cd = 0.49 + 0.065*wsp_10m
