'''
CALC.PY

Various utility function for some basic calculations

- Rounding while controlling floor/ceil for the last digit
'''

import numpy as np


def custom_round_ud(number, decimals, ud):
    """
    Round a number to a specified number of decimals, forcing rounding
    the last digit up or down.

    Parameters:
    - number (float): The number to be rounded.
    - decimals (int): The number of decimal places.
    - ud (str): Rounding direction. Use 'up' for rounding up or 'down'
    for rounding down.

    Returns:
    float: The rounded number.
    
    Raises:
    ValueError: If an invalid direction is provided.
    """
    factor = 10 ** decimals
    if ud == 'up':
        rounded_number = np.ceil(number * factor) / factor
    elif ud in ['dn', 'down']:
        rounded_number = np.floor(number * factor) / factor
    else:
        raise ValueError("Invalid direction. Use 'up' for rounding up or"
                        " 'down' for rounding down.")
                        
    return rounded_number