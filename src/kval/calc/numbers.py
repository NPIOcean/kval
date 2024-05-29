'''
ctd.calc.numbers

Various functions useful on numerical operations.

- Calculating OOM of a float.
- Rounding while controlling floor/ceil for the last digit.

'''
import numpy as np

def order_of_magnitude(number):
    """
    Calculate the order of magnitude of a given number.

    Parameters:
    - number (float): The input number for which to determine the order of magnitude.

    Returns:
    - int: The order of magnitude of the input number.
    """
    if number != 0:
        order = 0
        while abs(number) < 1:
            number *= 10
            order -= 1
        while abs(number) >= 10:
            number /= 10
            order += 1
        return order
    else:
        return 0  # or handle the case when the number is 0



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
        raise ValueError(f"Invalid direction argument ('{ud}'). Use 'up' for "
            "rounding up or 'down' for rounding down.")
                        
    return rounded_number