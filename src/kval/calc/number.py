"""
ctd.calc.numbers

Various functions useful for numerical operations:

- Calculating the order of magnitude (OOM) of a float.
- Rounding while controlling floor/ceil for the last digit.
"""

import numpy as np
from typing import Literal


def order_of_magnitude(number: float) -> int:
    """
    Calculate the order of magnitude of a given number.

    Parameters:
    - number: The input number for which to determine the order of magnitude.

    Returns:
    - The order of magnitude of the input number.
    """
    if number == 0:
        return 0

    order = 0
    while abs(number) < 1:
        number *= 10
        order -= 1
    while abs(number) >= 10:
        number /= 10
        order += 1

    return order


def custom_round_ud(number: float,
                    decimals: int,
                    ud: Literal['up', 'down', 'dn']) -> float:
    """
    Round a number to a specified number of decimal places, with control
    over the direction of rounding (up or down).

    Parameters:
    - number: The number to be rounded.
    - decimals: The number of decimal places.
    - ud: Rounding direction. Use 'up' for rounding up or 'down' for rounding
          down.

    Returns:
    - The rounded number.

    Raises:
    - ValueError: If an invalid direction is provided.
    """
    factor = 10 ** decimals
    if ud == 'up':
        rounded_number = np.ceil(number * factor) / factor
    elif ud in ['down', 'dn']:
        rounded_number = np.floor(number * factor) / factor
    else:
        raise ValueError(
            f"Invalid direction argument ('{ud}'). Use 'up' for rounding up "
            "or 'down' for rounding down."
        )

    return rounded_number
