'''
ctd.calc.numbers

Various functions useful on numrical operations.
'''

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
