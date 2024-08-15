import pytest
from kval.calc import number


# order_of_magnitude()

def test_order_of_magnitude_1():
    assert number.order_of_magnitude(2) == 0


def test_order_of_magnitude_2():
    assert number.order_of_magnitude(1111) == 3


# custom_round_ud

def test_custom_round_ud_1():
    test_cases_round_ud = [((3.3333, 3, 'up'), 3.334),
                           ((3.3333, 2, 'dn'), 3.33),
                           ((3.3333, 0, 'up'), 4.0)]

    # Perform the tests
    for input_, output_ in test_cases_round_ud:
        assert number.custom_round_ud(*input_) == output_


def test_custom_round_ud_2():
    """
    Test that a ValueError is raised when the third argument is not 'up' or
    'down'.
    """
    with pytest.raises(ValueError) as excinfo:
        number.custom_round_ud(3.333, 0, 'something_else')
    assert str(excinfo.value) == (
        "Invalid direction argument ('something_else'). Use 'up' for rounding"
        " up or 'down' for rounding down.")
