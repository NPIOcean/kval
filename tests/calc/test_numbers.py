import pytest
import numpy as np
from kval.calc import numbers


def test_numbers_1():
    assert numbers.order_of_magnitude(2) == 0


def test_numbers_2():
    assert numbers.order_of_magnitude(1111) == 3