import pytest
from kval.util import index

# Assuming index.indices_to_slices is imported or defined here


def test_indices_to_slices():
    # Test case 1: Empty list
    assert index.indices_to_slices([]) == []

    # Test case 2: No consecutive elements
    assert index.indices_to_slices([1, 3, 5, 7]) == [1, 3, 5, 7]

    # Test case 3: All consecutive elements
    assert index.indices_to_slices([1, 2, 3, 4]) == [slice(1, 5)]

    # Test case 4: Mix of consecutive and non-consecutive elements
    assert index.indices_to_slices([1, 2, 3, 4, 9, 11, 12, 13, 14, 19, 21]) == [
        slice(1, 5), 9, slice(11, 15), 19, 21
    ]

    # Test case 5: Single element list
    assert index.indices_to_slices([5]) == [5]

    # Test case 6: Edge case with only two elements, non-consecutive
    assert index.indices_to_slices([5, 10]) == [5, 10]

    # Test case 7: Edge case with two consecutive elements
    assert index.indices_to_slices([5, 6]) == [slice(5, 7)]

if __name__ == "__main__":
    pytest.main()
