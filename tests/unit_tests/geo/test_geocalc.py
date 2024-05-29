import numpy as np
import pytest
from kval.geo.geocalc import great_circle_distance, closest_coord

# Test great_circle_distance function
def test_great_circle_distance():
    
    # Define test cases
    earth_radius = 6378e3
    test_cases = [
        ((0, 0, 90, 0), np.pi*earth_radius/2),  # Distance between 90E and 0E on equator
        ((0, 0, 180, 0), np.pi*earth_radius), # Distance between date line and 0E on equator
        ((0, 0, 0, 90),  np.pi*earth_radius/2),  # Distance between equator and North pole
        ((0, 0, 0, -90),  np.pi*earth_radius/2),  # Distance between equator and South pole
        ((-180, 0, 180, 0), 0),  # Date line to date line (longitude wrap)
        ((-90, 0, 90, 0), np.pi*earth_radius),  # 90E to 90W
    ]
    
    # Perform the tests
    for (lon0, lat0, lon1, lat1), expected_distance in test_cases:
        assert np.isclose(great_circle_distance(lon0, lat0, lon1, lat1),
                           expected_distance)



# Test closest_coord function
def test_closest_coord_1():

    ## Case where lat and lon are 2D arrays
    lon = np.array([[5, 10, 15], 
                    [5, 10, 15]])
    lat = np.array([[40, 40, 40], 
                    [50, 50, 50]])
    
    # Test case where closest point is (0, 40)
    assert closest_coord(lon, lat, 0, 40) == (0, 0)
    
    # Test case where closest point is (8, 60)
    assert closest_coord(lon, lat, 8, 60) == (1, 1)
    
    # Test case where closest point is (1, 0)
    assert closest_coord(lon, lat, 15, 60) == (1, 2)


    ## Test case where lon and lat are 1D arrays of different
    lon = np.array([0, 1, 2, 3, 4, 5])
    lat = np.array([10, 11, 12, 13, 14, 15])
    
    # Test case where closest point is (2, 12)
    assert closest_coord(lon, lat, 2, 12) == (2, 2)
    
    # Test case where closest point is (-1, 17)
    assert closest_coord(lon, lat, -1, 17) == (0, 5)
    
    # Test case where closest point is (1, 0)
    assert closest_coord(lon, lat, 3.1, 13.8) == (3, 4)


def test_closest_coord_2():
    # Test case that should raise an error

    ## Case where lat is 1D and lon is 2D 
    lon = np.array([[5, 10, 15], 
                    [5, 10, 15]])
    lat = np.array([40, 40, 40, 4])
    
    # Use pytest.raises to check if ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        closest_coord(lon, lat, 14, 15)