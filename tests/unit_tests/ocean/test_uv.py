import pytest
import numpy as np
from kval.ocean.uv import rotate_uv, principal_angle # Adjust the import based on your file structure


######### TESTING rotate_uv ########

# Fixture to provide common test vectors
@pytest.fixture
def test_vectors_uvrot():
    return {
        'u': np.array([1.0, 0.0], dtype=float),
        'v': np.array([0.0, 1.0], dtype=float)
    }

@pytest.fixture
def test_vectors_zero_uvrot():
    return {
        'u': np.array([0.0, 0.0], dtype=float),
        'v': np.array([0.0, 0.0], dtype=float)
    }

def test_rotate_uv_basic(test_vectors_uvrot):

    u = test_vectors_uvrot['u']
    v = test_vectors_uvrot['v']
    angle = np.pi / 2  # 90 degrees in radians
    u_rot, v_rot = rotate_uv(u, v, angle)

    # Check the rotated vector
    assert np.allclose(u_rot, [0.0, 1.0])
    assert np.allclose(v_rot, [-1.0, 0.0])

def test_rotate_uv_degrees(test_vectors_uvrot):
    u = test_vectors_uvrot['u']
    v = test_vectors_uvrot['v']
    angle = 90  # degrees
    u_rot, v_rot = rotate_uv(u, v, angle, in_degrees=True)

    # Check the rotated vector
    assert np.allclose(u_rot, [0.0, 1.0])
    assert np.allclose(v_rot, [-1.0, 0.0])

def test_rotate_uv_zero_angle(test_vectors_uvrot):
    u = test_vectors_uvrot['u']
    v = test_vectors_uvrot['v']
    angle = 0  # radians
    u_rot, v_rot = rotate_uv(u, v, angle)

    # Check that rotation by zero angle leaves the vectors unchanged
    assert np.allclose(u_rot, u)
    assert np.allclose(v_rot, v)

def test_rotate_uv_negative_angle(test_vectors_uvrot):
    u = test_vectors_uvrot['u']
    v = test_vectors_uvrot['v']
    angle = -np.pi / 2 # -90 degrees in radians
    u_rot, v_rot = rotate_uv(u, v, angle)

    # Check that rotation works
    assert np.allclose(u_rot, [0.0, -1.0])
    assert np.allclose(v_rot, [1.0, 0.0])


######### TESTING principal_angle ########

# Fixture to provide common test vectors
@pytest.fixture
def test_vectors_pa():
    '''
    Define an uv dataset with all variability along the u axis
    '''
    L = 100
    u0 = np.array(np.random.randn(L), dtype=float)
    v0 = np.array(np.zeros(L), dtype=float)
    return {'u0': u0, 'v0': v0}


def test_principal_angle_basic(test_vectors_pa):
    u0 = test_vectors_pa['u0']
    v0 = test_vectors_pa['v0']

    for angle in [0, np.pi/3, -np.pi/4]:
        # Rotate CCW by `angle`
        u_rot, v_rot = rotate_uv(u0, v0, -angle)
        # Estimate the principal angle
        estimated_angle = principal_angle(u_rot, v_rot, in_degrees=False)
        assert np.allclose(estimated_angle, angle, atol=1e-3)

def test_principal_angle_basic_degrees(test_vectors_pa):
    u0 = test_vectors_pa['u0']
    v0 = test_vectors_pa['v0']

    for angle in [0, 30, -48]:
        # Rotate CCW by `angle`
        u_rot, v_rot = rotate_uv(u0, v0, -angle, in_degrees = True)
        # Estimate the principal angle
        estimated_angle = principal_angle(u_rot, v_rot, in_degrees = True)
        assert np.allclose(estimated_angle, angle, atol=1e-3)

def test_principal_angle_edge_cases(test_vectors_pa):
    u0 = test_vectors_pa['u0']
    v0 = test_vectors_pa['v0']

    # Angles close to the boundaries
    angles = [np.pi/2 - 1e-6, -np.pi/2 + 1e-6]
    for angle in angles:
        # Rotate CCW by `angle`
        u_rot, v_rot = rotate_uv(u0, v0, -angle)
        # Estimate the principal angle
        estimated_angle = principal_angle(u_rot, v_rot, in_degrees=False)
        assert np.allclose(estimated_angle, angle, atol=1e-3)


def test_principal_angle_zero_vectors():
    u = np.array([0, 0, 0])
    v = np.array([0, 0, 0])
    angle, major_std, minor_std = principal_angle(u, v, return_std=True)

    # Check that angle is 0
    assert np.isclose(angle, 0.0)

    # Standard deviations should be zero for zero vectors
    assert np.isclose(major_std, 0.0)
    assert np.isclose(minor_std, 0.0)


def test_principal_angle_with_nans():
    u = np.array([1, 2, np.nan])
    v = np.array([4, np.nan, 6])
    angle, major_std, minor_std = principal_angle(u, v, return_std=True)

    # Check the types and handle NaNs gracefully
    assert isinstance(angle, float)
    assert isinstance(major_std, float)
    assert isinstance(minor_std, float)

    # The angle should be computed without error even with NaNs
    assert not np.isnan(angle)
    assert not np.isnan(major_std)
    assert not np.isnan(minor_std)


def test_principal_angle_return_std(test_vectors_pa):
    u0 = test_vectors_pa['u0']
    v0 = test_vectors_pa['v0']

    for angle in [0, np.pi/4]:
        u_rot, v_rot = rotate_uv(u0, v0, angle)
        _, major_std, minor_std = principal_angle(u_rot, v_rot, return_std=True)

        # Ensure that standard deviations are calculated
        assert isinstance(major_std, float)
        assert isinstance(minor_std, float)

        # Check if standard deviations are within reasonable ranges
        assert np.isclose(major_std, np.std(u0))
        assert np.isclose(minor_std, 0)

