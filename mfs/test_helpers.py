import numpy as np
from numpy import cos, sin, sqrt, radians


from . import helpers

n = 100
angles = np.linspace(0, 360, n)
unit_x = np.array([1.0, 0.0, 0.0])
unit_y = np.array([0.0, 1.0, 0.0])
unit_z = np.array([0.0, 0.0, 1.0])


def test_rotation_x():
    """Unit tests for the roation-around-x matrix calculation
    Assumes right-hand rule. That means that, when viewed from positive infinity,
    positive angles mean a counter-clockwise rotation
    Cardinal matricies taken from the Wikipedia page
    https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions"""
    angle_90 = radians(90)
    angle_45 = radians(45)
    angle_30_minus = radians(-30)
    rotation_matrix_x_90 = helpers.rotate_around_x(angle_90)
    rotation_matrix_x_90_correct = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0],]
    )
    assert np.allclose(
        rotation_matrix_x_90, rotation_matrix_x_90_correct, atol=1e-12, rtol=1e-12
    )
    # Should be accurate to floating point precision, i.e. around 1e-16

    rotation_matrix_x_45 = helpers.rotate_around_x(angle_45)
    rotation_matrix_x_45_correct = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(angle_45), -sin(angle_45)],
            [0.0, sin(angle_45), cos(angle_45)],
        ]
    )
    assert np.allclose(
        rotation_matrix_x_45, rotation_matrix_x_45_correct, atol=1e-12, rtol=1e-12
    )

    rotation_matrix_x_30 = helpers.rotate_around_x(angle_30_minus)
    rotation_matrix_x_30_correct = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(angle_30_minus), -sin(angle_30_minus)],
            [0.0, sin(angle_30_minus), cos(angle_30_minus)],
        ]
    )
    assert np.allclose(
        rotation_matrix_x_30, rotation_matrix_x_30_correct, atol=1e-12, rtol=1e-12
    )
    return


def test_rotation_z():
    """Unit tests for the roation-around-x matrix calculation
    Assumes right-hand rule. That means that, when viewed from positive infinity,
    positive angles mean a counter-clockwise rotation"""
    angle_90 = radians(90)
    angle_45 = radians(45)
    angle_30_minus = radians(-30)
    rotation_matrix_z_90 = helpers.rotate_around_z(angle_90)
    rotation_matrix_z_90_correct = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],]
    )
    assert np.allclose(
        rotation_matrix_z_90, rotation_matrix_z_90_correct, atol=1e-12, rtol=1e-12
    )
    # Should be accurate to floating point precision, i.e. around 1e-16

    rotation_matrix_z_45 = helpers.rotate_around_z(angle_45)
    rotation_matrix_z_45_correct = np.array(
        [
            [cos(angle_45), -sin(angle_45), 0.0],
            [sin(angle_45), cos(angle_45), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(
        rotation_matrix_z_45, rotation_matrix_z_45_correct, atol=1e-12, rtol=1e-12
    )

    rotation_matrix_z_30 = helpers.rotate_around_z(angle_30_minus)
    rotation_matrix_z_30_correct = np.array(
        [
            [cos(angle_30_minus), -sin(angle_30_minus), 0.0],
            [sin(angle_30_minus), cos(angle_30_minus), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(
        rotation_matrix_z_30, rotation_matrix_z_30_correct, atol=1e-12, rtol=1e-12
    )
    return


def test_forward_transformation():
    """Integration tests for the mass rotation case, i.e. `rotate_to_dashed_frame`"""

    # verify that unit vectors are correctly rotated around a single axis
    # Reminder: theta is angle around Z-axis, phi is angle around X axis
    # Reminder: right-hand convention throughout
    #    i.e. rotation around an axis by a postive angle is counter-clockwise
    #    when viewed from positive infinity on that axis
    assert np.allclose(
        helpers.rotate_to_dashed_frame(unit_y, theta=radians(90), phi=0),
        -1 * unit_x,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(
        helpers.rotate_to_dashed_frame(unit_y, theta=radians(-90), phi=0),
        unit_x,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(
        helpers.rotate_to_dashed_frame(unit_y, theta=radians(45), phi=0),
        np.array([-1 / sqrt(2), 1 / sqrt(2), 0]),
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(
        helpers.rotate_to_dashed_frame(unit_x, theta=radians(90), phi=0),
        unit_y,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(
        helpers.rotate_to_dashed_frame(unit_x, theta=radians(30), phi=0),
        np.array([cos(radians(30)), sin(radians(30)), 0]),
        atol=1e-12,
        rtol=1e-12,
    )

    assert np.allclose(
        helpers.rotate_to_dashed_frame(unit_y, theta=0, phi=radians(90)),
        unit_z,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(
        helpers.rotate_to_dashed_frame(unit_y, theta=0, phi=radians(45)),
        np.array([0, 1 / sqrt(2), 1 / sqrt(2)]),
        atol=1e-12,
        rtol=1e-12,
    )

    # All rotations of unit_x or -unit_x, around X, should always have no effect
    # Likewise, all rotations of unit_z around Z, should alwaus have no effect
    for angle in radians(angles):
        assert np.array_equal(
            unit_x, helpers.rotate_to_dashed_frame(unit_x, theta=0, phi=angle)
        )
        assert np.array_equal(
            -unit_x, helpers.rotate_to_dashed_frame(-unit_x, theta=0, phi=angle)
        )
        assert np.array_equal(
            unit_z, helpers.rotate_to_dashed_frame(unit_z, theta=angle, phi=0)
        )
        assert np.array_equal(
            -unit_z, helpers.rotate_to_dashed_frame(-unit_z, theta=angle, phi=0)
        )

    # Verify that unit vectors are correctly rotated around two axes
    # This is where things are most likely to go wrong.
    # The transformation is defined as "first rotate around X, then rotate around Z".

    # This follows the two examples given in the docstring of rotate_to_dashed_frame,
    # i.e. unit_y, theta=90, phi=45
    # rotation 1a should be 1/sqrt(2) (0, 1, 1)
    # rotation 2a should be 1/sqrt(2) (-1, 0, 1)
    # rotation_1b and rotation_2b should be (-1, 0, 0)
    theta = radians(90)
    phi = radians(45)
    rotation_1a = helpers.rotate_to_dashed_frame(unit_y, theta=0, phi=phi)
    rotation_2a = helpers.rotate_to_dashed_frame(rotation_1a, theta=theta, phi=0)
    combined = helpers.rotate_to_dashed_frame(unit_y, theta=theta, phi=phi)

    rotation_1b = helpers.rotate_to_dashed_frame(unit_y, theta=theta, phi=0)
    rotation_2b = helpers.rotate_to_dashed_frame(rotation_1b, theta=0, phi=phi)

    assert np.allclose(
        rotation_1a, 1 / sqrt(2) * (unit_y + unit_z), atol=1e-12, rtol=1e-12
    )
    assert np.allclose(
        rotation_2a, 1 / sqrt(2) * (-unit_x + unit_z), atol=1e-12, rtol=1e-12
    )
    assert np.array_equal(rotation_2a, combined)

    assert not np.allclose(rotation_2b, combined, atol=1e-12, rtol=1e-12)
    assert not np.allclose(rotation_1b, rotation_1a, atol=1e-12, rtol=1e-12)
    assert np.allclose(rotation_1b, rotation_2b, atol=1e-12, rtol=1e-12)
    assert np.allclose(rotation_1b, -unit_x, atol=1e-12, rtol=1e-12)

    # And test a more complicated object, i.e. not a unit vector, not normalised
    # Begin with (1,1,1), and perform the same rotation
    # Rotation 1 (45° around X) should rotates into the YXZ plane, to (1,0, 1.414)
    # Rotation 2 (90° around Z) should rotate into the YZ plane to (0, 1, 1.414)
    theta = radians(90)
    phi = radians(45)
    unit = unit_x + unit_y + unit_z  # i.e. (1,1,1)
    rotation_1 = helpers.rotate_to_dashed_frame(unit, theta=0, phi=phi)
    rotation_2 = helpers.rotate_to_dashed_frame(rotation_1, theta=theta, phi=0)
    combined = helpers.rotate_to_dashed_frame(unit, theta=theta, phi=phi)

    assert np.allclose(rotation_1, (1, 0, sqrt(2)), atol=1e-12, rtol=1e-12)
    assert np.allclose(rotation_2, (0, 1, sqrt(2)), atol=1e-12, rtol=1e-12)
    assert np.allclose(rotation_2, combined, atol=1e-12, rtol=1e-12)

    return


def test_inverse_transformation():
    """Integration tests to ensure that the reverse transform, i.e.
    `rotate_to_normal_frame` always reverses the forward transform.
    
    For every possible angle, the effect of (forward, reverse) should always be
    to return the input, accurate to floating-point precision"""
    inputs = (unit_x, unit_y, unit_z, (2 * unit_x + 2.5 * unit_y - 0.3 * unit_z))
    for inp in inputs:
        for theta in radians(angles):
            for phi in radians(angles):
                forward = helpers.rotate_to_dashed_frame(inp, theta, phi)
                reverse = helpers.rotate_to_normal_frame(forward, theta, phi)
                assert np.allclose(inp, reverse, atol=1e-9, rtol=1e-9)
    return


def test_evaluate_axis_projection():
    """Unit tests for converting projection strings"""

    values = [
        ("x", (0, 2, 1)),
        ("xy", (0, 1, 2)),
        ("xyz", (0, 1, 2)),
        ("xz", (0, 2, 1)),
        ("yxz", (1, 0, 2)),
    ]
    for value in values:
        assert helpers.evaluate_axis_projection(value[0]) == value[1], value[0]
    return
