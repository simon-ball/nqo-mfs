import pytest
import numpy as np
from numpy import radians


from . import helpers, sources

n = 100
angles = np.linspace(0, 360, n)
unit_x = np.array([1.0, 0.0, 0.0])
unit_y = np.array([0.0, 1.0, 0.0])
unit_z = np.array([0.0, 0.0, 1.0])

"""Valid initialisation arguments"""
# Circular coil
dims_circ = {"radius": 1}

# Rectangular coil
dims_rect = {"axDash": 1, "azDash": 2}

# Rectangular permanent magnet
dims_perm = {"axDash": 1, "ayDash": 2, "azDash": 3}

# Coil pair
dims_pair = {
    "full spacing": 0.5,
    "spatially distributed": True,
    "axial layers": 2,
    "axial spacing": 0.1,
    "radial layers": 3,
    "radial spacing": 0.02,
    "configuration": "hh",
}
dims_pair_rect = {**dims_pair, **dims_rect, "shape": "rect"}
dims_pair_circ = {**dims_pair, **dims_circ, "shape": "circ"}

magnet_dims_mapping = [
    (sources.CircularCoil, dims_circ),
    (sources.RectangularCoil, dims_rect),
    (sources.PermanentMagnet, dims_perm),
    (sources.CoilPair, dims_pair_rect),
    (sources.CoilPair, dims_pair_circ),
]


def test_base_magnet_class():
    """Tests that the base class can be initialised, and offers the correct
    interface"""
    m = sources.Magnet(rDash=(0, 0, 0), dimsDash={}, theta=0, phi=0)
    with pytest.raises(NotImplementedError):
        m.get_BDash_field((0, 0, 0))
        m.plot_magnet_position(None, None)

    for angle in angles:
        m.phi = angle
        m.theta = angle
        assert np.isclose(radians(angle), m.phi)
        assert np.isclose(radians(angle), m.theta)
    return


def test_inits():
    """Test that each magnet can be initialised given correct dimensional arguments
    and that it cannot given missing or invalid dimensional arguments"""
    origin = (-1, -2, 3)
    strength = 1
    theta = 36
    phi = -60

    for dut, dims in magnet_dims_mapping:
        # Test that the magnet can be initialised
        dut(strength, origin, dims, theta, phi)
        for key in dims.keys():
            # Test that, given missing arguments, the magnet cannot be initialised
            dims_invalid = dims.copy()
            dims_invalid.pop(key)
            with pytest.raises(KeyError):
                dut(strength, origin, dims_invalid, theta, phi)
    return


def test_field_calculcation():
    """Test that the field calculcations work as expected"""
    for dut, dims in magnet_dims_mapping:
        rotation_consistency_t(dut, dims)
    return


###############################################################################
####
####            Test assistance functions


def rotation_consistency_t(cls_under_test, dims):
    """Given a magnet class to test, and a valid set of parameters, verify that
    the magnetic field calculation behaves **consistently** with rotation
    
    Note: this does NOT test the **correctness** of the magnetic field calculation,
    only its **consistency** under rotation"""
    origin = (-1, 2, 3)
    strength = 1
    m = cls_under_test(strength, origin, dimsDash=dims, theta=0, phi=0)

    # Check that the magnetic field at some specific location in the Dashed frame
    # is the same regardless of the angle of the magnet.
    # This must always be true, because `get_BDash_field` has no theta, phi
    # dependence
    rDash = unit_x + unit_y + unit_z
    static_field = m.get_BDash_field(rDash)
    for angle in angles:
        m2 = cls_under_test(strength, origin, dimsDash=dims, theta=angle, phi=0)
        assert np.array_equal(m2.get_BDash_field(rDash), static_field)
        m3 = cls_under_test(strength, origin, dimsDash=dims, theta=0, phi=angle)
        assert np.array_equal(m3.get_BDash_field(rDash), static_field)
        m4 = cls_under_test(
            strength, origin, dimsDash=dims, theta=-angle / 2, phi=angle
        )
        assert np.array_equal(m4.get_BDash_field(rDash), static_field)

    # Check that the magnetic field at some specific location in the global frame
    # behaves as expected as the magnets are rotated
    # Note: None of the operations are vectorised, so they generally must be run in for-loops
    sample_axis = np.zeros(
        (n, 3)
    )  # sample_axis[0] is a 3-element array, i.e. a position
    sample_axis[:, 0] = np.linspace(-5, 5, n)
    sample_axis[:, 1] = 1.5
    sample_axis[:, 2] = -0.3
    field = np.array([m.get_B_field(r) for r in sample_axis])

    for angle in angles:
        # The coil exists always in the same plane in the dashed frame
        # Starting with the case where the dashed frame equals the global frame,
        # take some axis or plane defned in the global frame and observe the magnetic field.
        # Now rotate the dashed frame through some angles
        # Based on the definition of the sample plane in the global plane,
        # calculcate a new sample plane such that it is at the same position and angle with respect to the coil
        # The magnetic field observed in the new sample plane _should_ be identical if the rotations are working correctly
        m2 = cls_under_test(strength, origin, dimsDash=dims, theta=angle, phi=0)
        new_axis = np.array(
            [helpers.rotate_to_normal_frame(r, angle, 0) for r in sample_axis]
        )
        new_field = np.array([m2.get_B_field(r) for r in new_axis])
        assert np.allclose(new_field, field, atol=1e-6, rtol=1e-6)

        m3 = cls_under_test(strength, origin, dimsDash=dims, theta=0, phi=angle)
        new_axis = np.array(
            [helpers.rotate_to_normal_frame(r, 0, angle) for r in sample_axis]
        )
        new_field = np.array([m3.get_B_field(r) for r in new_axis])
        assert np.allclose(new_field, field, atol=1e-6, rtol=1e-6)

    #        m4 = cls_under_test(
    #            strength, origin, dimsDash=dims, theta=-angle/3, phi=angle
    #        )
    #        new_axis = np.array(
    #            [helpers.rotate_to_normal_frame(r, theta=-angle/3, phi=angle) for r in sample_axis]
    #        )
    #        new_field = np.array([m4.get_B_field(r) for r in new_axis])
    #        assert np.allclose(new_field, field, atol=1e-6, rtol=1e-6)
    return
