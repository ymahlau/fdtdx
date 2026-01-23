# tests/core/test_core_misc.py

import jax.numpy as jnp

from fdtdx.core.misc import expand_to_3x3, pad_fields


def test_expand_to_3x3_shape():
    """Test output shape of expand_to_3x3 function."""

    test_none = expand_to_3x3(None)
    test_scalar = expand_to_3x3(1.0)
    test_isotropic = expand_to_3x3(jnp.ones((1, 10, 10, 10)))
    test_diagonal = expand_to_3x3(jnp.ones((3, 10, 10, 10)))
    test_full = expand_to_3x3(jnp.ones((9, 10, 10, 10)))

    assert test_none is None
    assert test_scalar.shape == (3, 3, 1, 1, 1)
    assert test_isotropic.shape == (3, 3, 10, 10, 10)
    assert test_diagonal.shape == (3, 3, 10, 10, 10)
    assert test_full.shape == (3, 3, 10, 10, 10)


def test_expand_to_3x3_value():
    """Test output value of expand_to_3x3 function."""

    spatial_shape = (10, 10, 10)
    zero = jnp.zeros(())
    zeros = jnp.zeros(spatial_shape)

    input_scalar = float(1.0)
    input_isotropic = jnp.ones((1, *spatial_shape))
    input_diagonal = jnp.ones((3, *spatial_shape))
    input_full = jnp.ones((9, *spatial_shape))

    test_none = expand_to_3x3(None)
    test_scalar = expand_to_3x3(input_scalar)
    test_isotropic = expand_to_3x3(input_isotropic)
    test_diagonal = expand_to_3x3(input_diagonal)
    test_full = expand_to_3x3(input_full)

    iso_a = input_isotropic[0]
    aniso_a, aniso_b, aniso_c = input_diagonal[0], input_diagonal[1], input_diagonal[2]

    assert test_none is None
    assert (
        test_scalar
        == jnp.array(
            [
                [[[[1.0]]], [[[zero]]], [[[zero]]]],
                [[[[zero]]], [[[1.0]]], [[[zero]]]],
                [[[[zero]]], [[[zero]]], [[[1.0]]]],
            ]
        )
    ).all()
    assert (
        test_isotropic
        == jnp.stack(
            [jnp.stack([iso_a, zeros, zeros]), jnp.stack([zeros, iso_a, zeros]), jnp.stack([zeros, zeros, iso_a])]
        )
    ).all()
    assert (
        test_diagonal
        == jnp.stack(
            [jnp.stack([aniso_a, zeros, zeros]), jnp.stack([zeros, aniso_b, zeros]), jnp.stack([zeros, zeros, aniso_c])]
        )
    ).all()
    assert (test_full == jnp.reshape(input_full, (3, 3) + spatial_shape)).all()


def test_pad_fields_shape():
    """Test output shape of pad_fields function with different periodic axes."""

    # Create input fields with shape (3, Nx, Ny, Nz)
    fields = jnp.ones((3, 10, 10, 10))
    periodic_axes = (True, True, True)
    padded_fields = pad_fields(fields, periodic_axes)
    assert padded_fields.shape == (3, 12, 12, 12)


def test_pad_fields_value():
    """Test output values of pad_fields function with different periodic axes."""

    # Create a simple test field with unique values
    fields = jnp.arange(3 * 5 * 5 * 5).reshape(3, 5, 5, 5).astype(jnp.float32)

    # Test with all periodic boundaries - wrap mode
    test_all_periodic = pad_fields(fields, (True, True, True))

    # Check that wrapped values are correct at boundaries
    # For periodic, the last value should wrap to the first
    assert jnp.allclose(test_all_periodic[:, 0, 1:-1, 1:-1], fields[:, -1, :, :])
    assert jnp.allclose(test_all_periodic[:, -1, 1:-1, 1:-1], fields[:, 0, :, :])
    assert jnp.allclose(test_all_periodic[:, 1:-1, 0, 1:-1], fields[:, :, -1, :])
    assert jnp.allclose(test_all_periodic[:, 1:-1, -1, 1:-1], fields[:, :, 0, :])
    assert jnp.allclose(test_all_periodic[:, 1:-1, 1:-1, 0], fields[:, :, :, -1])
    assert jnp.allclose(test_all_periodic[:, 1:-1, 1:-1, -1], fields[:, :, :, 0])

    # Test with no periodic boundaries - constant mode (zeros)
    test_no_periodic = pad_fields(fields, (False, False, False))

    # Check that padded values are zeros at boundaries
    assert jnp.allclose(test_no_periodic[:, 0, :, :], 0.0)
    assert jnp.allclose(test_no_periodic[:, -1, :, :], 0.0)
    assert jnp.allclose(test_no_periodic[:, :, 0, :], 0.0)
    assert jnp.allclose(test_no_periodic[:, :, -1, :], 0.0)
    assert jnp.allclose(test_no_periodic[:, :, :, 0], 0.0)
    assert jnp.allclose(test_no_periodic[:, :, :, -1], 0.0)

    # Check that inner values are preserved
    assert jnp.allclose(test_no_periodic[:, 1:-1, 1:-1, 1:-1], fields)

    # Test with mixed periodic boundaries (x periodic, y and z not)
    test_mixed = pad_fields(fields, (True, False, False))

    # X-axis should have wrapped values
    assert jnp.allclose(test_mixed[:, 0, 1:-1, 1:-1], fields[:, -1, :, :])
    assert jnp.allclose(test_mixed[:, -1, 1:-1, 1:-1], fields[:, 0, :, :])

    # Y and Z axes should have zeros
    assert jnp.allclose(test_mixed[:, :, 0, :], 0.0)
    assert jnp.allclose(test_mixed[:, :, -1, :], 0.0)
    assert jnp.allclose(test_mixed[:, :, :, 0], 0.0)
    assert jnp.allclose(test_mixed[:, :, :, -1], 0.0)
