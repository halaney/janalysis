"""Test the DCT implementation."""
import numpy
import scipy.fftpack
from janalysis.dct import dct2, dct2_twod_orthonormal


def test_dct2_simple():
    """Tests the 1D version of the dct2."""
    time_domain = [1, 1, 1, 1]
    expected = scipy.fftpack.dct(time_domain)
    actual = dct2(time_domain)
    numpy.testing.assert_array_almost_equal(actual, expected)


def test_dct2_same_length():
    """Tests the 1D version of the dct2 with n_point of equal length."""
    time_domain = [1, 1, 1, 1]
    expected = scipy.fftpack.dct(time_domain)
    actual = dct2(time_domain, len(time_domain))
    numpy.testing.assert_array_almost_equal(actual, expected)


def test_dct2_smaller_length():
    """Tests the 1D version of the dct2 with n_point less than input length."""
    time_domain = [1, 1, 1, 1]
    expected = scipy.fftpack.dct(time_domain, n=3)
    actual = dct2(time_domain, 3)
    numpy.testing.assert_array_almost_equal(actual, expected)


def test_dct2_bigger_length():
    """Tests the 1D version of the dct2 with n_point greater than length."""
    time_domain = [1, 1, 1, 1]
    expected = scipy.fftpack.dct(time_domain, n=8)
    actual = dct2(time_domain, 8)
    numpy.testing.assert_array_almost_equal(actual, expected)


def test_twod_dct2_dc_square():
    """Tests the 2d dct (orthonormalized) on a dc sequence."""
    time_domain = numpy.array([[1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1]])
    expected = scipy.fftpack.dct(scipy.fftpack.dct(time_domain.T, norm='ortho')
                                 .T, norm='ortho')
    actual = dct2_twod_orthonormal(time_domain)
    numpy.testing.assert_array_almost_equal(actual, expected)


def test_twod_dct2_smaller_square():
    """Tests the 2d dct (orthonormalized)."""
    time_domain = numpy.array([[1, 2, 3, 4],
                               [4, 3, 2, 1],
                               [1, 2, 3, 4],
                               [4, 3, 2, 1]])
    expected = scipy.fftpack.dct(scipy.fftpack.dct(time_domain.T, norm='ortho')
                                 .T, norm='ortho')
    actual = dct2_twod_orthonormal(time_domain)
    numpy.testing.assert_array_almost_equal(actual, expected)


def test_twod_dct2_non_square():
    """Tests the 2d dct (orthonormalized)."""
    time_domain = numpy.array([[1, 2, 3],
                               [4, 3, 2],
                               [1, 2, 3],
                               [4, 3, 2]])
    expected = scipy.fftpack.dct(scipy.fftpack.dct(time_domain.T, norm='ortho')
                                 .T, norm='ortho')
    actual = dct2_twod_orthonormal(time_domain)
    numpy.testing.assert_array_almost_equal(actual, expected)
