"""Test the DCT implementation."""
import numpy
import scipy.fftpack
from janalysis.dct import dct2


def test_1d_dct2_simple():
    """Tests the 1D version of the dct2."""
    time_domain = [1, 1, 1, 1]
    expected = scipy.fftpack.dct(time_domain)
    actual = dct2(time_domain)
    numpy.testing.assert_array_almost_equal(actual, expected)


def test_1d_dct2_simple_specific_length():
    """Tests the 1D version of the dct2 with n_point of equal length."""
    time_domain = [1, 1, 1, 1]
    expected = scipy.fftpack.dct(time_domain)
    actual = dct2(time_domain, len(time_domain))
    numpy.testing.assert_array_almost_equal(actual, expected)


def test_1d_dct2_smaller_length():
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
