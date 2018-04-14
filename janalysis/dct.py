"""Implements the DCT."""
from math import cos, pi, sqrt
import numpy
import scipy.fftpack


def dct2(x_list, n_point=None):
    """Implements the n_point dct2.
    :param x_list: The time domain sequence.
    :type x_list: list.
    :param n_point: The n point size of the DCT to be performed (default None
                    is to use the size of the input).
    :type n_point: int.
    :returns: numpy.array -- the transform representation of the sequence.
    """
    if not n_point:
        n_point = len(x_list)
    x_array = numpy.array(x_list)
    y_array = numpy.zeros(n_point)
    for index_k in range(n_point):
        for index_n in range(n_point):
            if index_n < x_array.size:
                temp = cos(pi * index_k * (2 * index_n + 1) / (2 * n_point))
                y_array[index_k] += x_array[index_n] * temp
        y_array[index_k] *= 2
    return y_array


def dct2_twod_orthonormal(two_d_array):
    """
    Implements the 2d dct2 as JPEG requires it (orthonormal, N points' size
    equal to row and columns).
    :param two_d_array: The time domain sequence.
    :type two_d_array: numpy.array.
    :returns: numpy.array -- the transform representation of the sequence.
    """
    matrix_dimensions = two_d_array.shape
    output = numpy.zeros(matrix_dimensions)
    for row_index, output_row in enumerate(output):
        for col_index in range(output_row.size):
            output[row_index][col_index] = _get_twod_value_dct2(row_index,
                                                                col_index,
                                                                two_d_array)
    return output


def dct2_scipy(two_d_array):
    """Implements the 2d orthonormal dct using scipy (much quicker)."""
    return scipy.fftpack.dct(scipy.fftpack.dct(two_d_array.T, norm='ortho')
                             .T, norm='ortho')


def _get_twod_value_dct2(output_row_index, output_column_index, two_d_array):
    """Does the actual computation for each entry in the 2d orthonormal dct.
    :param output_row_index: The row index of the entry to be calculated
    :type output_row_index: unsigned int.
    :param output_column_index: The column index of the entry to be calculated.
    :type output_column_index: unsigned int.
    :param two_d_array: The input 2d array that is being transformed.
    :type two_d_array: numpy.array.
    :returns: double -- The value calculated for the specific matrix entry.
    """
    value = 0
    row_size = two_d_array.shape[0]
    column_size = two_d_array.shape[1]
    for row_index, row in enumerate(two_d_array):
        for column_index in range(row.size):
            argx = (row_index+0.5) * output_row_index * pi / row_size
            argy = (column_index+0.5) * output_column_index * pi / column_size
            value += two_d_array[row_index][column_index]*cos(argx)*cos(argy)
    value *= 4  # No idea why this 4 needs to be here, but it does
    value *= _get_normalization_factor(output_row_index, row_size)
    value *= _get_normalization_factor(output_column_index, column_size)
    return value


def _get_normalization_factor(index, length):
    """Returns the normalization factor needed by the 2d dct to be orthonormal.
    :param index: The index.
    :type index: unsigned int.
    :param length: The length.
    :type length: unsigned int.
    :returns: double -- The normalization factor for this index.
    """
    norm = 0
    if index == 0:
        norm = sqrt(1 / (4 * length))
    else:
        norm = sqrt(1 / (2 * length))
    return norm
