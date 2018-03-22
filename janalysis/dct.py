"""Implements the DCT."""
from math import cos, pi
import numpy


def dct2(x_list, n_point=None):
    """Implements the n_point dct2.
    :param x_list: The time domain sequence.
    :type x_list: list.
    :param n_point: The n point size of the DCT to be performed (default None
                    is to use the size of the input).
    :type n_point: int.
    :returns: array -- the transform representation of the sequence.
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
