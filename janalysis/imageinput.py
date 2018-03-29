"""Handles reading in images and preparing them for compression."""
import numpy


def create_matrices_pixel_sequence(pixels, width, height):
    """Creates a list of 8x8 matrices from the pixel sequence.
    :param : The sequence of pixels representing the image/colorband.
    :type : list.
    :param width: the width of the original image (must be a multiple of 8).
    :type n_point: int.
    :param height: the height of the original image (must be a multiple of 8).
    :type n_point: int.
    :returns: list -- A list of the numpy.array matrices (from left to right).
    """
    if width % 8 != 0 or height % 8 != 0 or len(pixels) != width*height:
        # Not at all the best way to handle this
        assert width % 8 == 0
        assert height % 8 == 0
        assert len(pixels) == width*height
    matrices = []
    start_of_row = 0
    end_of_row = start_of_row + width

    # Jump down 8 rows once we get to the end of a row
    while start_of_row < width * height:
        # Create matrices across a row
        current = start_of_row
        while current < end_of_row:
            matrix = numpy.empty((8, 8))
            for column in range(current, current + 8):
                for row in range(8):
                    matrix[row][column-current] = pixels[column + row * width]
            matrices.append(matrix)
            current += 8
        start_of_row += 8 * width  # Jump down 8 rows
        end_of_row = start_of_row + width

    return matrices
