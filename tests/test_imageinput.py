"""Tests functions used to handle image input."""
import pytest
from janalysis.imageinput import create_matrices_pixel_sequence


def test_create_matrices_small():
    """Tests a perfect 8x8 sequence."""
    # Create a fake sequence
    width = 8
    height = 8
    sequence = [number for number in range(width*height)]

    # Examine all elements to ensure its correct
    matrices = create_matrices_pixel_sequence(sequence, width, height)
    matrix = matrices.pop()
    print(matrix)
    expected = 0
    for row in range(8):
        for column in range(8):
            assert matrix[row][column] == expected
            expected += 1


def test_create_matrices_8by16():
    """Tests a larger 8x16 sequence."""
    # Create a fake sequence
    width = 16
    height = 8
    sequence = [number for number in range(width * height)]

    # Examine elements to ensure it's correct
    matrices = create_matrices_pixel_sequence(sequence, width, height)
    matrix1 = matrices[0]
    matrix2 = matrices[1]
    print(matrix1)
    print(matrix2)
    for row in range(height):
        for column in range(width):
            if column < 8:
                assert matrix1[row][column] == sequence[row*width+column]
            else:
                assert matrix2[row][column-8] == sequence[row*width+column]


def test_create_matrices_16by8():
    """Tests a larger 16x8 sequence."""
    # Create a fake sequence
    width = 8
    height = 16
    sequence = [number for number in range(width * height)]

    # Examine elements to ensure it's correct
    matrices = create_matrices_pixel_sequence(sequence, width, height)
    matrix1 = matrices[0]
    matrix2 = matrices[1]
    print(matrix1)
    print(matrix2)
    for row in range(height):
        for column in range(width):
            if row < 8:
                assert matrix1[row][column] == sequence[row*width+column]
            else:
                assert matrix2[row-8][column] == sequence[row*width+column]


def test_create_matrices_16by24():
    """Tests a larger 16x24 sequence."""
    # Create a fake sequence
    width = 24
    height = 16
    sequence = [number for number in range(width * height)]

    # Examine elements to ensure it's correct
    matrices = create_matrices_pixel_sequence(sequence, width, height)
    matrix1 = matrices[0]
    matrix2 = matrices[1]
    matrix3 = matrices[2]
    matrix4 = matrices[3]
    matrix5 = matrices[4]
    matrix6 = matrices[5]
    print(matrix1)
    print(matrix2)
    print(matrix3)
    print(matrix4)
    print(matrix5)
    print(matrix6)
    for row in range(height):
        for column in range(width):
            if row < 8:
                if column < 8:
                    assert matrix1[row][column] == sequence[row*width+column]
                elif column < 16:
                    assert matrix2[row][column-8] == sequence[row*width+column]
                else:
                    assert matrix3[row][column-16] == sequence[
                        row*width+column]
            else:
                if column < 8:
                    assert matrix4[row-8][column] == sequence[row*width+column]
                elif column < 16:
                    assert matrix5[row-8][column-8] == sequence[
                        row*width+column]
                else:
                    assert matrix6[row-8][column-16] == sequence[
                        row*width+column]


def test_create_matrices_24by16():
    """Tests a larger 24x16 sequence."""
    # Create a fake sequence
    width = 16
    height = 24
    sequence = [number for number in range(width * height)]

    # Examine elements to ensure it's correct
    matrices = create_matrices_pixel_sequence(sequence, width, height)
    matrix1 = matrices[0]
    matrix2 = matrices[1]
    matrix3 = matrices[2]
    matrix4 = matrices[3]
    matrix5 = matrices[4]
    matrix6 = matrices[5]
    print(matrix1)
    print(matrix2)
    print(matrix3)
    print(matrix4)
    print(matrix5)
    print(matrix6)
    for row in range(height):
        for column in range(width):
            if row < 8:
                if column < 8:
                    assert matrix1[row][column] == sequence[row*width+column]
                elif column < 16:
                    assert matrix2[row][column-8] == sequence[row*width+column]
            elif row < 16:
                if column < 8:
                    assert matrix3[row-8][column] == sequence[row*width+column]
                elif column < 16:
                    assert matrix4[row-8][column-8] == sequence[
                        row*width+column]
            else:
                if column < 8:
                    assert matrix5[row-16][column] == sequence[
                        row*width+column]
                elif column < 16:
                    assert matrix6[row-16][column-8] == sequence[
                        row*width+column]


def test_create_matrices_invalid():
    """Tests than an exception is thrown if invalid input is used."""
    with pytest.raises(AssertionError):
        create_matrices_pixel_sequence([], 7, 8)
    with pytest.raises(AssertionError):
        create_matrices_pixel_sequence([], 8, 7)
    with pytest.raises(AssertionError):
        create_matrices_pixel_sequence([], 8, 8)


def test_create_matrices_really_big():
    """Tests random values from a very big sequence."""
    # Create a fake sequence (4K resolution)
    width = 3840
    height = 2160
    sequence = [number for number in range(width * height)]

    # Test some random spots
    matrices = create_matrices_pixel_sequence(sequence, width, height)
    matrix = matrices[0]
    assert matrix[0][0] == 0
    assert matrix[0][4] == 4
    matrix = matrices[479]
    assert matrix[0][7] == 3839
    matrix = matrices.pop()
    assert matrix[7][7] == 8294399
