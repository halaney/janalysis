"""Implements the JPEG compression algorithm."""
import numpy
from .dct import dct2_twod_orthonormal, dct2_scipy
from .huffman import JPEG_HUFFMAN_DC_LUM, JPEG_HUFFMAN_AC_LUM
from .imageinput import get_image, crop_image_to_multiple_eight, \
                        get_ycbcr_bands, get_pixels, \
                        create_matrices_pixel_sequence
from .utils import ZIGZAG_ORDER, get_huffman_table_bit_string, \
                   get_magnitude_dc, get_ones_complement_bit_string


# Make this a command line arg
USE_MY_DCT = False  # Too slow


# Currently just using one quantization table for the whole file
L_QUANTIZATION_TABLE = numpy.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])


def jpeg_encode(input_path):
    """Implements JPEG compression."""
    # Extract the data into pixel matrices of the Y, Cb, Cr components
    original = get_image(input_path)
    crop = crop_image_to_multiple_eight(original)
    width, height = crop.size
    lum, chromb, chromr = get_ycbcr_bands(crop)
    lum_matrices = create_matrices_pixel_sequence(get_pixels(lum),
                                                  width, height)
    chromb_matrices = create_matrices_pixel_sequence(get_pixels(chromb),
                                                     width, height)
    chromr_matrices = create_matrices_pixel_sequence(get_pixels(chromr),
                                                     width, height)

    # Shift them to be centered around 0
    all_matrices = [lum_matrices, chromb_matrices, chromr_matrices]
    for matrices in all_matrices:
        for matrix in matrices:
            matrix -= 128

    # Take DCT of all
    for matrices in all_matrices:
        for index, matrix in enumerate(matrices):
            if USE_MY_DCT:
                matrices[index] = dct2_twod_orthonormal(matrix)
            else:
                matrices[index] = dct2_scipy(matrix)

    # Quantize all
    for matrices in all_matrices:
        for index, matrix in enumerate(matrices):
            matrices[index] = matrix / L_QUANTIZATION_TABLE

    # Round (just cast everything to an integer, this doesn't have to be exact)
    for matrices in all_matrices:
        for index, matrix in enumerate(matrices):
            matrices[index] = matrix.astype(numpy.int32)

    # Subtract DC components
    for matrices in all_matrices:
        for index, matrix in reversed(list(enumerate(matrices))):
            if index == 0:
                continue  # Don't subtract the first DC term
            matrices[index][0][0] = matrix[0][0] - matrices[index - 1][0][0]

    # Interleave the components
    length_of_components = len(all_matrices[0])
    interleaved = []
    for index in range(length_of_components):
        interleaved.append(all_matrices[0][index])  # Append the Y component
        interleaved.append(all_matrices[1][index])  # Append the Cb component
        interleaved.append(all_matrices[2][index])  # Append the Cr component

    # Zigzag all into a list of lists (each list is the zigzag of a block)
    zigzaged_lists = []
    for matrix in interleaved:
        zigzaged_matrix = []
        for index in ZIGZAG_ORDER:
            zigzaged_matrix.append(matrix[index[0]][index[1]])
        zigzaged_lists.append(zigzaged_matrix)

    # Run length encode all
    run_lengthed_lists = []
    for serial_list in zigzaged_lists:
        serial_index = 0
        run_length = []
        while serial_index < 64:
            if serial_index == 0:
                # DC component is treated differently
                # DC should be a 1 byte code representing length of following bits
                # the literal bits are a 1's complement representation of the actual mag
                value = serial_list[serial_index]
                run_length.append((get_magnitude_dc(value),
                                   get_ones_complement_bit_string(value)))
                serial_index += 1
                continue
            # AC component
            # Has a 1 byte code and literal bits following
            # The upper 4 bits of the code represent the ZRL (how many zero's)
            # The lower 4 bits of the code represent the length of the following bits
            # the literal bits are a 1's complement representation of the actual mag
            # Special code 0016 all remaining values in data unit are zero (is this just 00???)
            # Special code F0 represents 16 zeros
            # Neither special code is followed by the literal bits
            zero_count = 0
            while serial_index < 64 and serial_list[serial_index] == 0:
                zero_count += 1
                serial_index += 1
            if serial_index == 64:  # Rest of the block is zero
                run_length.append((0x00, ''))
                break
            while zero_count > 15:  # Encode as 16 zeroes as needed till nonzero
                run_length.append((0xF0, ''))
                zero_count -= 16
            nonzero_value = serial_list[serial_index]
            zero_count <<= 4
            zrl = zero_count | get_magnitude_dc(nonzero_value)
            run_length.append((zrl,
                               get_ones_complement_bit_string(nonzero_value)))
            serial_index += 1
        run_lengthed_lists.append(run_length)

    # Huffman encode the whole scan
    for run_length in run_lengthed_lists:
        for index, tup in enumerate(run_length):
            # Just using the luminance tables for simplicity
            if index == 0:
                tup = run_length[index]
                run_length[index] = (JPEG_HUFFMAN_DC_LUM[tup[0]], tup[1])
            else:
                tup = run_length[index]
                run_length[index] = (JPEG_HUFFMAN_AC_LUM[tup[0]], tup[1])

    # Format file, add DQT, DHT, and scan
    file_string = bin(0xFFD8)[2:].zfill(16)  # SOI

    file_string += bin(0xFFE0)[2:].zfill(16)  # APP0
    file_string += bin(0x0010)[2:].zfill(16)  # Length of APP0, including the length (16 bytes)
    file_string += bin(0x4A46494600)[2:].zfill(40)  # 'JFIF\0'
    file_string += bin(0x0102)[2:].zfill(16)  # JFIF version 1.02
    file_string += bin(0x01)[2:].zfill(8)  # Units (DPI)
    file_string += bin(0x0040)[2:].zfill(16)  # Arbitrary X DPI
    file_string += bin(0x0040)[2:].zfill(16)  # Arbitrary Y DPI
    file_string += bin(0x00)[2:].zfill(8)  # X thumbnail length
    file_string += bin(0x00)[2:].zfill(8)  # Y thumbnail length

    # Encode the quantization table
    dqt = []
    for index in ZIGZAG_ORDER:
        dqt.append(L_QUANTIZATION_TABLE[index[0]][index[1]])
    file_string += bin(0xFFDB)[2:].zfill(16)  # DQT Marker
    file_string += bin(0x0043)[2:].zfill(16)  # Length (67), including the length bytes
    file_string += bin(0x00)[2:].zfill(8)  # Table value sizes (0 means 1 byte) and table identifier 0
    for value in dqt:
        file_string += (bin(value)[2:]).zfill(8)  # Add each table value

    # Encode the huffman table
    # TODO: FIX THIS
    file_string += bin(0xFFC4)[2:].zfill(16)  # DHT MARKER

    # Create bitstring representation for the huffman tables
    dc_table_string = bin(0x00)[2:].zfill(8)  # DC table, identifier 0
    dc_table = get_huffman_table_bit_string(JPEG_HUFFMAN_DC_LUM)
    ac_table_string = bin(0x10)[2:].zfill(8)  # AC table, identifier 0
    ac_table = get_huffman_table_bit_string(JPEG_HUFFMAN_AC_LUM)

    # Write the length of the huffman tables plus the 2 bytes of the length bytes
    length = int(((len(dc_table) + len(ac_table)) / 8)) + 2 + 2
    dc_table_string += dc_table
    ac_table_string += ac_table
    file_string += (bin(length)[2:]).zfill(16)

    # Write the huffman tables
    file_string += dc_table_string + ac_table_string

    # Start of frame
    file_string += bin(0xFFC0)[2:].zfill(16)  # SOF0 marker
    file_string += bin(0x0011)[2:].zfill(16)  # Length
    file_string += bin(0x08)[2:].zfill(8)  # Sample precision (8 bits)
    file_string += (bin(height)[2:]).zfill(16)  # 2 bytes for height
    file_string += (bin(width)[2:]).zfill(16)  # 2 bytes for width
    file_string += (bin(0x03)[2:]).zfill(8)  # Number of components
    # Y component for SOF
    file_string += bin(0x01)[2:].zfill(8)  # 1 is the Y indicator for JFIF files
    file_string += bin(0x11)[2:].zfill(8)  # Sampling frequency of 1 to 1
    file_string += bin(0x00)[2:].zfill(8)  # Quantization table identifier
    # Cb component for SOF
    file_string += bin(0x02)[2:].zfill(8)  # 2 is the Cb indicator for JFIF files
    file_string += bin(0x11)[2:].zfill(8)  # Sampling frequency of 1 to 1
    file_string += bin(0x00)[2:].zfill(8)  # Quantization table identifier
    # Cr component for SOF
    file_string += bin(0x03)[2:].zfill(8)  # 3 is the Cb indicator for JFIF files
    file_string += bin(0x11)[2:].zfill(8)  # Sampling frequency of 1 to 1
    file_string += bin(0x00)[2:].zfill(8)  # Quantization table identifier

    # Scan
    file_string += bin(0xFFDA)[2:].zfill(16)  # Scan component
    file_string += bin(0x000C)[2:].zfill(16)  # Length
    file_string += bin(0x03)[2:].zfill(8)  # Component count
    file_string += bin(0x01)[2:].zfill(8)  # Y component
    file_string += bin(0x00)[2:].zfill(8)  # DC and AC huffman table identifiers
    file_string += bin(0x02)[2:].zfill(8)  # Cb component
    file_string += bin(0x00)[2:].zfill(8)  # DC and AC huffman table identifiers
    file_string += bin(0x03)[2:].zfill(8)  # Cr component
    file_string += bin(0x00)[2:].zfill(8)  # DC and AC huffman table identifiers
    file_string += bin(0x00)[2:].zfill(8)  # Spectral selection start
    file_string += bin(0x3F)[2:].zfill(8)  # Spectral selection end (63)
    file_string += bin(0x00)[2:].zfill(8)  # Successive approximation

    # Dump the scan data
    for run_length in run_lengthed_lists:
        for mag, literal in run_length:
            file_string += mag + literal

    # Add throw away bits to make it a byte
    if len(file_string) % 8:
        for bit in range(8 - len(file_string) % 8):
            file_string += '1'

    # EOI
    file_string += bin(0xFFD9)[2:].zfill(16)

    # Split our massive and inefficient bitstring into bytes
    bytez = [file_string[index:index+8] for
             index in range(0, len(file_string), 8)]
    bytez = [int(byte, 2) for byte in bytez]

    # Write all the bites to a file
    with open('test.jpg', 'wb') as fp:
        fp.write(bytes(bytez))


    # Temp print to show this works
    print('{} {} {}'.format(len(lum_matrices), len(chromb_matrices),
                            len(chromr_matrices)))
    print('{}'.format(len(run_lengthed_lists)))
    # print(lum_matrices)
    # print(chromb_matrices)
    # print(chromr_matrices)
    # print(zigzaged_lists)
    # print(run_lengthed_lists)






