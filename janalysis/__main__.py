"""Defines a cli entrypoint for the encoder to run."""
import os
import sys
from .jpeg import jpeg_encode, L_QUANTIZATION_TABLE


def main():
    """Gets the options for converting a PNG to a JPEG."""
    inputfile = input('Enter PNG file to convert to a jpeg: ')
    outputfile = input('Enter desired name of output file: ')
    print('Original quantization matrix:')
    print(L_QUANTIZATION_TABLE)
    scalefactor = float(input('Enter scaling factor for quantization matrix: '))
    jpeg_encode(inputfile, scalefactor, outputfile)
    original_size = os.path.getsize(inputfile)
    new_size = os.path.getsize(outputfile)
    print('PNG file size: {} bytes'.format(original_size))
    print('JPEG file size: {} bytes'.format(new_size))
    percent_smaller = 100*float(original_size-new_size)/original_size
    print('JPEG file size is {:.2f}% smaller than the PNG file size'.format(percent_smaller))
    sys.exit(0)


if __name__ == '__main__':
    main()
