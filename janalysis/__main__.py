"""Defines a cli entrypoint for the encoder to run."""
import sys
from .jpeg import jpeg_encode


def main():
    """Gets the filepath from the first command line arg and encodes the image
    to a JPEG.
    """
    jpeg_encode(sys.argv[1])
    sys.exit(0)


if __name__ == '__main__':
    main()
