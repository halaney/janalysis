# janalysis
JPEG Compression implementation for ECE493 (Digital Signal Processing)

This program converts a PNG file (or any image file that is represented using
RGB and is decodable by Pillow) to a JPEG/JFIF file. The entrypoint defined
is called jpeg-encoder, calling this will prompt for input path, output path,
which DCT to use, and for a scaling factor to affect the quality. This was done
as a project for the course, and the main intent was to examine how the JPEG
compression process works. It is definitely not done in the most efficient manner
but is a fairly straight forward implementation of the process and could be used
by others as a reference.
