// example.c++ -- Illustrate use of ImageWriter

#include <iostream>
#include <fstream>
#include <cstdlib>

#include "ImageWriter.h"

int main(int argc, char* argv[])
{
	if (argc < 7)
		std::cerr << "Usage: " << argv[0] << " numRows numCols R G B outputImageFile\n";
	else
	{
		int numRows = atoi(argv[1]);
		int numCols = atoi(argv[2]);
		double RGB[3] = { atof(argv[3]), atof(argv[4]), atof(argv[5]) };
		int numChannels = 3; // R, G, B
		ImageWriter* iw = ImageWriter::create(argv[6], numCols, numRows, numChannels);
		if (iw == nullptr)
			exit(1);

		// We would launch a GPU kernel to get the data to be written; let's just
		// use a placeholder here:
		unsigned char* image = new unsigned char[numRows * numCols * numChannels];
		for (int r=0 ; r<numRows ; r++)
		{
			for (int c=0 ; c<numCols ; c++)
			{
				for (int chan=0 ; chan<numChannels ; chan++)
				{
					int loc = r*numCols*numChannels + c*numChannels + chan;
					// In your GPU code, you will either have the kernel return a buffer of unsigned char,
					// or return a float or double buffer and do the following:
					unsigned char pixelVal = static_cast<unsigned char>(RGB[chan]*255.0 + 0.5);
					// In any event, place the unsigned char into the buffer to be written to the output
					// image file.  It MUST be a one-byte 0..255 value.
					image[loc] = pixelVal;
				}
			}
		}

		iw->writeImage(image);
		iw->closeImageFile();
		delete iw;
		delete [] image;
	}
	return 0;
}
