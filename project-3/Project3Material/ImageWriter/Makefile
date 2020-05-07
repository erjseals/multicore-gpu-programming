# If you are using cuda for project 3, use the following:
# CPP = /usr/local/gcc-4.9.3/bin/g++ -std=c++11 -fPIC
# If your project is OpenCL, use:
CPP = g++ -std=c++11 -fPIC

INCLUDES = -I../Packed3DArray -I/usr/include/x86_64-linux-gnu

CFLAGS = -O -c $(INCLUDES)

OBJS = ImageWriter.o BMPImageWriter.o JPEGImageWriter.o PNGImageWriter.o

../lib/libCOGLImageWriter.so: libCOGLImageWriter.so
	cp libCOGLImageWriter.so ../lib/

libCOGLImageWriter.so: $(OBJS)
	g++ -shared -o libCOGLImageWriter.so $(OBJS) -lpng -ljpeg

ImageWriter.o: ImageWriter.c++
	$(CPP) $(CFLAGS) ImageWriter.c++

BMPImageWriter.o: BMPImageWriter.c++
	$(CPP) $(CFLAGS) BMPImageWriter.c++

JPEGImageWriter.o: JPEGImageWriter.c++
	$(CPP) $(CFLAGS) JPEGImageWriter.c++

PNGImageWriter.o: PNGImageWriter.c++
	$(CPP) $(CFLAGS) PNGImageWriter.c++
