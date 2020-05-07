# uncomment the following lines if running on linux
CF_EXTRA = -Wno-deprecated-declarations
OPENCL_LIB = -lOpenCL
#uncomment the following lines if running on Mac OS
# CF_EXTRA = -DCL_SILENCE_DEPRECATION
# OPENCL_LIB = -framework OpenCL
#end: linux versus Mac OS

PROG = project3

CPP = g++ -fPIC -std=c++11
LINK = g++ -fPIC

INC = -I../Packed3DArray -I../ImageWriter
CFLAGS = -g -c $(INC)

LIB = ../lib/libCOGLImageWriter.so

$(PROG): $(PROG).o $(LIB)
	$(LINK) -g -o $(PROG) $(PROG).o $(LIB)

../lib/libCOGLImageWriter.so: ../ImageWriter/ImageWriter.h ../ImageWriter/ImageWriter.c++
	(cd ../ImageWriter; make)

$(PROG).o: $(PROG).c++
	$(CPP) $(CFLAGS) $(PROG).c++

readSource.o: readSource.c++
	g++ -c -g -std=c++11 $(CF_EXTRA) readSource.c++
