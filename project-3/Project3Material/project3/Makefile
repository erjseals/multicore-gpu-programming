all: project3

# uncomment the following lines if running on linux
CF_EXTRA = -Wno-deprecated-declarations
OPENCL_LIB = -lOpenCL
#uncomment the following lines if running on Mac OS
# CF_EXTRA = -DCL_SILENCE_DEPRECATION
# OPENCL_LIB = -framework OpenCL
#end: linux versus Mac OS

CPP = g++ -fPIC -std=c++11
LINK = g++ -fPIC

INC = -I../Packed3DArray -I../ImageWriter
CFLAGS = -g -c $(INC)
C_FLAGS = -c -g -std=c++11 $(CF_EXTRA)

LIB = ../lib/libCOGLImageWriter.so

project3: project3.o readSource.o $(LIB)
	$(LINK) -g -o project3 project3.o $(OPENCL_LIB) $(LIB)

../lib/libCOGLImageWriter.so: ../ImageWriter/ImageWriter.h ../ImageWriter/ImageWriter.c++
	(cd ../ImageWriter; make)

project3.o: project3.c++
	$(CPP) $(CFLAGS) project3.c++

readSource.o: readSource.c++
	g++ $(C_FLAGS) readSource.c++