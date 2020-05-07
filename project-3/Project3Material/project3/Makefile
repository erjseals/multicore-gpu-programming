# uncomment the following lines if running on linux
CF_EXTRA = -Wno-deprecated-declarations
OPENCL_LIB = -lOpenCL
#uncomment the following lines if running on Mac OS
# CF_EXTRA = -DCL_SILENCE_DEPRECATION
# OPENCL_LIB = -framework OpenCL
#end: linux versus Mac OS

LINK = g++ -fPIC

INC = -I../Packed3DArray -I../ImageWriter

LIB = ../lib/libCOGLImageWriter.so

C_FLAGS = -c -g -std=c++11 $(CF_EXTRA)

OBJS = project3.o readSource.o

project3: $(OBJS) $(LIB)
	$(LINK) -g -o $(OBJS) project3 $(LIB) $(OPENCL_LIB)

../lib/libCOGLImageWriter.so: ../ImageWriter/ImageWriter.h ../ImageWriter/ImageWriter.c++
	(cd ../ImageWriter; make)

project3.o: project3.c++
	g++ $(C_FLAGS) project3.c++

readSource.o: readSource.c++
	g++ $(C_FLAGS) readSource.c++
