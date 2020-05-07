# uncomment the following lines if running on linux
CF_EXTRA = -Wno-deprecated-declarations
OPENCL_LIB = -lOpenCL
#uncomment the following lines if running on Mac OS
# CF_EXTRA = -DCL_SILENCE_DEPRECATION
# OPENCL_LIB = -framework OpenCL
#end: linux versus Mac OS

C_FLAGS = -c -g -std=c++11 $(CF_EXTRA)

LIB = ../lib/libCOGLImageWriter.so
INC = -I../Packed3DArray -I../ImageWriter

OBJS = project3.o readSource.o

project3: $(OBJS) $(LIB)
	g++ -fPIC -g -o project3 $(OBJS) $(LIB) $(OPENCL_LIB)

../lib/libCOGLImageWriter.so: ../ImageWriter/ImageWriter.h ../ImageWriter/ImageWriter.c++
	(cd ../ImageWriter; make)

project3.o: project3.c++
	g++ -fPIC $(C_FLAGS) $(INC) project3.c++

readSource.o: readSource.c++
	g++ -fPIC $(C_FLAGS) readSource.c++
