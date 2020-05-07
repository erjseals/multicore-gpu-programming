all: daxpy saxpy

# uncomment the following lines if running on linux
CF_EXTRA = -Wno-deprecated-declarations
OPENCL_LIB = -lOpenCL
#uncomment the following lines if running on Mac OS
# CF_EXTRA = -DCL_SILENCE_DEPRECATION
# OPENCL_LIB = -framework OpenCL
#end: linux versus Mac OS

C_FLAGS = -c -g -std=c++11 $(CF_EXTRA)

OBJS_D = daxpy.o readSource.o
OBJS_S = saxpy.o readSource.o

daxpy: $(OBJS_D)
	g++ -g $(OBJS_D) -o daxpy $(OPENCL_LIB)

daxpy.o: daxpy.c++
	g++ $(C_FLAGS) daxpy.c++

saxpy: $(OBJS_S)
	g++ -g $(OBJS_S) -o saxpy $(OPENCL_LIB)

saxpy.o: saxpy.c++
	g++ $(C_FLAGS) saxpy.c++

readSource.o: readSource.c++
	g++ $(C_FLAGS) readSource.c++
