# uncomment the following lines if running on linux
CF_EXTRA = -Wno-deprecated-declarations
OPENCL_LIB = -lOpenCL
#uncomment the following lines if running on Mac OS
# CF_EXTRA = -DCL_SILENCE_DEPRECATION
# OPENCL_LIB = -framework OpenCL
#end: linux versus Mac OS

C_FLAGS = -c -g -std=c++11 $(CF_EXTRA)

OBJS = matrixMultiplyV1.o readSource.o

matrixMultiplyV1: $(OBJS)
	g++ -g $(OBJS) -o matrixMultiplyV1 $(OPENCL_LIB)

matrixMultiplyV1.o: matrixMultiplyV1.c++
	g++ $(C_FLAGS) matrixMultiplyV1.c++

readSource.o: readSource.c++
	g++ $(C_FLAGS) readSource.c++
