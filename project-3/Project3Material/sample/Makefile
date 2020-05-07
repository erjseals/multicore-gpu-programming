# uncomment the following lines if running on linux
CF_EXTRA = -Wno-deprecated-declarations
OPENCL_LIB = -lOpenCL
#uncomment the following lines if running on Mac OS
# CF_EXTRA = -DCL_SILENCE_DEPRECATION
# OPENCL_LIB = -framework OpenCL
#end: linux versus Mac OS

C_FLAGS = -c -g -std=c++11 $(CF_EXTRA)

OBJS = HelloOpenCL.o readSource.o

HelloOpenCL: $(OBJS)
	g++ -g $(OBJS) -o HelloOpenCL $(OPENCL_LIB)

HelloOpenCL.o: HelloOpenCL.c++
	g++ $(C_FLAGS) HelloOpenCL.c++

readSource.o: readSource.c++
	g++ $(C_FLAGS) readSource.c++
