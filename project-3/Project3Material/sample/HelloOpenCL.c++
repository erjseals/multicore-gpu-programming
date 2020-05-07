// The OpenCL version of Hello, World

#include <iostream>
#include <string>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

struct NameTable
{
	std::string name;
	int value;
};

const char* readSource(const char* fileName);

// A couple simple utility functions:
bool debug = false;
void checkStatus(std::string where, cl_int status, bool abortOnError)
{
	if (debug || (status != 0))
		std::cout << "Step " << where << ", status = " << status << '\n';
	if ((status != 0) && abortOnError)
		exit(1);
}

void reportPlatformInformation(const cl_platform_id& platformIn)
{
	NameTable what[] = {
		{ "CL_PLATFORM_PROFILE:    ", CL_PLATFORM_PROFILE },
		{ "CL_PLATFORM_VERSION:    ", CL_PLATFORM_VERSION },
		{ "CL_PLATFORM_NAME:       ", CL_PLATFORM_NAME },
		{ "CL_PLATFORM_VENDOR:     ", CL_PLATFORM_VENDOR },
		{ "CL_PLATFORM_EXTENSIONS: ", CL_PLATFORM_EXTENSIONS },
		{ "", 0 }
	};
	size_t size;
	char* buf = nullptr;
	int bufLength = 0;
	std::cout << "===============================================\n";
	std::cout << "========== PLATFORM INFORMATION ===============\n";
	std::cout << "===============================================\n";
	for (int i=0 ; what[i].value != 0 ; i++)
	{
		clGetPlatformInfo(platformIn, what[i].value, 0, nullptr, &size);
		if (size > bufLength)
		{
			if (buf != nullptr)
				delete [] buf;
			buf = new char[size];
			bufLength = size;
		}
		clGetPlatformInfo(platformIn, what[i].value, bufLength, buf, &size);
		std::cout << what[i].name << buf << '\n';
	}
	std::cout << "================= END =========================\n\n";
	if (buf != nullptr)
		delete [] buf;
}

void showProgramBuildLog(cl_program pgm, cl_device_id dev)
{
	size_t size;
	clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
	char* log = new char[size+1];
	clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, size+1, log, nullptr);
	std::cout << "LOG:\n" << log << "\n\n";
	delete [] log;
}

// Typical OpenCL startup

// Some global state variables (These would be better packaged as
// instance variables of some class.)
// 1) Platforms
cl_uint numPlatforms = 0;
cl_platform_id* platforms = nullptr;
cl_platform_id curPlatform;
// 2) Devices
cl_uint numDevices = 0;
cl_device_id* devices = nullptr;

// Return value is device index to use; -1 ==> no available devices
int typicalOpenCLProlog(cl_device_type desiredDeviceType)
{
	// ----------------------------------------------------
	// Discover and initialize the platforms
	// ----------------------------------------------------

	cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
	checkStatus("clGetPlatformIDs-0", status, true);
	if (numPlatforms <= 0)
	{
		std::cout << "No platforms!\n";
		return -1;
	}

	platforms = new cl_platform_id[numPlatforms];
 
	status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
	checkStatus("clGetPlatformIDs-1", status, true);
	int which = 0;
	if (numPlatforms > 1)
	{
		std::cout << "Found " << numPlatforms << " platforms:\n";
		for (int i=0 ; i<numPlatforms ; i++)
		{
			std::cout << i << ": ";
			reportPlatformInformation(platforms[i]);
		}
		which = -1;
		while ((which < 0) || (which >= numPlatforms))
		{
			std::cout << "Which platform do you want to use? ";
			std::cin >> which;
		}
	}
	curPlatform = platforms[which];

	std::cout << "Selected platform:\n";
	reportPlatformInformation(curPlatform);

	// ------------------------------------------------------------------
	// Discover and initialize the devices on a specific platform
	// ------------------------------------------------------------------

	status = clGetDeviceIDs(curPlatform, desiredDeviceType, 0, nullptr, &numDevices);
	checkStatus("clGetDeviceIDs-0", status, true);
	if (numDevices <= 0)
	{
		std::cout << "No devices on platform!\n";
		return -1;
	}

	devices = new cl_device_id[numDevices];

	status = clGetDeviceIDs(curPlatform, desiredDeviceType, numDevices, devices, nullptr);
	checkStatus("clGetDeviceIDs-1", status, true);
	int devIndex = 0;
	if (numDevices > 1)
	{
		size_t nameLength;
		for (int idx=0 ; idx<numDevices ; idx++)
		{
			clGetDeviceInfo(devices[idx], CL_DEVICE_NAME, 0, nullptr, &nameLength);
			char* name = new char[nameLength+1];
			clGetDeviceInfo(devices[idx], CL_DEVICE_NAME, nameLength+1, name, nullptr);
			// You can also query lots of other things about the device capability,
			// for example, CL_DEVICE_EXTENSIONS to see if "cl_khr_fp64" is included.
			// (See also the first line of saxpy.cl.)
			std::cout << "Device " << idx << ": " << name << '\n';
			delete [] name;
		}
		devIndex = -1;
		while ((devIndex < 0) || (devIndex >= numDevices))
		{
			std::cout << "Which device do you want to use? ";
			std::cin >> devIndex;
		}
	}
	else if (numDevices <= 0)
		std::cout << "No devices found\n";
	else
		std::cout << "Only one device detected\n";
	return devIndex;
}

int main(int argc, char* argv[])
{
	// In "real" OpenCL programs, the number of dimensions for the kernel
	// is determined based on what maps best to the problem.  In this
	// simple demo program, we will be seeing different ways to lay out
	// kernel grids, so we will allow a command line parameter to be used
	// to set the desired dimension.
	int numDimsToUse = 1;
	if (argc > 1)
		numDimsToUse = atoi(argv[1]);

	int useDevice = typicalOpenCLProlog(CL_DEVICE_TYPE_ALL);
	if (useDevice < 0)
		return 0;

	//-------------------------------------------------------------------
	// Create a context for the selected device
	//-------------------------------------------------------------------

	cl_int status;
	cl_context context = clCreateContext(nullptr, 1, &devices[useDevice],
		nullptr, nullptr, &status);
	checkStatus("clCreateContext", status, true);

	//-------------------------------------------------------------
	// Create a command queue for the device in the current context
	//-------------------------------------------------------------

	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[useDevice], 
		0, &status);
	checkStatus("clCreateCommandQueue", status, true);

	//-----------------------------------------------------
	// Create, compile, and link the program
	//----------------------------------------------------- 

	const char* programSource[] = { readSource("HelloOpenCL.cl") };
	cl_program program = clCreateProgramWithSource(context, 
		1, programSource, nullptr, &status);
	checkStatus("clCreateProgramWithSource", status, true);
	status = clBuildProgram(program, 1, &devices[useDevice], 
		nullptr, nullptr, nullptr);
	if (status != 0)
		showProgramBuildLog(program, devices[useDevice]);
	checkStatus("clBuildProgram", status, true);

	//-----------------------------------------------------------
	// Create a kernel from one of the __kernel functions
	//         in the source that was built.
	//-----------------------------------------------------------

	cl_kernel kernel = clCreateKernel(program, "helloOpenCL", &status);

	//-----------------------------------------------------
	// Configure the work-item structure
	//----------------------------------------------------- 

	size_t globalWorkSize[] = { 64, 32, 32 };
	// if numDimsToUse == 3, then the following declaration means
	// our Work Groups will form a 2 x 4 x 8 array.
	size_t localWorkSize[] = { 32, 8, 4 };
	if (numDimsToUse == 1)
		// our Work Groups will form a 1D array of length 2.
		localWorkSize[0] = 32;
	else if (numDimsToUse == 2)
	{
		// the following declaration means our Work Groups will form
		// a 2 x 2 array.
		localWorkSize[0] = 32;
		localWorkSize[1] = 16;
	}

	//-----------------------------------------------------
	// Enqueue the kernel for execution
	//----------------------------------------------------- 

	float a = 10.0; // will be unused; no-parameter kernels are not supported
	status = clSetKernelArg(kernel, 0, sizeof(float), &a);
	status = clEnqueueNDRangeKernel(cmdQueue, kernel,
		numDimsToUse,
		nullptr, globalWorkSize, // globalOffset, globalSize
		localWorkSize, // This is allowed to be nullptr
		0, nullptr, nullptr); // event information, if needed
	checkStatus("clEnqueueNDRangeKernel", status, true);

	// block until all commands have finished execution
	clFinish(cmdQueue);

	//-----------------------------------------------------
	// Release OpenCL resources
	//----------------------------------------------------- 

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);

	// Free host resources
	delete [] platforms;
	delete [] devices;

	return 0;
}
