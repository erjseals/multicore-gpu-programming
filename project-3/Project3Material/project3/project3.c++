#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <string.h>
#include <math.h>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "ImageWriter.h"\

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

void lookAtDeviceCharacteristics(cl_device_id dev)
{
	cl_ulong gms;
	clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &gms, nullptr);
	cl_ulong lms;
	clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, nullptr);
	size_t mwgs;
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mwgs, nullptr);
	cl_uint maxCUs;
	clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxCUs, nullptr);

	size_t extLength;
	clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &extLength);
	char* extString = new char[extLength+1];
	clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, extLength+1, extString, nullptr);

	std::cout << "Device global mem size:     " << gms << '\n';
	std::cout << "Device local mem size:      " << lms << '\n';
	std::cout << "Device max work group size: " << mwgs << '\n';
	std::cout << "Device max compute units:   " << maxCUs << '\n';
	std::cout << "Device extensions string:   " << extString << '\n';
	std::cout << '\n';

	delete [] extString;
}

void lookAtKernelCharacteristics(cl_kernel kernel, cl_device_id dev)
{
	cl_ulong lms = -1;
	clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lms, nullptr);
	cl_ulong pms = -1;
	clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &pms, nullptr);
	size_t warpSize = -1;
	clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &warpSize, nullptr);
	size_t maxWorkGroupSize = -1;
	clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr);

	std::cout << "Kernel local memory size:   " << lms << '\n';
	std::cout << "Kernel private memory size: " << pms << '\n';
	std::cout << "Kernel warpSize:            " << warpSize << '\n';
	std::cout << "Kernel max work group size: " << maxWorkGroupSize << '\n';
	std::cout << '\n';
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
			// (See also the first line of daxpy.cl.)
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

void doTheKernelLaunch(cl_device_id dev, double a, double* h_X, double* h_Y,
	size_t arraySize, double* h_Z)
{
	// --------------------------------------------------
	// Create a context for the one chosen device
	// --------------------------------------------------
	
	cl_int status;
	cl_context context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
	checkStatus("clCreateContext", status, true);

	// ------------------------------------------------------------
	// Create a command queue for one device in the context
	// (There is one queue per device per context.)
	// ------------------------------------------------------------

	cl_command_queue cmdQueue = clCreateCommandQueue(context, dev, 0, &status);
	checkStatus("clCreateCommandQueue", status, true);

	// ---------------------------------------------------------
	// Create device buffers associated with the context
	// ---------------------------------------------------------

	size_t datasize = arraySize * sizeof(double);

	cl_mem bufferX = clCreateBuffer( // Input array on the device
		context, CL_MEM_READ_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-X", status, true);

	cl_mem bufferY = clCreateBuffer( // Input array on the device
		context, CL_MEM_READ_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-Y", status, true);

	cl_mem bufferZ = clCreateBuffer( // Output array on the device
		context, CL_MEM_WRITE_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-Z", status, true);

	// ------------------------------------------------------
	// Use the command queue to encode requests to write host
	// data to the device buffers
	// ------------------------------------------------------

	status = clEnqueueWriteBuffer(cmdQueue, 
		bufferX, CL_FALSE, 0, datasize,                         
		h_X, 0, nullptr, nullptr);
	checkStatus("clEnqueueWriteBuffer-X", status, true);

	status = clEnqueueWriteBuffer(cmdQueue, 
		bufferY, CL_FALSE, 0, datasize,                                  
		h_Y, 0, nullptr, nullptr);
	checkStatus("clEnqueueWriteBuffer-Y", status, true);

	// ----------------------------------------------------
	// Create, compile, and link the program
	// ----------------------------------------------------

	const char* programSource[] = { readSource("daxpy.cl") };
	cl_program program = clCreateProgramWithSource(context, 
		1, programSource, nullptr, &status);
	checkStatus("clCreateProgramWithSource", status, true);

	status = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
	if (status != 0)
		showProgramBuildLog(program, dev);
	checkStatus("clBuildProgram", status, true);

	// ---------------------------------------------------------------------
	// Create a kernel using a "__kernel" function in the ".cl" file
	// ---------------------------------------------------------------------

	cl_kernel kernel = clCreateKernel(program, "daxpy", &status);

	lookAtDeviceCharacteristics(dev); // Just FYI - has no bearing on this program
	lookAtKernelCharacteristics(kernel, dev); // Just FYI - has no bearing on this program

	// ----------------------------------------------------
	// Set the kernel arguments
	// ----------------------------------------------------

	status = clSetKernelArg(kernel, 0, sizeof(double), &a);
	checkStatus("clSetKernelArg-0", status, true);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferX);
	checkStatus("clSetKernelArg-1", status, true);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferY);
	checkStatus("clSetKernelArg-2", status, true);
	status = clSetKernelArg(kernel, 3, sizeof(int), &arraySize);
	checkStatus("clSetKernelArg-3", status, true);
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferZ);
	checkStatus("clSetKernelArg-4", status, true);

	// ------------------------------------------------------
	// Configure the work-item structure
	// These are arrays of length n where n=dimension of the
	// computational grid. (n <= 3) Here we are preparing a
	// 1-dimensional computational grid (i.e., n=1)
	// ------------------------------------------------------

	size_t globalWorkSize[] = { arraySize };
	size_t* globalWorkOffset = nullptr; // ==> offset=0 in all dims
	size_t* localWorkSize = nullptr; // ==> OpenCL runtime will pick sizes

	// ---------------------------------------------------
	// Enqueue the kernel for execution
	// ----------------------------------------------------

	status = clEnqueueNDRangeKernel(cmdQueue, kernel,
		1, // dimension of kernel
		globalWorkOffset, globalWorkSize,
		localWorkSize,
		0, nullptr, nullptr); // event information, if desired
	checkStatus("clEnqueueNDRangeKernel", status, true);

	// ----------------------------------------------------
	// Read the output buffer back to the host
	// ----------------------------------------------------

	clEnqueueReadBuffer(cmdQueue, 
		bufferZ, CL_TRUE, 0, datasize, 
		h_Z, 0, nullptr, nullptr);

	// ----------------------------------------------------
	// Release OpenCL resources
	// ----------------------------------------------------

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferX);
	clReleaseMemObject(bufferY);
	clReleaseMemObject(bufferZ);
	clReleaseContext(context);

	// Free host resources
	delete [] platforms;
	delete [] devices;
}

double* do_daxpy(cl_device_id curDevice, size_t arraySize)
{
	double a = 2.0;
	double* X = new double[arraySize];
	double* Y = new double[arraySize];
	double* Z = new double[arraySize];
	for (int i=0 ; i<arraySize ; i++)
	{
		X[i] = 1000.0;
		Y[i] =   10.0;
		Z[i] = -999.99;
	}
	doTheKernelLaunch(curDevice, a, X, Y, arraySize, Z);
	for (int i=0 ; i<arraySize ; i++)
		std::cout << Z[i] << " = " << a << " * " << X[i] << "  +  " << Y[i] << '\n';
	delete [] X;
	delete [] Y;
	return Z;
}

int main(int argc, char* argv[])
{
	cl_device_type devType = CL_DEVICE_TYPE_DEFAULT;
	if (argc > 1)
	{
		for (int i=1 ; i<argc ; i++)
		{
			if (strcmp("-debug", argv[i]) == 0)
				debug = true;
			else if (argv[i][0] == '-')
			{
				switch (argv[i][1])
				{
					case 'a':
						devType = CL_DEVICE_TYPE_ALL;
						break;
					case 'c':
						devType = CL_DEVICE_TYPE_CPU;
						break;
					case 'g':
						devType = CL_DEVICE_TYPE_GPU;
						break;
				}
			}
		}
	}
	int deviceIndex = typicalOpenCLProlog(devType);
	if (deviceIndex < 0)
		return 0;
			
	double* Z = do_daxpy(devices[deviceIndex], 20);
	// ...
	delete [] Z;

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
