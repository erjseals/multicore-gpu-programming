// System includes
#include <iostream>
#include <string>
#include <string.h>
#include <math.h>
#include <tuple>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "ImageWriter.h"

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
	//-----------------------------------------------------
	// Discover and query the platforms
	//-----------------------------------------------------

	cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
	checkStatus("clGetPlatformIDs-0", status, true);

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

	std::cout << "Selected platform: ";
	reportPlatformInformation(curPlatform);

	//----------------------------------------------------------
	// Discover and initialize the devices on a platform
	//----------------------------------------------------------

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
	// Find a device that supports double precision arithmetic
	int* possibleDevs = new int[numDevices];
	int nPossibleDevs = 0;
	std::cout << "\nLooking for a device that supports double precision...\n";
	for (int idx=0 ; idx<numDevices ; idx++)
	{
		size_t extLength;
		clGetDeviceInfo(devices[idx], CL_DEVICE_EXTENSIONS, 0, nullptr, &extLength);
		char* extString = new char[extLength+1];
		clGetDeviceInfo(devices[idx], CL_DEVICE_EXTENSIONS, extLength+1, extString, nullptr);
		const char* fp64 = strstr(extString, "cl_khr_fp64");
		if (fp64 != nullptr) // this device supports double precision
			possibleDevs[nPossibleDevs++] = idx;
		delete [] extString;
	}
	if (nPossibleDevs == 0)
	{
		std::cerr << "\nNo device supports double precision.\n";
		return -1;
	}
	size_t nameLength;
	for (int i=0 ; i<nPossibleDevs ; i++)
	{
		clGetDeviceInfo(devices[possibleDevs[i]], CL_DEVICE_NAME, 0, nullptr, &nameLength);
		char* name = new char[nameLength+1];
		clGetDeviceInfo(devices[possibleDevs[i]], CL_DEVICE_NAME, nameLength+1, name, nullptr);
		std::cout << "Device " << i << ": [" << name << "] supports double precision.\n";
		delete [] name;
	}
	if (nPossibleDevs == 1)
	{
		std::cout << "\nNo other device in the requested device category supports double precision.\n"
		          << "You may want to try the -a command line option to see if there are others.\n"
		          << "For now, I will use the one I found.\n";
		return possibleDevs[0];
	}
	int devIndex = -1;
	while ((devIndex < 0) || (devIndex >= nPossibleDevs))
	{
		std::cout << "Which device do you want to use? ";
		std::cin >> devIndex;
	}
	return possibleDevs[devIndex];
}

void doTheKernelLaunch(cl_device_id dev, double* A, double* B, double* C, size_t N)
{
	//------------------------------------------------------------------------
	// Create a context for some or all of the devices on the platform
	// (Here we are including all devices.)
	//------------------------------------------------------------------------

	cl_int status;
	cl_context context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
	checkStatus("clCreateContext", status, true);

	//-------------------------------------------------------------
	// Create a command queue for one device in the context
	// (There is one queue per device per context.)
	//-------------------------------------------------------------

	cl_command_queue cmdQueue = clCreateCommandQueue(context, dev, 0, &status);
	checkStatus("clCreateCommandQueue", status, true);

	//----------------------------------------------------------
	// Create device buffers associated with the context
	//----------------------------------------------------------

	size_t datasize = N * N * sizeof(double);

	cl_mem d_A = clCreateBuffer( // Input array on the device
		context, CL_MEM_READ_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-A", status, true);

	cl_mem d_B = clCreateBuffer( // Input array on the device
		context, CL_MEM_READ_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-B", status, true);

	cl_mem d_C = clCreateBuffer( // Output array on the device
		context, CL_MEM_WRITE_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-C", status, true);

	//-----------------------------------------------------
	// Use the command queue to encode requests to
	//         write host data to the device buffers
	//----------------------------------------------------- 

	status = clEnqueueWriteBuffer(cmdQueue, 
		d_A, CL_FALSE, 0, datasize,                         
		A, 0, nullptr, nullptr);
	checkStatus("clEnqueueWriteBuffer-A", status, true);

	status = clEnqueueWriteBuffer(cmdQueue, 
		d_B, CL_FALSE, 0, datasize,                                  
		B, 0, nullptr, nullptr);
	checkStatus("clEnqueueWriteBuffer-B", status, true);

	//-----------------------------------------------------
	// Create, compile, and link the program
	//----------------------------------------------------- 

	const char* programSource[] = { readSource("project3.cl") };
	cl_program program = clCreateProgramWithSource(context, 
		1, programSource, nullptr, &status);
	checkStatus("clCreateProgramWithSource", status, true);

	status = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
	if (status != 0)
		showProgramBuildLog(program, dev);
	checkStatus("clBuildProgram", status, true);

	//----------------------------------------------------------------------
	// Create a kernel using a "__kernel" function in the ".cl" file
	//----------------------------------------------------------------------

	cl_kernel kernel = clCreateKernel(program, "project3", &status);

	//-----------------------------------------------------
	// Set the kernel arguments
	//----------------------------------------------------- 

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	checkStatus("clSetKernelArg-A", status, true);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
	checkStatus("clSetKernelArg-B", status, true);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
	checkStatus("clSetKernelArg-C", status, true);
	status = clSetKernelArg(kernel, 3, sizeof(int), &N);
	checkStatus("clSetKernelArg-N", status, true);

	//-----------------------------------------------------
	// Configure the work-item structure
	//----------------------------------------------------- 

	size_t localWorkSize[] = { 16, 16 };
	size_t globalWorkSize[2];
	// Global work size needs to be at least NxN, but it must
	// also be a multiple of local size in each dimension:
	for (int d=0 ; d<2 ; d++)
	{
		globalWorkSize[d] = N;
		if (globalWorkSize[d]%localWorkSize[d] != 0)
			globalWorkSize[d] = ((N / localWorkSize[d]) + 1) * localWorkSize[d];
	}

	//-----------------------------------------------------
	// Enqueue the kernel for execution
	//----------------------------------------------------- 

	status = clEnqueueNDRangeKernel(cmdQueue, kernel,
		2, // number dimensions in grid
		nullptr, globalWorkSize, // globalOffset, globalSize
		localWorkSize,
		0, nullptr, nullptr); // event information, if needed
	checkStatus("clEnqueueNDRangeKernel", status, true);

	//-----------------------------------------------------
	// Read the output buffer back to the host
	//----------------------------------------------------- 

	clEnqueueReadBuffer(cmdQueue, 
		d_C, CL_TRUE, 0, datasize, 
		C, 0, nullptr, nullptr);

	//-----------------------------------------------------
	// Release OpenCL resources
	//----------------------------------------------------- 

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
	clReleaseContext(context);

	// Free host resources
	delete [] platforms;
	delete [] devices;
}

double* do_project3(cl_device_id dev, size_t N)
{
	double* X = new double[N*N];
	double* Y = new double[N*N];
	double* Z = new double[N*N];
	for (int row=0 ; row<N ; row++)
		for (int col=0 ; col<N ; col++)
		{
			// make X be 2*I
			X[row*N + col] = (row == col) ? 2.0 : 0.0;
			Y[row*N + col] = 17.5;
		}
	doTheKernelLaunch(dev, X, Y, Z, N);

	delete [] X;
	delete [] Y;
	return Z;
}

void print(std::string label, double* M, size_t N)
{
	std::cout << label << ":\n";
	for (int row=0 ; row<N ; row++)
	{
		for (int col=0 ; col<N ; col++)
		{
			std::cout << M[row*N + col] << " ";
		}
		std::cout << '\n';
	}
}

int main(int argc, char* argv[])
{
	if(argc < 4)
		std::cerr << "Usage: " << argv[0] << " M/R params.txt imageFileOut.png\n";

	int nRows, nCols, MaxIterations, MaxLengthSquared;
	double realMin, realMax, imagMin, imagMax;
	double JReal, JImag;
	double COLOR_1[3] = {0.0, 0.0, 0.0};
	double COLOR_2[3] = {0.0, 0.0, 0.0};
	double COLOR_3[3] = {0.0, 0.0, 0.0};

	//**********************************************
	//Read the file
	std::ifstream inFile(argv[2]);
	std::string line;
	int value;
	double data;

	//We know exactly the format of the file

	//line 1
	std::getline(inFile, line);
	std::stringstream lineStream(line);

	lineStream >> value;
	nRows = value;

	lineStream >> value;
	nCols = value;

	//line 2
	std::getline(inFile, line);
	lineStream(line);

	lineStream >> value;
	MaxIterations = value;

	//line 3
	std::getline(inFile, line);
	lineStream(line);

	lineStream >> value;
	MaxLengthSquared = value;

	//line 4
	std::getline(inFile, line);
	lineStream(line);

	lineStream >> data;
	realMin = data;

	lineStream >> data;
	realMax = data;

	//line 5
	std::getline(inFile, line);
	lineStream(line);

	lineStream >> data;
	imagMin = data;

	lineStream >> data;
	imagMax = data;

	//line 6
	std::getline(inFile, line);
	lineStream(line);

	lineStream >> data;
	JReal = data;

	lineStream >> data;
	JImag = data;

	//line 7
	std::getline(inFile, line);
	lineStream(line);

	lineStream >> data;
	COLOR_1[0] = data;

	lineStream >> data;
	COLOR_1[1] = data;

	lineStream >> data;
	COLOR_1[2] = data;

	//line 8
	std::getline(inFile, line);
	lineStream(line);

	lineStream >> data;
	COLOR_2[0] = data;

	lineStream >> data;
	COLOR_2[1] = data;

	lineStream >> data;
	COLOR_2[2] = data;

	//line 9
	std::getline(inFile, line);
	lineStream(line);

	lineStream >> data;
	COLOR_3[0] = data;

	lineStream >> data;
	COLOR_3[1] = data;

	lineStream >> data;
	COLOR_3[2] = data;

	//*************************
	//test results
	std::cout 	<< nRows << " " << nCols << '\n' 
				<< MaxIterations << '\n' << MaxLengthSquared << '\n'
				<< realMin << ' ' << realMax << '\n'
				<< imagMin << ' ' << imagMax << '\n'
				<< JReal   << ' ' << JImag   << '\n'
				<< COLOR_1[0] << ' ' << COLOR_1[1] << ' ' << COLOR_1[2] << '\n'
				<< COLOR_2[0] << ' ' << COLOR_2[1] << ' ' << COLOR_2[2] << '\n'
				<< COLOR_3[0] << ' ' << COLOR_3[1] << ' ' << COLOR_3[2] << '\n';

	

	cl_device_type devType = CL_DEVICE_TYPE_DEFAULT;
	size_t N = 20;
	bool doPrint = true;

	for (int i=1 ; i<argc ; i++)
	{
		if (strcmp("-debug", argv[i]) == 0)
			debug = true;
		else if (strcmp(argv[i], "-a") == 0)
			devType = CL_DEVICE_TYPE_ALL;
		else if (strcmp(argv[i], "-c") == 0)
			devType = CL_DEVICE_TYPE_CPU;
		else if (strcmp(argv[i], "-g") == 0)
			devType = CL_DEVICE_TYPE_GPU;
		else if (strcmp(argv[i], "-n") == 0)
			N = atoi(argv[++i]);
		else if (strcmp(argv[i], "-noprint") == 0)
			doPrint = false;
	}

	int devIndex = typicalOpenCLProlog(devType);
	if (devIndex >= 0)
	{
		double* C = do_project3(devices[devIndex], N);
		if (doPrint)
			print("The product is", C, N);
		delete [] C;
	}

	///////////////////////////////////////////////
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
