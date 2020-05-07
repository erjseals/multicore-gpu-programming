#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
void project3(__global double* ret, int nRows, int nCols, int MaxIterations, int MaxLengthSquared,
	double realMin, double realMax, double imagMin, double imagMax, __global double * COLOR_1, __global double * COLOR_2, __global double * COLOR_3)
{
	// Get the work-item's unique ID
	int col = get_global_id(0);
	int row = get_global_id(1);
	if ((row < nRows) && (col < nCols))
	{
		ret[row*nCols + col] = row*nCols + col;
	}
}
