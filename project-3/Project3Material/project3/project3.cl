#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
void project3(__global double* ret, int nRows, int nCols)
{
	// Get the work-item's unique ID
	int col = get_global_id(0);
	int row = get_global_id(1);
	if ((row < nRows) && (col < nCols))
	{
		C[row*nRows + col] = row*nRows + col;
	}
}
