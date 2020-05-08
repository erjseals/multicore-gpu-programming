#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
void project3(__global double* ret, int nRows, int nCols, __global int * intValues, __global double * doubleValues, __global double * colors)
{
	// Get the work-item's unique ID
	int col = get_global_id(0);
	int row = get_global_id(1);

	int MaxIterations 	 = intValues[0];
	int MaxLengthSquared = intValues[1];
	double realMin = doubleValues[0];
	double realMax = doubleValues[1];
	double imagMin = doubleValues[2];
	double imagMax = doubleValues[3];
	double COLOR_1[3] = { colors[0], colors[1], colors[2]}
	double COLOR_2[3] = { colors[3], colors[4], colors[5]}
	double COLOR_3[3] = { colors[6], colors[7], colors[8]}

	double QReal = realMin + (col/(nCols-1))*(realMax - realMin);
	double QImag = imagMin + (row/(nRows-1))*(imagMax - imagMin);

	double RReal = QReal;
	double RImag = QImag;
	double SReal = QReal;
	double SImag = QImag;
	
	bool maxReached = false;
	int actualNumberIterations = 0;
	double colorRet[3];

	if ((row < nRows) && (col < nCols))
	{
		for(int i = 0 ; i < MaxIterations ; i++) {
			double XReal = (RReal*RReal - RImag*RImag) + SReal; 
			double XImag = 2*RReal*RImag + SImag;

			if( (XReal*XReal + XImag*XImag) > MaxLengthSquared && !maxReached){
				maxReached = true;
				actualNumberIterations = i + 1;
			}
			RReal = XReal;
			RImag = XImag;
		}
		if(maxReached) {
			colorRet[0] = COLOR_1[0];
			colorRet[1] = COLOR_1[1];
			colorRet[2] = COLOR_1[2];
		}
		else {
			double f = actualNumberIterations/MaxIterations;
			colorRet[0] = (1.0 - f)*COLOR_2[0] + f*COLOR_3[0];
			colorRet[1] = (1.0 - f)*COLOR_2[1] + f*COLOR_3[1];
			colorRet[2] = (1.0 - f)*COLOR_2[2] + f*COLOR_3[2];
		}

		ret[(row*nCol + col) * 3]     = colorRet[0];
		ret[(row*nCol + col) * 3 + 1] = colorRet[1];
		ret[(row*nCol + col) * 3 + 2] = colorRet[2];
	}
}
