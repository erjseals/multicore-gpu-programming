#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
void project3(__global double* ret, int nRows, int nCols, int MaxIterations, int MaxLengthSquared, double realMin, double realMax, 
					double imagMin, double imagMax, double jReal, double jImag, __global double * COLORS, char MorJ)
{
	// Get the work-item's unique ID
	int col = get_global_id(0);
	int row = get_global_id(1);

	double COLOR_1[3] = { COLORS[0], COLORS[1], COLORS[2]};
	double COLOR_2[3] = { COLORS[3], COLORS[4], COLORS[5]};
	double COLOR_3[3] = { COLORS[6], COLORS[7], COLORS[8]};

	double RReal = realMin + (((double)col)/((double)nCols-1))*(realMax - realMin);
	double RImag = imagMin + ((double)row/((double)nRows-1))*(imagMax - imagMin);
	double SReal;
	double SImag;

	if(MorJ == 'M'){
		SReal = RReal;
		SImag = RImag;
	}
	else {
		SReal = jReal;
		SImag = jImag;
	}
	
	bool maxReached = false;
	int actualNumberIterations = 0;
	double colorRet[3];
	double f;

	double XReal;
	double XImag;

	int i = 0;

	if ((row < nRows) && (col < nCols))
	{
		for(i = 0 ; i < MaxIterations ; i++) {
			XReal = (RReal*RReal - RImag*RImag) + SReal; 
			XImag = 2*RReal*RImag + SImag;

			if(!maxReached) {
				if( (XReal*XReal + XImag*XImag) > MaxLengthSquared ) {
					maxReached = true;
					actualNumberIterations = i;
				}
			}

			RReal = XReal;
			RImag = XImag;
		}
		if(!maxReached) {
			colorRet[0] = COLOR_1[0];
			colorRet[1] = COLOR_1[1];
			colorRet[2] = COLOR_1[2];
		}
		else {
		 f = ((double)actualNumberIterations)/((double)MaxIterations);
			colorRet[0] = (1.0 - f)*COLOR_2[0] + f*COLOR_3[0];
			colorRet[1] = (1.0 - f)*COLOR_2[1] + f*COLOR_3[1];
			colorRet[2] = (1.0 - f)*COLOR_2[2] + f*COLOR_3[2];
		}

		ret[(row*nCols + col) * 3]     = colorRet[0];
		ret[(row*nCols + col) * 3 + 1] = colorRet[1];
		ret[(row*nCols + col) * 3 + 2] = colorRet[2];
	}
}
