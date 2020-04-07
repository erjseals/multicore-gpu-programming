// mpiArgcArgv.c++ - Illustrating how parameters can be supplied to processes.
//                   Try running with various combinations of the following
//                   mpirun parameters:
//    -output-filename someName
//    -tag-output
//    -timestamp-output

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>

int NUMBER_OF_CITIES = 500;
std::string FILENAME = "500_Cities__City-level_Data__GIS_Friendly_Format_.csv";

void read(std::string filename, std::vector<std::vector<std::string>>& dataStorage) {


	std::ifstream inFile(filename);
    std::string line;
    int value;

	int row = 0, col = 0;

	//search the line, seperating by commas (except when the case of a tuple)
	while(std::getline(inFile, line)) {
		

		//new line, increase number of row
		dataStorage.push_back(std::vector<std::string>());

		//push an empty string to start
		dataStorage[row].push_back(std::string());
		//reset index
		col = 0;
		bool openParan = false;

		//get size of the line of string
		int size = line.size();
		for(int i = 0 ; i < size ; i++) {
			//check if paran opened
			if(line[i] == '\"') {
				if(openParan){
					openParan = false;
				}
				else {
					openParan = true;
				}
			}
			else if(line[i] == ',') {
				if(openParan) {
					dataStorage[row][col].push_back(line[i]);
				}
				else {
					col++;
					dataStorage[row].push_back(std::string());
				}
			}
			else {
				dataStorage[row][col].push_back(line[i]);
			}
		}
		row++;
	}
}

double getMax(double * doubleArray, int rank, int &index) {
	index = 0;
	double max = doubleArray[rank*NUMBER_OF_CITIES];
	for(int i = rank * NUMBER_OF_CITIES  ; i < rank * NUMBER_OF_CITIES + 500 ; i++){
		if(doubleArray[i] > max){
			max = doubleArray[i];
			index = i;
		}
	}
	return max;
}

double getMin(double * doubleArray, int rank, int &index) {
	index = 0;
	double min = doubleArray[rank*NUMBER_OF_CITIES];
	for(int i = rank * NUMBER_OF_CITIES  ; i < rank * NUMBER_OF_CITIES + 500 ; i++){
		if(doubleArray[i] < min){
			min = doubleArray[i];
			index = i;
		}
	}
	return min;
}

double getAvg(double * doubleArray, int rank) {
	double avg = doubleArray[rank*NUMBER_OF_CITIES];
	for(int i = rank * NUMBER_OF_CITIES + 1  ; i < rank * NUMBER_OF_CITIES + 500 ; i++){
		avg += doubleArray[i];
	}
	avg /= NUMBER_OF_CITIES;
	return avg;
}


int convertColumn(const std::string& col){
	int ret = 0, size = col.size();

	//Example "BC", is (26^1)(2) + (26^0)(3)
	for(auto i = size - 1 ; i >= 0 ; i--) {
		
		ret += pow(26,i) * ((int)col[size-i-1] - (int)'A' + 1);
	}

	//subtract 1 because A is actually 0
	return ret - 1;
}

void do_rank_0_work(int communicatorSize, int numberArgs, int * colArray, int operation) {
	std::vector<std::vector<std::string>> dataStorage;
	std::string ** array;
	read(FILENAME, dataStorage);
	//convert to an array because MPI does not handle vectors well
	//will hopefully have time to fix this later
	array = new std::string*[dataStorage.size()];
	int rowSize = dataStorage.size();
	int colSize = dataStorage[0].size();


	//will only need the columns that are actually used
	for(int i = 0 ; i < rowSize ; i++) {
		array[i] = new std::string[numberArgs];
	}

	for(int i = 0 ; i < rowSize ; i++) {
		for(int j = 0 ; j < numberArgs ; j++) {
			array[i][j] = dataStorage[i][colArray[j]];
		}
	}

	double * doubleArray = new double[NUMBER_OF_CITIES * numberArgs];
	for(int j = 0 ; j < numberArgs ; j++) {
		for(int i = 0 ; i < NUMBER_OF_CITIES ; i++)
			doubleArray[j * NUMBER_OF_CITIES + i] = std::stod(array[i+1][j]);
	}

	MPI_Bcast(doubleArray, NUMBER_OF_CITIES * numberArgs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Required data has been sent to everyone, so they should be able to start working.
	// Now I will get going on my piece:


	double result;
	int indexResult;

	if(operation == 0) {
		result = getMax(doubleArray, 0, indexResult);
	}else if(operation == 1) {
		result = getMin(doubleArray, 0, indexResult);
	}else if(operation == 2) {
		result = getAvg(doubleArray, 0);
	}else ;


	double * resultsArray = new double[numberArgs];
	resultsArray[0] = result;
	MPI_Gather(MPI_IN_PLACE, 0, MPI_DOUBLE, resultsArray, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	for(int i = 0 ; i < numberArgs ; i++) {
		if(operation == 0) {
			std::cout << "max ";
		}else if(operation == 1) {
			std::cout << "min ";
		}else if(operation == 2) {
			std::cout << "avg ";
		}else ;
		std::cout << array[0][i] << " = " << resultsArray[i] << '\n';
		
	}

	delete [] resultsArray;
	delete [] doubleArray;
	for(int i = 0 ; i < rowSize ; i++) {
		delete [] array[i];
	}
	delete array;
	delete [] colArray;
}


void do_rank_i_work(int rank, int numberArgs, int operation) {
	double * doubleArray = new double[NUMBER_OF_CITIES * numberArgs];

	MPI_Bcast(doubleArray, NUMBER_OF_CITIES * numberArgs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double result;
	int indexResult;

	if(operation == 0) {
		result = getMax(doubleArray, rank, indexResult);
	}else if(operation == 1) {
		result = getMin(doubleArray, rank, indexResult);
	}else if(operation == 2) {
		result = getAvg(doubleArray, rank);
	}else ;


	MPI_Gather(&result, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	delete [] doubleArray;
}

void do_rank_0_workS(int communicatorSize, int numberArgs, int column, int operation, double compare) {
	std::vector<std::vector<std::string>> dataStorage;
	std::string * array, * states, * cities;
	read(FILENAME, dataStorage);
	//convert to an array because MPI does not handle vectors well
	//will hopefully have time to fix this later
	array = new std::string[dataStorage.size()];

	int rowSize = dataStorage.size();
	int colSize = dataStorage[0].size();

	for(int i = 0 ; i < rowSize ; i++) {
		array[i] = dataStorage[i][column];
	}

	double * doubleArray = new double[NUMBER_OF_CITIES];
	for(int i = 0 ; i < NUMBER_OF_CITIES ; i++)
		doubleArray[i] = std::stod(array[i+1]);

	int sizePerProcess = NUMBER_OF_CITIES / communicatorSize;
	double * reducedArray = new double[sizePerProcess];

	MPI_Scatter(doubleArray, sizePerProcess, MPI_DOUBLE, reducedArray, sizePerProcess, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Required data has been sent to everyone, so they should be able to start working.
	// Now I will get going on my piece:

	double globalResult;

	//calculate the local value
	double result = 0;
	double min = reducedArray[0];
	double max = reducedArray[0];
	double countLT = 0, countGT = 0;

	int indexMax = 0, indexMin = 0;

	for(int i = 0 ; i < sizePerProcess ; i++) {
		if(reducedArray[i] > max){
			max = reducedArray[i];
			indexMax = i;
		}
		if(reducedArray[i] < min){
			min = reducedArray[i];
			indexMin = i;
		}
		result += reducedArray[i];
		if(reducedArray[i] > compare)
			countGT++;
		if(reducedArray[i] < compare)
			countLT++;
	}

	if(operation == 0)
		MPI_Reduce(&max, &globalResult, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	else if(operation == 1)
		MPI_Reduce(&min, &globalResult, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	else if(operation == 2)
		MPI_Reduce(&result, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	else if(operation == 3)
		MPI_Reduce(&countGT, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	else if(operation == 4)
		MPI_Reduce(&countLT, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(operation == 0)
		std::cout << array[0] << " = " << globalResult << '\n';
	else if(operation == 1)
		std::cout << array[0] << " = " << globalResult << '\n';
	else if(operation == 2){
		globalResult /= NUMBER_OF_CITIES;
		std::cout << "Average " <<  array[0] << " = " << globalResult << '\n';
	}
	else if(operation == 3)
		std::cout << "Number cities with " << array[0] << " gt " << compare << " = " << globalResult << '\n';
	else if(operation == 4)
		std::cout << "Number cities with " << array[0] << " lt " << compare << " = " << globalResult << '\n';

	delete [] array;
	delete [] reducedArray;
	delete [] doubleArray;
}


void do_rank_i_workS(int communicatorSize, int rank, int numberArgs, int operation, double compare) {
	int sizePerProcess = NUMBER_OF_CITIES / communicatorSize;
	double * reducedArray = new double[sizePerProcess];

	MPI_Scatter(nullptr, sizePerProcess, MPI_DOUBLE, reducedArray, sizePerProcess, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double globalResult;

	//calculate the local value
	double result = 0;
	double min = reducedArray[0];
	double max = reducedArray[0];
	double countLT = 0, countGT = 0;

	int indexMax = 0, indexMin = 0;

	for(int i = 0 ; i < sizePerProcess ; i++) {
		if(reducedArray[i] > max){
			max = reducedArray[i];
			indexMax = i;
		}
		if(reducedArray[i] < min){
			min = reducedArray[i];
			indexMin = i;
		}
		result += reducedArray[i];
		if(reducedArray[i] > compare)
			countGT++;
		if(reducedArray[i] < compare)
			countLT++;
	}

	if(operation == 0)
		MPI_Reduce(&max, &globalResult, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	else if(operation == 1)
		MPI_Reduce(&min, &globalResult, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	else if(operation == 2)
		MPI_Reduce(&result, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	else if(operation == 3)
		MPI_Reduce(&countGT, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	else if(operation == 4)
		MPI_Reduce(&countLT, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	delete [] reducedArray;
}

void bProcess(int rank, int communicatorSize, int argc, char **argv){
	int numberArgs = argc - 3;
	int index = 3;

	int * colArray = new int[numberArgs];
	for(int i = 0 ; i < numberArgs ; i++) {
		colArray[i] = convertColumn(argv[index++]);
	}


	int operation;
	if(strcmp(argv[2], "max") == 0) {
		operation = 0;
	}else if(strcmp(argv[2], "min") == 0) {
		operation = 1;
	}else if(strcmp(argv[2], "avg") == 0) {
		operation = 2;
	}else {
		return;
	}

	if(rank == 0)
		do_rank_0_work(communicatorSize, numberArgs, colArray, operation);
	else 
		do_rank_i_work(rank, numberArgs, operation);
}


void sProcess(int rank, int communicatorSize, int argc, char **argv){
	int numberArgs = argc - 3;
	int index = 3;

	int * colArray = new int[numberArgs];
	for(int i = 0 ; i < numberArgs ; i++) {
		colArray[i] = convertColumn(argv[index++]);
	}

	int column, operation;
	column = convertColumn(argv[3]);

	std::string comparison = "0.0";
	double compare = 0.0;

	if(strcmp(argv[2], "max") == 0)
		operation = 0;
	else if(strcmp(argv[2], "min") == 0)
		operation = 1;
	else if(strcmp(argv[2], "avg") == 0)
		operation = 2;
	else if(argc == 6 && strcmp(argv[2], "number") == 0){
		if(strcmp(argv[4], "gt") == 0)
			operation = 3;
		else if(strcmp(argv[4], "lt") == 0 )
			operation = 4;
		comparison = argv[5];
		compare = std::stod(comparison);
	}

	

	if(rank == 0)
		do_rank_0_workS(communicatorSize, numberArgs, column, operation, compare);
	else 
		do_rank_i_workS(communicatorSize, rank, numberArgs, operation, compare);
}


int main (int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank, communicatorSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // unique ID for this process; 0<=rank<N where:
	MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize); // N=communicatorSize (size of "world")


	if(argc < 4) {
		if(rank == 0) 
			std::cerr << "Need at least 4 input params\n";
	}


	else if(strcmp(argv[1], "bg") == 0) {
		if( (argc - 3) != communicatorSize) {
			if(rank == 0)
				std::cerr << "Communicator size not valid for broadcast!\n";
		}
		else
			bProcess(rank, communicatorSize, argc, argv);
	}


	else if(strcmp(argv[1], "sr") == 0) {
		if((NUMBER_OF_CITIES % communicatorSize) != 0){
			if(rank == 0)
				std::cerr << "Communicator size must divide number of cities\n";
		}
		else 
			sProcess(rank, communicatorSize, argc, argv);
	}


	else {
		//do nothing
	}

	MPI_Finalize();
	return 0;
}