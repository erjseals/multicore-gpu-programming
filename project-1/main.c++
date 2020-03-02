#include "Handler.h"
#include "Barrier.h"

#include <thread>

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <fstream>
#include <sstream>

Barrier b;


void read(std::string filename, 
          int &numberOfTrains, 
          int &numberOfStations, 
          std::vector<std::vector<int>> & trainRoutes, 
          std::vector<int> & trainStops){

    std::ifstream inFile(filename);
    std::string line;

    std::getline(inFile, line);
    std::stringstream lineStreamFirst(line);
    int value;

    //first iteration gets meta data
    lineStreamFirst >> value;
    numberOfTrains = value;

    lineStreamFirst >> value;
    numberOfStations = value;

    while(std::getline(inFile, line)) {
        std::stringstream lineStream(line);

        lineStream >> value;
        trainStops.push_back(value);

        std::vector<int> train;

        while(lineStream >> value)
            train.push_back(value);
        
        trainRoutes.push_back(train);
    }
}

void work(int assignment)
{
    b.barrier(assignment);
    std::cout << "Threads started " << assignment << "\n";
}

void spawnThreads(std::thread** t, int numberOfTrains){
	int nTrains = numberOfTrains;

	t = new std::thread*[nTrains];
	for (int i=0 ; i<nTrains ; i++)
		// All parameters to the std::thread constructor after the
		// first are passed as parameters to the function identified
		// by the first parameter:
		t[i] = new std::thread(work, i);
}

void cleanUpThreads(std::thread** t, int nTrains){
    for (int i=0 ; i<nTrains ; i++)
		delete t[i];
	delete [] t;
}

int main(int argc, char* argv[]) {
    if(argc < 2)
        std::cout << "Incorrect number of parameters!\n";
    else 
    {
      int numberOfTrains, numberOfStations;
      std::vector<std::vector<int>> trainRoutes;
      std::vector<int> trainStops;

      read(argv[1], numberOfTrains, numberOfStations, trainRoutes, trainStops);

      std::thread** t;

      spawnThreads(t, numberOfTrains);

      cleanUpThreads(t,numberOfTrains);
    }
  return(0);
}
