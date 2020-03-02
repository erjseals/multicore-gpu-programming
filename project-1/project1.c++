#include "Barrier.h"

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <fstream>
#include <sstream>

Barrier b;
std::mutex coutMutex;
std::mutex trainCountMutex;
std::mutex trainCompletionMutex;

int step;

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


//numberOfTrains is by reference to manage the barrier with trains completing their journey
void work(int assignment, int &numberOfTrains, std::vector<int> trainRoute)
{
    int routeLength = trainRoute.size();
    bool routeCompleted = false;

    coutMutex.lock();
    std::cout << "Train " << (assignment+65) << " waiting to start\n";
    coutMutex.unlock();

    b.barrier(numberOfTrains);

    int currentStation = 0;
    int nextStation = 0;

    // while(!routeCompleted) {
    //     //get path
    //     int from =  
        
    //     //try to get the mutex
    //     //if(trackLocks);


    // }
}

void cleanUpThreads(std::thread** t, int numberOfTrains){
    for (int i = 0 ; i < numberOfTrains ; i++)
		delete t[i];
	delete [] t;
}

void cleanUpArray(std::mutex ** trackLocks, int numberOfTrains) {
    for (auto i = 0 ; i < numberOfTrains ; i++)
        delete trackLocks[i];
    delete trackLocks;
}


int main(int argc, char* argv[]) {
    if(argc < 2)
        std::cout << "Incorrect number of parameters!\n";
    else 
    {
        int numberOfTrains, numberOfStations;
        int buildTrains;
        
        std::vector<std::vector<int>> trainRoutes;
        std::vector<int> trainStops;

        read(argv[1], numberOfTrains, numberOfStations, trainRoutes, trainStops);

        step = 0;
        buildTrains = numberOfTrains;

        //Build mutex array
        std::mutex ** trackLocks;
        trackLocks = new std::mutex * [buildTrains];
        for(auto i = 0 ; i < buildTrains ; i++)
            trackLocks[i] = new std::mutex[buildTrains];
        
        //Spawn the threads
        std::thread** t = new std::thread*[buildTrains];
        for (int i = 0 ; i < buildTrains ; i++)
		    t[i] = new std::thread(work, i, numberOfTrains, trainRoutes[i]);

        
        //Wait for all threads to finish
        for (int i=0 ; i<numberOfTrains ; i++){
            coutMutex.lock();
            std::cout << "thread complete " << i << "\n";
            coutMutex.unlock();
		// Wait until the i-th thread completes. It will complete
		// when the given function ("work" in this case) exits.
		    t[i]->join();
        }

        cleanUpArray(trackLocks, buildTrains);
        cleanUpThreads(t, numberOfTrains);
    }
  return(0);
}
