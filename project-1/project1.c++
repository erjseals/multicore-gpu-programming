#include "Barrier.h"

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <iostream>
#include <string>
#include <array>
#include <fstream>
#include <sstream>

Barrier b;
std::mutex coutMutex;

int step;

void read(std::string filename, 
          int &numberOfTrains, 
          int &numberOfStations, 
          int ** &trainRoutes,
          int * &trainRouteSize){

    std::ifstream inFile(filename);
    std::string line;
    int value;

    //get the first line and create a string stream (default deliminator is the space)
    std::getline(inFile, line);
    std::stringstream lineStreamFirst(line);

    //first iteration gets meta data
    lineStreamFirst >> value;
    numberOfTrains = value;

    lineStreamFirst >> value;
    numberOfStations = value;

    //create a 2d array for the routes
    trainRoutes = new int * [numberOfTrains];

    //create a 1d array for the length of the routes
    trainRouteSize = new int [numberOfTrains];

    int train = 0;

    for(auto i = 0 ; i < numberOfTrains ; i++) {
        std::getline(inFile, line);
        std::stringstream lineStream(line);

        lineStream >> value;
        trainRoutes[train] = new int [value];
        trainRouteSize[train] = value;

        int stop = 0;

        while(lineStream >> value){
            trainRoutes[train][stop] = value;
            stop++;
        }
        train++;
    }
}


//numberOfTrains is by reference to manage the barrier with trains completing their journey
void work(int assignment, int numberOfTrains)
{
    coutMutex.lock();
    std::cout << "Train " << (char)(assignment+65) << " waiting to start\n";
    coutMutex.unlock();
    
    b.barrier(numberOfTrains);

    coutMutex.lock();
    std::cout << "Train " << (char)(assignment+65) << " starting\n";
    coutMutex.unlock();
}

int main(int argc, char* argv[]) { 
    if(argc < 2)
        return 0;

    int numberOfTrains, numberOfStations, numTrains;
    int ** trainRoutes;
    int * trainRouteSize;
    step = 0;

    read(argv[1], numberOfTrains, numberOfStations, trainRoutes, trainRouteSize);

    //buildTrains is a non-changing of initial total
    //used to clean up memory later on
    numTrains = numberOfTrains;

    //Build 2D mutex array
    std::mutex ** trackLocks = new std::mutex *[numTrains];
    for (int i = 0 ; i < numTrains ; i++) 
        trackLocks[i] = new std::mutex[numTrains];
    
    //Spawn the threads
    std::thread** t = new std::thread*[numTrains];
    for (int i = 0 ; i < numTrains ; i++)
        t[i] = new std::thread(work, i, numberOfTrains);


    //Wait for all threads to finish
    for (int i = 0 ; i < numTrains ; i++){
    // Wait until the i-th thread completes. It will complete
    // when the given function ("work" in this case) exits.
        t[i]->join();
        
        coutMutex.lock();
        std::cout << "Train complete " << (char)(i+65) << "\n";
        coutMutex.unlock();
    }

    for (int i = 0 ; i < numTrains ; i++){
        delete [] trainRoutes[i];
        delete t[i];
        delete [] trackLocks[i];
    }
    delete [] trainRoutes;
    delete [] t;
    delete [] trackLocks;
    
  return(0);
}
