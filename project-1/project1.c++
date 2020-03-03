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
std::mutex trainCountMutex;
std::mutex stepMutex;

int step;
int trainCount;

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

void work(int assignment, int numberOfDestinations, int * trainRoute, std::mutex ** trackLocks, int * timeStepFinished)
{
    coutMutex.lock();
    std::cout << "Train " << (char)(assignment+65) << " waiting to start\n";
    coutMutex.unlock();
    
    //Will first use this barrier as a way of synchronization
    b.barrier(trainCount);

    //Order of events:
    // 1. check if finished, if so just go to barrier
    // 2. grab the mutex for the track the train wishes to travel
    // 3. a) if the mutex is aquired, move the train forward and then go to barrier
    //    b) if the mutex is locked, go to the barrier
    //    c) grab a local copy of the current number of trains
    // 4. First Barrier
    //    a) release all track mutex locks
    //    b) check if a train is complete
    //    c) if complete, decrement active trains
    //       also, the thread will exit (after first calling the barrier)
    //    d) go to second barrier with the local copy generated in 2.c
    // 5. Second Barrier
    //    a) go to 1.  

    int currentStation = 0;
    int nextStation    = 0;

    if(numberOfDestinations > 1)
        nextStation++;

    bool finished = false;

    int low  = 0;
    int high = 0;

    int currentTrainCount = 0;

    bool hasStepMutex = false;

    while(!finished) {

        if(stepMutex.try_lock())
            step++;

        currentTrainCount = trainCount;
        //BEFORE FIRST BARRIER
        if(currentStation == nextStation)
            b.barrier(currentTrainCount);
        else 
        {
            //try to grab the mutex, always lowest number to high
            //example, if a train travels from 5 to 4, the train grabs the mutex at location [4][5]
            low  = (trainRoute[currentStation] > trainRoute[nextStation]) ? trainRoute[nextStation] : trainRoute[currentStation];
            high = (trainRoute[currentStation] > trainRoute[nextStation]) ? trainRoute[currentStation] : trainRoute[nextStation];

            //must try the lock or the code will hang on failure
            //and need to hold on to the mutex until after the first barrier
            if(trackLocks[low][high].try_lock()) {
                coutMutex.lock();
                std::cout << "At time step: " << step << " train " << (char)(assignment+65) << " is going from station " << trainRoute[currentStation] << " to station " << trainRoute[nextStation] << '\n';
                coutMutex.unlock();
                currentStation++;
                nextStation++;
                if(nextStation == numberOfDestinations)
                    nextStation--;

                b.barrier(currentTrainCount);
            }
            //did not receive the mutex
            //will have to wait
            else {
                coutMutex.lock();
                std::cout << "At time step: " << step << " train " << (char)(assignment+65) << " must stay at station " << trainRoute[currentStation] << '\n';
                coutMutex.unlock();
                b.barrier(currentTrainCount);
            }
        }

        //FIRST BARRIER
        
        stepMutex.unlock();

        //release mutex
        trackLocks[low][high].unlock();
        //check completion
        if(currentStation == nextStation) 
        {
            finished = true;
            trainCountMutex.lock();
            trainCount--;
            trainCountMutex.unlock();
            timeStepFinished[assignment] = step;
        }

        b.barrier(currentTrainCount);
    }

    




    // coutMutex.lock();
    // std::cout << "Train " << (char)(assignment+65) << ": " << numberOfDestinations << '\n';
    // for(auto i = 0 ; i < numberOfDestinations ; i++) {
    //     std::cout << trainRoute[i] << ' ';
    // }
    // std::cout << '\n';
    // coutMutex.unlock();
}

int main(int argc, char* argv[]) { 
    if(argc < 2)
        return 0;

    std::cout << "Starting simulation...\n";

    int numberOfTrains, numberOfStations;
    int ** trainRoutes;
    int * trainRouteSize;
    step = 0;

    read(argv[1], numberOfTrains, numberOfStations, trainRoutes, trainRouteSize);

    trainCount = numberOfTrains;

    int * timeStepFinished = new int[numberOfTrains];

    //Build 2D mutex array
    std::mutex ** trackLocks = new std::mutex *[numberOfStations];
    for (int i = 0 ; i < numberOfStations ; i++) 
        trackLocks[i] = new std::mutex[numberOfStations];
    
    //Spawn the threads
    std::thread** t = new std::thread*[numberOfTrains];
    for (int i = 0 ; i < numberOfTrains ; i++)
        t[i] = new std::thread(work, i, trainRouteSize[i], trainRoutes[i], trackLocks, timeStepFinished);





    //Wait for all threads to finish
    for (int i = 0 ; i < numberOfTrains ; i++){
    // Wait until the i-th thread completes. It will complete
    // when the given function ("work" in this case) exits.
        t[i]->join();
        
        coutMutex.lock();
        std::cout << "Train " << (char)(i+65) << " completed its route at time step " << timeStepFinished[i] << "\n";
        coutMutex.unlock();
    }

    for (int i = 0 ; i < numberOfTrains ; i++){
        delete [] trainRoutes[i];
        delete t[i];
        delete [] trackLocks[i];
    }
    delete [] trainRoutes;
    delete [] t;
    delete [] trackLocks;
    
  return(0);
}
