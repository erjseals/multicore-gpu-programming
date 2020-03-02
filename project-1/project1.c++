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
          int ** trainRoutes,
          int * trainRouteSize){

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

    trainRoutes = new int * [numberOfTrains];
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

    bool routeCompleted = false;

    coutMutex.lock();
    std::cout << "Train " << (char)(assignment+65) << " waiting to start\n";
    coutMutex.unlock();

    // while(!routeCompleted) {
    //     //get path
    //     int from =  
        
    //     //try to get the mutex
    //     //if(trackLocks);


    // }
}

void cleanUpArray(int ** trainRoutes, int numberOfTrains) {
    for (int i = 0 ; i < numberOfTrains ; i++){
		delete [] trainRoutes[i];
    }
	delete [] trainRoutes;
    
 }

void cleanUpThreads(std::thread** t, int numberOfTrains){
    for (int i = 0 ; i < numberOfTrains ; i++)
		delete [] t[i];
	delete [] t;
}

void cleanUpMutexArray(std::mutex ** trackLocks, int numberOfTrains) {
    for (auto i = 0 ; i < numberOfTrains ; i++)
        delete [] trackLocks[i];
    delete [] trackLocks;
}


int main(int argc, char* argv[]) { 
    if(argc < 2)
        std::cout << "Incorrect number of parameters!\n";
    else 
    {
        int numberOfTrains, numberOfStations, buildTrains;
        int ** trainRoutes;
        int * trainRouteSize;
        step = 0;

        read(argv[1], numberOfTrains, numberOfStations, trainRoutes, trainRouteSize);

        for(auto i = 0 ; i < numberOfTrains ; i++) {
            for(auto j = 0 ; j < trainRouteSize[i] ; j++){
                std::cout << trainRoutes[i][j] << ' ';
            }
            std::cout << std::endl;
        }


        //buildTrains is a non-changing of initial total
        //used to clean up memory later on
        buildTrains = numberOfTrains;

        

        //Build 2D mutex array
        std::mutex ** trackLocks = new std::mutex *[numberOfTrains];
        for (int i = 0 ; i < buildTrains ; i++) 
            trackLocks[i] = new std::mutex;

        
        
        //Spawn the threads
        std::thread** t = new std::thread*[buildTrains];
        for (int i = 0 ; i < buildTrains ; i++)
		    t[i] = new std::thread(work, i, numberOfTrains);

        
        //Wait for all threads to finish
        for (int i = 0 ; i < buildTrains ; i++){
            coutMutex.lock();
            std::cout << "Train complete " << (char)(i+65) << "\n";
            coutMutex.unlock();
		// Wait until the i-th thread completes. It will complete
		// when the given function ("work" in this case) exits.
		    t[i]->join();
        }

        for (int i = 0 ; i < numberOfTrains ; i++){
            std::cout << trainRoutes[i] << '\n';
		    delete [] trainRoutes[i];
        }
	    delete [] trainRoutes;

        // for (int i = 0 ; i < numberOfTrains ; i++)
		//     delete [] t[i];
	    // delete [] t;

        // for (int i = 0 ; i < numberOfTrains ; i++)
		//     delete [] trackLocks[i];
	    // delete [] trackLocks;
    }
  return(0);
}
