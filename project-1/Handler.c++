#include "Handler.h"

Handler::Handler(std::string filename) {
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

void Handler::test(){
    std::cout << "numberOfTrains: "   << numberOfTrains   << "\n";
    std::cout << "numberOfStations: " << numberOfStations << "\n";

    int count = 0;
    for(auto train : trainRoutes) {
        
        std::cout << "NumberStops: " << trainStops[count] << " : ";
        for(auto stop : train) {
            std::cout << stop << ' ';
        }
        std::cout << std::endl;
        count++;
    }
}