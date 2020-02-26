#ifndef HANDLER_H
#define HANDLER_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class Handler {
private:
    int numberOfTrains, numberOfStations;
    std::vector<std::vector<int>> trainRoutes;
    std::vector<int> trainStops;
public:
    Handler(std::string filename);
    void test();
};

#endif