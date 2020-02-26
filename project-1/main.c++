#include "Handler.h"

#include <iostream>
#include <string>
#include <array>

int main(int argc, char* argv[]) {
    if(argc < 2)
        std::cout << "Incorrect number of parameters!\n";
    else 
    {
      Handler handler(argv[1]);
    }
  return(0);
}
