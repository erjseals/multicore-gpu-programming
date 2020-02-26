CPP = g++ -g -fPIC
LINK = g++ -g -fPIC

# Uncomment the following if building on EECS linux machines
C_FLAGS =
L_FLAGS = -lpthread
# Uncomment the following if building on a Macintosh
# C_FLAGS = -std=c++11
# L_FLAGS =

project1: main.o Handler.o
	$(LINK) main.o Handler.o -o project1 $(L_FLAGS)
Handler.o: Handler.h Handler.c++
	$(CPP) $(C_FLAGS) -c Handler.c++
main.o: main.c++
	$(CPP) $(C_FLAGS) -c main.c++
clean:
	-rm project1 *.o