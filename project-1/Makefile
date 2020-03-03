CPP = g++ -g -fPIC
LINK = g++ -g -fPIC

# Uncomment the following if building on EECS linux machines
C_FLAGS =
L_FLAGS = -lpthread
# Uncomment the following if building on a Macintosh
# C_FLAGS = -std=c++11
# L_FLAGS =

project1: project1.o
	$(LINK) project1.o -o project1 $(L_FLAGS)
project1.o: project1.c++
	$(CPP) $(C_FLAGS) -c project1.c++
clean:
	-rm project1 *.o