PROG = example

CPP = g++ -fPIC -std=c++11
LINK = g++ -fPIC

INC = -I../Packed3DArray -I../ImageWriter
CFLAGS = -g -c $(INC)

LIB = ../lib/libCOGLImageWriter.so

$(PROG): $(PROG).o $(LIB)
	$(LINK) -g -o $(PROG) $(PROG).o $(LIB)

../lib/libCOGLImageWriter.so: ../ImageWriter/ImageWriter.h ../ImageWriter/ImageWriter.c++
	(cd ../ImageWriter; make)

$(PROG).o: $(PROG).c++
	$(CPP) $(CFLAGS) $(PROG).c++
