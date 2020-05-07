#include <stdio.h>
#include <cstdlib>

// A common utility extracted (and slightly modified) from the source
// code associated with the book "Heterogeneous Computing with OpenCL".

// This function reads a text file and returns a pointer to its
// contents as stored in the heap.  The caller is responsible for issuing
//     delete [] ptr;
// on the returned pointer.
const char* readSource(const char* kernelPath)
{
   printf("Program file is: %s\n", kernelPath);

   FILE* fp = fopen(kernelPath, "rb");
   if(!fp)
   {
      printf("Could not open kernel file\n");
      exit(-1);
   }
   int status = fseek(fp, 0, SEEK_END);
   if(status != 0)
   {
      printf("Error seeking to end of file\n");
      exit(-1);
   }
   long int size = ftell(fp);
   if(size < 0)
   {
      printf("Error getting file position\n");
      exit(-1);
   }

   rewind(fp);

   char* source = new char[size + 1];

   for (int i = 0; i < size+1; i++)
   {
      source[i]='\0';
   }

   fread(source, 1, size, fp);
   source[size] = '\0';

   return source;
}

