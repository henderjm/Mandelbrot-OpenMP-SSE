CC=g++
CFLAGS=-c -Wall -ansi -fopenmp -O3
LDFLAGS=-fopenmp
SOURCES=main.cpp Screen.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=mandelbrot
LIBS=-lSDL -fopenmp

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) $(LIBS)  -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@



clean:
	rm -f *.o $(EXECUTABLE)
