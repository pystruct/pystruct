OBJS =
CC = g++
DEBUG = -g
INCLUDES = -I../.. 
LIBS = -L/usr/local/lib/ -L../../ad3/
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 $(INCLUDES)
LFLAGS = $(LIBS) -lad3

all : simple_grid

simple_grid : $(OBJS) simple_grid.o 
	$(CC) $(OBJS) simple_grid.o $(LFLAGS) -o simple_grid

simple_grid.o: simple_grid.cpp FactorSequence.h
	$(CC) $(CFLAGS) simple_grid.cpp

clean:
	rm -f *.o *~ simple_grid
