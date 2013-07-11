OBJS = FactorTree.o
CC = g++
DEBUG = -g
INCLUDES = -I../.. 
LIBS = -L/usr/local/lib/ -L../../ad3/
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 $(INCLUDES)
LFLAGS = $(LIBS) -lad3

all : simple_parser

simple_parser : $(OBJS) simple_parser.o 
	$(CC) $(OBJS) simple_parser.o $(LFLAGS) -o simple_parser

simple_parser.o: simple_parser.cpp FactorHeadAutomaton.h FactorTree.h
	$(CC) $(CFLAGS) simple_parser.cpp

FactorTree.o: FactorTree.cpp FactorTree.h
	$(CC) $(CFLAGS) FactorTree.cpp

clean:
	rm -f *.o *~ simple_parser
