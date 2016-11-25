OBJS =
CC = g++
DEBUG = -g
INCLUDES = -I../.. 
LIBS = -L/usr/local/lib/ -L../../ad3/
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 $(INCLUDES)
LFLAGS = $(LIBS) -lad3

all : simple_coref

simple_coref : $(OBJS) simple_coref.o 
	$(CC) $(OBJS) simple_coref.o $(LFLAGS) -o simple_coref

simple_coref.o: simple_coref.cpp
	$(CC) $(CFLAGS) simple_coref.cpp

clean:
	rm -f *.o *~ simple_coref
