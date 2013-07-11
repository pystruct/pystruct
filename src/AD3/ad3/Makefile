OBJS = FactorGraph.o Factor.o GenericFactor.o Utils.o
CC = g++
DEBUG = -g
INCLUDES = -I./ad3/ -I../Eigen
LIBS = -L/usr/local/lib/ -L./
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 $(INCLUDES) -fPIC
LFLAGS = $(LIBS) -lpthread

all : libad3.a

libad3.a : $(OBJS)
	ar rcs libad3.a $(OBJS)

FactorGraph.o: FactorGraph.h FactorGraph.cpp FactorDense.h Factor.h \
	MultiVariable.h Utils.h
	$(CC) $(CFLAGS) FactorGraph.cpp

GenericFactor.o: GenericFactor.h Factor.h GenericFactor.cpp Utils.h
	$(CC) $(CFLAGS) GenericFactor.cpp

Factor.o: Factor.h Factor.cpp Utils.h
	$(CC) $(CFLAGS) Factor.cpp

Utils.o: Utils.h Utils.cpp
	$(CC) $(CFLAGS) Utils.cpp

clean:
	rm -f *.o *~ libad3.a
