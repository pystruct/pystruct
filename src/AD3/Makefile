EXAMPLE_DENSE = examples/dense
EXAMPLE_PARSING = examples/parsing
EXAMPLE_LOGIC = examples/logic
AD3 = ad3
OBJS = FactorTree.o
CC = g++
INCLUDES = -I. -I./$(AD3) -I./$(EXAMPLE_DENSE) -I./$(EXAMPLE_PARSING) \
	-I./$(EXAMPLE_LOGIC)
LIBS = -L/usr/local/lib -L./$(AD3)
DEBUG = -g
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 $(INCLUDES)
LFLAGS = $(LIBS) -lad3

all: libad3 ad3_multi simple_grid simple_parser simple_coref

ad3_multi: $(OBJS) ad3_multi.o 
	$(CC) $(OBJS) ad3_multi.o $(LFLAGS) -o ad3_multi

ad3_multi.o: ad3_multi.cpp
	$(CC) $(CFLAGS) ad3_multi.cpp

FactorTree.o: $(EXAMPLE_PARSING)/FactorTree.cpp
	$(CC) $(CFLAGS) $(EXAMPLE_PARSING)/FactorTree.cpp

simple_grid: 
	cd $(EXAMPLE_DENSE) && $(MAKE)

simple_parser: 
	cd $(EXAMPLE_PARSING) && $(MAKE)

simple_coref: 
	cd $(EXAMPLE_LOGIC) && $(MAKE)

libad3:
	cd $(AD3) && $(MAKE)

clean:
	rm -f *.o *~ ad3_multi
	cd $(AD3) && $(MAKE) clean
	cd $(EXAMPLE_DENSE) && $(MAKE) clean
	cd $(EXAMPLE_PARSING) && $(MAKE) clean
	cd $(EXAMPLE_LOGIC) && $(MAKE) clean
	
