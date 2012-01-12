#============================================================================
# Name        : Opt-MLP-lib
# Author      : Erik Reed
# Description : Implementation of an optimized, multithreaded, 
#               single and multi-layer perceptron library in C++. 
#               Fast neural network.
#============================================================================

CC      = g++
CFLAGS  = -Wall -g
LDFLAGS = 
OBJECTS = main.o Perceptron.o DataSet.o MLPerceptron.o

all: opt-mlp

opt-mlp: $(OBJECTS)
	$(CC) ${CFLAGS} -o $@ $^ $(LDFLAGS)


%.o: %.cpp
	${CC} ${CFLAGS} -c $< -o $@

.PHONY: clean run memcheck
run: all
	./opt-mlp

clean:
	rm -f *.o opt-mlp

memcheck: all
	valgrind --tool=memcheck --leak-check=full --show-reachable=yes ./opt-mlp
