CC      = g++
CFLAGS  = -Wall -g
LDFLAGS = 

all: opt-mlp

opt-mlp: main.o
	$(CC) -o $@ $^ $(LDFLAGS)

opt-mlp.o: OptMLP.hpp NeuralNetwork.hpp DataUtils.hpp DataUtils.cpp Perceptron.cpp main.cpp
	$(CC) -c $(CFLAGS) $<

.PHONY: clean run memcheck
run: all
	./opt-mlp

clean:
	rm -f *.o opt-mlp

memcheck: all
	valgrind --tool=memcheck --leak-check=full --show-reachable=yes ./opt-mlp
