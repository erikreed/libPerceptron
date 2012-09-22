#============================================================================
# Name        : 
# Author      : Erik Reed
# Description : 
#============================================================================

CC      = g++
CFLAGS  = -Wall -g #NDEBUG
LDFLAGS = 
OBJECTS = testPerceptron.o Perceptron.o DataSet.o MLPerceptron.o NaiveBayes.o

all: main

main: $(OBJECTS)
	$(CC) ${CFLAGS} -o $@ $^ $(LDFLAGS)


%.o: %.cpp
	${CC} ${CFLAGS} -c $< -o $@

.PHONY: clean run memcheck
run: all
	./main

clean:
	rm -f *.o opt-mlp

memcheck: all
	valgrind --tool=memcheck --leak-check=full --show-reachable=yes ./opt-mlp
