CC      = g++
CFLAGS  = -Wall -g
LDFLAGS = 

all: opt-mlp

opt-mlp: main.o
	$(CC) -o $@ $^ $(LDFLAGS)
# headers below
opt-mlp.o: main.cpp 
	$(CC) -c $(CFLAGS) $<

.PHONY: clean run memcheck
run: all
	./opt-mlp

clean:
	rm -f *.o opt-mlp

memcheck: all
	valgrind --tool=memcheck --leak-check=full --show-reachable=yes ./opt-mlp
