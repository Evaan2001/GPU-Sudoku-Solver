CC := nvcc
CFLAGS := -g

all: sudoku

clean:
	rm -f sudoku

sudoku: sudoku.cu util.h Makefile
	$(CC) $(CFLAGS) -o sudoku sudoku.cu

.PHONY: all clean zip format