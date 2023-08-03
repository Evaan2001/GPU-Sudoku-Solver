<p  align="center">
<img  src="https://github.com/Evaan2001/GPU-Sudoku-Solver/assets/82547698/da367d87-3376-4228-8d82-6d2f91bfc3e9"
width = "900"/>

</p>
<h3 align="center">
We'll solve a lot of sudoku boards – and solve them pretty darn quickly! 
</h3>

<p align="center">
Here's a highly parallelized program in CUDA (Nvidia's parallel computing platform) that was able to solve 1,750,000 sudoku boards/sec by applying constraint propagation on a GPU (Graphical Processing Unit).
</p>

<h2 align="center"> 
What Do We Need?
</h2>
 
<p  align="center">
CUDA is a programming model developed by Nvidia for general computing on GPUs. To run CUDA files, you'll need two things –
</p>

1) The nvcc compiler, which stands for **Nv**idia **C**uda **C**ompiler
2) A Nvidia graphics card (sadly, this means no Macs released after 2015; I used the Nvidia Quadro K1200 GPU which sells for about $100)

<h2 align="center"> 
An Overview Of The Problem-Solving Logic
</h2>

<align="center">
Let's discuss **what on earth constraint propagation is**! 

<p  align="center">
In Sudoku, each number 1-9 must appear exactly once in each row, column, and designated 3x3 block. If you can determine the value of a particular cell, then you can eliminate that value as a possibility from all other cells in the same row, column, and 3x3 block.
</p>

<p  align="center">
This elimination process can reveal new clues that further reduce the possibilities in other cells. By **repeatedly applying these constraints**, you may solve some cells entirely and make others easier to figure out. Constraint propagation can be an efficient way to make progress in solving a Sudoku puzzle by **systematically considering the implications of the known numbers**.
</p>

<h2 align="center"> 
Any Limitations?
</h2>

<p  align="center">
Yes, there is one important thing. Constraint propagation can be super efficiently implemented on a GPU kernel. However, this logic doesn't guarantee to solve all sudoku boards. It can certainly solve quite a few sudoku puzzles but fails against some of the more tricky ones. I tested my program on 5 different data sets of varying sizes and here are the results:

1.  100/100 for the tiny set
2.  1000/1000 for the small set
3.  9997/10000 for the medium set
4.  99959/100000 for the large set <- This is still a 99.96% solving rate (I'm happy with this!) and just takes 0.057 seconds in total (now I'm really happy 😂)!
</p>

<h2 align="center"> 
Programming Implementation
</h2>
 
1. We'll represent a board as an array of 81 cells.
2. Each cell is encoded as a 16-bit integer. Bits 1-9 of the integer indicate whether the values 1, 2, ..., 8, or 9 could appear in the given cell. For example, if bit 3 is set to 1, then the cell may hold a three. Cells that have multiple possible values will have multiple bits set to 1. So if a cell can have 1,3, 8, or 9, we'll encode it as `0000 0011 0000 1010` (Note that the rightmost bit is the 0th bit, so we ignore it as we are only concerned with bits 1-9).
3. We are gonna associate a 9*9 block of threads with each sudoku board; this means that we will have 1 thread for each cell of the board. 
4. Each thread will begin by getting the current cell value (the 16-bit integer). If only 1 bit of the value is set to 1, that means the cell is already solved for and no more work needs to be done for that cell. If however multiple bits are on, this means that we need to find a number for that cell. So we will first scan through the relevant row, column, and 3*3 cell block to see if we can reduce the number of possibilities or, ideally, solve the cell. If we do reduce the number of possibilities or solve the cell, we will update the cell value and call the kernel to run again for all of the threads. The idea is that if a particular cell has fewer possibilities, this information may be useful for other cells, so we want to run calculations for all cells again. If no cell is updated, that either means the board is solved or it is not solvable via constraint propagation; either way, we quit the program. 
5. A note on scanning for possibilities. We optimize this process by using the logical 'and' operator `&` and **idempotence**. This refers to the property of some operations that can be applied multiple times without changing the result beyond the initial application. For example, adding zero to a number will never change the result, regardless of how many times you do it. In our case, let's say cell A can have either 1,2, or 3. It's encoded as `A = 0000 0000 0000 1110`. Now suppose we see that another cell in that row, say B, has a value of 3. So B is encoded as `B = 0000 0000 0000 1000`. Clearly, A cannot have a value of 3 and we need to update this information. We can update this information by executing `A = A & ~B`. Basically, A can only have values that B can't, so invert the bits of B and bitwise AND them with the bits of A. For example, `A = A & ~B` = `A = A & ~(0000 0000 0000 1000)` = `A = A & (1111 1111 1111 0111)` = `A = (0000 0000 0000 1110) & (1111 1111 1111 0111)` = `A = 0000 0000 0000 0110`. As we see, we just updated A's value to only encode the possibilities of 1 and 2 (3 is gone). The idempotence property lies in the fact that we can repeat this process for newer values without changing the results of previous operations; even when the cell is solved, we can still run this operation without changing anything – much like adding 0 to a number.
6. In my code, I use the function `cell_to_digit` to help with the implementation of the idempotence logic. It takes in an encoded cell and returns which bit is active for that cell; it will return 0 if multiple bits are active. So `cell_to_digit(A)` will return `0` and `cell_to_digit(B)` will return `3`.  This ensures that the cell we will be using to eliminate possibilities has indeed been solved and is not just a sum of possible values. I use this function as follows: `A = A & ~(1 << cell_to_digit(B))`.  This means we have `A = A & ~(1 << cell_to_digit(B))` = `A = A & ~(1 << 3)` = `A = A & ~(1000)`. Note that 1000 is the same as  0000 0000 0000 1000. And with this, we can proceed the way we did in step 5.
7. The provided code reads a set of sudoku from an input file and passes them on to be solved. The code will read  `BATCH_SIZE`  sudoku boards at once, then invoke the  `solve_boards`  function. Note: I've found that setting `BATCH_SIZE = 128 * 200` produces the fastest results but do feel free to experiment.

<h2 align="center"> 
Files
</h2>
 
<p  align="center">
Here's what you'll find –
</p>

1. *input* – a directory/folder that has 4 CSV files. Each file has sudoku boards where a blank cell is represented by 0. The tiny file has 100 boards, the small has 1000, the medium has 10,000, and the large has 100,000 boards.
2. *Makefile* – This compiles the program and prepares a file that you can run from the terminal.
3. *sudoku.cu* – The CUDA file that has all of the main code
4. *util.h* – A utility file that helps with time calculations to measure performance.

<h2 align="center"> 
How To Run This
</h2>


1. To run the program, first download *sudoku.cu*, *util.h*, *Makefile*, and the *inputs* sub-directory/sub-folder from my repository and store them in one directory/folder. So if you name your folder "sudoku-solver", it should have *sudoku.cu*, *util.h*, *Makefile*, and the *inputs* folder. If you open *input*, you'll see the 4 individual input CSV files. Also note that depending on your browser, the *Makefile* might get downloaded as a text file, so you'll have *Makefile.txt*. If so, remove the *txt* extension as Makefiles don't have any extensions.
2. Now install nvcc (the compiler for CUDA) if you don't have it already. A common practice is to download the entire CUDA toolkit that you can find [here](https://developer.nvidia.com/cuda-downloads).
3. Once you have that set up, open a terminal window and navigate your way to the directory containing the files you downloaded. 
4. Now just type `make all`. This will create a new executable file named *sudoku*.
5. Now all that's left is running the executable with the input file. If you want to run this code on the tiny input set, you'll have to type `./sudoku inputs/tiny.csv` in the terminal. Analogously, `./sudoku inputs/small.csv` runs the program on the small input set, `./sudoku inputs/medium.csv` runs the program on the medium input set, and `./sudoku inputs/large.csv` runs the program on the large input set, 
6. Wait for the magic to happen. 🙂
