<p  align="center">
<img  src="https://github.com/Evaan2001/GPU-Sudoku-Solver/assets/82547698/da367d87-3376-4228-8d82-6d2f91bfc3e9"
width = "900"/>

</p>
<h3 align="center">
We'll solve a lot of sudoku boards – and solve them pretty darn quickly! 
</h3>

<p align="center">
Here's a highly parallelized program in CUDA (Nvidia's parallel computing platform) that was able to solve 1,750,000 sudoku boards/sec by applying constraint propagation on a GPU (Graphical Processing Units).
</p>

<h2 align="center"> 
What Do We Need?
</h2>
 
<p  align="center">
CUDA is a programming model developed by Nvidia for general computing on GPUs. To run cuda files, you'll need two things –
</p>

1) The nvcc compiler, which stands for **Nv**idia **C**uda **C**ompiler
2) A Nvidia graphics card (I used the Nvidia Quadro K1200 GPU which sells for about $100)

<h2 align="center"> 
An Overview Of The Problem Solving Logic
</h2>

<p  align="center">
Let's talk about **what on earth is constraint propogation**! Here's what I lazily copied from ChatGPT. 😌
</p>

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
Yes, there is one important thing. Constraint propogation can be super efficiently implemented on a GPU kernel. However, this logic doesn't guarantee solving all sudoku boards. It can certainly solve quite a few sudoku puzzles but fails against some of the more tricky ones. I tested my program on 5 different data sets of varying sizes and here are the results:

1.  100/100 for the tiny set
2.  1000/1000 for the small set
3.  9997/10000 for the medium set
4.  99959/100000 for the large set <- this is still a 99.96% solving rate (I'm happy with this!) and just takes 0.057 seconds in total (now I'm really happy 😂)!
</p>

<h2 align="center"> 
Programming Implemention
</h2>
 
<p  align="center">
Te
</p>

<h2 align="center"> 
Files
</h2>
 
<p  align="center">
Here's what you'll find –
</p>

1. *input* – a directory/folder that has 5 csv files. Each file has sudoku boards where a blank cells is represented by 0. The tiny file has 100 boards, small has 1000, medium has 10,000, and large has 100,000 boards.
2. *Makefile* – This compiles the program and preprares a file that you can run from the terminal.
3. *sudoku.cu* – 
4. *util.h* – A utilities file that helps with time calculations to measure performance.

<h2 align="center"> 
How To Run This
</h2>


1. To run the program, first download *sudoku.cu*, *util.h*, *Makefile*, and the *inputs* sub-directory/sub-folder from my repository and store them in one directory/folder. So if you name your folder "sudoku-solver", it should have *sudoku.cu*, *util.h*, *Makefile*, and the *inputs* folder. If you open *input*, you'll see the 5 individual input csv files.
2. Now install nvcc (the compiler for CUDA) if you don't have it already. A common practice is to download the entire CUDA toolkit that you can find [here](https://developer.nvidia.com/cuda-downloads).
3. Once you have that set up, open a terminal window and navigate your way to the directory containing the files you downloaded. 
4. Now just type `make all`.
5. Wait for the magic to happen. 🙂