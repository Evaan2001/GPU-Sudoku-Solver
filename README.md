<p align="center">
<img src="https://github.com/Evaan2001/GPU-Sudoku-Solver/assets/82547698/4ef944d0-9bc9-4d0d-9b88-b9fdfe8c0f97" 
 width="200" />
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
An Overview Of The Programming Logic
</h2>

Before we go there, let's talk about **what on earth is constraint propogation**! Here's what I lazily copied from ChatGPT. 😌

In Sudoku, each number 1-9 must appear exactly once in each row, column, and designated 3x3 block. If you can determine the value of a particular cell, then you can eliminate that value as a possibility from all other cells in the same row, column, and 3x3 block.

This elimination process can reveal new clues that further reduce the possibilities in other cells. By **repeatedly applying these constraints**, you may solve some cells entirely and make others easier to figure out. Constraint propagation can be an efficient way to make progress in solving a Sudoku puzzle by **systematically considering the implications of the known numbers**.

<h2 align="center"> 
Any Limitations?
</h2>

<p  align="center">
Yes there is one important thing. Constraint propogation can be super efficiently implemented on a GPU kernel. However, this logic doesn't guarantee solving all sudoku boards. It can certainly solve quite a few sudoku puzzles but fails against some of the more tricky ones. I tested my program on 5 different data sets and here are the results:

1.  100/100 for the tiny set
2.  1000/1000 for the small set
3.  9997/10000 for the medium set
4.  99959/100000 for the large set
5.  999671/1000000 for the huge set <- this is still a 99.96% solving rate (I'm happy with this!) and just takes 0.57 seconds in total!
</p>
