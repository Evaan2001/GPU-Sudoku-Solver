<p align="center">
<img src="Screenshot 2023-08-02 at 5 44 00 PM" src="https://github.com/Evaan2001/GPU-Sudoku-Solver/assets/82547698/ea6d1533-3d04-4094-a5c9-47585da9971d"/>
</p>

<h3 align="center">
We'll solve a lot of sudoku boards – and solve them pretty darn quickly! 
</h3>

<p align="center">
Here's a highly parallelized program in CUDA (Nvidia's parallel computing platform) that was able to solve 1,750,000 sudoku boards/sec by applying constraint propagation on a GPU (Graphical Processing Units).
</p>

<h1 align="center"> 
What Do We Need?
</h1>
 
<p align="center">
CUDA is a programming model developed by NVIDIA  for general computing on GPUs. To run cuda files, you'll need two things –
</p>

1) The nvcc compiler, which stands for **Nv**idia **C**uda **C**ompiler

2) A NVidia graphics card (I used the NVidia Quadro K1200 GPU which sells for about $100)
