#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include "util.h"

// The width and height of a sudoku board
#define BOARD_DIM 9

// The width and heigh of a square group in a sudoku board
#define GROUP_DIM 3

// The number of boards to pass to the solver at one time
#define BATCH_SIZE (128*200)

/**
 * A board is an array of 81 cells. Each cell is encoded as a 16-bit integer.
 * Read about this encoding in the documentation for the digit_to_cell and
 * cell_to_digit functions' documentation.
 *
 * Boards are stored as a one-dimensional array. It doesn't matter if you use
 * row-major or column-major form (that just corresponds to a rotation of the
 * sudoku board) but you will need to convert column and row to a single index
 * when accessing the board to propagate constraints.
 */
typedef struct board {
  uint16_t cells[BOARD_DIM * BOARD_DIM];
} board_t;

// Declare a few functions. Documentation is with the function definition.
void print_board(board_t* board);
__host__ __device__ uint16_t digit_to_cell(int digit);
__host__ __device__ int cell_to_digit(uint16_t cell);


/**
 * The function solve_boards calls this GPU kernel to parallely solve sudoku boards. Each block
 * corresponds to a board and has 9*9 threads (each thread corresponds to one cell on a board).
 * This kernel only implements constraint propagation as a way to solve the boards. As a result,
 * not all boards are guaranteed to be solved.
 *
 * \param boards      An array of boards to be solved.
 */


__global__ void solver_kernel(board_t* boards) {

  // Get the x,y coordinate of a cell from thread data
  int x = threadIdx.x;
  int y = threadIdx.y;

  // calculate the index that the cell will be located inside a board's cells array
  size_t index = (9 * threadIdx.x) + threadIdx.y;
  // get the current cell value, i.e., range of possibile numbers that could go in that cell
  uint16_t current_cell = boards[blockIdx.x].cells[index];
  // create another variable to hold a potentially new cell value
  uint16_t new_cell = current_cell;

  int predicate; // predicate for __sync_threads_count(); gets incremented if the new_cell value is different from the original value and so tells a block to run the threads again
  int val = 0; // stores the value held in a cell
  int i = 0, j = 0, block_x = 0, block_y = 0; // loop control variables
  do{
    predicate = 0;

    if(cell_to_digit(new_cell) == 0){ // if we are working with a blank cell

      // go through the relevant row
      for(i=0; i<9; i++){ // i here shuffles through the different columns in a row
        if(i==y)
            continue;
        val = cell_to_digit(boards[blockIdx.x].cells[9*x + i]);
        if (val != 0) {
          /* using idempotence (look at the description of cell encoding in the documentation for digit_to_cell) to narrow down the possible values for the cell */
          new_cell = new_cell & ~(1 << val);
        }
      }

      // go through the relevant column
      for(i=0; i<9; i++){ // i here shuffles through the different rows in a column
        if(i==x)
            continue;
        val = cell_to_digit(boards[blockIdx.x].cells[9*i + y]);
        if (val != 0) {
          // same usage of idempotence as in the case of going through the relevant row
          new_cell = new_cell & ~(1 << val);
        }
      }

      // Find out the x,y coordinates of the top-left corner of the relevant block
      block_x = x - x%3;
      block_y = y - y%3;

      // Search the block
      for(i=block_x; i<block_x+3; i++){
        for(j=block_y; j<block_y+3; j++){
          if(i==x && j==y)
              continue;
          val = cell_to_digit(boards[blockIdx.x].cells[9*i + j]);
          if (val != 0) {
            // same usage of idempotence as in the case of going through the relevant row/column
            new_cell = new_cell & ~(1 << val);
          }
        }
      }

      if (boards[blockIdx.x].cells[index] != new_cell)
      // if we have a new value for the cell (this happens if another thread solved the value for its cell in a previous run
        predicate++; // increment the predicate as a non-zero predicate signals a block to run again

      boards[blockIdx.x].cells[index] = new_cell; // update the current cell value

    }
  }while(__syncthreads_count(predicate) != 0); // run the loop while any thread reports making progress
}


/**
 * Take an array of boards and solve them all. The number of boards will be no
 * more than BATCH_SIZE, but may be less if the total number of input boards
 * is not evenly-divisible by BATCH_SIZE.
 *
 * TODO: Implement this function! You will need to add a GPU kernel, and you
 *       will almost certainly want to write helper functions; that is fine.
 *       However, you should not modify any other functions in this file.
 *
 * \param boards      An array of boards that should be solved.
 * \param num_boards  The numebr of boards in the boards array
 */
void solve_boards(board_t* boards, size_t num_boards) {

  dim3 threads(9,9); // to configure the layout of threads in a block

  board_t* gpu_boards; // to store the GPU's copy of the boards to solve

  // malloc space for gpu_boards
  if(cudaMalloc(&gpu_boards, sizeof(board_t) * num_boards) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate space for %d boards\n", num_boards);
    exit(2);
  }

  // copy over the boards from the CPU to GPU
  if(cudaMemcpy(gpu_boards, boards, sizeof(board_t) * num_boards, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy boards to the GPU\n");
  }

  // Call the kernel
  solver_kernel<<<num_boards, threads>>>(gpu_boards);

  // Wait for the threads to finish
  if(cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  // Copy the solved boards from the GPU back to the CPU
  if(cudaMemcpy(boards, gpu_boards, sizeof(board_t) * num_boards, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy solved boards from the GPU\n");
  }

  // Free memory malloced for GPU
  cudaFree(gpu_boards);

}

/**
 * Take as input an integer value 0-9 (inclusive) and convert it to the encoded
 * cell form used for solving the sudoku. This encoding uses bits 1-9 to
 * indicate which values may appear in this cell.
 *
 * For example, if bit 3 is set to 1, then the cell may hold a three. Cells that
 * have multiple possible values will have multiple bits set.
 *
 * The input digit 0 is treated specially. This value indicates a blank cell,
 * where any value from one to nine is possible.
 *
 * \param digit   An integer value 0-9 inclusive
 * \returns       The encoded form of digit using bits to indicate which values
 *                may appear in this cell.
 */
__host__ __device__ uint16_t digit_to_cell(int digit) {
  if (digit == 0) {
    // A zero indicates a blank cell. Numbers 1-9 are possible, so set bits 1-9.
    return 0x3FE;
  } else {
    // Otherwise we have a fixed value. Set the corresponding bit in the board.
    return 1 << digit;
  }
}

/*
 * Convert an encoded cell back to its digit form. A cell with two or more
 * possible values will be encoded as a zero. Cells with one possible value
 * will be converted to that value.
 *
 * For example, if the provided cell has only bit three set, this function will
 * return the value 3.
 *
 * \param cell  An encoded cell that uses bits to indicate which values could
 *              appear at this point in the board.
 * \returns     The value that must appear in the cell if there is only one
 *              possibility, or zero otherwise.
 */
__host__ __device__ int cell_to_digit(uint16_t cell) {
  // Get the index of the least-significant bit in this cell's value
  #if defined(__CUDA_ARCH__)
    int msb = __clz(cell);
    int lsb = sizeof(unsigned int) * 8 - msb - 1;
  #else
    int lsb = __builtin_ctz(cell);
  #endif

  // Is there only one possible value for this cell? If so, return it.
  // Otherwise return zero.
  if (cell == 1 << lsb)
    return lsb;
  else
    return 0;
}

/**
 * Read in a sudoku board from a string. Boards are represented as an array of
 * 81 16-bit integers. Each integer corresponds to a cell in the board. Bits
 * 1-9 of the integer indicate whether the values 1, 2, ..., 8, or 9 could
 * appear in the given cell. A zero in the input indicates a blank cell, where
 * any value could appear.
 *
 * \param output  The location where the board will be written
 * \param str     The input string that encodes the board
 * \returns       true if parsing succeeds, false otherwise
 */
bool read_board(board_t* output, const char* str) {
  for (int index = 0; index < BOARD_DIM * BOARD_DIM; index++) {
    if (str[index] < '0' || str[index] > '9') return false;

    // Convert the character value to an equivalent integer
    int value = str[index] - '0';

    // Set the value in the board
    output->cells[index] = digit_to_cell(value);
  }

  return true;
}

/**
 * Print a sudoku board. Any cell with a single possible value is printed. All
 * cells with two or more possible values are printed as blanks.
 *
 * \param board   The sudoku board to print
 */
void print_board(board_t* board) {
  for (int row = 0; row < BOARD_DIM; row++) {
    // Print horizontal dividers
    if (row != 0 && row % GROUP_DIM == 0) {
      for (int col = 0; col < BOARD_DIM * 2 + BOARD_DIM / GROUP_DIM; col++) {
        printf("-");
      }
      printf("\n");
    }

    for (int col = 0; col < BOARD_DIM; col++) {
      // Print vertical dividers
      if (col != 0 && col % GROUP_DIM == 0) printf("| ");

      // Compute the index of this cell in the board array
      int index = col + row * BOARD_DIM;

      // Get the index of the least-significant bit in this cell's value
      int digit = cell_to_digit(board->cells[index]);

      // Print the digit if it's not a zero. Otherwise print a blank.
      if (digit != 0)
        printf("%d ", digit);
      else
        printf("  ");
    }
    printf("\n");
  }
  printf("\n");
}

/**
 * Check through a batch of boards to see how many were solved correctly.
 *
 * \param boards        An array of (hopefully) solved boards
 * \param solutions     An array of solution boards
 * \param num_boards    The number of boards and solutions
 * \param solved_count  Output: A pointer to the count of solved boards.
 * \param error:count   Output: A pointer to the count of incorrect boards.
 */
void check_solutions(board_t* boards,
                     board_t* solutions,
                     size_t num_boards,
                     size_t* solved_count,
                     size_t* error_count) {
  // Loop over all the boards in this batch
  for (int i = 0; i < num_boards; i++) {
    // Does the board match the solution?
    if (memcmp(&boards[i], &solutions[i], sizeof(board_t)) == 0) {
      // Yes. Record a solved board
      (*solved_count)++;
    } else {
      // No. Make sure the board doesn't have any constraints that rule out
      // values that are supposed to appear in the solution.
      bool valid = true;
      for (int j = 0; j < BOARD_DIM * BOARD_DIM; j++) {
        if ((boards[i].cells[j] & solutions[i].cells[j]) == 0) {
          valid = false;
        }
      }

      // If the board contains an incorrect constraint, record an error
      if (!valid) (*error_count)++;
    }
  }
}

/**
 * Entry point for the program
 */
int main(int argc, char** argv) {
  // Check arguments
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <input file name>\n", argv[0]);
    exit(1);
  }

  // Try to open the input file
  FILE* input = fopen(argv[1], "r");
  if (input == NULL) {
    fprintf(stderr, "Failed to open input file %s.\n", argv[1]);
    perror(NULL);
    exit(2);
  }

  // Keep track of total boards, boards solved, and incorrect outputs
  size_t board_count = 0;
  size_t solved_count = 0;
  size_t error_count = 0;

  // Keep track of time spent solving
  size_t solving_time = 0;

  // Reserve space for a batch of boards and solutions
  board_t boards[BATCH_SIZE];
  board_t solutions[BATCH_SIZE];

  // Keep track of how many boards we've read in this batch
  size_t batch_count = 0;

  // Read the input file line-by-line
  char* line = NULL;
  size_t line_capacity = 0;
  while (getline(&line, &line_capacity, input) > 0) {
    // Read in the starting board
    if (!read_board(&boards[batch_count], line)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    // Read in the solution board
    if (!read_board(&solutions[batch_count], line + BOARD_DIM * BOARD_DIM + 1)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    // Move to the next index in the batch
    batch_count++;

    // Also increment the total count of boards
    board_count++;

    // If we finished a batch, run the solver
    if (batch_count == BATCH_SIZE) {
      size_t start_time = time_ms();
      solve_boards(boards, batch_count);
      solving_time += time_ms() - start_time;

      check_solutions(boards, solutions, batch_count, &solved_count, &error_count);

      // Reset the batch count
      batch_count = 0;
    }
  }

  // Check if there's an incomplete batch to solve
  if (batch_count > 0) {
    size_t start_time = time_ms();
    solve_boards(boards, batch_count);
    solving_time += time_ms() - start_time;

    check_solutions(boards, solutions, batch_count, &solved_count, &error_count);
  }

  // Print stats
  double seconds = (double)solving_time / 1000;
  double solving_rate = (double)solved_count / seconds;

  // Don't print nan when solver is not implemented
  if (seconds < 0.01) solving_rate = 0;

  printf("Boards: %lu\n", board_count);
  printf("Boards Solved: %lu\n", solved_count);
  printf("Errors: %lu\n", error_count);
  printf("Total Solving Time: %lums\n", solving_time);
  printf("Solving Rate: %.2f sudoku/second\n", solving_rate);

  return 0;
}
