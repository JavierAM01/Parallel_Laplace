# :zap: Parallel Laplace Code :zap:

![Visualization](images/gif.gif)

## Overview

This repository contains an implementation of a parallelized solution for the Laplace equation using MPI (Message Passing Interface) and Python. The Laplace equation is a fundamental partial differential equation that appears in various fields such as physics, engineering, and mathematics. This implementation distributes the computational workload across multiple processes, significantly speeding up the calculation on large grids.

## Why Parallelization Matters

Parallelization is increasingly important in today's world as datasets grow larger and computations become more complex. Leveraging multiple processing units allows for faster computations by dividing tasks among different processors, thus reducing the overall runtime. In scientific computing, parallelization is essential for solving problems like fluid dynamics, image processing, or machine learning on large datasets. In this project, parallelization is used to efficiently solve the Laplace equation over a large grid, demonstrating the power of distributed computing.

## Code Breakdown

### MPI Setup

```python
COMM = MPI.COMM_WORLD
PE_num = COMM.Get_rank()
PE_size = COMM.Get_size()
```
- **COMM**: Sets up the communicator between processes.
- **PE_num**: Determines the rank of the process (i.e., which processor this is).
- **PE_size**: The total number of processing elements (PEs).

### Grid Initialization and Distribution

The grid is divided among processes, with each process responsible for a portion of the grid:
```python
def create_info():
    # Number of rows each process will handle
    ...
    return all_rows, my_AB
```
This function determines how many rows each process will handle and ensures "ghost cells" are added for communication between adjacent processes.

### Laplace Calculation

The Laplace equation is iteratively solved using the following:
```python
while ( dt > MAX_TEMP_ERROR ) and ( iteration < max_iterations ):
    my_temperature[1:-1, 1:-1] = 0.25 * ( my_temperature_last[:-2, 1:-1] + my_temperature_last[2:, 1:-1] + 
                                          my_temperature_last[1:-1, :-2] + my_temperature_last[1:-1, 2:]  )
    ...
```
Each process calculates the new temperature values for its portion of the grid, then exchanges the boundary information with neighboring processes to ensure continuity across the entire grid.

### Synchronization and Final Output

After all iterations, the results from each process are gathered into the root process:
```python
COMM.Gatherv(flat_X, [gathered_data, total_elements, displacements, MPI.DOUBLE], root=0)
```
The results are then combined into the final matrix and outputted for visualization.

## Results

The result includes:
- A visual representation of the temperature distribution.
- Scaled versions of the output for better detail.
- Execution time and configuration details such as the number of iterations and the maximum temperature error.

## How to Run

To run the code on multiple processes, use the `mpiexec` command:
```bash
mpirun -n <number_of_processes> python main.py
```
Make sure you have `mpi4py` installed:
```bash
pip install mpi4py
```

## Visualization

This repository also includes a visualization of the final temperature distribution of the grid, helping you better understand the results.

---

Explore the power of parallel computing with this practical implementation of the Laplace equation!
