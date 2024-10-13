import numpy as np
import math
import time

import matplotlib.pyplot as plt
from PIL import Image
import os, shutil

from mpi4py import MPI



COMM = MPI.COMM_WORLD
PE_num = COMM.Get_rank()
PE_size = COMM.Get_size()

ROWS , COLUMNS = 1000, 1000
MAX_TEMP_ERROR = 0.01


def output(data, string_output):

    # create folder for results (if it exists is going to erase the previous results)
    folder = f"output3"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    # save string
    with open(os.path.join(folder, f"output.yml"), "w") as file:
        file.write(string_output)

    # save image
    plt.imshow(data, cmap="plasma")
    plt.colorbar()
    plt.savefig(os.path.join(folder, f'plate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # save image scaled - for better details
    plt.imshow(100*np.log(data+1)/np.log(101), cmap="plasma")
    plt.colorbar()
    plt.savefig(os.path.join(folder, f'plate_scaled.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # save matrix
    data.tofile(os.path.join(folder, f"plate.out"))


def create_info():
    # find the number of rows for every PE
    n = (ROWS+2) // PE_size
    r = (ROWS+2) % PE_size
    all_rows = [n+3]*r + [n+2]*(PE_size-r)   # we are going to distribute (+1) to "r" PEs and (+2) to every PE for the ghost ceels
    # take care of exceptions
    all_rows[0] -= 1
    all_rows[-1] -= 1

    # find the range [A,B] for the rows of the original matrix in which this PE should be put in 
    # to be able to init the data
    ranges = [n+1]*r + [n]*(PE_size-r)
    s = sum(ranges[:PE_num])
    my_AB = [s, s+ranges[PE_num]]
    # take care of the extra edges
    if PE_num != 0:
        my_AB[0] -= 1
    if PE_num != PE_size-1:
        my_AB[1] += 1

    # return info
    return all_rows, my_AB

ALL_ROWS, my_AB = create_info()
my_ROWS = ALL_ROWS[PE_num]
my_COLS = COLUMNS + 2


my_temperature = np.empty(( my_ROWS , my_COLS ))
my_temperature_last = np.empty(( my_ROWS , my_COLS ))




# ----------------------------------------------------------------------
# -------------------     INIT LOCAL MATRIX     ------------------------
# ----------------------------------------------------------------------

def initialize_temperature(temp):
    temp[:,:] = 0

    #Set right side boundary condition
    ii = 1
    for i in range(my_AB[0]+1, my_AB[1]-1):
        temp[ ii , my_COLS-1 ] = 100 * math.sin( ( (3.14159/2) /ROWS ) * i )
        ii += 1

    #Set bottom boundary condition (only for the last PE)
    if PE_num == PE_size-1:
        for i in range(1, my_COLS-1):
            temp[ -1 , i ] = 100 * math.sin( ( (3.14159/2) /COLUMNS ) * i )



initialize_temperature(my_temperature_last)

# ----------------------------------------------------------------------
# -------------------     LAPLACE EQUATION      ------------------------
# ----------------------------------------------------------------------


if PE_num == 0:
    max_iterations = int (input("Maximum iterations: "))
else:
    max_iterations = None

max_iterations = COMM.bcast(max_iterations, root=0)


# timer for the algorithm
t0 = time.time()

dt = 100
iteration = 0

while ( dt > MAX_TEMP_ERROR ) and ( iteration < max_iterations ):
    
    # calc new temperature
    my_temperature[1:-1, 1:-1] = 0.25 * ( my_temperature_last[:-2, 1:-1] + my_temperature_last[2:, 1:-1] + 
                                                    my_temperature_last[1:-1, :-2] + my_temperature_last[1:-1, 2:]  )
    dt = 0

    # share edges
    if PE_num != 0:
        COMM.Send([my_temperature[1, 1:-1], MPI.DOUBLE], dest=PE_num-1)
    if PE_num != PE_size - 1:
        COMM.Send([my_temperature[-2, 1:-1], MPI.DOUBLE], dest=PE_num+1)
    if PE_num != 0:
        COMM.Recv([my_temperature[0, 1:my_COLS-1], MPI.DOUBLE], source=PE_num-1)
    if PE_num != PE_size - 1:
        COMM.Recv([my_temperature[-1, 1:my_COLS-1], MPI.DOUBLE], source=PE_num+1)

    # calc: |old - new| and update "last temperature + shared edges"
    dt = np.max(my_temperature[1:-1, 1:-1] - my_temperature_last[1:-1, 1:-1])
    start = 0
    finish = my_ROWS
    if PE_num == 0:
        start = 1
    elif PE_num == PE_size - 1:
        finish = my_ROWS-1
    my_temperature_last[ start:finish, 1:-1 ] = my_temperature [ start:finish, 1:-1 ]


    iteration += 1

    # collect all "dt" to avoid one PE scaping and the others getting stacked
    dt = COMM.allreduce(dt, op=MPI.MAX)



t1 = time.time()



# ----------------------------------------------------------------------
# -------------------  JOIN MATRIXES TO PE = 0  ------------------------
# ----------------------------------------------------------------------

if PE_num == 0:
    total_elements = [(nr-2) * my_COLS for nr in ALL_ROWS]  # Total elements from each PE
    total_elements[0] += my_COLS
    total_elements[-1] += my_COLS
    gathered_data = np.empty(sum(total_elements), dtype='float64')  # Flat buffer for all data
    displacements = np.cumsum([0] + total_elements[:-1])  # Offsets for each PE's data
else:
    gathered_data = None
    total_elements = None
    displacements = None

# get rid of the extra edges & flat data to send
if PE_num == 0:
    flat_X = my_temperature_last[:-1].flatten()
elif PE_num == PE_size-1:
    flat_X = my_temperature_last[1:].flatten()
else:
    flat_X = my_temperature_last[1:-1].flatten()


COMM.Gatherv(flat_X, [gathered_data, total_elements, displacements, MPI.DOUBLE], root=0)

if PE_num == 0:
    # Now reconstruct the matrices based on the known shapes
    gd = []  # List to hold the gathered matrices
    start = 0
    for i, nr in enumerate(ALL_ROWS):
        elements = total_elements[i] 
        shape = (elements // my_COLS, my_COLS)
        matrix = gathered_data[start:start + elements].reshape(shape)
        gd.append(matrix)  # Store matrix in the list gd
        start += elements
        
    final_matrix = np.concatenate(gd, axis=0)
    
    t2 = time.time()

    # description of the script results and configurations 
    string_output = f"""
Configuration:
  PE_size: {PE_size}
  Plate_dimensions:
    x: {ROWS}
    y: {COLUMNS}
  Max_iterations: {max_iterations}
  Max_error: {MAX_TEMP_ERROR}

Output:
   iterations: {iteration}
   time: {t2-t0}
   time_prior_joining_local_PEs_result: {t1-t0}
    """
    
    print(final_matrix.shape)

    output(data=final_matrix, string_output=string_output)
