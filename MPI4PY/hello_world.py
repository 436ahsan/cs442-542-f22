# https://mpi4py.readthedocs.io/en/stable/tutorial.html

# To Run : mpirun -n <num_procs> python3 hello_world.py

from mpi4py import MPI
import numpy as np


if __name__=="__main__":
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    print("Hello World from rank %d of %d\n"%(rank, num_procs))

    MPI.Finalize()

