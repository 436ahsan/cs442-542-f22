# https://mpi4py.readthedocs.io/en/stable/tutorial.html

# To run : mpirun -n 2 python3 mpi_send.py

from mpi4py import MPI
import numpy as np

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    size = np.random.randint(1000)
    tag = 1234

    if rank % 2 == 0:
        comm.send(size, dest=rank+1, tag=tag)
    else:
        size = comm.recv(source=rank-1, tag=tag)

