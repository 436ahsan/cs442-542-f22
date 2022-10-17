import numpy as np
import scipy.sparse as sp
from scipy.io import mmread, mmwrite

A = mmread("Dubcova2.mtx")
A = A.tocsr()

n_procs = 16
n_rows = A.shape[0]
first_row = 0
for i in range(n_procs):
    local_rows = n_rows / n_procs
    if (n_rows % n_procs > i):
        local_rows += 1
    rowptr = list()
    cols = list()
    data = list()
    rowptr.append(0)
    last_row = (int)(first_row + local_rows)
    for row in range(first_row, last_row):
        for j in range(A.indptr[row], A.indptr[row+1]):
            cols.append(A.indices[j])
            data.append(A.data[j])
        rowptr.append(len(cols))
    A_local = sp.csr_matrix((data, cols, rowptr))
    mmwrite("Dubcova2_%d.mtx"%i, A_local)
    first_row = last_row

