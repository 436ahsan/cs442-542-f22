import numpy as np
import scipy.sparse as sp
import matplotlib.pylab as plt
from scipy.io import mmread
import glob

files = glob.glob("*.mtx")
for filename in files:
    A = mmread(filename)
    plt.spy(A, markersize=.3, rasterized=True)
    plt.savefig"%s.pdf"%filename)
