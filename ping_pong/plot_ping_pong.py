import numpy as np
import pyfancyplot.plot as plot

n_sizes = 19

## Parse Output
f= open("mac_ping_pong.out")
sizes = list()
times = list()
for i in range(n_sizes):
    sizes.append(2**i * 4)
    times.append(np.inf)
ctr = 0
for line in f:
    list_words = (line.rsplit('\n')[0]).rsplit(' ')
    size = (int)((list_words[1].rsplit(','))[0])
    time = (float)(list_words[3])
    if (time < times[ctr]):
        times[ctr] = time
    ctr += 1
    if ctr == len(sizes):
        ctr = 0
f.close()


## Plot Times
plot.add_luke_options()
plot.line_plot(times, sizes)
plot.add_labels("Message Size (Bytes)", "Measured Time")
plot.set_scale('log', 'log')
plot.save_plot("ping_pong_plot.pdf")


## Calculate Alpha/Beta (for eager and rendezvous)
eager_lim = 8192
eager_A = list()
rend_A = list()
eager_b = list()
rend_b = list()
for i in range(n_sizes):
    if sizes[i] < eager_lim:
        eager_A.append([1, sizes[i]])
        eager_b.append(times[i])
    else:
        rend_A.append([1, sizes[i]])
        rend_b.append(times[i])

A = np.matrix(eager_A)
b = np.array(eager_b)
eager_alpha, eager_beta = np.linalg.lstsq(A, b)[0]

A = np.matrix(rend_A)
b = np.array(rend_b)
rend_alpha, rend_beta = np.linalg.lstsq(A, b)[0]

print("Eager: alpha %e, beta %e\nRend: alpha %e, beta %e\n" %(eager_alpha, eager_beta, rend_alpha, rend_beta))


