import multiprocessing as mp
import numpy as np
from scipy import special
import csv

def get_times(N_list, times_per_N, N_scale):
    times = []
    for n in N_list:
        times += [n*t*N_scale for t in range(times_per_N)]
    return list(set(times))


def get_init_sites(setup):
    '''
    List of site indices at which particles occupy initially
    '''
    if setup == "Domwall":
        return [i for i in range(N)]
    if setup == "Cake":
        return [i for i in range(L/4, L/4 + N)]

def get_exact_corrs(t):
    '''
    Exact correlation matrix at time t
    '''
    mat = np.zeros((L,L), dtype = complex)
    for m in range(L):
        for n in range(m,L):
            # total = 0.0
            # for j in init_sites:
            #     total += special.jv(m-j, 2*t)*special.jv(n-j, 2*t)
            # # total *= 1j**(n - m)
            # mat[m][n] = total
            # mat[n][m] = np.conjugate(total)
            c_mn = special.jv(m, 2*t)*special.jv(n+1, 2*t) - special.jv(m+1, 2*t)*special.jv(n, 2*t)
            c_mn *= 1j**(n - m) * t / (n - m)
            mat[m][n] = c_mn
            mat[n][m] = np.conjugate(c_mn)
    return mat


def main(t):
    exact_corrs = get_exact_corrs(t)               # exact density matrix

    with open(setup + "-exact"  + "/exact-t" + str(t), 'w') as exact_file:
        writer = csv.writer(exact_file)
        [writer.writerow(r) for r in exact_corrs]

    # s = "\nExact correlation matrix at time t = " + str(t) + ":" + "\n" + str(exact_corrs) + "\n"*2
    # print s


'''
Initial conditions
'''

N = 1000                                                  # Number of particles
L = 2*N
setup = "Domwall"

N_list = [10, 20, 50, 100, 200, 500, 1000]
N_list = [10, 20, 50]
times_per_N = 12
N_scale = 1.0/20

times = get_times(N_list, times_per_N, N_scale)
init_sites = get_init_sites(setup)                      # List of sites of initial positions

for t in times:
    main(t)

# # Parallelization
# num = 4                                                 # Number of CPU's
# p = mp.Pool(num)
# p.map(main, times)                                      # Parallelize main(t) for t arguments
