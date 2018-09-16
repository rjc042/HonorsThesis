# #!/usr/remote/python-2.7/bin/python
# ^ (uncomment for linux)

import numpy as np
import math as math
import cmath as cmath
import itertools
import time
import sys
import multiprocessing as mp
from scipy import special
import time
import csv

### Noninteracting, Zero T ###


def read_csv(t):
    '''
    Read in csv
    '''
    with open("Domwall-exact/exact-t" + str(t), 'r') as csvfile:
        reader = csv.reader(csvfile)
        mat = np.array([np.array([complex(e) for e in r], dtype = complex) for r in reader])
    return mat[0:L, 0:L]

def get_bases():
    '''
    Construct computational basis states
    '''
    bases = []
    for i in range(L):
        row = '0'*i + '1' + '0'*(L-i-1)
        bases.append(row)
    return bases

def get_init_sites(setup):
    '''
    List of site indices at which particles occupy initially
    '''
    if setup == "Domwall":
        return [i for i in range(N)]
    if setup == "Cake":
        return [i for i in range(L/4, L/4 + N)]



def expval(lbasis, rbasis, cdag, c):
    '''
    Correlation function expectation value
    '''
    check_c = float(rbasis[c])
    shift = c - cdag
    if shift >= 0:
        rbasis = int(rbasis, 2) << shift
    else:
        rbasis = int(rbasis, 2) >> -shift
    return check_c*float(int(lbasis, 2) == rbasis)


def trace_dist(A,B):
    '''
    Calculate the trace distance between matrices A and B
    '''
    evals = np.linalg.eigvals(A-B)
    return (1.0/L)*sum([abs(l) for l in evals])



def get_effham(t):
    '''
    Construct effective Hamiltonian at time t
    '''
    effham = P + t*Q
    return effham


def get_eff_state(effham, track):
    '''
    State of Effective Hamiltonian
    Returns list of N single particle eigenstates
    '''
    effevals, effevecs = np.linalg.eigh(effham)         # L energy values, L eigenstates
    effevecs = np.transpose(effevecs)                   # eigenvectors are the columns effevecs[:,i]

    eff_state = []                                   # list of energies corresponding to initial positions
    for i in track:
        eff_state.append(effevecs[i])
    return eff_state


def get_eff_corrs(eff_state):
    '''
    Correlation matrix via emergent Hamiltonian
    '''
    eff_corrs = np.zeros((L,L), dtype = complex)
    for m in range(L):
        for n in range(m,L):
            total = 0.0
            for q in range(N):                              # sum over N corresponding single particle eigenstates
                phi0q = eff_state[q]                         # single particle eigenstate
                total += np.conjugate(phi0q[m]) * phi0q[n]    # multiply CC of qth single particle eigenstate at site m by value at site n
            eff_corrs[m][n] = total
            eff_corrs[n][m] = np.conjugate(total)        # antisymmetric matrix
    return eff_corrs



def get_effham_consts():
    '''
    Compute constants in effective Hamiltonian
    '''
    P = np.zeros((L,L), dtype = complex)
    Q = np.zeros((L,L), dtype = complex)
    for m in range(L):
        for n in range(m,L):
            sumP = (L - N) * expval(bases[m], bases[n], L-1, L-1)
            sumQ = 0.0
            for j in range(L-1):
                sumP += (j - N + 1) * expval(bases[m], bases[n], j, j)
                sumQ += 1j * expval(bases[m], bases[n], j, j+1) - 1j * expval(bases[m], bases[n], j+1, j)
            P[m][n], Q[m][n]  = sumP, sumQ
            P[n][m] = np.conjugate(sumP)
            Q[n][m] = np.conjugate(sumQ)
    return [P, Q]



def track_indices(effevals, effevecs):
    sorted_evals = sorted(effevals)[:N:]
    track = []
    for ev in sorted_evals:
        ind = effevals.index(ev)
        track.append(ind)
    return track




def main(t):
    effham = get_effham(t)                            # construct effham at time t
    eff_state = get_eff_state(effham, track)                 # construct effham state from effham
    eff_corrs = get_eff_corrs(eff_state)              # effham density matrix

    exact_corrs = read_csv(t)                         # exact density matrix

    '''
    TRACE DISTANCES
    '''

    trace_dist_file = open(setup + "/N" + str(N) + "/Tdists", 'a')
    trace_dist_file.write(str(t) + " " + str(trace_dist(eff_corrs, exact_corrs)) + "\n")
    trace_dist_file.close()

    '''
    DENSITIES
    '''

    dens_file = open(setup + "/N" + str(N) + "/dens", 'a')
    dens_list = []
    for i in range(L):
        dens_list += [str(eff_corrs[i][i]), str(exact_corrs[i][i]), str(t), "\n"]
    dens_file.write(" ".join(dens_list))
    dens_file.close()

    '''
    MIDPOINT CORRELATIONS
    '''

    corrs1d_file = open(setup + "/N" + str(N) + "/corrs", 'a')
    midcorrs_list = []
    for i in range(L/2):
        midcorrs_list += [str(eff_corrs[0][i]), str(exact_corrs[0][i]), str(t), "\n"]
    corrs1d_file.write(" ".join(midcorrs_list))
    corrs1d_file.close()

    '''
    Print results
    '''

    s1 = "Effective correlation matrix at time t  = " + str(t) + ":" + "\n" + str(eff_corrs) + "\n"*2
    # s1 = "Exact correlation matrix at time t = " + str(t) + ":" + "\n" + str(exact_corrs) + "\n"*2
    # sd = ""
    # for m in range(L):
        # sd += str(eff_corrs[m][m]) + " " + str(exact_corrs[m][m]) + "\n"
    # s1 += "\n"*2
    # s1 += "Diagonals (effham, exact) at time t = " + str(t) + ":" + "\n" + sd
    # s1 += "\n" + "="*75 + "\n"
    print s1

'''
Initial conditions
'''
start = time.time()

N = 10                                                          # Number of particles
L = 2*N
setup = "Domwall"

num_times = 5
times_scale = 1.0
times = [i*times_scale for i in range(num_times)]

init_sites = get_init_sites(setup)                              # list of sites of initial positions
bases = get_bases()

effham_consts = get_effham_consts()
P = effham_consts[0]
Q = effham_consts[1]

effevals, effevecs = np.linalg.eigh(P)                          # LcN energy eigenvalues, eigenstates
effevals = list(effevals)
effevecs = np.transpose(effevecs)                               # eigenvectors are the columns effevecs[:,i]
track = track_indices(effevals, effevecs)

# print "\nBases:" + "\n" + str(bases) + "\n"
# print "Initial Effective Hamiltonian (P):" + "\n" + str(P) + "\n"
# print "P Energy Eigenvalues: \n" + str(effevals) + "\n \n" + "P Energy Eigenvectors: \n" + str(effevecs) + "\n"
# print "Current Operator (Q):" + "\n" + str(Q) + "\n"
# print "Tracking Eigenstate indices: " + str(track) + "\n"
# print "Starting main... \n" + "="*75 + "\n" + "="*75 + "\n"


# Clear output files
open(setup + "/N" + str(N) + "/Tdists", 'w').close()
open(setup + "/N" + str(N) + "/dens", 'w').close()
open(setup + "/N" + str(N) + "/corrs", 'w').close()

num = 4                                                           # number of CPU's
p = mp.Pool(num)
p.map(main, times)                                                # parallelize main(t) for t arguments

# for t in times:
#     main(t)

end = time.time()
print "OLD: ", end - start
