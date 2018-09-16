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
import operator as op

'''
Infinite temperature
Center half-filled domain wall (L = 4N)
'''


def ncr(n, r):
    '''
    Choose function
    '''
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

def get_bases():
    '''
    Construct computational basis states
    '''
    bases = []
    for i in range(L):
        row = '0'*i + '1' + '0'*(L - i - 1)
        bases.append(row)
    return bases


def get_middle_bases(iterable, window):
    '''
    Construct computational basis states
    '''
    bases = itertools.permutations(iterable, len(window))
    bases = list(set(bases))
    bases = ['0'*N + ''.join(i) + '0'*N for i in bases]
    bases = sorted(bases)[::-1]
    return bases


def get_initial_state(middle_bases):
    psi0_list = []
    for state in middle_bases:
        psi0 = [0]*L
        for j in range(len(state)):
            if state[j] == '1':
                psi0[j] = 1.0
        psi0 = [1.0/NcN2*i for i in psi0]
        psi0_list.append(psi0)
    return psi0_list


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


def diagonalize(ham):
    '''
    Return energy evals and estates of given matrix
    '''
    evals, evecs = np.linalg.eigh(ham)         # L energy values, L eigenstates
    evals = list(evals)
    evecs = np.transpose(evecs)
    return [evals, evecs]


def get_exact_densmat(t, mixed_psi0):
    '''
    Exact density matrix at time t
    '''
    mat = np.zeros((L,L), dtype = complex)
    for m in range(L):
        for n in range(m,L):
            total = 0.0
            for psi0 in mixed_psi0:
                init_sites = []
                for j in range(len(psi0)):
                    if psi0[j] != 0.0:
                        init_sites.append(j)
                for xj in init_sites:
                    total +=  1j ** (n - m) * special.jv(m-xj, 2*t) * special.jv(n-xj, 2*t)
            total *= 1.0 / NcN2
            mat[m][n] = total
            mat[n][m] = np.conjugate(total)
    return mat


def get_effham(t):
    '''
    Construct effective Hamiltonian at time t
    '''
    effham = P + t*Q
    return effham


def get_eff_state(effham):
    '''
    State of Effective Hamiltonian
    Returns list of N single particle eigenstates
    '''
    eigens = diagonalize(effham)
    effevals, effevecs = eigens[0], eigens[1]

    mixed_eff_state = []                                # list of list of energies corresponding to initial positions
    for state0 in mixed_psi0:                        # sum over each contributing state in mixed state
        eff_state = []
        for i in range(len(state0)):                #
            if state0[i] != 0.0:
                # weighted_estate = [state0[i]*val for val in list(effevecs[i])]
                # eff_state.append(weighted_estate)
                eff_state.append(list(effevecs[i]))
        mixed_eff_state.append(eff_state)
    return mixed_eff_state


# def get_eff_densmat(effham, eff_state):
#     '''
#     Density matrix via effective Hamiltonian
#     '''
#     eff_densmat = np.zeros((L,L), dtype = complex)
#     for i in range(L):
#         for j in range(i,L):
#             total = 0.0
#             for p in range(NcN2):
#                 middle_basis = middle_bases[p]
#                 for q in range(N):                              # sum over N corresponding single particle eigenstates
#                     phi0q = eff_state[q]                         # single particle eigenstate
#                     total += weights[q]*np.conjugate(phi0q[i])*phi0q[j]    # multiply CC of qth single particle eigenstate at site m by value at site n
#                 eff_densmat[i][j] = total
#                 eff_densmat[j][i] = np.conjugate(total)        # antisymmetric matrix
#     return eff_densmat


def get_eff_dens1d(effham, mixed_eff_state):
    """ Density matrix diagonals """
    eff_dens = np.zeros(L, dtype = complex)
    for m in range(L):
        total = 0.0
        for eff_state in mixed_eff_state:
            for p in range(len(eff_state)):                          # sum over pure states (tracking eigenstates)
                # phip = np.inner(bases1[m], eff_state[p])
                phip = eff_state[p][m]
                total += np.conjugate(phip) * phip
                # total += np.conjugate(phipq[m])*phip[m]
        eff_dens[m] = 1.0 / NcN2 * total
    # eff_dens *= 1.0/NcN2
    return eff_dens

def get_eff_corrs1d(effham, mixed_eff_state):
    """ Density matrix top row """
    eff_corrs = np.zeros(L, dtype = complex)
    for m in range(L):
        total = 0.0
        for eff_state in mixed_eff_state:
            for p in range(len(eff_state)):                          # sum over pure states (tracking eigenstates)
                # phip = np.inner(bases1[m], eff_state[p])
                phip = eff_state[p]
                total += np.conjugate(phip[0]) * phip[m]
                # total += np.conjugate(phipq[m])*phip[m]
        eff_corrs[m] = 1.0 / NcN2 * total
    # eff_dens *= 1.0/NcN2
    return eff_corrs



def get_effham_consts(bases1):
    '''
    Compute constants in effective Hamiltonian
    '''
    P = np.zeros((L,L), dtype = complex)
    Q = np.zeros((L,L), dtype = complex)
    for m in range(L):
        for n in range(m,L):
            sumP = (L - N) * expval(bases1[m], bases1[n], L-1, L-1)
            sumQ = 0.0
            for j in range(L-1):
                sumP += (j - N + 1) * expval(bases1[m], bases1[n], j, j)
                sumQ += 1j * expval(bases1[m], bases1[n], j, j+1) - 1j * expval(bases1[m], bases1[n], j+1, j)
            P[m][n], Q[m][n]  = sumP, sumQ
            P[n][m] = np.conjugate(sumP)
            Q[n][m] = np.conjugate(sumQ)
    return [P, Q]



def main(t):
    effham = get_effham(t)                                           # Construct effham at time t
    mixed_eff_state = get_eff_state(effham)                          # Construct effham state from effham
    # eff_densmat = get_eff_densmat(effham, mixed_eff_state)          # Effective density matrix
    eff_dens1d = get_eff_dens1d(effham, mixed_eff_state)
    eff_corrs1d = get_eff_corrs1d(effham, mixed_eff_state)

    exact_densmat = get_exact_densmat(t, mixed_psi0)                 # Exact density matrix

    '''
    DENSITIES
    '''

    dens_file = open(setup + "/N" + str(N) + "/dens", 'a')
    dens_list = []
    for i in range(L):
        dens_list += [str(eff_dens1d[i]), str(exact_densmat[i][i]), str(t), "\n"]
    dens_file.write(" ".join(dens_list))
    dens_file.close()

    '''
    MIDPOINT CORRELATIONS
    '''

    midcorrs_file = open(setup + "/N" + str(N) + "/corrs", 'a')
    midcorrs_list = []
    for i in range(L):
        midcorrs_list += [str(eff_corrs1d[i]), str(exact_densmat[0][i]), str(t), "\n"]
    midcorrs_file.write(" ".join(midcorrs_list))
    midcorrs_file.close()

    '''
    Print results
    '''

    p = "t = " + str(t) + "\n"*2
    p += "Eff.Ham. at time t = " + str(t) + ": \n" + str(effham) + "\n"*2
    p +=  "(Mixed) Effective State at time t = " + str(t) + ":\n"
    peff_state = ""
    for s in mixed_eff_state:
        peff_state += str(s) + "\n"
    p += peff_state
    p += "\nExact correlation matrix at time t = " + str(t) + ": \n" + str(exact_densmat) + "\n"
    # p += "\nEffective correlation matrix at time t = " + str(t) + ": \n" + str(eff_densmat) + "\n"*2
    dens_string = ""
    for m in range(L):
        dens_string += "%0s %25s \n" % (str(eff_dens1d[m]), str(exact_densmat[m][m]))
        # dens_string += str(eff_dens1d[m]) + " " + str(exact_densmat[m][m]) + "\n"
    p += "\nDiagonals (effham, exact) at time t = " + str(t) + ":" + "\n" + dens_string
    corrs_string = ""
    for m in range(L):
        corrs_string += "%0s %25s \n" % (str(eff_corrs1d[m]), str(exact_densmat[0][m]))
        # dens_string += str(eff_dens1d[m]) + " " + str(exact_densmat[m][m]) + "\n"
    p += "\nMidpoint Correlations (effham, exact) at time t = " + str(t) + ":" + "\n" + corrs_string
    p += "="*75 + "\n" + "="*75 + "\n"
    print p

'''
Initial conditions
'''

N = 4
L = 4*N
setup = "Domwall"

bases1 = get_bases()
iterable = '1'*N + '0'*(L-N)

# init_sites = [j for j in range(L/4, L/4+2)]                              # list of sites of initial positions
window = [j for j in range(L/4, 3*L/4)]

iterable = '0'*N + '1'*N
middle_bases = get_middle_bases(iterable, window)
NcN2 = len(middle_bases)
# weights = [1.0/NcN2]*len(middle_bases)

mixed_psi0 = get_initial_state(middle_bases)

eff_constants = get_effham_consts(bases1)
P = eff_constants[0]
Q = eff_constants[1]

Peigens = diagonalize(P)
Pevals, Pestates = Peigens[0], Peigens[1]

# track = trackIndices(Pevals, Pestates)

# Clear output files
open(setup + "/N" + str(N) + "/dens", 'w').close()
open(setup + "/N" + str(N) + "/corrs", 'w').close()




basis_str = "\nBases: \n"
for base in bases1:
    basis_str += str(base) + "\n"

midbasis_str = "\nMiddle Bases:\n"
for midbase in middle_bases:
    midbasis_str += str(midbase) + "\n"

init_state_str = "\nInitial State:\n"
for state in mixed_psi0:
    init_state_str += str(state) + "\n"

# print basis_str
print midbasis_str
print init_state_str
# print "\nInitial Mixed State psi0: \n" + str(psi0) + "\n"
print "\nTracking Eigenstate indices: " + str(window) + "\n"
print "\nInitial Emergent Hamiltonian (P):" + "\n" + str(P) + "\n"
print "\nP Energy Eigenvalues: \n" + str(Pevals) + "\n \n" + "P Energy Eigenstates: \n" + str(Pestates) + "\n"
print "\nCurrent Operator (Q):" + "\n" + str(Q) + "\n"
print "\nStarting main... \n" + "="*75 + "\n" + "="*75 + "\n"


num_times = 5                                                # number of times
times = [i*0.1*N for i in range(num_times)]                    # get data for these times

# for t in timeArgs:
#     main(t)


num_cpu = 4                                                           # number of CPU's
p = mp.Pool(num_cpu)
p.map(main, times)                                                # parallelize main(t) for t arguments
