# #!/usr/remote/python-2.7/bin/python
# ^ (uncomment for linux)

import numpy as np
import math as math
import cmath as cmath
import itertools
import random
import time
import sys
import multiprocessing as mp
from scipy import special
import time
from copy import deepcopy



def get_bases(iterable):
    '''
    Construct computational basis states
    '''
    bases = itertools.permutations(iterable, L)
    bases = list(set(bases))
    bases = [''.join(i) for i in bases]
    bases = sorted(bases)[::-1]
    return bases


def expval(lbasis, rbasis, cdag, c):
    check_c = float(rbasis[c])
    if cdag != c:
        check_cdag = 1.0 - float(rbasis[cdag])
        mask = ['0']*L
        mask[cdag], mask[c] = '1', '1'
        mask = ''.join(mask)
        rbasis = bin(int(mask,2)^int(rbasis,2))[2:].zfill(L)
        return check_c*check_cdag*float(lbasis == rbasis)
    else:
        return check_c*float(lbasis == rbasis)


def get_exact_dens_mat(t):
    '''
    Exact density matrix at time t
    '''
    exact_dens_mat = np.zeros((L, L), dtype = complex)
    for i in range(L):
        for j in range(i, L):
            total = 0.0
            for a in range(LcN):                                # double sum over all LcN eigenstates
                E_a = energy_evals[a]
                phi_a = energy_estates[a]
                for b in range(LcN):
                    E_b = energy_evals[b]
                    phi_b = energy_estates[b]
                    psi_overlap = np.inner(psi0, np.conjugate(phi_a))*np.inner(phi_b, np.conjugate(psi0))
                    expcc = 0.0
                    for m in range(LcN):
                        for n in range(LcN):
                            num_between = 0.0
                            for k in range(i+1,j):
                                if bases[n][k] == '1':
                                    num_between += 1
                            expcc += (-1)**num_between*np.conjugate(phi_a[m])*phi_b[n]*expval(bases[m], bases[n], i, j)
                    total += cmath.exp(1j*t*(E_a - E_b))*psi_overlap*expcc
            exact_dens_mat[i][j] = total
            exact_dens_mat[j][i] = np.conjugate(total)
    return exact_dens_mat



def diag_ham(ham):
    '''
    Diagonalize Hamiltonian
    '''
    energy_evals, energy_estates = np.linalg.eigh(ham)            # LcN energy eigenvalues, eigenstates
    energy_evals = list(energy_evals)
    energy_estates = np.transpose(energy_estates)           # eigenvectors are the columns effevecs[:,i]
    return [energy_evals, energy_estates]


def hj(j, m, n, J, V):
    current_sum = expval(bases[m], bases[n], j+1, j) + expval(bases[m], bases[n], j, j+1)
    pot_sum = (-0.5)*(expval(bases[m], bases[n], j, j) + expval(bases[m], bases[n], j+1, j+1))
    pot_sum += expval(bases[n], bases[n], j+1, j+1)*expval(bases[m], bases[n], j, j)
    if m == n:
        pot_sum += 0.25
    return -J*current_sum + V*pot_sum



def get_ham(J,V):
    '''
    Construct time-evolved Hamiltonian
    '''
    ham = np.zeros((LcN, LcN), dtype = np.complex128)
    for m in range(LcN):
        for n in range(m,LcN):
            ham_mn = 0.0
            for j in range(L-1):
                ham_mn += hj(j, m, n, J, V)
            ham[m][n] = ham_mn
            ham[n][m] = ham_mn
    return ham


def get_P(J,V):
    '''
    Construct initial Hamiltonian P
    '''
    P = np.zeros((LcN, LcN), dtype = np.complex128)
    for m in range(LcN):
        for n in range(m,LcN):
            P_mn = 0.0
            for j in range(L-1):
                P_mn += (j-N+1)*hj(j, m, n, J, V)
            P[m][n] = P_mn
            P[n][m] = P_mn
    P += new_matrix
    return P


def get_Q(J,V):
    '''
    Construct Q
    '''
    Q = np.zeros((LcN, LcN), dtype = np.complex128)
    for m in range(LcN):
        for n in range(m,LcN):
            current_sum, pot_sum = 0.0, 0.0
            for j in range(L-2):
                current_sum += (1-2*int(bases[n][j+1]))*(expval(bases[m], bases[n], j+2, j) - expval(bases[m], bases[n], j, j+2))
                pot_sum += 0.5*(expval(bases[m], bases[n], j, j+1) - expval(bases[m], bases[n], j+1, j))
                pot_sum += 0.5*(expval(bases[m], bases[n], j+1, j+2) - expval(bases[m], bases[n], j+2, j+1))
                pot_sum += expval(bases[n], bases[n], j+2, j+2)*(expval(bases[m], bases[n], j+1, j) - expval(bases[m], bases[n], j, j+1))
                pot_sum += expval(bases[n], bases[n], j, j)*(expval(bases[m], bases[n], j+2, j+1) - expval(bases[m], bases[n], j+1, j+2))
            Q[m][n] = 1j*(J**2*current_sum - V*pot_sum)
            Q[n][m] = np.conjugate(Q[m][n])
    return Q


def get_new_matrix(offset):
    mat = np.zeros((LcN, LcN), dtype = np.complex128)
    for m in range(LcN):
        for n in range(LcN):
            mat[m][n] = offset*expval(bases[m], bases[n], 0, 0)
    return mat


def get_effham(t):
    return P + t*Q


def get_effGS(effham, t):
    '''
    Ground state of effective Hamiltonian
    In energy basis
    '''
    effevals, effevecs = np.linalg.eigh(effham)                 # LcN energy eigenvalues, eigenstates
    effevals = list(effevals)
    effevecs = np.transpose(effevecs)                     # eigenvectors are the columns effevecs[:,i]

    state = effevecs[track_index]
    # state[-1], state[-2] = complex(0.0), complex(0.0)

    s = "Effective Energy Eigenvalues at time t = " + str(t) + ":" + "\n"
    s += str(effevals) + "\n"*2
    s += "Effective Evecs at time t = " + str(t) + ":" "\n"
    s += str(effevecs) + "\n"*2 + "-"*75 + "\n"*2
    print s

    return state


def get_eff_dens(effGState):
    '''
    Density matrix via effective Hamiltonian
    '''
    eff_densmat = np.zeros((L,L), dtype = np.complex128)
    for i in range(L):
        for j in range(i, L):
            eff_ij = 0.0
            for m in range(LcN):                              # sum over LcN fock bases
                for n in range(LcN):
                    num_between = 0.0
                    for k in range(i+1,j):
                        if bases[n][k] == '1':
                            num_between += 1
                    add_term = (-1)**num_between*effGState[m]*np.conjugate(effGState[n])*expval(bases[m], bases[n], i, j)
                    eff_ij += add_term

                    s =  "i = " + str(i) + " j = " + str(j) + " m = " + str(m) + " n = " + str(n) + "\n"
                    s += "bases[m] = " + str(bases[m]) + " bases[n] = " + str(bases[n]) + " expval = " + str(expval(bases[m], bases[n], i, j)) + "\n"
                    s += "GS[m] = " + str(np.conjugate(effGState[m])) + " GS[n] = " + str(effGState[n]) + "\n"
                    s += "Adding: " + str(add_term) + " \neff_ij = " + str(eff_ij)
                    # if i == j and add_term != complex(0.0, 0.0):
                    #     pass
                    #     print s + "\n"
                # print ""
            eff_densmat[i][j] = eff_ij
            eff_densmat[j][i] = np.conjugate(eff_ij)
    return eff_densmat

# def get_middle_bases(bases, left_index, right_index):
#     '''
#     Return list of bases with unoccupied endpoints
#     '''
#     middle_bases = []
#     for base in bases:
#         if base[left_index] == '0' and base[right_index] == '0':
#             middle_bases.append(base)
#     return middle_bases



def main(t):
    effham = get_effham(t)
    effGS = get_effGS(effham, t)
    eff_dens_mat = get_eff_dens(effGS)

    exact_dens_mat = get_exact_dens_mat(t)


    p = "t = " + str(t) + "\n"*2
    p += "Eff.Ham. at time t = " + str(t) + ": \n" + str(effham) + "\n"*2
    p +=  "Eff. GS at time t = " + str(t) + ": \n" + str(effGS) + "\n"*2
    p += "Exact correlation matrix at time t = " + str(t) + ": \n" + str(exact_dens_mat) + "\n"*2
    p += "Effective correlation matrix at time t = " + str(t) + ": \n" + str(eff_dens_mat) + "\n"*2
    p += "="*75 + "\n" + "="*75 + "\n"
    print p

    '''
    Densities
    '''

    # dens_file = open("N" + str(N) + "/dir-dens/dens-t" + str(t), 'w')
    dens_file = open(setup + "/N" + str(N) + "/dens", 'a')
    for i in range(L):
        dens_file.write(str(exact_dens_mat[i][i]) + " " + str(eff_dens_mat[i][i]) + " " + str(t) + "\n")
    dens_file.close()

    '''
    Midpoint Correlations
    '''

    # midcorrs_file = open("N" + str(N) + "/dir-corrs/corrs-t" + str(t), 'w')
    midcorrs_file = open(setup + "/N" + str(N) + "/corrs", 'a')
    for i in range(L/2):
        midcorrs_file.write(str(exact_dens_mat[0][i]) + " " + str(eff_dens_mat[0][i]) + " " + str(t) + "\n")
    midcorrs_file.close()


'''
Initial conditions
'''


N = 6
J = 1.0
V = 1.5
L = 2*N
setup = "Domwall"

num_times = 10                                                       # Number of times
times = [0.2*t*N for t in range(num_times)]


iterable = '1'*N + '0'*(L-N)
bases = get_bases(iterable)                                           # Array of basis states
LcN = len(bases)

offset = -10**(-3)
new_matrix = get_new_matrix(offset)

P = get_P(J,V)
Q = get_Q(J,V)


effevals, effevecs = np.linalg.eigh(P)                                # LcN energy eigenvalues, eigenstates
effevals = list(effevals)
effevecs = np.transpose(effevecs)                                     # Eigenvectors are the columns effevecs[:,i]
track_index = effevals.index(np.float128(offset))

psi0 = effevecs[track_index]

ham = get_ham(J,V)                                                   # Construct Hamiltonian
eigen_list = diag_ham(ham)                                             # Diagonalize Hamiltonian
energy_evals, energy_estates = eigen_list[0], eigen_list[1]


print "\nBases:" + "\n" + str(bases) + "\n"
print "Tracking eigenstate index: " + str(track_index) + "\n"
print "P Energy Eigenvalues: \n" + str(effevals) + "\n \n" + "P Energy Eigenvectors: \n" + str(effevecs) + "\n"
print "Initial state in Fock state representation (psi0):" + "\n" + str(psi0) + "\n"
print "Initial Hamiltonian (P):" + "\n" + str(P) + "\n"
print "Current Operator (Q):" + "\n" + str(Q) + "\n"
print "Hamiltonian (H):" + "\n" + str(ham) + "\n"
# print "Exact energy eigenvalues:" + "\n" + str(energy_evals) + "\n"
# print "Exact energy eigenstates:" + "\n" + str(energy_estates) + "\n"
print "Starting main... \n" + "="*75 + "\n" + "="*75 + "\n"


# Clear output files
open(setup + "/N" + str(N) + "/dens", 'w').close()
open(setup + "/N" + str(N) + "/corrs", 'w').close()

num_cpu = 4                 # Number of CPU's
p = mp.Pool(num_cpu)
p.map(main, times)          # Parallelize main(t) for t arguments
