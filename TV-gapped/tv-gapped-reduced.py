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



def basisArr(iterable):
    '''
    Construct computational basis states
    '''
    bases = itertools.permutations(iterable, L)
    bases = list(set(bases))
    bases = [''.join(i) for i in bases]
    bases = sorted(bases)[::-1]
    return bases


def expVal(lbasis, rbasis, cdag, c):
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



def exact_dens(t):
    '''
    Exact density matrix at time t
    '''
    densityMat = np.zeros(L, dtype = np.complex128)
    for d in range(L):
        total = 0.0
        for a in range(LcN):                                # double sum over all LcN eigenstates
            E_a = energy_evals[a]
            phi_a = energy_estates[a]
            for b in range(LcN):
                E_b = energy_evals[b]
                phi_b = energy_estates[b]
                psiOverlap = np.inner(psi0, np.conjugate(phi_a))*np.inner(phi_b, np.conjugate(psi0))
                expcc = 0.0
                for m in range(LcN):
                    for n in range(LcN):
                        # num_between = 0.0
                        # for k in range(i+1,j):
                        #     if bases[n][k] == '1':
                        #         num_between += 1
                        expcc += np.conjugate(phi_a[m])*phi_b[n]*expVal(bases[m], bases[n], d, d)
                total += cmath.exp(1j*t*(E_a - E_b))*psiOverlap*expcc
        densityMat[d] = total
    return densityMat





def diagHam(hamilt):
    '''
    Diagonalize Hamiltonian
    '''
    energy_evals, energy_estates = np.linalg.eigh(hamilt)            # LcN energy eigenvalues, eigenstates
    energy_evals = list(energy_evals)
    energy_estates = np.transpose(energy_estates)           # eigenvectors are the columns effevecs[:,i]
    return [energy_evals, energy_estates]


def hj(j, m, n, J, V):
    current_sum = expVal(bases[m], bases[n], j+1, j) + expVal(bases[m], bases[n], j, j+1)
    # pot_sum = (-0.5)*(expVal(bases[m], bases[n], j, j) + expVal(bases[m], bases[n], j+1, j+1))
    if m==n:
        pot_sum = (expVal(bases[m], bases[n], j+1, j+1)-0.5)*(expVal(bases[m], bases[n], j, j)-0.5)
    else:
        pot_sum = 0
    return -J*current_sum + V*pot_sum



def ham_func(J,V):
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


def P_func(J,V):
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
    P += newMat
    return P


def Q_func(J,V):
    '''
    Construct Q
    '''
    Q = np.zeros((LcN, LcN), dtype = np.complex128)
    for m in range(LcN):
        for n in range(m,LcN):
            current_sum, pot_sum = 0.0, 0.0
            for j in range(L-2):
                current_sum += (1-2*int(bases[n][j+1]))*(expVal(bases[m], bases[n], j+2, j) - expVal(bases[m], bases[n], j, j+2))
                pot_sum += 0.5*(expVal(bases[m], bases[n], j, j+1) - expVal(bases[m], bases[n], j+1, j))
                pot_sum += 0.5*(expVal(bases[m], bases[n], j+1, j+2) - expVal(bases[m], bases[n], j+2, j+1))
                pot_sum += expVal(bases[n], bases[n], j+2, j+2)*(expVal(bases[m], bases[n], j+1, j) - expVal(bases[m], bases[n], j, j+1))
                pot_sum += expVal(bases[n], bases[n], j, j)*(expVal(bases[m], bases[n], j+2, j+1) - expVal(bases[m], bases[n], j+1, j+2))
            Q[m][n] = 1j*(J**2*current_sum - V*pot_sum)
            Q[n][m] = np.conjugate(Q[m][n])
    return Q


def newMatrix(offset):
    mat = np.zeros((LcN, LcN), dtype = np.complex128)
    for m in range(LcN):
        for n in range(LcN):
            mat[m][n] = offset*expVal(bases[m], bases[n], 0, 0)
    return mat


def effHam_func(t):
    return P + t*Q


def effGS_func(effHamilt, t):
    '''
    Ground state of effective Hamiltonian
    In energy basis
    '''
    effevals, effevecs = np.linalg.eigh(effHamilt)                 # LcN energy eigenvalues, eigenstates
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



def eff_dens(effGState):
    '''
    Density matrix via effective Hamiltonian
    '''
    effDenMat = np.zeros(L, dtype = np.complex128)
    for d in range(L):
        eff_dd = 0.0
        for m in range(LcN):                              # sum over LcN fock bases
            for n in range(LcN):
                add_term = effGState[m]*np.conjugate(effGState[n])*expVal(bases[m], bases[n], d, d)
                eff_dd += add_term
        effDenMat[d] = eff_dd
    return effDenMat

def eff_corrs(effGState):
    '''
    corrs via effective Hamiltonian
    '''
    effDenMat = np.zeros(L, dtype = np.complex128)
    left_site = L/2 - 1
    for j in range(left_site+1, L):
        eff_0j = 0.0
        for m in range(LcN):                              # sum over LcN fock bases
            for n in range(LcN):
                num_between = 0
                for k in range(left_site+1, j):
                    if bases[n][k] == '1':
                        num_between += 1
                # if expVal(bases[m],bases[n],left_site,j) == 1.0:
                #     print j
                #     print num_between
                #     print bases[m]
                #     print bases[n]
                #     print expVal(bases[m],bases[n],left_site,j)
                add_term = (-1)**num_between*effGState[m]*np.conjugate(effGState[n])*expVal(bases[m], bases[n], left_site, j)
                eff_0j += add_term
        effDenMat[j] = eff_0j
    print "eff before:" + str(effDenMat)
    effDenMat = effDenMat[L/2:L]
    print "eff:" + str(effDenMat)
    return effDenMat

def exact_corrs(t):
    '''
    Exact correlations at time t
    '''
    densityMat = np.zeros(L, dtype = np.complex128)
    left_site = L/2 - 1
    for j in range(left_site+1,L):
        total = 0.0
        for a in range(LcN):                                # double sum over all LcN eigenstates
            E_a = energy_evals[a]
            phi_a = energy_estates[a]
            for b in range(LcN):
                E_b = energy_evals[b]
                phi_b = energy_estates[b]
                psiOverlap = np.inner(psi0, np.conjugate(phi_a))*np.inner(phi_b, np.conjugate(psi0))
                expcc = 0.0
                for m in range(LcN):
                    for n in range(LcN):
                        num_between = 0
                        for k in range(left_site+1,j):
                            if bases[n][k] == '1':
                                num_between += 1
                        expcc += (-1)**num_between*phi_a[n]*np.conjugate(phi_b[m])*expVal(bases[m], bases[n], left_site, j)
                total += cmath.exp(-1j*t*(E_a - E_b))*psiOverlap*expcc
        densityMat[j] = total
    print "exact before:" + str(densityMat)
    densityMat = densityMat[L/2:L]
    print "exact:" + str(densityMat)
    return densityMat

def middleBases(bases, left_index, right_index):
    '''
    Return list of bases with unoccupied endpoints
    '''
    middle_bases = []
    for base in bases:
        if base[left_index] == '0' and base[right_index] == '0':
            middle_bases.append(base)
    return middle_bases



def main(t):
    effHam = effHam_func(t)
    effGS = effGS_func(effHam, t)
    # effDensMat = effDensMat_func(effGS)
    eff_dens_ = eff_dens(effGS)
    eff_corrs_ = eff_corrs(effGS)

    # densityMat = densityMatFunc(t)
    exact_dens_ = exact_dens(t)
    exact_corrs_ = exact_corrs(t)


    p = "t = " + str(t) + "\n"*2
    # p += "Eff.Ham. at time t = " + str(t) + ": \n" + str(effHam) + "\n"*2
    # p +=  "Eff. GS at time t = " + str(t) + ": \n" + str(effGS) + "\n"*2
    p += "Exact correlation matrix at time t = " + str(t) + ": \n" + str(exact_corrs_) + "\n"*2
    p += "Effective correlation matrix at time t = " + str(t) + ": \n" + str(eff_corrs_) + "\n"*2
    p += "="*75 + "\n" + "="*75 + "\n"
    print p

    '''
    Densities
    '''

    dens_file = open("N" + str(N) + "/dir-dens-gapped/dens-t" + str(t), 'w')
    for i in range(L):
        # dens_file.write(str(densityMat[i][i]) + " " + str(effDensMat[i][i]) + " " + str(t) + "\n")
        dens_file.write(str(exact_dens_[i]) + " " + str(eff_dens_[i]) + " " + str(t) + "\n")
    dens_file.close()

    '''
    Midpoint Correlations
    '''

    midCorrs_file = open("N" + str(N) + "/dir-corrs-gapped/corrs-t" + str(t), 'w')
    for i in range(L/2):
        # midCorrs_file.write(str(densityMat[0][i]) + " " + str(effDensMat[0][i]) + " " + str(t) + "\n")
        midCorrs_file.write(str(exact_corrs_[i]) + " " + str(eff_corrs_[i]) + " " + str(t) + "\n")
    midCorrs_file.close()


'''
Initial conditions
'''


N = 3
J = 1.0
V = 1.2
L = 2*N


iterable = '1'*N + '0'*(L-N)
bases = basisArr(iterable)                                            # array of basis states
LcN = len(bases)

offset = -10**(-3)
newMat = newMatrix(offset)

P = P_func(J,V)
Q = Q_func(J,V)


effevals, effevecs = np.linalg.eigh(P)                 # LcN energy eigenvalues, eigenstates
effevals = list(np.round(effevals,3))
print effevals
effevecs = np.transpose(effevecs)                     # eigenvectors are the columns effevecs[:,i]
track_index = effevals.index(np.float(offset))
# track_index = effevals.index(np.float128(0.0))

psi0 = effevecs[track_index]

ham = ham_func(J,V)                                                   # construct Hamiltonian
eigen_list = diagHam(ham)                                             # diagonalize Hamiltonian
energy_evals, energy_estates = eigen_list[0], eigen_list[1]


# print "\nBases:" + "\n" + str(bases) + "\n"
print "Tracking eigenstate index: " + str(track_index) + "\n"
# print "P Energy Eigenvalues: \n" + str(effevals) + "\n \n" + "P Energy Eigenvectors: \n" + str(effevecs) + "\n"
print "Initial state in Fock state representation (psi0):" + "\n" + str(psi0) + "\n"
# print "Initial Hamiltonian (P):" + "\n" + str(P) + "\n"
# print "Current Operator (Q):" + "\n" + str(Q) + "\n"
# print "Hamiltonian (H):" + "\n" + str(ham) + "\n"
print "Exact energy eigenvalues:" + "\n" + str(energy_evals) + "\n"
# print "Exact energy eigenstates:" + "\n" + str(energy_estates) + "\n"
print "Starting main... \n" + "="*75 + "\n" + "="*75 + "\n"



num_timeSteps = 10                                                     # number of times
timeArgs = [i*0.02*N for i in range(num_timeSteps)]                    # get data for these times

# for t in timeArgs:
#     main(t)

num_cpu = 4                 # number of CPU's
p = mp.Pool(num_cpu)
p.map(main, timeArgs)      # parallelize main(t) for t arguments
