import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import operator as op


def ncr(n, r):
    '''
    Choose function
    '''
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom


def plotDens(effDens, dens, t):
    '''
    DENSITY PLOT
    '''

    fig2 = plt.figure()

    ax = subplot(1,1,1)
    for i in range(num_times):
        effPlot = ax.plot(xlst, effDens[i*L:(i+1)*L:], 'o', markersize = 4.5, label = "t = " + str(t[i*L]))
        c = effPlot[0].get_color()
        exact_plot = ax.plot(xlst, dens[i*L:(i+1)*L:], '--', color = c, label = "Ex")

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))          # sort both labels and handles by labels
    lines_legend = ax.legend([handles[0], handles[num_times]], ["Exact", "Eff."], loc = 3, numpoints = 1)
    plt.gca().add_artist(lines_legend)

    handles = handles[num_times:]
    labels = labels[num_times:]
    times_legend = ax.legend(handles, labels, loc = 'upper right', numpoints = 1, fontsize = 12)        # ax.legend(handles, labels)
    plt.gca().add_artist(times_legend)

    fig2.suptitle(r"$\langle c^{\dag}_{n} c_{n} \rangle$ Density vs. Site (Gapped)", fontsize = 16)
    plt.title("L = " + str(L) + ", N = " + str(N), fontsize = 10)
    plt.xlabel("Density Matrix Diagonal (n)", fontsize = 18)
    plt.ylabel("Density", fontsize = 18)
    plt.ylim([0,max(dens)+0.1])
    fig2.savefig(setup + "/N" + str(N) + "/plot-dens.pdf")
    plt.show()


def plotCorrs(midCorrs, effMidCorrs, t):
    '''
    CORRELATION PLOT
    '''
    fig = plt.figure()

    ax = subplot(1,1,1)
    for time in range(1,num_times):
        effPlot = ax.loglog(xdiff, effMidCorrs[time*L/2:(time+1)*L/2:], 'o', label = 't = ' + str(t[time*L/2]))
        c = effPlot[0].get_color()
        exact_plot = ax.loglog(xdiff, midCorrs[time*L/2:(time + 1)*L/2:], '--', color = c, label = str("Ex"))
    ax.plot(xdiff, [1/(np.pi*float(x)) for x in xdiff], 'r-')

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))          # sort both labels and handles by labels
    lines_legend = ax.legend([handles[0], handles[num_times]], ["Exact", "Eff."], loc = 4, numpoints = 1)
    plt.gca().add_artist(lines_legend)

    handles = handles[num_times:]
    labels = labels[num_times:]
    times_legend = ax.legend(handles, labels, loc = 'lower left', numpoints = 1, fontsize = 12)        # ax.legend(handles, labels)
    plt.gca().add_artist(times_legend)

    fig.suptitle(r"Time Evolved $\langle c^{\dag}_{0} c_{n} \rangle$ Single particle correlations", fontsize = 16)
    plt.title("L = " + str(L) + ", N = " + str(N), fontsize = 10)
    # plt.legend(loc = 'upper right', numpoints = 1)
    plt.xlabel("Site (n)", fontsize = 18)
    plt.ylabel("Correlation", fontsize = 18)
    plt.xlim(min(xdiff),max(xdiff) + 1)
    # plt.ylim([0,max(midCorrs)+1])
    fig.savefig(setup + "/N" + str(N) + "/plot-corrs.pdf")
    plt.show()


'''
Initial conditions
'''

N = 5
V = 4.0
J = 1.0
setup = "Domwall"

L = 2*N
LcN = ncr(L,N)

xlst = [i for i in range(-N,N)]
xdiff = [i for i in range(1, int(L/2) + 1)]       # ???

exactDens, effDens, tDens = np.absolute(np.loadtxt(setup + "/N" + str(N) + "/dens", dtype = complex, unpack = True))
exactMidCorrs, effMidCorrs, tCorrs = np.absolute(np.loadtxt(setup + "/N" + str(N) + "/corrs", dtype = complex, unpack = True))

num_times = len(effDens)/L

plotDens(exactDens, effDens, tDens)
plotCorrs(exactMidCorrs, effMidCorrs, tCorrs)
