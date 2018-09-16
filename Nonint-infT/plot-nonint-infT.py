import numpy as np
import matplotlib.pyplot as plt
from pylab import *


def corrs_plot(eff_midcorrs, exact_midcorrs, t):
    '''
    CORRELATION PLOT
    '''

    fig = plt.figure()
    fig.suptitle(r"Time Evolved $\langle c^{\dag}_{0} c_{n} \rangle$ Single particle correlations", fontsize = 16)
    plt.title("L = " + str(L) + ", N = " + str(N), fontsize = 10)

    ax = subplot(1,1,1)
    for time in range(1,num_times):
        # xdiff = [i for i in range(len(eff_midcorrs[0:(time+1)*(L/2):]))]
        effplot = ax.loglog(xdiff, eff_midcorrs[time*L:(time+1)*L:], 'o', label = 't = ' + str(t[time*L]))
        c = effplot[0].get_color()
        ax.loglog(xdiff, exact_midcorrs[time*L:(time+1)*L:], '--', color = c, label = "Ex")

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))          # sort both labels and handles by labels
    lines_legend = ax.legend([handles[0], handles[num_times-1]], ["Exact", "Eff."], loc = 4, numpoints = 1)
    plt.gca().add_artist(lines_legend)

    handles = handles[num_times-1:]
    labels = labels[num_times-1:]
    times_legend = ax.legend(handles, labels, loc = 'upper right', numpoints = 1, fontsize = 8)        # ax.legend(handles, labels)
    plt.gca().add_artist(times_legend)

    plt.plot(xdiff, [1/(np.pi*float(x)) for x in xdiff], 'r-')
    # plt.legend(loc = 'upper right', numpoints = 1)
    plt.xlabel("Site (n)", fontsize = 18)
    plt.ylabel("Correlation", fontsize = 18)
    plt.xlim([min(xdiff), max(xdiff)])
    plt.ylim([10e-12, 1])
    fig.savefig(setup + "/N" + str(N) + "/plot-corrs.pdf")
    plt.show()


def dens_plot(eff_dens, exact_dens, t):
    '''
    DENSITY PLOT
    '''
    fig2 = plt.figure()
    fig2.suptitle(r"$\langle c^{\dag}_{n} c_{n} \rangle$ Density vs. Site", fontsize = 16)
    plt.title("L = " + str(L) + ", N = " + str(N), fontsize = 10)

    ax = subplot(1,1,1)
    for i in range(num_times):
        effplot = ax.plot(xlst, eff_dens[i*L:(i+1)*L:], 'o', label = "t = " + str(t[i*L]))
        c = effplot[0].get_color()
        ax.plot(xlst, exact_dens[i*L:(i+1)*L:], '--', color = c, label = "Ex")

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))          # sort both labels and handles by labels
    lines_legend = ax.legend([handles[0], handles[num_times]], ["Exact", "Eff."], loc = 8, numpoints = 1)
    plt.gca().add_artist(lines_legend)

    handles = handles[num_times:]
    labels = labels[num_times:]
    times_legend = ax.legend(handles, labels, loc = 'upper right', numpoints = 1, fontsize = 8)        # ax.legend(handles, labels)
    plt.gca().add_artist(times_legend)

    # handles, labels = ax.get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))                    # sort both labels and handles by labels
    # ax.legend(handles, labels, loc = 'upper right', numpoints = 1, fontsize = 8)                # ax.legend(handles, labels)
    plt.xlabel("Site (n)", fontsize = 18)
    plt.ylabel("Density", fontsize = 18)
    plt.ylim([0, max(exact_dens) + 0.05])
    fig2.savefig(setup + "/N" + str(N) + "/plot-dens.pdf")
    plt.show()






'''
Initial conditions
'''

N = 5
L = 4*N
setup = "Domwall"

xlst = [i for i in range(-L/2,L/2)]         # window
xdiff = [i for i in range(1,L+1)]       # ???

eff_dens, exact_dens, tdens = np.absolute(np.loadtxt(setup + "/N" + str(N) + "/dens", dtype = complex, unpack = True))
eff_midcorrs, exact_midcorrs, tcorrs = np.absolute(np.loadtxt(setup + "/N" + str(N) + "/corrs", dtype = complex, unpack = True))
# tTdists, Tdists = np.loadtxt("N" + str(N) + "/Tdists", unpack = True)

num_times = len(eff_dens)/L

dens_plot(eff_dens, exact_dens, tdens)
corrs_plot(eff_midcorrs, exact_midcorrs, tcorrs)
# tracesPlot(Tdists, tTdists)
