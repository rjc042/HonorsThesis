import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.lines as mlines


def corrs_plot(eff_midcorrs, exact_midcorrs, alltimes):
    '''
    CORRELATION PLOT
    '''
    # corr_times = [2*i for i in range(2,6)]                     # plot for these times xN
    corr_times = [1,2,4,6]
    num_corr_times = len(corr_times)

    fig = plt.figure()
    fig.suptitle(r"Time Evolved $\langle c^{\dag}_{0} c_{n} \rangle$ Single particle correlations", fontsize = 16)
    plt.title("L = " + str(L) + ", N = " + str(N), fontsize = 10)

    ax = subplot(1,1,1)
    for time in corr_times:
        effplot = ax.loglog(xdiff, eff_midcorrs[time*N:(time+1)*(L/2):], 'o', markersize = 4.5, label = 't = ' + str(alltimes[time*(L/2)]))
        # effplot = ax.loglog(xdiff, eff_midcorrs[time*N:(time+1)*(L/2):], 'o', markersize = 4.5, label = 't = ' + str(time))
        c = effplot[0].get_color()
        exact_plot = ax.loglog(xdiff, exact_midcorrs[time*N:(time+1)*(L/2):], '--', color = c, label = "Ex")
        # exact_plot = ax.loglog(xdiff, exact_midcorrs[time*N:(time+1)*(L/2):], '--', markersize = 4.5, label = 't = ' + str(alltimes[time*(L/2)]))
        # plt.loglog(xdiff, exact_midcorrs[time*N:(time+1)*(L/2):], '--', label = 't = ' + str(t[time*(L/2)]))

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))          # sort both labels and handles by labels
    lines_legend = ax.legend([handles[0], handles[num_corr_times]], ["Exact", "Eff."], loc = 3, numpoints = 1)
    plt.gca().add_artist(lines_legend)

    handles = handles[num_corr_times:]
    labels = labels[num_corr_times:]
    times_legend = ax.legend(handles, labels, loc = 'upper right', numpoints = 1, fontsize = 8)        # ax.legend(handles, labels)
    plt.gca().add_artist(times_legend)

    ax.plot(xdiff, [1/(np.pi*float(x)) for x in xdiff], 'r-')
    # ax.legend(loc = 'upper right', numpoints = 1)
    plt.xlabel("Site (n)", fontsize = 18)
    # plt.xticks(np.arange(min(xdiff), max(xdiff) + 1, 1.0))
    plt.ylabel("Single Particle Correlation", fontsize = 18)
    plt.ylim(10e-7, 1)
    fig.savefig(setup + "/N" + str(N) + "/plot-corrs.pdf")
    plt.show()


def dens_plot(eff_dens, exact_dens, t):
    '''
    DENSITY PLOT
    '''
    fig2 = plt.figure()
    fig2.suptitle(r"$\langle c^{\dag}_{n} c_{n}\rangle$ Density vs. Site", fontsize = 16)
    plt.title("L = " + str(L) + ", N = " + str(N), fontsize = 10)
    times_list = []
    # plot_lines = []
    num_dens_times = 0

    ax = subplot(1,1,1)
    for i in range(num_times):
        if t[i*L] in goodtimes:
            # effplot = ax.plot(xlst, eff_dens[i*L:(i+1)*L:], 'o', label = "t = " + str(t[i*L]) + " Eff.")
            effplot = ax.plot(xlst, eff_dens[i*L:(i+1)*L:], 'o', label = "t = " + str(t[i*L]))
            c = effplot[0].get_color()
            exactplot = ax.plot(xlst, exact_dens[i*L:(i+1)*L:], '--', color = c, label = "Ex")
            num_dens_times += 1


    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key = lambda t: t[0]))          # sort both labels and handles by labels
    lines_legend = ax.legend([handles[0], handles[num_dens_times]], ["Exact", "Eff."], loc = 3, numpoints = 1)
    plt.gca().add_artist(lines_legend)

    handles = handles[num_dens_times:]
    labels = labels[num_dens_times:]
    times_legend = ax.legend(handles, labels, loc = 'upper right', numpoints = 1, fontsize = 8)        # ax.legend(handles, labels)
    plt.gca().add_artist(times_legend)


    plt.xlabel("Site (n)", fontsize = 18)
    plt.ylabel("Density", fontsize = 18)
    fig2.savefig(setup + "/N" + str(N) + "/plot-dens.pdf")
    plt.show()


def tracesPlot(Tdists, t):
    '''
    TRACE DISTANCE PLOT
    '''

    fig3 = plt.figure()
    fig3.suptitle("Trace Distance over Time", fontsize = 16)
    plt.title("L = " + str(L) + ", N = " + str(N), fontsize = 10)
    plt.loglog(t, Tdist, 'bo', markersize = 10)
    plt.xlabel("Time", fontsize = 18)
    plt.xticks(fontsize = 15)
    plt.xlim([min(t) - 0.5, max(t) + 0.5])
    plt.ylabel("Eff.Ham. - Exact", fontsize = 16)
    plt.yticks(fontsize = 15)
    fig3.savefig(setup + "/N" + str(N) + "/plot-Tdists.pdf")
    plt.show()





'''
Initial conditions
'''

N = 10
L = 2*N
setup = "Domwall"

xlst = [i for i in range(-N, N)]         # window
xdiff = [i for i in range(1, N+1)]       # ???

eff_dens, exact_dens, tdens = np.absolute(np.loadtxt(setup + "/N" + str(N) + "/dens", dtype = complex, unpack = True))
eff_midcorrs, exact_midcorrs, tcorrs = np.absolute(np.loadtxt(setup + "/N" + str(N) + "/corrs", dtype = complex, unpack = True))
tTdists, Tdist = np.loadtxt(setup + "/N" + str(N) + "/Tdists", unpack = True)

num_times = len(eff_dens)/L
num_goodtimes = 7
goodtimes = [i*100 for i in range(num_goodtimes)]

dens_plot(eff_dens, exact_dens, tdens)
corrs_plot(eff_midcorrs, exact_midcorrs, tcorrs)
# tracesPlot(Tdist, tTdists)
