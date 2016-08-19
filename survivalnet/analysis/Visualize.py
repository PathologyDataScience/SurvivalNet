from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from textwrap import wrap

from . import RiskCohort
from . import RiskCluster

# define colors for positive risk (red) and negative risk (blue)
REDFACE = '#DE2D26'
BLUEFACE = '#3182BD'
REDEDGE = '#DE2D26'
BLUEEDGE = '#3182BD'
MEDIAN = '#000000'
WHISKER = '#AAAAAA'
POINTS = '#000000'
GRID = '#BBBBBB'

# layout constants general
WRAP = 20  # number of characters for text wrapping
SPACING = 0.2  # margin

# layout constants for boxplot
BOX_FH = 4  # boxplot figure width
BOX_FW = 8  # boxplot figure height
JITTER = 0.08

# layout constants for pairwise feature plot
PAIR_FW = 10

# layout constants for survival plot
SURV_FW = 6
SURV_FH = 6


def Visualize(Model, Normalized, Raw, Symbols, Survival, Censored,
              GeneSet=False, N=30, Tau=0.05, Path=None):
    """
    Generate visualizations of risk profiles. Backpropagation is used to

    Parameters:
    -----------

    Model : class
    Model defined by finetuning

    Features : array_like
    An N x P array containing the normalized (z-scored) features used in model
    finetuning. Contains P features from N total patients.

    Symbols : array_like
    P-length list of strings describing model inputs

    N : integer
    Number of features to analyze.
    """

    # modify duplicate symbols where needed - append index to each instance
    Prefix = [Symbol[0:str.rfind(str(Symbol), '_')] for Symbol in Symbols]
    Suffix = [Symbol[str.rfind(str(Symbol), '_'):] for Symbol in Symbols]

    # copy prefixes
    Corrected = Prefix[:]

    # append index to each duplicate instance
    for i in np.arange(len(Prefix)):
        if Prefix.count(Prefix[i]) > 1:
            Corrected[i] = Prefix[i] + '.' + \
                str(Prefix[0:i+1].count(Prefix[i])) + Suffix[i]
        else:
            Corrected[i] = Prefix[i] + Suffix[i]

    # generate risk derivative profiles for cohort
    Gradients = RiskCohort(Model, Normalized)

    # generate ranked box plot series
    RBFig = RankedBox(Gradients, Symbols, N)

    # generate paired scatter plot
    PSFig = PairScatter(Gradients, Symbols, N)

    # generate cluster plot
    CFig = RiskCluster(Gradients, Raw, Symbols, N, Tau)

    # generate Kaplan-Meier plots for individual features
    KMFigs, KMNames = KMPlots(Raw, Symbols, Survival, Censored, N)

    # save figures
    if Path is not None:

        # save standard figures
        RBFig.savefig(Path + 'RankedBox.pdf')
        PSFig.savefig(Path + 'PairedScatter.pdf')
        CFig.savefig(Path + 'Heatmap.pdf')
        for i, Figure in enumerate(KMFigs):
            Figure.savefig(Path + 'KM.' + KMNames[i] + '.pdf')


def RankedBox(Gradients, Symbols, N=30):
    """
    Generates boxplot series of feature gradients ranked by absolute magnitude.

    Parameters:
    ----------

    Gradients: numpy matrix
    a matrix containing feature weights.

    Symbols: numpy nd array
    a matrix of feature Symbols.

    N: integer value
    number of featurs to display in barchart.

    Returns
    -------
    Figure : figure handle
        Handle to figure used for saving image to disk i.e.
        Figure.savefig('heatmap.pdf')
    """

    # generate mean values
    Means = np.asarray(np.mean(Gradients, axis=0))

    # sort features by mean absolute gradient
    Order = np.argsort(-np.abs(Means))

    # generate figure and add axes
    Figure = plt.figure(figsize=(BOX_FW, BOX_FH), facecolor='white')
    Axes = Figure.add_axes([SPACING, SPACING, 1-2*SPACING, 1-2*SPACING],
                           frame_on=False)
    Axes.set_axis_bgcolor('white')

    # generate boxplots
    Box = Axes.boxplot(Gradients[:, Order[0:N]],
                       patch_artist=True,
                       showfliers=False)

    # set global properties
    plt.setp(Box['medians'], color=MEDIAN, linewidth=1)
    plt.setp(Box['whiskers'], color=WHISKER, linewidth=1, linestyle='-')
    plt.setp(Box['caps'], color=WHISKER, linewidth=1)

    # modify box styling
    for i, box in enumerate(Box['boxes']):
        if Means[Order[i]] <= 0:
            box.set(color=BLUEEDGE, linewidth=2)
            box.set(facecolor=BLUEFACE)
        else:
            box.set(color=REDEDGE, linewidth=2)
            box.set(facecolor=REDFACE)

    # add jittered data overlays
    for i in np.arange(N):
        plt.scatter(np.random.normal(i+1, JITTER, size=Gradients.shape[0]),
                    Gradients[:, Order[i]], color=POINTS, alpha=0.2,
                    marker='o', s=2,
                    zorder=100)

    # set limits
    Axes.set_ylim(1.05 * Gradients.min(), 1.05 * Gradients.max())

    # format x axis
    plt.xlabel('Model Features')
    Fixed = _FixSymbols(Symbols)
    Names = plt.setp(Axes, xticklabels=[Fixed[Order[i]] for i in np.arange(N)])
    plt.setp(Names, rotation=90, fontsize=10)
    Axes.set_xticks(np.linspace(1.5, N-0.5, N-1), minor=True)
    Axes.xaxis.set_ticks_position('bottom')

    # format y axis
    plt.ylabel('Risk Gradient')
    Axes.yaxis.set_ticks_position('left')

    # add grid lines and zero line
    Axes.xaxis.grid(True, color=GRID, linestyle='-', which='minor')
    plt.axhline(0, color='black')

    return Figure


def PairScatter(Gradients, Symbols, N=30):
    """
    Generates boxplot series of feature gradients ranked by absolute magnitude.

    Parameters:
    ----------

    Risk_Gradients: numpy matrix
    a matrix containing feature weights.

    Symbols: numpy nd array
    a matrix of feature Symbols.

    N: integer value
    number of featurs to display in barchart.

    """

    # calculate means, standard deviations
    Means = np.asarray(np.mean(Gradients, axis=0))
    Std = np.asarray(np.std(Gradients, axis=0))

    # sort features by mean absolute gradient
    Order = np.argsort(-np.abs(Means))

    # generate subplots
    Figure, Axes = plt.subplots(nrows=N, ncols=N,
                                figsize=(PAIR_FW, PAIR_FW),
                                facecolor='white')
    Figure.subplots_adjust(hspace=SPACING, wspace=SPACING, bottom=SPACING)

    # remove axes and ticks
    for ax in Axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # generate scatter plots in lower triangular portion
    for i, j in zip(*np.triu_indices_from(Axes, k=1)):
        Axes[i, j].scatter((Gradients[:, Order[j]]-Means[Order[j]]) /
                           Std[Order[j]],
                           (Gradients[:, Order[i]]-Means[Order[i]]) /
                           Std[Order[i]],
                           color=POINTS, alpha=0.2, marker='o', s=2)
        Smooth = lowess((Gradients[:, Order[j]]-Means[Order[j]]) /
                        Std[Order[j]],
                        (Gradients[:, Order[i]]-Means[Order[i]]) /
                        Std[Order[i]])
        Axes[i, j].plot(Smooth[:, 1], Smooth[:, 0], color='red')

    # generate histograms on diagonal
    Fixed = _FixSymbols(Symbols, WRAP)
    for i in np.arange(N):
        if Means[Order[i]] <= 0:
            Axes[i, i].hist(Gradients[:, Order[i]],
                            facecolor=BLUEFACE,
                            alpha=0.8)
        else:
            Axes[i, i].hist(Gradients[:, Order[i]],
                            facecolor=REDFACE,
                            alpha=0.8)
        Axes[i, i].annotate(Fixed[Order[i]], (0, 0),
                            xycoords='axes fraction',
                            ha='right', va='top',
                            rotation=45)

    # delete unused axes
    for i, j in zip(*np.tril_indices_from(Axes, k=-1)):
        Figure.delaxes(Axes[i, j])

    return Figure


def KMPlots(Raw, Symbols, Survival, Censored, N=30):
    """
    Generates KM plots for individual features ranked by absolute magnitude.

    Parameters:
    ----------

    Gradients: numpy matrix
    a matrix containing feature weights.

    Symbols: numpy nd array
    a matrix of feature Symbols.

    N: integer value
    number of featurs to display in barchart.

    Returns
    -------
    Figures : figure handle
    List containing handles to figures.

    Names : array_like
    List of feature names for figures in 'Figures'

    Notes
    -----
    Note this uses feature values as opposed to back-propagated risk gradients.
    """

    # initialize list of figures and names
    Figures = []
    Names = []

    # generate mean values
    Means = np.asarray(np.mean(Raw, axis=0))

    # sort features by mean absolute gradient
    Order = np.argsort(-np.abs(Means))

    # generate Kaplan Meier fitter
    kmf = KaplanMeierFitter()

    # generate KM plot for each feature
    for count, i in enumerate(Order[0:N]):

        # generate figure and axes
        Figures.append(plt.figure(figsize=(SURV_FW, SURV_FH),
                                  facecolor='white'))
        Axes = Figures[count].add_axes([SPACING, SPACING,
                                        1-2*SPACING, 1-2*SPACING])

        # generate names
        Names.append(Symbols[i])

        # extract suffix to classify feature
        Suffix = Symbols[i][str.rfind(str(Symbols[i]), '_'):].strip()

        if Suffix == '_Clinical':

            # get unique values to determine if binary or continuous
            Unique = np.unique(Raw[:, i])

            # process based on variable type
            if Unique.size == 2:

                # extract and plot mutant and wild-type survival profiles
                kmf.fit(Survival[Raw[:, i] == Unique[0]],
                        Censored[Raw[:, i] == Unique[0]] == 1,
                        label=Symbols[i] + str(Unique[0]))
                kmf.plot(ax=Axes)
                kmf.fit(Survival[Raw[:, i] == Unique[1]],
                        Censored[Raw[:, i] == Unique[1]] == 1,
                        label=Symbols[i] + str(Unique[1]))
                kmf.plot(ax=Axes)
                plt.ylim(0, 1)

            else:

                # determine median value
                Median = np.median(Raw[:, i])

                # extract and altered and unaltered survival profiles
                kmf.fit(Survival[Raw[:, i] > Median],
                        Censored[Raw[:, i] > Median] == 1,
                        label=Symbols[i] + " > " + str(Median))
                kmf.plot(ax=Axes)
                kmf.fit(Survival[Raw[:, i] <= Median],
                        Censored[Raw[:, i] <= Median] == 1,
                        label=Symbols[i] + " <= " + str(Median))
                kmf.plot(ax=Axes)
                plt.ylim(0, 1)

        elif Suffix == '_Mut':

            # extract and plot mutant and wild-type survival profiles
            kmf.fit(Survival[Raw[:, i] == 1],
                    Censored[Raw[:, i] == 1] == 1,
                    label=Symbols[i] + " Mutant")
            kmf.plot(ax=Axes)
            kmf.fit(Survival[Raw[:, i] == 0],
                    Censored[Raw[:, i] == 0] == 1,
                    label=Symbols[i] + " Mutant")
            kmf.plot(ax=Axes)
            plt.ylim(0, 1)

        elif Suffix == '_CNV':

            # determine if alteration is amplification or deletion
            Amplified = np.mean(Raw[:, i]) > 0

            # extract and plot altered and unaltered survival profiles
            if Amplified:
                kmf.fit(Survival[Raw[:, i] > 0],
                        Censored[Raw[:, i] > 0] == 1,
                        label=Symbols[i] + " Amplified")
                kmf.plot(ax=Axes)
                kmf.fit(Survival[Raw[:, i] <= 0],
                        Censored[Raw[:, i] <= 0] == 1,
                        label=Symbols[i] + " not Amplified")
                kmf.plot(ax=Axes)
            else:
                kmf.fit(Survival[Raw[:, i] < 0],
                        Censored[Raw[:, i] < 0] == 1,
                        label=Symbols[i] + " Deleted")
                kmf.plot(ax=Axes)
                kmf.fit(Survival[Raw[:, i] >= 0],
                        Censored[Raw[:, i] >= 0] == 1,
                        label=Symbols[i] + " not Deleted")
                kmf.plot(ax=Axes)
            plt.ylim(0, 1)

        elif Suffix == '_CNVArm':

            # determine if alteration is amplification or deletion
            Amplified = np.mean(Raw[:, i]) > 0

            # extract and plot altered and unaltered survival profiles
            if Amplified:
                kmf.fit(Survival[Raw[:, i] > 0.25],
                        Censored[Raw[:, i] > 0.25] == 1,
                        label=Symbols[i] + " Amplified")
                kmf.plot(ax=Axes)
                kmf.fit(Survival[Raw[:, i] <= 0.25],
                        Censored[Raw[:, i] <= 0.25] == 1,
                        label=Symbols[i] + " not Amplified")
                kmf.plot(ax=Axes)
            else:
                kmf.fit(Survival[Raw[:, i] < -0.25],
                        Censored[Raw[:, i] < -0.25] == 1,
                        label=Symbols[i] + " Deleted")
                kmf.plot(ax=Axes)
                kmf.fit(Survival[Raw[:, i] >= -0.25],
                        Censored[Raw[:, i] >= -0.25] == 1,
                        label=Symbols[i] + " not Deleted")
                kmf.plot(ax=Axes)
            plt.ylim(0, 1)

        elif (Suffix == '_Protein') or (Suffix == '_mRNA'):

            # determine median expression
            Median = np.median(Raw[:, i])

            # extract and altered and unaltered survival profiles
            kmf.fit(Survival[Raw[:, i] > Median],
                    Censored[Raw[:, i] > Median] == 1,
                    label=Symbols[i] + " Higher Expression")
            kmf.plot(ax=Axes)
            kmf.fit(Survival[Raw[:, i] <= Median],
                    Censored[Raw[:, i] <= Median] == 1,
                    label=Symbols[i] + " Lower Expression")
            kmf.plot(ax=Axes)
            plt.ylim(0, 1)

        else:
            raise ValueError('Unrecognized feature type')

    return Figures, Names


def _FixSymbols(Symbols, Length=WRAP):
    """
    Removes trailing and leading whitespace and wraps long labels
    """

    # remove whitespace and wrap
    Fixed = ['\n'.join(wrap(Symbol.strip().replace('_', ' '), Length))
             for Symbol in Symbols]

    return Fixed
