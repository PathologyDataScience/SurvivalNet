import numpy as np
import plotly as py
import plotly.graph_objs as go

from . import RiskCohort

# define colors for positive risk (red) and negative risk (blue)
Red = 'rgba(222,45,38,0.8)'
Blue = 'rgb(49,130,189)'


def Visualize(Model, Features, Symbols, N=30):
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
    Gradients = RiskCohort(Model, Features)

    # generate ranked bar chart
    RankedBar(Gradients, Symbols, N)

    # generate ranked box plot series
    RankedBar(Gradients, Symbols, N)

def RankedBar(Gradients, Symbols, N=30):
    """
    Generates bar chart of feature gradients ranked by absolute magnitude.

    Parameters:
    ----------

    Risk_Gradients: numpy matrix
    a matrix containing feature weights.

    Symbols: numpy nd array
    a matrix of feature Symbols.

    N: integer value
    number of featurs to display in barchart.

    """

    # calculate means, standard deviations if multiple sample provided
    if(Gradients.shape[0] > 1):
        Mean = np.asarray(np.mean(Gradients, axis=0))
        Std = np.asarray(np.std(Gradients, axis=0))
        data = zip(Symbols, Mean[0], Std[0])
    else:
        data = zip(Symbols, np.asarray(Gradients)[0])

    # sort by mean gradient for cohorts, gradient for individual samples
    data = sorted(data, key=lambda x: np.abs(x[1]), reverse=True)

    # generate variables for visualization
    if(Gradients.shape[1] > 1):
        Means = [X[1] for X in data[0:N]]
        Stdevs = [X[2] for X in data[0:N]]
        Colors = [Red if X[1] > 0 else Blue for X in data[0:N]]
        Labels = [X[0] for X in data[0:N]]
    else:
        Values = [X[1] for X in data[0:N]]
        Colors = [Red if X[1] > 0 else Blue for X in data[0:N]]
        Labels = [X[0] for X in data[0:N]]

    # generate plot
    if(Gradients.shape[1] > 1):
        trace = [go.Bar(x=Labels, y=Means, type='bar',
                 error_y=dict(type='data', array=Stdevs, visible=True),
                 name='Risk Gradient',
                 marker=dict(color=Colors))]
    else:
        trace = [go.Bar(x=Labels, y=Values, type='bar',
                 name='Risk Gradient',
                 marker=dict(color=Colors))]
    py.offline.plot(trace, filename='RankedBar')


def RankedBox(Gradients, Symbols, N=30):
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

    # generate mean values
    Means = np.asarray(np.mean(Gradients, axis=0))[0]

    # generate colors
    Colors = [Red if mean > 0 else Blue for mean in Means]

    # zip data
    data = zip(Symbols, Means, Colors, list(np.array(Gradients).transpose()))

    # sort by mean gradient for cohorts, gradient for individual samples
    data = sorted(data, key=lambda x: np.abs(x[1]), reverse=True)

    # generate boxplot traces
    Traces = []
    for Symbol, Mean, Color, Points in data[0:N]:
        Traces.append(go.Box(y=Points,
                             name=Symbol,
                             jitter=0.5,
                             whiskerwidth=0.2,
                             boxpoints='all',
                             fillcolor=Color,
                             marker=dict(size=1, color=Color),
                             line=dict(width=1),))

    py.offline.plot(Traces, filename='RankedBox')


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
    Means = np.asarray(np.mean(Gradients, axis=0))[0]
    Std = np.asarray(np.std(Gradients, axis=0))[0]

    # zip data
    data = zip(Symbols, Means, Std, list(np.array(Gradients).transpose()))

    # sort by mean gradient for cohorts, gradient for individual samples
    data = sorted(data, key=lambda x: np.abs(x[1]), reverse=True)

    # generate layout for subplots
    Figure = py.tools.make_subplots(rows=N, cols=N)

    # layout subplots
    for i in np.arange(N):

        # append scatter plot for each variable pair
        for j in np.arange(i):
            Figure.append_trace(go.Scatter(x=data[i][3] / data[i][2],
                                           y=data[j][3] / data[j][2],
                                           mode='markers',
                                           marker=dict(color='grey',
                                                       size=1)),
                                i+1, j+1)

        # add histograms on diagonal
        Figure.append_trace(go.Histogram(x=np.array(Gradients[:, i] /
                                                    Std[i]).squeeze(),
                                         marker=dict(color='red')),
                            i+1, i+1)

    # generate plot
    py.offline.plot(Figure, filename='PairScatter')
