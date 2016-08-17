import numpy as np
import plotly as py
import plotly.graph_objs as go


def viztool(Risk_Gradients, Symbols, N):
    """
    Generates barchart of N number of features has larger absolute weights
    value. providing profiles of feature weights will leads to the barchart of
    N number of features has a larger mean absoulte value with errorbars.

    Parameters:
    ----------

    Risk_Gradients: numpy matrix
    a matrix containing feature weights.

    Symbols: numpy nd array
    a matrix of feature Symbols.

    N: integer value
    number of featurs to be shown in barchart.

    """

    if(Risk_Gradients.shape[0] > 1):
               
        Mean_Risk_Gradient = np.asarray(np.mean(Risk_Gradients, axis=0))
        std_Risk_Graident = np.asarray(np.std(Risk_Gradients, axis=0))
        std_Risk_Graident = std_Risk_Graident / np.sqrt(Risk_Gradients.shape[0])
        data = zip(Symbols, Mean_Risk_Gradient[0], std_Risk_Graident[0])

    else:
        data = zip(Symbols, np.asarray(Risk_Gradients)[0])

    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

    def Draw_barchart(data, N):

        X = []
        labels = []
        colors = []
        errorbars = []
        Red = 'rgba(222,45,38,0.8)'
        Blue = 'rgb(49,130,189)'
        i = 0
        j = 0
        if(len(data[0]) > 2):
            while(j + i < N):
                if(data[i][1] > np.absolute(data[(len(data) - 1) - j][1])):
                    X.append(data[i][1])
                    labels.append(data[i][0])
                    errorbars.append(data[i][2])
                    colors.append(Red)
                    i = i + 1
                elif(data[i][1] < np.absolute(data[(len(data) - 1) - j][1])):
                    X.append(data[(len(data) - 1) - j][1])
                    labels.append(data[(len(data) - 1) - j][0])
                    errorbars.append(data[j][2])
                    colors.append(Blue)
                    j = j + 1
                else:
                    X.append(data[i][1])
                    labels.append(data[i][0])
                    errorbars.append(data[i][2])
                    colors.append(Red)
                    X.append(data[(len(data) - 1) - j][1])
                    labels.append(data[(len(data) - 1) - j][0])
                    errorbars.append(data[j][2])
                    colors.append(Blue)
                    i = i + 1
                    j = j + 1

            trace = [go.Bar(x=labels, y=X, type='bar',
                            error_y=dict(type='data', array=errorbars,
                                         visible=True),
                            name='Sensitivity_analysis',
                            marker=dict(color=colors))]
            for lab in labels:
                print lab.strip()
            py.offline.plot(trace, filename='basic-bar')

        else:
            while(j + i < N):
                if(data[i][1] > np.absolute(data[(len(data) - 1) - j][1])):
                    X.append(data[i][1])
                    labels.append(data[i][0])
                    colors.append(Red)
                    i = i + 1
                elif(data[i][1] < np.absolute(data[(len(data) - 1) - j][1])):
                    X.append(data[(len(data) - 1) - j][1])
                    labels.append(data[(len(data) - 1) - j][0])
                    colors.append(Blue)
                    j = j + 1
                else:
                    X.append(data[i][1])
                    labels.append(data[i][0])
                    colors.append(Red)
                    X.append(data[(len(data) - 1) - j][1])
                    labels.append(data[(len(data) - 1) - j][0])
                    colors.append(Blue)
                    i = i + 1
                    j = j + 1
            text_file = open("Gene_Output.txt", "w")
            for i in range(N):
                text_file.write(str((labels[i]).strip())+'\t'+str(X[i])+'\n')
            text_file.close()
            trace = [go.Bar(x=labels, y=X, name='Sensitivity_analysis',
                            marker=dict(color=colors))]

            py.offline.plot(trace, filename='basic-bar2')
    Draw_barchart(sorted_data, N)
