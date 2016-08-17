import matplotlib as mpl
import matplotlib.pyplot as pylab
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
from scipy.stats import chisquare
from scipy.stats.mstats import kruskalwallis

WINDOW_HEIGHT = 30
WINDOW_WIDTH = 30
[SAMPLE_X, SAMPLE_Y, SAMPLE_W, SAMPLE_H] = [0.4, 0.72, 0.6, 0.2]
[FEATURE_X, FEATURE_Y, FEATURE_W, FEATURE_H] = [0.05, 0.1, 0.2, 0.5]
[HEATMAP_X, HEATMAP_Y, HEATMAP_W, HEATMAP_H] = [0.4, 0.1, 0.6, 0.5]
[gcb_x, gcb_y, gcb_w, gcb_h] = [0.4, 0.63, HEATMAP_W, 0.15]
[cncb_x, cncb_y, cncb_w, cncb_h] = [0.4, 0.63, HEATMAP_W, 0.15]

#[HEATMAP_X, HEATMAP_Y, HEATMAP_W, HEATMAP_H] = [0.4, 0.9, 0.6, 0.5]


def RiskCluster(Gradients, Raw, Symbols, N=30, Tau=0.05):
    """
    """

    # calculate rank order of features
    Order = np.argsort(-np.abs(np.mean(Gradients, axis=0)))

    # copy data, re-order, normalize
    Normalized = Gradients[:, Order[0:N]].copy()
    Normalized = (Normalized - np.mean(Normalized, axis=0)) / \
        np.std(Normalized, axis=0)

    # transpose so that samples are in columns
    Normalized = Normalized.transpose()

    # generate figure
    Figure = pylab.figure(figsize=(WINDOW_WIDTH, WINDOW_HEIGHT))

    # cluster samples and generate dendrogram
    SampleDist = dist.pdist(Normalized.T, 'correlation')
    SampleDist = dist.squareform(SampleDist)
    SampleHandle = Figure.add_axes([SAMPLE_X, SAMPLE_Y, SAMPLE_W, SAMPLE_H],
                                   frame_on=True)
    SampleLinkage = sch.linkage(SampleDist, method='average',
                                metric='correlation')
    SampleDendrogram = sch.dendrogram(SampleLinkage)
    SampleIndices = sch.fcluster(SampleLinkage, 0.7*max(SampleLinkage[:, 2]),
                                 'distance')
    SampleHandle.set_xticks([])
    SampleHandle.set_yticks([])

    # cluster features and generate dendrogram
    FeatureDist = dist.pdist(Normalized, 'correlation')
    FeatureDist = dist.squareform(FeatureDist)
    FeatureHandle = Figure.add_axes([FEATURE_X, FEATURE_Y,
                                     FEATURE_W, FEATURE_H],
                                    frame_on=True)
    FeatureLinkage = sch.linkage(FeatureDist, method='average',
                                 metric='correlation')
    FeatureDendrogram = sch.dendrogram(FeatureLinkage, orientation='right')
    FeatureHandle.set_xticks([])
    FeatureHandle.set_yticks([])

    # reorder input matrices based on clustering and capture order
    Reordered = Normalized[:, SampleDendrogram['leaves']]
    Reordered = Reordered[FeatureDendrogram['leaves'], :]
    SampleOrder = SampleIndices[SampleDendrogram['leaves']]

    # generate heatmap
    Heatmap = Figure.add_axes([HEATMAP_X, HEATMAP_Y, HEATMAP_W, HEATMAP_H])
    Heatmap.matshow(Reordered, aspect='auto', origin='lower',
                    cmap=pylab.cm.bwr)
    Heatmap.set_xticks([])
    Heatmap.set_yticks([])

    # capture cluster associations
    Significant = ClusterAssociations(Raw, Symbols, SampleIndices, Tau)

    # extract mutation values from raw features
    SigMut = [Symbol for Symbol in Significant if
              Symbol.strip()[-4:] == "_Mut"]
    Indices = [i for i, Symbol in enumerate(Symbols) if Symbol in set(SigMut)]
    Mutations = Raw[:, Indices]
    Mutations = Mutations[SampleOrder, :].T

    # add significant mutation tracks to plot
    gm = Figure.add_axes([gcb_x, gcb_y, gcb_w, gcb_h])
    cmap_g = mpl.colors.ListedColormap(['k', 'w'])
    gm.matshow(Mutations, aspect='auto', origin='lower', cmap=cmap_g)
    for i in range(len(SigMut)):
        gm.text(gcb_x - 10, gcb_y + (i) - 0.5,
                SigMut[i], fontsize=6)
    gm.set_xticks([])
    gm.set_yticks([])

    # extract CNV values from raw features
    SigCNV = [Symbol for Symbol in Significant if
              Symbol.strip()[-4:] == "_CNV"]
    Indices = [i for i, Symbol in enumerate(Symbols) if Symbol in set(SigCNV)]
    CNVs = Raw[:, Indices]
    CNVs = CNVs[SampleOrder, :].T

    # add significant CNV tracks to plot
    cnv = Figure.add_axes([cncb_x, cncb_y, cncb_w, cncb_h])
    cnv.matshow(CNVs, aspect='auto', origin='lower', cmap=pylab.cm.bwr)
    for i in range(len(SigCNV)):
        cnv.text(cncb_x - 10, cncb_y + (i) - 0.5,
                 SigCNV[i], fontsize=6)
    cnv.set_xticks([])
    cnv.set_yticks([])

    Figure.show()
    Figure.savefig('heatmap.png')

    # return cluster labels
    return SampleIndices


def ClusterAssociations(Raw, Symbols, Labels, Tau=0.05):
    """
    """

    # initialize list of symbols with significant associations
    Significant = []

    # get feature type from 'Symbols'
    Suffix = [Symbol[str.rfind(str(Symbol), '_')+1:] for Symbol in Symbols]

    # identify mutations and CNVs
    Mutations = [index for index, x in enumerate(Suffix) if x == "Mut"]
    CNVs = [index for index, x in enumerate(Suffix) if x == "CNV"]

    # test mutation associations
    for i in np.arange(len(Mutations)):

        # build contingency table - expected and observed
        Observed = np.zeros((2, np.max(Labels)))
        for j in np.arange(1, np.max(Labels)+1):
            Observed[0, j-1] = np.sum(Raw[Labels == j, Mutations[i]] == 0)
            Observed[1, j-1] = np.sum(Raw[Labels == j, Mutations[i]] == 1)
        RowSum = np.sum(Observed, axis=0)
        ColSum = np.sum(Observed, axis=1)
        Expected = np.outer(ColSum, RowSum) / np.sum(Observed.flatten())

        # perform test
        stat, p = chisquare(Observed, Expected, ddof=1, axis=None)
        if p < Tau:
            Significant.append(Symbols[Mutations[i]])

    # copy number associations
    for i in np.arange(len(CNVs)):

        # separate out CNV values by cluster and perform test
        if(np.max(Labels) == 2):
            CNV1 = Raw[Labels == 1, CNVs[i]]
            CNV2 = Raw[Labels == 2, CNVs[i]]
            stat, p = kruskalwallis(CNV1, CNV2)
        elif(np.max(Labels) == 3):
            CNV1 = Raw[Labels == 1, CNVs[i]]
            CNV2 = Raw[Labels == 2, CNVs[i]]
            CNV3 = Raw[Labels == 3, CNVs[i]]
            stat, p = kruskalwallis(CNV1, CNV2, CNV3)
        elif(np.max(Labels) == 4):
            CNV1 = Raw[Labels == 1, CNVs[i]]
            CNV2 = Raw[Labels == 2, CNVs[i]]
            CNV3 = Raw[Labels == 3, CNVs[i]]
            CNV4 = Raw[Labels == 4, CNVs[i]]
            stat, p = kruskalwallis(CNV1, CNV2, CNV3, CNV4)
        elif(np.max(Labels) == 5):
            CNV1 = Raw[Labels == 1, CNVs[i]]
            CNV2 = Raw[Labels == 2, CNVs[i]]
            CNV3 = Raw[Labels == 3, CNVs[i]]
            CNV4 = Raw[Labels == 4, CNVs[i]]
            CNV5 = Raw[Labels == 5, CNVs[i]]
            stat, p = kruskalwallis(CNV1, CNV2, CNV3, CNV4, CNV5)
        if p < Tau:
            Significant.append(Symbols[CNVs[i]])

    # return names of features with significant associations
    return Significant
