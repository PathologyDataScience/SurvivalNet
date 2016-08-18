import numpy as np
import scipy as sp


def SSGSEA(Gradients, Symbols, Sets, Alpha=0.0, Normalize=False):
    """
    Generates enrichment scores from feature gradients using single-sample gene
    set enrichment analysis. Enrichment scores of feature gradients help to
    identify pathways that are enriched with positive or negative prognostic
    features.

    Parameters
    ----------
    Gradients : array_like
    Numpy array containing feature/sample gradients obtained by Risk_Cohort.
    Features are in columns and samples are in rows.

    Symbols : array_like
    List containing strings describing features. See Notes below for
    restrictions on symbol names.

    Sets : array_like
    A list of lists, each containing the gene symbols for each gene set.

    Alpha : double
    Exponential weighting used in gene set enrichment analysis.
    Default value = 0.

    Normalized : bool
    Flag indicating whether to normalize across samples (True), or to analyze
    each sample independently.

    Returns
    -------
    ES : array_like
    Enrichment scores for each gene set in each sample. Each column represents
    a gene set, each row a sample.

    Notes
    -----
    Gene sets can be obtained from the Molecular Signatures Database (MSigDB)
    at http://software.broadinstitute.org/gsea/msigdb/.

    Reference
    ---------
    Barbie et al "Systematic RNA interference reveals that oncogenic KRAS-
    driven cancers require TBK1", Nature. 2009 Nov 5; 462(7269): 108-112.

    See Also
    --------
    ReadGMT, RiskCohort
    """

    # total number of genes
    N = len(Symbols)

    # allocate enrichment scores
    ES = np.nan((Gradients.shape[0], len(Sets)))

    # rank expression values within each sample
    Ranks = np.zeros((Gradients.shape[0], len(Sets)))
    if not Normalize:  # rank gradient values
        for j in np.arange(Gradients.shape[0]):
            Ranks[j, :] = sp.stats.rankdata(-Gradients[j, :])
    else:  # transform gradients to percentiles - then rank
        Percentiles = np.zeros((Gradients.shape[0], len(Sets)))
        for j in np.arange(Gradients.shape[1]):
            Percentiles[:, j] = sp.stats.rankdata(-Gradients[:, j]) \
                / Gradients.shape[0]
        for j in np.arange(Gradients.shape[0]):
            Ranks[j, :] = sp.stats.rankdata(-Gradients[j, :])

    # generate index arrays for gene sets
    Indices = np.zeros((len(Sets), N))
    for i in np.arange(len(Sets)):
        Indices[i, _MapSymbols(Symbols, Sets[i])] = 1.0

    # iterate over each sample
    for i in np.arange(Gradients.shape[0]):

        # calculate ECDFs for gene sets
        PGw = np.cumsum(Indices[:, Ranks[i, :].astype(np.uint) - 1] *
                        np.outer(np.ones((len(Sets), 1)),
                                 np.linspace(1, N, N) ** Alpha),
                        axis=1)
        Total = np.sum(Indices * (np.outer(np.ones((Indices.shape[0], 1)),
                                           Ranks[i, :]) ** Alpha), axis=1)
        PGw = PGw / np.outer(Total, np.ones((1, N)))

        # calculate ECDFs for other genes
        PNG = np.cumsum((1-Indices), axis=1)
        PNG = PNG / np.outer(N-np.sum(Indices, axis=1), np.ones((1, N)))

        # calculate enrichment score
        ES[i, :] = np.sum(PGw - PNG, axis=1) / N

    return ES


def _MapSymbols(Symbols, Set):
    return np.array([i for i, Symbol in enumerate(Symbols)
                     if Symbol in Set])
