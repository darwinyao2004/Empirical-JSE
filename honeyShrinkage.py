# Copyright (c) 2022, Jan K. Schluesener, github.com/jkschluesener
# Copyright (c) 2014, Olivier Ledoit and Michael Wolf
# All rights reserved.

# BSD 2-clause license

import numpy as np
from warnings import warn
from typing import Tuple
from numpy.typing import ArrayLike


def covCor(X: ArrayLike, shrink: float = None) -> Tuple[ArrayLike, float]:
    """A form of shrinkage coefficient estimation that shrinks towards
    a constant correlation. Usually, a Ledoit-Wolf estimator shrinks
    towards a common variance.

    The coefficient is computed using the method described in this publication:
    O. Ledoit and M. Wolf, “Honey, I Shrunk the Sample Covariance Matrix,”
    JPM, vol. 30, no. 4, pp. 110–119, Jul. 2004, doi: 10.3905/jpm.2004.110.

    This is a direct translation of the `covCor` matlab function
    by Ledoit and Wolf, derived from their work and also carries a
    BSD 2-clause license.

    Parameters
    ----------
    X : ArrayLike
        Data to etimte shrinkage on, shape (n_samples, n_features)
    shrink : float, optional
        Constant for shrinkage, the default `None` computes it.

    Returns
    -------
    ArrayLike
        Invertible covariance matrix estimator
    Float
        Shrinkage target
    """

    n_samples, n_features = X.shape

    if n_features == 1:
        warn('Input Data has only feature sample')
        return 0

    if n_samples == 1:
        warn('Input Data has only one sample')
        return None

    X = np.asarray(X)

    # de-mean
    X -= X.mean(0)

    # sample covariance matrix
    sample = (1 / n_samples) * (X.T @ X)

    # compute prior
    var = np.diag(sample)
    sqrtvar = np.sqrt(var)

    # squared priors as 2D
    var2 = np.repeat(var.copy()[:, np.newaxis], n_features, axis=1)
    sqrtvar2 = np.repeat(sqrtvar.copy()[:, np.newaxis], n_features, axis=1)

    rBar = (np.sum(np.sum(sample / (sqrtvar2 * sqrtvar2.T))) - n_features) / (n_features * (n_features - 1))
    prior = rBar * sqrtvar2 * sqrtvar2.T

    if shrink == None:
        np.fill_diagonal(prior, var)

        # compute shrinkage
        X2 = X ** 2

        # pi-hat
        phiMat = X2.T @ X2 / n_samples - 2. * (X.T @ X) * sample / n_samples + sample ** 2
        phi = phiMat.sum()

        # rho-hat
        term1 = ((X ** 3).T @ X) / n_samples
        help = X.T @ X / n_samples
        helpDiag = np.diag(help)
        term2 = np.repeat(helpDiag.copy()[:, np.newaxis], n_features, axis=1) * sample
        term3 = help * var2
        term4 = var2 * sample

        thetaMat = term1 - term2 - term3 + term4
        np.fill_diagonal(thetaMat, 0)
        rho = np.diag(phiMat).sum() + rBar * (((1. / sqrtvar[:, np.newaxis]) * sqrtvar2.T) * thetaMat).sum()

        # gamma-hat
        gamma = np.linalg.norm(sample-prior, ord='fro') ** 2

        # shrinkage constant
        kappa = (phi - rho) / gamma
        shrinkage = np.max([0, np.min([1, kappa / n_samples])])

    else:
        shrinkage = shrink

    sigma = shrinkage * prior + (1 - shrinkage) * sample

    return sigma, shrinkage