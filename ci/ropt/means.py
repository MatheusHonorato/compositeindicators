import numpy as np

def weighted_arithimetic_mean(coef, X):
    """
    Calculates the weighted arithmetic mean of the input data X using the provided coefficients.
    Parameters:
    coef : list
        Coefficients for the weighted arithmetic mean.
    X : list
        Input data for which the weighted arithmetic mean is calculated.
    Returns: float
        The weighted arithmetic mean of the input data X.
    """
    return np.dot(coef, X)
def normal_weighted_arithimetic_mean(coef, X):
    """
    Calculates the normal weighted arithmetic mean of the input data X using the provided coefficients.
    Parameters:
    coef : list
        Coefficients for the normal weighted arithmetic mean.
    X : list
        Input data for which the normal weighted arithmetic mean is calculated.
    Returns: float
        The normal weighted arithmetic mean of the input data X.
    """
    return np.dot(coef, X)/np.sum(coef)

def weighted_geometric_mean(coef, X):
    """
    Calculates the weighted geometric mean of the input data X using the provided coefficients.
    Parameters:
    coef : list
        Coefficients for the weighted geometric mean.
    X : list
        Input data for which the weighted geometric mean is calculated.
    Returns: float
        The weighted geometric mean of the input data X.
    """
    return np.exp(np.dot(coef,np.log(X)))
def safe_weighted_geometric_mean(coef, X):
    """
    Calculates the weighted geometric mean of the input data X using the provided coefficients.
    If any value in X is zero, it returns 0 to avoid division by zero errors
    Parameters:
    coef : list
        Coefficients for the weighted geometric mean.
    X : list
        Input data for which the weighted geometric mean is calculated.
    Returns: float
        The weighted geometric mean of the input data X.
    """
    if any((v==0 for v in X)):
        return 0
    return weighted_geometric_mean(coef, X)
    
def normal_weighted_geometric_mean(coef, X):
    """
    Calculates the normal weighted geometric mean of the input data X using the provided coefficients.
    Parameters:
    coef : list
        Coefficients for the normal weighted geometric mean.
    X : list
        Input data for which the normal weighted geometric mean is calculated.
    Returns: float
        The normal weighted geometric mean of the input data X.
    """
    return np.exp(np.dot(coef,np.log(X))/np.sum(coef))
def safe_normal_weighted_geometric_mean(coef, X):
    """
    Calculates the normal weighted geometric mean of the input data X using the provided coefficients.
    If any value in X is zero, it returns 0 to avoid division by zero errors
    Parameters:
    coef : list
        Coefficients for the normal weighted geometric mean.
    X : list
        Input data for which the normal weighted geometric mean is calculated.
    Returns: float
        The normal weighted geometric mean of the input data X.
    """
    if any((v==0 for v in X)):
        return 0
    return normal_weighted_geometric_mean(coef, X)

def weighted_harmonic_mean(coef, X):
    """
    Calculates the weighted harmonic mean of the input data X using the provided coefficients.
    Parameters:
    coef : list
        Coefficients for the weighted harmonic mean.
    X : list
        Input data for which the weighted harmonic mean is calculated.
    Returns: float
        The weighted harmonic mean of the input data X.
    """
    return 1/(np.sum([c/x for c,x in zip(coef, X)]))
def safe_weighted_harmonic_mean(coef, X):
    """
    Calculates the weighted harmonic mean of the input data X using the provided coefficients.
    If any value in X is zero, it returns 0 to avoid division by zero errors.
    Parameters:
    coef : list
        Coefficients for the weighted harmonic mean.
    X : list
        Input data for which the weighted harmonic mean is calculated.
    Returns: float
        The weighted harmonic mean of the input data X.
    """
    if all((v==0 for v in coef)) or any((v==0 for v in X)):
        return 0
    return weighted_harmonic_mean(coef, X)

def normal_weighted_harmonic_mean(coef, X):
    """
    Calculates the normal weighted harmonic mean of the input data X using the provided coefficients.
    Parameters:
    coef : list
        Coefficients for the normal weighted harmonic mean.
    X : list
        Input data for which the normal weighted harmonic mean is calculated.
    Returns: float
        The normal weighted harmonic mean of the input data X.
    """
    return np.sum(coef)/(np.sum([c/x for c,x in zip(coef, X)]))
def safe_normal_weighted_harmonic_mean(coef, X):
    """
    Calculates the normal weighted harmonic mean of the input data X using the provided coefficients.
    If any value in X is zero, it returns 0 to avoid division by zero errors.
    Parameters:
    coef : list
        Coefficients for the normal weighted harmonic mean.
    X : list
        Input data for which the normal weighted harmonic mean is calculated.
    Returns: float
        The normal weighted harmonic mean of the input data X.
    """
    if all((v==0 for v in coef)) or any((v==0 for v in X)):
        return 0
    return normal_weighted_harmonic_mean(coef, X)

def weighted_harmonic_mean_correction(coef, X):
    """
    Calculates the weighted harmonic mean of the input data x using the provided coefficients.
    If any value in x is zero, it returns 0 to avoid division by zero errors.
    Parameters:
    coef : list
        Coefficients for the weighted harmonic mean.
    x : list
        Input data for which the weighted harmonic mean is calculated.
    Returns: float
        The weighted harmonic mean of the input data x.
    """
    ncoef, nX = zip(*((c,x) for c,x in zip(coef,X) if x != 0 and c != 0))
    return weighted_harmonic_mean(ncoef,nX) * len(nX) / len(X)

def normal_weighted_harmonic_mean_correction(coef, X):
    """
    Calculates the normal weighted harmonic mean of the input data x using the provided coefficients.
    If any value in x is zero, it returns 0 to avoid division by zero errors.
    Parameters:
    coef : list
        Coefficients for the normal weighted harmonic mean.
    x : list
        Input data for which the normal weighted harmonic mean is calculated.
    Returns: float
        The normal weighted harmonic mean of the input data x.
    """
    ncoef, nX = zip(*((c,x) for c,x in zip(coef,X) if x != 0 and c != 0))
    return normal_weighted_harmonic_mean(ncoef,nX) * len(nX) / len(X)
