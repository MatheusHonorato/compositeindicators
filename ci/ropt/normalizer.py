import pandas as pd
from .normalization import Maximize, Minimize

def series_outlier_removal(data, n=4):
    """
    Removes outliers from a Series based on the interquartile range (IQR) method.
    This function calculates the IQR and removes values that are outside the range of Q1 - 1.5 * IQR to Q3 + 1.5 * IQR.
    Optional parameter 'n' specifies the number of parts to divide the data for outlier detection. 4 creates the quartiles.
    Parameters:
    data : Series
        A Series containing the data from which outliers are to be removed.
    n : int, optional
        The number of parts to divide the data for outlier detection. Default is 4.
    Returns: Series
        A Series with outliers clamped.
    """
    dt = data.sort_values(ignore_index=True)
    
    mi = len(data) / n
    ma = dt[round((n-1) * mi)]
    mi = dt[round(mi)-1]
    
    outlier = (ma-mi)*1.5
    ma += outlier
    mi -= outlier
    
    if ma == mi:
        return data
    return data.apply(lambda x: max(mi, min(ma, x)))

def outlier_removal(data, n=4):
    """
    Removes outliers from each column in a DataFrame based on the interquartile range (IQR) method.
    This function applies the series_outlier_removal function to each column in the DataFrame.
    Optional parameter 'n' specifies the number of parts to divide the data for outlier detection. 4 creates the quartiles.
    Parameters:
    data : DataFrame
        A pandas DataFrame containing the data from which outliers are to be removed.
    n : int, optional
        The number of parts to divide the data for outlier detection. Default is 4.
    Returns: DataFrame
        A DataFrame with outliers clamped in each column.
    """
    return data.apply(lambda col: series_outlier_removal(col,n))

def series_normalize(data, opt_function=Maximize):
    """
    Normalizes a Series using the specified optimization function.
    Parameters:
    data : Series
        A Series containing the data to be normalized.
    opt_function : function, optional
        The optimization function to be applied for normalization. Default is Maximize.
    Returns: Series
        A Series with normalized values.
    """
    return data.apply(opt_function(data))

def normalize(data, opt_function=Maximize):
    """
    Normalizes each column in a DataFrame using the specified optimization function.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to be normalized.
    opt_function : function, optional
        The optimization function to be applied for normalization. Default is Maximize.
    Returns: DataFrame
        A DataFrame with normalized values in each column.
    """
    return data.apply(lambda col: series_normalize(col, opt_function=opt_function))

def robust_normalize(data, max_function=Maximize, min_function=Minimize, reference_series=None):
    """
    Normalizes each column in a DataFrame using robust methods based on correlation with a reference series.
    If the correlation with the reference series is positive, it uses the max_function; otherwise, it uses the min_function for normalization.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to be normalized.
    max_function : function, optional
        The function to be used for maximization. Default is Maximize.
    min_function : function, optional
        The function to be used for minimization. Default is Minimize.
    reference_series : Series, optional
        A Series to be used as a reference for correlation. If None, the last column of the DataFrame is used.
    Returns: DataFrame
        A DataFrame with normalized values in each column based on the robust methods.
    """
    if reference_series is None:
        reference_series = data[data.columns[-1]]

    return data.apply(lambda col: series_normalize(
        col, opt_function=max_function if col.corr(reference_series) >= 0 else min_function
    ))


def direction_normalize(data, direction, max_function=Maximize, min_function=Minimize):
    """
    Normalizes each column in a DataFrame using robust methods based on correlation with a reference series.
    If the correlation with the reference series is positive, it uses the max_function; otherwise, it uses the min_function for normalization.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to be normalized.
    max_function : function, optional
        The function to be used for maximization. Default is Maximize.
    min_function : function, optional
        The function to be used for minimization. Default is Minimize.
    reference_series : Series, optional
        A Series to be used as a reference for correlation. If None, the last column of the DataFrame is used.
    Returns: DataFrame
        A DataFrame with normalized values in each column based on the robust methods.
    """
    return data.apply(lambda col: series_normalize(
        col, opt_function=max_function if direction[data.columns.get_indexer([col.name])[0]] else min_function
    ))

def importance_factors(data, factors):
    """
    Applies importance factors to each column in a DataFrame.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to which importance factors will be applied.
    factors : list
        A list of importance factors corresponding to each column in the DataFrame.
    Returns: DataFrame
        A DataFrame with each value multiplied by the corresponding factor.
    """
    return data.apply(lambda row: pd.Series([r ** factor for r, factor in zip(row,factors)], index=data.columns), axis=1)

def series_ranking(data, minimize=False):
    """
    Ranks the values in a Series, assigning ranks based on the higher values.
    Parameters:
    data : Series
        A Series containing the data to be ranked.
    minimize : bool, optional
        If True, ranks based on lower values. Default is False.
    Returns: Series
        A Series with ranks assigned to each value of the original Series.
    """
    dt = pd.DataFrame(data)
    dt['__index'] = dt.index
    dt = dt.sort_values(data.name or 0, ignore_index=True, ascending=minimize)
    dt['__rank'] = dt.index + 1
    dt = dt.sort_values('__index', ignore_index=True)
    return dt['__rank']
    
def series_ranking_normalize(data, minimize=False):
    """
    Normalizes the ranks of a Series.
    Parameters:
    data : Series
        A Series containing the data to be ranked and normalized.
    minimize : bool, optional
        If True, ranks based on lower values. Default is False.
    Returns: Series
        A Series with normalized ranks.
    """
    dt = series_ranking(data, minimize=minimize)
    return (max(dt)-dt)/max(dt)

def ranking(data, minimize=False):
    """
    Ranks the values of a DataFrame, assigning ranks based on higher values.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to be ranked.
    minimize : bool, optional
        If True, ranks based on lower values. Default is False.
    Returns: DataFrame
        A DataFrame with ranks assigned to each value of the original DataFrame.
    """  
    return data.apply(lambda x: series_ranking(x, minimize=minimize))

def ranking_normalize(data, minimize=False):
    """
    Normalizes the ranks of a DataFrame.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to be ranked and normalized.
    minimize : bool, optional
        If True, ranks based on lower value. Default is False.
    Returns: DataFrame
        A DataFrame with normalized ranks.
    """
    return data.apply(lambda x: series_ranking_normalize(x, minimize=minimize))

def robust_ranking(data, reference_series=None):
    """
    Ranks each column in a DataFrame based on its correlation with a reference series.
    If the correlation with the reference series is positive, it ranks based on higher values; otherwise, it ranks based on lower values.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to be ranked.
    reference_series : Series, optional
        A Series to be used as a reference for correlation. If None, the last column of the DataFrame is used.
    Returns: DataFrame
        A DataFrame with ranks assigned to each value of the original DataFrame.
    """
    if reference_series is None:
        reference_series = data[data.columns[-1]]
        
    return data.apply(lambda col: series_ranking(col, minimize=col.corr(reference_series)<0))
    
def robust_ranking_normalize(data, reference_series=None):
    """
    Normalizes the ranks of each column in a DataFrame based on its correlation with a reference series.
    If the correlation with the reference series is positive, it normalizes based on higher values; otherwise, it normalizes based on lower values.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to be ranked and normalized.
    reference_series : Series, optional
        A Series to be used as a reference for correlation. If None, the last column of the DataFrame is used.
    Returns: DataFrame
        A DataFrame with normalized ranks.
    """
    if reference_series is None:
        reference_series = data[data.columns[-1]]
        
    return data.apply(lambda col: series_ranking_normalize(col, minimize=col.corr(reference_series)<0))
