from ropt.normalizer import series_ranking, series_ranking_normalize

def MaxCorrel(reference):
    """
    Returns the evaluation function of the objective that maximizes correlation of the indicator with a reference series.
    Parameters:
    reference : Series
        A Series to be used as a reference for correlation.
    Returns: function
        The evaluation function of the objective that maximizes correlation of an indicator with the reference series.
    """
    def Evaluation(ci):
        return ci.corr(reference)
    return Evaluation

def PCA(data):
    """
    Returns the evaluation function of the objective that maximizes the variance of the data.
    Parameters:
    data : DataFrame
        A DataFrame containing the data to be evaluated.
    Returns: function
        The evaluation function of the objective that maximizes the variance of the data.
    """
    def Evaluation(ci):
        variance = data.apply(lambda col: 1 if sum(col)==0 else ci.corr(col)**2)
        return sum(variance)/len(variance)
    return Evaluation
    
def MinUncertanty(others):
    """
    Returns the evaluation function of the objective that minimizes uncertainty
    Parameters:
    others : list of Series
        A list of Series to be used as a reference for uncertainty.
    Returns: function
        The evaluation function of the objective that minimizes uncertainty.
    """
    others = [series_ranking(o) for o in others]
    # print(others)
    def Evaluation(ci):
        ci = series_ranking(ci)
        # print(ci)
        return -2*sum((sum((abs(val-oval) for val,oval in zip(ci,other))) for other in others))/(len(ci)*len(ci)*len(others))
    return Evaluation
    
# def MinContinuousUncertanty(others):
#     """
#     Returns the evaluation function of the objective that minimizes a version of continuous uncertainty
#     Parameters:
#     others : list of Series
#         A list of Series to be used as a reference for uncertainty.
#     Returns: function
#         The evaluation function of the objective that minimizes uncertainty.
#     """
#     others = [series_ranking_normalize(o) for o in others]
#     def Evaluation(ci):
#         return -2*sum((sum((abs(val-oval) for val,oval in zip(ci,other))) for other in others))/(len(ci)*len(ci)*len(others))
#     return Evaluation
    
def MinContinuousUncertanty(others):
    """
    Returns the evaluation function of the objective that minimizes a version of continuous uncertainty
    Parameters:
    others : list of Series
        A list of Series to be used as a reference for uncertainty.
    Returns: function
        The evaluation function of the objective that minimizes uncertainty.
    """
    # others = [series_ranking_normalize(o) for o in others]
    def Evaluation(ci):
        m_dist = [min((abs(val-oval) for val, oval in zip(ci,other))) for other in others]
        return -2*sum((sum((abs(val-oval)*(1-m_d) for val,oval in zip(ci,other))) for m_d, other in zip(m_dist,others)))/(len(ci)*len(others))
    return Evaluation

# def MaxShannon():
#     """
#     Returns the evaluation function of the objective that maximizes shannon entropy.
#     Returns: function
#         The evaluation function of the objective that maximizes entropy.
#     """
#     def Evaluation(ci):
        
#     return Evaluation
    

def MaxEntropy():
    """
    Returns the evaluation function of the objective that maximizes entropy.
    Returns: function
        The evaluation function of the objective that maximizes entropy.
    """
    def Evaluation(ci):
        ci = ci.sort_values(ignore_index=True)
        max_entropy_x2 = (ci[len(ci)-1]-ci[0])*len(ci)
        if max_entropy_x2 == 0:
            return 1
        d = (ci[len(ci)-1]-ci[0])/(len(ci)-1)
        ideal = [ci[0]+n*d for n in range(len(ci))]
        system_entropy = sum((abs(c-val) for c, val in zip(ci,ideal)))
        return (1-2*system_entropy/max_entropy_x2)
    return Evaluation