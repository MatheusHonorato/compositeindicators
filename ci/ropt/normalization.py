from numpy import exp

def Maximize(data):
    """
    Normalizes the data to a range of 0 to 1 using the MinMax method for maximization.
    Parameters:
    data : Series
        Input data to be normalized.
    Returns: function
        A function to be applied in a series using DataFrame.apply(Maximize(data)).
    """
    ma = max(data)
    mi = min(data)
    def Evaluation(value):
        return (value-mi)/(ma-mi)
    if ma==mi:
        def Evaluation(value):
            return value
    return Evaluation

def Minimize(data):
    """
    Normalizes the data to a range of 0 to 1 using the MinMax method for minimization.
    Parameters:
    data : Series
        Input data to be normalized.
    Returns: function
        A function to be applied in a series using DataFrame.apply(Minimize(data)).
    """
    ma = max(data)
    mi = min(data)
    def Evaluation(value):
        return (ma-value)/(ma-mi)
    if ma==mi:
        def Evaluation(value):
            return value
    return Evaluation
    
def UncertainMaximize(data, uncertanty=0.05):
    """
    Normalizes the data to a range of uncertanty to 1 - uncertanty using the MinMax method for maximization.
    Parameters:
    data : Series
        Input data to be normalized.
    uncertanty : float, optional
        The uncertainty factor to be applied to the maximum and minimum values. Default is 0.05 (5%).
    Returns: function
        A function to be applied in a series using DataFrame.apply(UncertainMaximize(data)).
    """
    ma = max(data)
    mi = min(data)
    diff = (ma-mi)*uncertanty
    ma = ma + diff
    mi = mi + diff
    def Evaluation(value):
        return (value-mi)/(ma-mi)
    return Evaluation

def UncertainMinimize(data, uncertanty=0.05):
    """
    Normalizes the data to a range of uncertanty to 1 - uncertanty using the MinMax method for minimization.
    Parameters:
    data : Series
        Input data to be normalized.
    uncertanty : float, optional
        The uncertainty factor to be applied to the maximum and minimum values. Default is 0.05 (5%).
    Returns: function
        A function to be applied in a series using DataFrame.apply(UncertainMinimize(data)).
    """
    ma = max(data)
    mi = min(data)
    diff = (ma-mi)*uncertanty
    ma = ma + diff
    mi = mi + diff
    def Evaluation(value):
        return (ma-value)/(ma-mi)
    return Evaluation
    
def RatioMaximize(data):
    """
    Normalizes the data to a range of 0 to 1 using the ratio method for maximization.
    Parameters:
    data : Series
        Input data to be normalized.
    Returns: function
        A function to be applied in a series using DataFrame.apply(RatioMaximize(data)).
    """
    ma = max(data)
    def Evaluation(value):
        return value/ma
    return Evaluation
    
def SafeRatioMaximize(data):
    """
    Normalizes the data to a range of 0 to 1 using the ratio method for maximization.
    If the maximum value in the data is zero, it returns a function that applies Maximize to the data.
    Parameters:
    data : Series
        Input data to be normalized.
    Returns: function
        A function to be applied in a series using DataFrame.apply(SafeRatioMaximize(data)).
    """
    return Maximize(data) if max(data) == 0 else RatioMaximize(data)

def RatioMinimize(data):
    """
    Normalizes the data to a range of 0 to 1 using the ratio method for minimization.
    Parameters:
    data : Series
        Input data to be normalized.
    Returns: function
        A function to be applied in a series using DataFrame.apply(RatioMinimize(data)).
    """
    mi = min(data)
    def Evaluation(value):
        return mi/value
    return Evaluation
    
def SafeRatioMinimize(data):
    """
    Normalizes the data to a range of 0 to 1 using the ratio method for minimization.
    If the minimum value in the data is zero, it returns a function that applies Minimize to the data.
    Parameters:
    data : Series
        Input data to be normalized.
    Returns: function
        A function to be applied in a series using DataFrame.apply(SafeRatioMinimize(data)).
    """
    return Minimize(data) if min(data) == 0 else RatioMinimize(data)

def LogisticMaximize(data):
    """
    Normalizes the data using a logistic function for maximization.
    Parameters:
    data : Series
        Input data to be normalized.
    Returns: function
        A function to be applied in a series using DataFrame.apply(LogisticMaximize(data)).
    """
    std = data.std()
    mean = data.mean()
    def Evaluation(value):
        return 1/(1+exp(-((value-mean)/std)))
    return Evaluation
    
def LogisticMinimize(data):
    """
    Normalizes the data using a logistic function for minimization.
    Parameters:
    data : Series
        Input data to be normalized.
    Returns: function
        A function to be applied in a series using DataFrame.apply(LogisticMinimize(data)).
    """
    std = data.std()
    mean = data.mean()
    def Evaluation(value):
        return 1/(1+exp(-((mean-value)/std)))
    return Evaluation