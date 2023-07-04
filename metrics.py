def MSE(A, B):
    """
    MSE metric evaluation according to the formula (for two arrays):
    MSE(A,B) = (1/n)\sum (A_i-B_i)^2
    """

    #Flatten the numpy structures to allow for the evaluation of matrices
    return ((A.flatten() - B.flatten())**2).mean()