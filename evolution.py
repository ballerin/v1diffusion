import numpy as np
from tqdm import tqdm
from v1diffusion import operators

def parallel_diffusion(u0, T, dt=0.1, beta=0.5, model=0, progress=True):
    """
    Hypoelliptic evolution in PTR2 according to Boscain-2016
    
    This method takes an input an image lifted in PTR2 and applies the
    hypoelliptic heat operator \Delta_H at time intervals of dt until final time
    T is reached

    u0: image in PTR2
    T: time to run the evolution
    dt: time interval
    beta: weight coefficient
    progress: to either show or not tqdm animated progress bar

    @returns: an image in PTR2
    """

    u1 = u0 #Initialization of u1 in case the loop is not ran

    t=0
    for step in tqdm(range(0, int(T/dt)), disable=(not progress)):
        u1 = u1+dt*operators.hypoelliptic_heat_operator(u1, beta, model)
        t+=dt
    
    return np.asarray(u1)


def orthogonal_diffusion(u0, T, dt=0.1, beta=0.5, model=0, progress=True):
    u1 = u0.copy() #Initialization of u1 in case the loop is not ran

    t=0
    for step in tqdm(range(0, int(T/dt)), disable=(not progress)):
        u1 = u1+dt*operators.orthogonal_operator(u1, beta, model)
        t+=dt
    
    return u1


def parallel_concentration(u0, T, dt=0.01, beta=0.5, model=0, progress=True):
    """
    Handler function to regress the hypoelliptic evolution with negative dt
    """
    u1 = u0 #Initialization of u1 in case the loop is not ran

    t=0
    for step in tqdm(range(0, int(T/dt)), disable=(not progress)):
        u1 = u1-dt*operators.hypoelliptic_heat_operator(u1, beta, model)
        t+=dt
    
    return np.asarray(u1)


def orthogonal_concentration(u0, T, dt=0.01, beta=0.5, model=0, progress=True):
    """
    Handler function to regress the hypoelliptic evolution with negative dt
    """
    u1 = u0 #Initialization of u1 in case the loop is not ran

    t=0
    for step in tqdm(range(0, int(T/dt)), disable=(not progress)):
        u1 = u1-dt*operators.orthogonal_operator(u1, beta, model)
        t+=dt
    
    return np.asarray(u1)


def v1_unsharp_filter(u, C=1, T=1, beta=1, model=0):
    """
    Applies the unsharp filter as described in Ballerin-Grong-23


    u: image in PTR2/SE2
    T: final time
    beta: weight coefficient
    """
    u2 = orthogonal_concentration(u, T, beta=beta, model=model)
    return u-C*(u-u2)


def evolve_vc_hypoelliptic(u0, img, T, a0=1, b0=1, a1=1, b1=1, sigma=1,
 epss=0.1, dt=0.1, progress=True):
    """
    Hypoelliptic evolution with varying coefficients in PTR2 
    
    This method takes an input an image lifted in PTR2 and applies
    the hypoelliptic heat operator \Delta_H with varying coefficients
    at time intervals of dt until final time T is reached

    u0: image in PTR2
    img: the original 2D greyscale image
    T: time to run the evolution
    a0: starting weight
    b0: starting weight
    a1: starting weight
    a2: starting weight
    sigma: weight
    epss: weight
    dt: time interval
    progress: to either show or not tqdm animated progress bar

    @returns: an image in PTR2
    """

    u1 = u0.copy() #Initialization of u1 in case the loop is not ran

    eps = np.exp(-img**2/sigma)
    t=0
    for step in tqdm(range(0, int(T/dt)), disable=(not progress)):
        a = np.zeros(u0.shape[:2])
        b = np.zeros(u0.shape[:2])
        a_mask = (a0+a1*eps)/(a0+a1)>epss
        b_mask = (b0+b1*eps)/(b0+b1)>epss
        a[a_mask] = a0 + a1*eps[a_mask]
        b[b_mask] = b0 + b1*eps[b_mask]

        u1 = u0+dt*operators.vc_hypoelliptic_heat_operator(u0, a, b)
        u0 = u1.copy()
        t+=dt

    return u1