import numpy as np
import cv2
from v1diffusion import utils

EPS = 0.0001

def forma(ang):
    """
    DEPRECATED
    Function to check if an angle is close to \pi/2 up to an error
    Originally written by Rossi and Boscain


    ang: angle from -\infty to +\infty

    @returns: if the angle is in [\pi/2-passo, \pi/2+passo]
    """

    #First adjust the angle so that it is in [-\pi,\pi]
    if ang>np.pi:
        ang-=np.pi
    if ang<0:
        ang+=np.pi
    passo = np.pi/8 #The granularity of the angle discretization

    return ang<(np.pi/2+passo) and (ang>np.pi/2-passo)

def lift_gradient_rossi(img, theta_size=50):
    """
    Orginal implementation of the lifting procedure by Rossi and Boscain
    This has been ported from MATLAB as a reference and is therefore not optimized

    img: original 2D image
    theta_size: how many discrete angles are considered

    @returns: the lift in SE(2)
    """

    #Compute the gradient
    GRADx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5, borderType=cv2.BORDER_REPLICATE)
    GRADy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5, borderType=cv2.BORDER_REPLICATE)

    #Initialize the image
    u = np.zeros((img.shape[0],img.shape[1],theta_size))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            #General case for a non-vertical gradient
            if GRADx[i,j] != 0:
                for k in range(theta_size):
                    u[i,j,k] = img[i,j]*forma(k/theta_size*np.pi-np.arctan(GRADy[i,j]/GRADx[i,j]))

            #Special case of vertical gradient
            elif GRADy[i,j] != 0:
                for k in range(theta_size):
                    u[i,j,k] = img[i,j]*forma(k/theta_size*np.pi-np.pi/2)

            #If the gradient is zero then every angle is considered
            else:
                u[i,j,:] = img[i,j]
    return u

def lift_gradient(img, model="SE2", criterium="spot", theta_size=30):
    """
    Lifting procedure using gradient

    img:        original 2D image
    model:      "SE2" or "PTR2"
    criterium:  "spot" or "distributed", spot considers only one point in the
                discrete space, whereas distributed spreads the image over a
                portion of the angles
    theta_size: how many discrete angles are considered

    @returns: the lift in the chosen mode
    """

    #Sanity check:
    if model not in ("SE2", "PTR2"):
        raise Exception("Method not found")

    if criterium=="spot":
        if model=="PTR2":
            hsector = np.pi/theta_size/2 + EPS #half sector+ an epsilon
        elif model=="SE2":
            hsector = np.pi/theta_size + EPS #half sector+ an epsilon
    elif criterium == "distributed":
        hsector = np.pi/8 #This is a fixed value suggested by Rossi

    #Compute the gradient
    GRADx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5, borderType=cv2.BORDER_REPLICATE)
    GRADy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5, borderType=cv2.BORDER_REPLICATE)
    Theta = np.arctan2(GRADy, GRADx)

    #Since we want the direction parallel to the level set and not perpendicular
    Theta = np.add(Theta, np.pi/2) #This is necessary

    #PTR2 -> modulus \pi
    #SE(2) -> modulus 2\pi
    if model=="PTR2":
        Theta = np.mod(Theta,np.pi)
    elif model=="SE2":
        Theta = np.mod(Theta,2*np.pi)

    #Initialize the image
    u = np.zeros((img.shape[0],img.shape[1],theta_size), dtype=np.float64)

    for i in range(img.shape[0]): #vertical
        for j in range(img.shape[1]): #horizontal

            #non-zero gradient
            if np.abs(GRADx[i,j])>EPS or np.abs(GRADy[i,j])>EPS:
                #consider all the points in (i,j) close enough to Theta(i,j)
                if model=="PTR2":
                    if criterium=="spot":
                        u[i,j,utils.ang_d(np.arange(theta_size)*np.pi/theta_size, Theta[i,j], unit="P1")<hsector] = img[i,j]
                    elif criterium=="distributed":
                        u[i,j,utils.ang_d(np.arange(theta_size)*np.pi/theta_size, Theta[i,j], unit="P1")<hsector] = img[i,j]/np.size(u[i,j,utils.ang_d(np.arange(theta_size)*np.pi/theta_size, Theta[i,j])<hsector])
                elif model=="SE2":
                    if criterium=="spot":
                        u[i,j,utils.ang_d(np.arange(theta_size)*2*np.pi/theta_size, Theta[i,j])<hsector] = img[i,j]
                    elif criterium=="distributed":
                        u[i,j,utils.ang_d(np.arange(theta_size)*2*np.pi/theta_size, Theta[i,j])<hsector] = img[i,j]/np.size(u[i,j,utils.ang_d(np.arange(theta_size)*2*np.pi/theta_size, Theta[i,j])<hsector])

            #zero-gradient
            else:
                if model=="PTR2":
                    #all the points in the lift are set to 1/N of the image
                    u[i,j,:] = img[i,j]/theta_size
                elif model=="SE2":
                    u[i,j,:] = img[i,j]/theta_size
    return u

def lift_gaussian(img, sigma, model="SE2", theta_size=30):
    """
    Gaussian lift
    """
    
    imgR = np.repeat(img[..., None], theta_size, axis=2)
    Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5, borderType=cv2.BORDER_REPLICATE)
    u = np.zeros((img.shape[0],img.shape[1],theta_size))
    S = np.repeat((Ix**2 + Iy**2)[...,None], theta_size, axis=2)
    v = np.arange(theta_size)*np.pi/theta_size
    COS = np.repeat(np.repeat(np.cos(v)[None,...], img.shape[1], axis=0)[None,...], img.shape[0], axis=0)
    SIN = np.repeat(np.repeat(np.sin(v)[None,...], img.shape[1], axis=0)[None,...], img.shape[0], axis=0)

    #Compute the magnitude of the gradient in the direction (cos, sin)
    A = ((np.repeat(Ix[...,None],theta_size, axis=2) * COS + np.repeat(Iy[...,None],theta_size, axis=2) * SIN)**2)

    #Only divide by the magnitude where such magnitude is >0. Otherwise the value of the image is used
    A = np.divide(A, S, where=S!=0)
    A[S==0]=imgR[S==0]

    #Construct the Gaussian
    u = imgR * np.exp(- A/(4* sigma**2))
    
    return u

def project_gaussian(u):
    """
    Inverse of the Gaussian lift
    """
    Q = -np.cumsum(np.gradient(np.log(u+EPS), axis=2, edge_order=2), axis=2) #EPS is needed numerically when the value of the image is 0
    Qmin = np.min(Q, axis=2)
    mu = np.add(Q, np.repeat(Qmin[...,None],u.shape[2], axis=2))
    I = 1/u.shape[2] * np.sum(u*np.exp(-mu), axis=2)
    return I 



def lift_gabor(img, theta_size=50, ksize=9, lambd=0.5, gamma=0.4, psi=0.1):
    """
    Lift based on the gabor filter kernel
    """
    u = np.zeros((img.shape[0],img.shape[1], theta_size))
    for step in range(0, theta_size):
        theta = np.pi/theta_size*step
        kern = cv2.getGaborKernel((ksize, ksize), sigma=1, theta=theta, lambd=lambd, gamma=0.4, psi=0.1, ktype=cv2.CV_32F)
        u[:,:,step] = cv2.filter2D(img, -1, kern)
    return u

def project_max(u):
    """
    Projection based on the non-maxima suppression presented by Citt-Sarti(2006)
    img(i,j) = max_{\theta} u(i,j,\theta)

    u: an image in PTR2

    @returns: a 2D image
    """
    img = np.amax(u, axis=2)
    return img

def project_integral(u):
    """
    Alternative projection method that computes the integral over \theta at
    every point

    u: an image in PTR2

    @returns: a 2D image
    """
    img = np.sum(u, axis=2)
    return img

def project_inverse(u, method="max", ksize=9, lambd=0.5, gamma=0.4, psi=0.1):
    """
    TODO
    """
    u1 = u.copy()
    for step in range(0, u.shape[2]):
        theta = np.pi/u.shape[2]*step
        kern = cv2.getGaborKernel((ksize, ksize), sigma=1, theta=theta, lambd=lambd, gamma=0.4, psi=0.1, ktype=cv2.CV_32F)
        
        kern_inv = np.real(np.fft.ifft2(np.true_divide(1,np.fft.fft2(kern))))
        #kern_inv[kern_inv >= 1] = 1
        #kern_inv[kern_inv < -1] = 1
        
        u1[:,:,step] = cv2.filter2D(u[:,:,step], -1, kern_inv)
    if method == "max":
        return project_max(u1)
    elif method == "integral":
        return project_integral(u1)
    else:
        raise Exception("Method not found")