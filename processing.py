from cv2 import STEREO_SGBM_MODE_HH
import numpy as np
from v1diffusion import utils
from v1diffusion import transforms
from v1diffusion import evolution

def AHE(img, mask, T1, T2=0.5, a0=1, b0=1, a1=1, b1=1, sigma=1, epss=0.1, dt1=0.1, dt2=0.1, progress=True):
    """
    AHE algorithm as described by Boscain-2018 exploiting the knowledge of the
    position of the corruption

    img: input greyscale 2D image (0 is white, 1 is black)
    mask: corruption characteristic function (0 is image, 1 is corruption)
    T1: time to run the strong smoothing
    T2: time to run the weak smoothing
    a0:
    b0:
    a1:
    b1:
    sigma:
    dt1: time interval for strong smoothing
    dt2: time interval for weak smoothing
    progress: to either show or not tqdm animated progress bar

    @returns: the restored 2D image
    """
    I = img.copy()
    
    #Sanity check
    if I.shape != mask.shape:
        raise Exception("Mask size mismatch")

    if np.max(I)>1:
        I = I.astype(np.float32)
        I /= 255
    
    #Step 1, simple averaging
    I_f = utils.fill_corruption_with_bfs_avg(I, mask)

    #Step 2, strong smoothing
    uf1 = evolution.evolve_vc_hypoelliptic(transforms.lift_gradient(I_f),I_f, T1, a0=a0, b0=b0, a1=a1, b1=b1, sigma=sigma, epss=epss, dt=dt1)
    result1 = transforms.project_max(uf1)
    if np.max(result1)>1:
        result1 /= np.max(result1)

    #Step 3, advanced averaging
    result1 = (I_f+result1)/2

    #Step 4, weak smoothing
    uf2 = evolution.evolve_vc_hypoelliptic(transforms.lift_gradient(result1),result1, T2, dt=dt2)
    result2 = transforms.project_max(uf2)
    if np.max(result2)>1:
        result2 /= np.max(result2)

    return result2

def diffusion_with_restarts(img, mask, T=1):
    """
    TO IMPLEMENT
    """
    I = img.copy()

    #Sanity check
    if I.shape != mask.shape:
        raise Exception("Mask size mismatch")

    if np.max(I)>1:
        I = I.astype(np.float32)
        I /= 255

    uf2 = evolution.evolve_vc_hypoelliptic(transforms.lift_gradient(result1),result1, T2, dt=dt2)
    result2 = transforms.project_max(uf2)
    if np.max(result2)>1:
        result2 /= np.max(result2)

    return result2