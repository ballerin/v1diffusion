import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def fill_corruption_with_avg(img, mask,step_mask=None, ksize = 9):
    """
    Method to fill a corrupted area of a 2D image with an approximation computed
    starting from a neighbor of every corrupted pixel

    Direct use of this method is to be avoided, as a point positioned in a
    fully corrupted neighborhood will stay corrupted

    img: input greyscale 2D image (0 is white, 1 is black)
    mask: corruption characteristic function (0 is image, 1 is corruption)
    step_mask: boundary of the corrupted area. If None, the whole mask is used
    ksize: kernel size for neighbors

    @returns: A filled 2D image
    """

    #If the boundary mask is not provided then the whole mask is used as boundary
    if step_mask is None:
        step_mask = mask

    I = img.copy()
    m = mask.astype(np.bool8)
    sm = step_mask.astype(np.bool8)
    I[m] = 0
    kern = np.ones((ksize, ksize))
    
    #Compute the number of pixels contributing to the avg, and then the avg for every pixel
    avg_not_mask = cv2.filter2D(1-mask, -1, kern)
    blurred = cv2.filter2D(I, -1, kern)
    
    #For every point in the step mask (boundary), substitute the pixel with the average
    I[sm]=blurred[sm]/avg_not_mask[sm]
    return I

def fill_corruption_with_bfs_avg(img, mask, ksize = 9):
    """
    Method to fill a corrupted area of a 2D image with an approximation computed
    starting from a neighbor of every corrupted pixel. The procedure is done
    recursively considering at each step the boundary of the corrupted area in
    a BFS (Breadth-First-Search) style

    img: input greyscale 2D image (0 is white, 1 is black)
    mask: corruption characteristic function (0 is image, 1 is corruption)
    ksize: kernel size for neighbors

    @returns: A filled 2D image
    """
    """
    I = img.copy()
    m = mask.copy()
    step_mask = (np.abs(np.gradient(m, axis=0))+np.abs(np.gradient(m, axis=1)))*m>0
    while np.sum(step_mask)>0:
        I = fill_corruption_with_avg(I, m, step_mask)
        m[step_mask.astype(np.bool8)] = False
        step_mask = (np.abs(np.gradient(m, axis=0))+np.abs(np.gradient(m, axis=1)))*m>0
    return I
    """
    I = img.copy()
    m = mask.copy()
    flag = True
    while flag:
        flag = False
        step_mask = np.zeros(m.shape)
        for y in range(I.shape[0]):
            for x in range(I.shape[1]):
                if y==0:
                    if m[y,x]==1 and m[y+1,x]==0:
                        step_mask[y,x]=1
                elif y==I.shape[0]-1:
                    if m[y,x]==1 and m[y-1,x]==0:
                        step_mask[y,x]=1
                else:
                    if m[y,x]==1 and (m[y-1,x]==0 or m[y+1,x]==0):
                        step_mask[y,x]=1
                if x==0:
                    if m[y,x]==1 and m[y,x+1]==0:
                        step_mask[y,x]=1
                elif x==I.shape[1]-1:
                    if m[y,x]==1 and m[y,x-1]==0:
                        step_mask[y,x]=1
                else:
                    if m[y,x]==1 and (m[y,x-1]==0 or m[y,x+1]==0):
                        step_mask[y,x]=1
        if np.sum(step_mask)>0:
            flag = True
            I = fill_corruption_with_avg(I, m, step_mask,ksize)
            m[step_mask.astype(np.bool8)] = False
    return I

def imshow(img, r=None, mode = 2):
    """
    Handler method to show a greyscale numpy/opencv image using matplotlib

    TODO: extend the method to support color images

    img: a greyscale image
    r: range expressed as a tuple (min, max). If not provided (0,1) is used

    @returns: None
    """
    plt.figure(figsize=(8,8))
    if r is None:
        r= (0,1)
    if mode==1:
        plt.imshow(img, vmin=r[0], vmax=r[1], cmap="gray")
    elif mode==2:
        plt.imshow(1-img, vmin=r[0], vmax=r[1], cmap="gray")
    plt.axis('off')
    plt.show()

def save_animation(stack, filename, T=5000, delay=0, reverse=True, r=None):
    """
    Utility function to save an animation from a stack of greyscale images

    stack:  a list of greyscale images
    filename: name/path of the animation
    T:  duration of the animation in milliseconds
    delay:  delay at the end of the animation on a still image
    reverse:    boolean, if the images are presented as an inverse greyscale,
                0 for white and 1 for black
    r: range expressed as a tuple (min, max). If not provided (0,1) is used
    """

    if r is None:
        r= (0,1)
    
    duration = int(T/len(stack))

    frames = []
    for i in stack:
        j = i.copy()
        j[j<r[0]]=r[0]
        j[j>r[1]]=r[1]
        frames.append(Image.fromarray(np.uint8((1-j)*255),'L').convert('P'))
    for i in range(int(delay/duration)):
        frames.append(frames[-1])
    frames[0].save(filename,save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)


def ang_d(a,b, unit="radians", abs=True):
    """
    Distance on S^1 the unit circle, expressed in angle
    Works with integers, floats and numpy arrays

    a:  first angle
    b:  second angle
    unit:   "radians" or "degrees", unit of measure of the angles
    abs: if the required distance is regardless of orientation

    @returns: the distance on S^1 between the two angles
    """

    u = np.pi #radians

    if unit=="degrees":
        u = 180
    
    if unit=="P1":
        u = np.pi/2

    c = (a - b) % (2*u)
    if type(a)==type(1) or type(a)==type(1.0): #Integer/float case
        if c > u:
            c -= 2*u
    else: #Numpy case
        c[c>u] = c[c>u]-2*u
    if abs:
        c = np.abs(c)
    return c    