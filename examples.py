import numpy as np
import cv2

def generate_example(number = 0, blur=False):
    """
    Example 0 : eye
    Example 1 : horizontal lines
    Example 2 : vertical lines
    Example 3 : diagonal lines 1
    Example 4 : diagonal lines 2 (TODO)
    Example 5 : circle
    """
    if number<0 or number>5:
        raise Exception("Example does not exist")

    if number == 0:
        return generate_example_0(blur)
    elif number == 1:
        return generate_example_1(blur)
    elif number == 2:
        return generate_example_2(blur)
    elif number == 3:
        return generate_example_3(blur)
    elif number == 4:
        return generate_example_4(blur)
    elif number == 5:
        return generate_example_5(blur)
    else:
        raise Exception("Example does not exist")
    

def generate_example_0(blur=False):
    """
    Example 0 : eye
    """
    I = cv2.imread("../digip/images/eye.jpg",cv2.IMREAD_GRAYSCALE).astype(np.float32)
    I/=255
    mask = np.zeros(I.shape)
    mask[60:64,100:180]=1
    I = (1-I)*(1-mask)
    if blur:
        I = cv2.GaussianBlur(I,(15,15),0.4)
    return I

def generate_example_1(blur=False):
    """
    Example 1 : horizontal lines
    """
    I = np.zeros((256,256), dtype=np.float32)
    I[30:-30:10,:] = 1
    I[31:-29:10,:] = 1
    I[:,100:110] = 0
    if blur:
        I = cv2.GaussianBlur(I,(15,15),0.4)
    return I

def generate_example_2(blur=False):
    """
    Example 2 : vertical lines
    """
    I = np.zeros((256,256), dtype=np.float32)
    I[:,30:-30:10] = 1
    I[:,31:-29:10] = 1
    I[100:110,:] = 0
    if blur:
        I = cv2.GaussianBlur(I,(15,15),0.4)
    return I

def generate_example_3(blur=False):
    """
    Example 3 : diagonal lines 1
    """
    I = np.zeros((256,256), dtype=np.float32)
    x=np.linspace(0,255,256)
    y=np.linspace(0,255,256)
    X,Y = np.meshgrid(x,y)
    V = X+Y
    I[V%30<2]=1
    I[150:160,100:256]=0
    if blur:
        I = cv2.GaussianBlur(I,(15,15),0.8)
    return I

def generate_example_4(blur=False):
    """
    TODO
    Example 4 : diagonal lines 2
    """
    I = np.zeros((256,256), dtype=np.float32)
    x=np.linspace(0,255,256)
    y=np.linspace(0,255,256)
    X,Y = np.meshgrid(x,y)
    V = X+Y
    I[V%30<2]=1
    I[150:160,100:256]=0
    if blur:
        I = cv2.GaussianBlur(I,(15,15),0.8)
    return I

def generate_example_5(blur=False):
    """
    Example 5 : circle
    """
    I = np.zeros((256,256), dtype=np.float32)
    x=np.linspace(0,255,256)
    y=np.linspace(0,255,256)
    X,Y = np.meshgrid(x,y)
    V = (X-128)**2+(Y-128)**2
    I[(V>4800)*(V<5000)]=1
    I[150:160,100:256]=0
    if blur:
        I = cv2.GaussianBlur(I,(15,15),1.8)
    return I