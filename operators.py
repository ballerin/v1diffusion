#import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

@partial(jit, static_argnames=['n','model','order'])
def dX1(u, n=1, model=0, order=2):
    """
    model:
    0: SE(2)
    1: PTR2

    Directional derivative X_1 in SE2/PTR2
    X_1(x,y,\theta) = \cos(\theta)\partial_x + \sin(\theta)\partial_y 

    From the documentation of numpy.gradient:
    The gradient is computed using second order accurate central differences in
    the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries. The returned gradient
    hence has the same shape as the input array
    """
    u2 = u
    for i in range(n):
        #Construct a constant boundary for Neumann condition
        
        if model==1:
            COS = jnp.meshgrid(jnp.arange(0,u2.shape[1]), jnp.arange(0,u2.shape[0]), jnp.cos(jnp.linspace(0,jnp.pi,u2.shape[2])))[2]
            SIN = jnp.meshgrid(jnp.arange(0,u2.shape[1]), jnp.arange(0,u2.shape[0]), jnp.sin(jnp.linspace(0,jnp.pi,u2.shape[2])))[2]
        elif model==0:
            COS = jnp.meshgrid(jnp.arange(0,u2.shape[1]), jnp.arange(0,u2.shape[0]), jnp.cos(jnp.linspace(0,2*jnp.pi,u2.shape[2])))[2]
            SIN = jnp.meshgrid(jnp.arange(0,u2.shape[1]), jnp.arange(0,u2.shape[0]), jnp.sin(jnp.linspace(0,2*jnp.pi,u2.shape[2])))[2]
        else:
            raise Exception("Method not found")
        
        if order == 1:
            dx = jnp.diff(u2, axis=1, append=u2[:,-1:,:])
            dy = jnp.diff(u2, axis=0, append=u2[-1:,:,:])
        elif order == 2:
            dx = jnp.gradient(u2, axis=1)
            dy = jnp.gradient(u2, axis=0)
            
        #Force Neumann boundary conditions
        dx = dx.at[:,0,:].set(0)
        dx = dx.at[:,-1,:].set(0)
        dy = dy.at[0,:,:].set(0)
        dy = dy.at[-1,:,:].set(0)

        u2 = (COS*dx+SIN*dy)
    return u2

@partial(jit, static_argnames=['n','model'])
def dX2(u, n=1, model=0):
    """
    model:
    0: SE(2)
    1: PTR2

    Directional derivative X_2 in SE2/PTR2
    X_2(x,y,\theta) = \partial_\theta
    """
    u2 = u
    u3 = jnp.zeros((u2.shape[0],u2.shape[1], u2.shape[2]+4))
    for i in range(n):
        #Periodic boundary
        u3 = u3.at[:,:,2:-2].set(u2)
        u3 = u3.at[:,:,:2].set(u2[:,:,-2:])
        u3 = u3.at[:,:,-2:].set(u2[:,:,:2])

        u2 = jnp.gradient(u3, axis=2)[:,:,2:-2]
    return u2

@partial(jit, static_argnames=['n','model','order'])
def dX3(u, n=1, model=0, order=2):
    """
    model:
    0: SE(2)
    1: PTR2

    Directional derivative X_3 in SE2/PTR2
    X_3(x,y,\theta) = -\sin(\theta)\partial_x + \cos(\theta)\partial_y 
    """
    u2 = u
    for i in range(n):
        #Construct a constant boundary for Neumann condition
        
        if model==0:
            COS = jnp.meshgrid(jnp.arange(0,u2.shape[1]), jnp.arange(0,u2.shape[0]), jnp.cos(jnp.linspace(0,jnp.pi,u2.shape[2])))[2]
            SIN = jnp.meshgrid(jnp.arange(0,u2.shape[1]), jnp.arange(0,u2.shape[0]), jnp.sin(jnp.linspace(0,jnp.pi,u2.shape[2])))[2]
        elif model==1:
            COS = jnp.meshgrid(jnp.arange(0,u2.shape[1]), jnp.arange(0,u2.shape[0]), jnp.cos(jnp.linspace(0,2*jnp.pi,u2.shape[2])))[2]
            SIN = jnp.meshgrid(jnp.arange(0,u2.shape[1]), jnp.arange(0,u2.shape[0]), jnp.sin(jnp.linspace(0,2*jnp.pi,u2.shape[2])))[2]
        else:
            raise Exception("Method not found")
        
        if order == 1:
            dx = jnp.diff(u2, axis=1, append=u2[:,-1:,:])
            dy = jnp.diff(u2, axis=0, append=u2[-1:,:,:])
        elif order == 2:
            dx = jnp.gradient(u2, axis=1)
            dy = jnp.gradient(u2, axis=0)
            
        #Force Neumann boundary conditions
        dx = dx.at[:,0,:].set(0)
        dx = dx.at[:,-1,:].set(0)
        dy = dy.at[0,:,:].set(0)
        dy = dy.at[-1,:,:].set(0)

        u2 = (-SIN*dx+COS*dy)
    return u2

@partial(jit, static_argnames=['beta','model'])
def hypoelliptic_heat_operator(u, beta=1, model=0):
    """
    Applies the hypoelliptic heat operator X_1^2 + \beta X_2^2 to a lifted image
    in PTR2

    u: image in PTR2
    beta: weight coefficient
    """

    return dX1(u,n=2, model=model, order=2) + beta**2 * dX2(u, n=2, model=model)

@partial(jit, static_argnames=['beta','model'])
def orthogonal_operator(u, beta=1, model=0):
    """
    Applies the hypoelliptic heat operator X_3^2 + \beta X_2^2 to a lifted image
    in PTR2. This is the orthogonal wrt to the level lines


    u: image in PTR2
    beta: weight coefficient
    """

    return dX3(u,n=2, model=model, order=2) + beta**2 * dX2(u, n=2, model=model)

@jit
def vc_hypoelliptic_heat_operator(u, a, b):
    """
    Applies the hypoelliptic heat operator aX_1^2 + bX_2^2 to a lifted image
    in PTR2

    u: image in PTR2
    a: weight coefficient
    b: weight coefficient
    """

    u1 = dX1(u,n=2)
    u2 = dX2(u,n=2)
    
    uu = jnp.zeros(u.shape)
    for i in range(u.shape[2]):
        uu[:,:,i] = a*u1[:,:,i] + b*u2[:,:,i]
             
    return uu
