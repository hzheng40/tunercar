import numpy as np
from numpy import loadtxt
from numpy import genfromtxt
import scipy.interpolate as interpolate

def generate_way_points(track):

    if track == 'corner':
        return corner()
    elif track == 'ellipse':
        return ellipse()
    elif track == 'ellipse3D':
        return ellipse3D()
    elif track == 'wave':
        return wave() 
    elif track == 'qph':
        return qph()
    elif track == 'rectangular2D':
        return rectangular2D()
    elif track == 'rectangular3D':
        return rectangular3D()        
    elif track == 'random':
        return random()  
    elif track == 'levine':
        return levine()
    else:
        raise NotImplementedError("wrong track")


def corner():
    x = np.linspace(0,50, num=50)
    x2 = np.ones(50)*50
    x = np.concatenate((x, x2), axis=0)
    y = np.zeros(50)
    y2 = np.linspace(0,10, num=50)
    y = np.concatenate((y, y2), axis=0)
    return x, y


def ellipse():
    t = np.linspace(0, 2*np.pi, num=100)
    x = 10*np.cos(t)
    y = 5*np.sin(t)
    return x, y


def ellipse3D():
    t = np.linspace(1e-10, 2*np.pi, num=100)
    x = 10*np.sin(t)
    y = 5*np.cos(t)
    psi = np.arctan(-x/4/y)
    return x, y, psi


def wave():
    x = np.linspace(0, 6*np.pi, num=200)
    y = 5*np.sin(t+np.pi/2)
    return x, y


def qph():
    return loadtxt("../tracks/qph.txt", comments="#", delimiter=",", unpack=False)


def rectangular2D():
    from tracks import rectangular2D
    x, y = rectangular2D(width=2, length=20, breadth=10).trajectory(num=100) 
    return x, y


def rectangular3D():
    from tracks import rectangular3D
    x, y, psi = rectangular3D(width=2, length=20, breadth=10).trajectory(num=400) 
    return x, y, psi

def levine():
    start_point = 100
    end_point = 1
    smooth = 5
    wp_raw = genfromtxt("../tracks/levine.csv", delimiter= ',')
    tck, u = interpolate.splprep([wp_raw[start_point:-end_point,0],
                  wp_raw[start_point:-end_point,1]], 
                                  s = smooth)
    point_spacing = np.arange(0, 1.0, 0.002)
    wp = np.asarray(interpolate.splev(point_spacing, tck))
    return wp[0,:], wp[1,:]

def random():
    return loadtxt("../tracks/random.txt", comments="#", delimiter=",", unpack=False)
