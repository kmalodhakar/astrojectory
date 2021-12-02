import numpy as np
import math as m

def cart2sph(coords_3d):
    assert isinstance(coords_3d,tuple)
    x = coords_3d[0]
    y = coords_3d[1]
    z = coords_3d[2]
    tmp = x**2 + y**2
    r = m.sqrt(tmp + z**2)               # r
    theta = m.atan2(z,m.sqrt(tmp))     # theta
    phi = m.atan2(y,x)                           # phi
    return r, theta, phi

print(cart2sph((3,4,5)))