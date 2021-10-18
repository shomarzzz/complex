import matplotlib.pyplot as plt
import numpy as np
from numba import jit
def imgtocomp():
    a = np.zeros((2000, 2000), dtype=complex)
    dr = 0.0025
    di = 0.0025
    centre = (1000, 1000)
    k = 2000
    for row in range(2000):
        for col in range(k):
            ru = (col-centre[0])*dr
            iu = (k-(centre [1]+row))*di*1j
            a[row][col] = ru+iu
    return a
"""
@jit(nopython=True)
def mdb(a:complex):
    b = a
    d = 0.2-0.5*1j
    c = 0
    for i in range(1000):
        if abs(b)>=2:
            return [255, 255, 255]
        b = b**2+d
    return [0, 0, 0]
"""
@jit(nopython=True)
def mdb(a:complex):
    b = a
    q = 0
    for i in range(1000):
        if abs(q) >= 2:
            return [255, 255, 255]
        q = q**2+b
    return [0, 0, 0]
b = imgtocomp()
print(b)
pic = np.zeros((2000, 2000, 3), dtype='uint8')
for row in range(len(b)):
    for col in range(len(b[0])):
        pic[row][col] = mdb(b[row][col])
plt.imshow(pic)
plt.show()