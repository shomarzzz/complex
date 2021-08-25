import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from math import floor
import numba
# params
d = int(input("give resulotion"))
img = mpimg.imread("ImageToMap.jpg")
center = [round(len(img)/2), round(len(img[0])/2)]
new = np.zeros((len(img)+2*d, len(img[0])+2*d, 3), dtype=int)

dr = .01
di = .01
plt.imshow(img)
plt.show()
print(img.shape)
@numba.jit(nopython=True)
def imgtocomp(new, a, b):
    def func(z):
        return np.log(z)
    center = [a, b]
    a = len(img)
    b = len(img[0])
    sc = len(new[0])
    cent = [center[0]+d, center[1]+d]
    sl = len(new)
    for row in range(a):
        for col in range(b):
            real = col - center[0]
            imag = b-(center[1]+row)
            z = real*dr + imag*di*1j
            z = func(z)
            r = round(z.real/dr)
            c = round(z.imag/di)
            rr = r + cent[0]
            cc = sl - cent[1] - c
            if rr < sc and rr >= 0:
                if cc < sl and cc >= 0:
                    new[cc][rr] = img[row][col]
    return new

new = imgtocomp(new, center[0], center[1])
plt.imshow(new)
plt.show()
