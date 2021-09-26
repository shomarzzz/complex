import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import jit

class ComplexImage:
    def __init__(self, function, IMG_PATH:str, CENTRE_INPUT:tuple, dr_di_input) -> None:
        self.function = function
        self.img = mpimg.imread(IMG_PATH)
        self.output_size = OUTPUT_SIZE
        self.centreI = CENTRE_INPUT
        self.centreO = OUTPUT_CENTRE
        if dr_di_output == None:
            self.dro = dr_di_input[0]
            self.dio = dr_di_input[1]
        else:
            self.dro = dr_di_output[0]
            self.dio = dr_di_output[1] 
        self.dri = dr_di_input[0]
        self.dii = dr_di_input[1]
        self.input = np.zeros((len(self.img), len(self.img[0])), dtype=complex)
        self.imgtocomp()
        self.input = function(self.input)
        
    def new(self, new, cent, dr, di, output_size):
        self.output_img = np.ones((output_size[0], output_size[1], self.img.shape[2]), dtype=np.uint8)*0
        self.out_img = self.comptoimg(self.img, self.input, output_img, output_size, dr, di)
        
    def imgtocomp(self):
        dr = self.dri
        di = self.dii
        centre = self.centreI
        k = len(self.img[0])
        for row in range(len(self.img)):
            for col in range(k):
                ru = (col-centre[0])*dr
                iu = (k-(centre [1]+row))*di*1j
                self.input[row][col] = ru+iu
    
    @staticmethod
    @jit(nopython=True)
    def comptoimg(img, comp, new, cent, dr, di):
        a = len(comp)
        b = len(comp[0])
        sc = len(new[0])
        sl = len(new)
        for row in range(a):
            for col in range(b):
                z = comp[row][col]
                r = round(z.real/dr)
                c = round(z.imag/di)
                rr = r + cent[0]
                cc = sl - cent[1] - c
                if rr < sc and rr >= 0:
                    if cc < sl and cc >= 0:
                        new[cc][rr] = img[row][col]
        return new

    def show(self, output=True):
        if output:
            plt.imshow(self.output_img)
        else:
            plt.imshow(self.img)
        plt.show()

    def save(self, name:str):
        plt.imsave(name, self.output_img)

if __name__=="__main__":
    def func(x):
        return np.sqrt(x)
    img = ComplexImage(func, "ImageToMap.jpg", (1000, 1000), (500, 500), (200, 200), (.01, .01))
    img.show(True)
    img.save("output.jpg")
