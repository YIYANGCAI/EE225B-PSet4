import cv2 as cv
import numpy as np
import os
import pdb

class StatisticFilters(object):
    def __init__(self):
        super(StatisticFilters).__init__()
    
    def imread(self, path):
        img = cv.imread(path)
        img_grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img, img_grayscaled
    
    def imsave(self, path, img):
        cv.imwrite(path, img)

    def filter4e(self, img, m, n):
        # default assertion:
        # kernel size is odd: 3*3, 5*5, etc
        # channel number: 1
        kernel_size = m
        padding_size = int((kernel_size-1)/2)
        print(padding_size)
        h, w = img.shape[:2]
        img=np.pad(img, ((padding_size,padding_size),(padding_size,padding_size)), mode='edge') 
        # keep the same size of input
        out = np.zeros((h, w))
        print(img.shape)
        for row in range(0, h):
            for col in range(0, w):
                window = img[row: row+kernel_size, col:col+kernel_size]
                #####################
                # Here, users can define different operations to process the window
                # give the result to out[row, col]
                # Write the functions here
                # 
                # val = Function(window)
                # out[row, col] = val
                #
                #
                #
                # end of your code
                #####################
        out = np.uint8(out)
        return out

    def minFilter4e(self, img, m, n):
        # default assertion:
        # kernel size is odd: 3*3, 5*5, etc
        # channel number: 1
        kernel_size = m
        padding_size = int((kernel_size-1)/2)
        print(padding_size)
        h, w = img.shape[:2]
        img=np.pad(img, ((padding_size,padding_size),(padding_size,padding_size)), mode='edge') 
        # keep the same size of input
        out = np.zeros((h, w))
        print(img.shape)
        for row in range(0, h):
            for col in range(0, w):
                window = img[row: row+kernel_size, col:col+kernel_size]
                #m2 = kernel
                #val = (m1*m2).sum()
                # minimum 
                val = np.min(window)
                out[row,col] = val
        out = np.uint8(out)
        return out

    def maxFilter4e(self, img, m, n):
        # default assertion:
        # kernel size is odd: 3*3, 5*5, etc
        # channel number: 1
        kernel_size = m
        padding_size = int((kernel_size-1)/2)
        print(padding_size)
        h, w = img.shape[:2]
        img=np.pad(img, ((padding_size,padding_size),(padding_size,padding_size)), mode='edge') 
        # keep the same size of input
        out = np.zeros((h, w))
        print(img.shape)
        for row in range(0, h):
            for col in range(0, w):
                window = img[row: row+kernel_size, col:col+kernel_size]
                #m2 = kernel
                #val = (m1*m2).sum()
                # maximum 
                val = np.max(window)
                out[row,col] = val
        out = np.uint8(out)
        return out

    def medianFilter4e(self, img, m, n):
        # default assertion:
        # kernel size is odd: 3*3, 5*5, etc
        # channel number: 1
        kernel_size = m
        padding_size = int((kernel_size-1)/2)
        print(padding_size)
        h, w = img.shape[:2]
        img=np.pad(img, ((padding_size,padding_size),(padding_size,padding_size)), mode='edge') 
        # keep the same size of input
        out = np.zeros((h, w))
        print(img.shape)
        for row in range(0, h):
            for col in range(0, w):
                window = img[row: row+kernel_size, col:col+kernel_size]
                #m2 = kernel
                #val = (m1*m2).sum()
                # median
                val = np.median(window)
                out[row,col] = val
        out = np.uint8(out)
        return out


if __name__ == "__main__":
    filters = StatisticFilters()
    #pdb.set_trace()
    # problem (e)
    path_1 = './hubble.tif'
    img1, img1_gray = filters.imread(path_1)
    out_1 = filters.minFilter4e(img1_gray, 17, 17)
    filters.imsave('./result_1.jpg', out_1)
    print("sub problem e is in ./result_1.jpg")
    # problem (f)
    path_2 = './circuitboard-saltandpep.tif'
    img2, img2_gray = filters.imread(path_2)
    out_2 = filters.medianFilter4e(img2_gray, 3,3)
    print("median filter first pass result saved at ./result_2.jpg")
    filters.imsave('./result_2.jpg', out_2)
    out_3 = filters.medianFilter4e(out_2, 3,3)
    print("median filter second pass result saved at ./result_3.jpg")
    filters.imsave('./result_3.jpg', out_3)
    out_4 = filters.medianFilter4e(out_3, 3,3)
    print("median filter third pass result saved at ./result_4.jpg")
    filters.imsave('./result_4.jpg', out_4)