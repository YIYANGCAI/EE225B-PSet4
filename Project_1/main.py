import cv2 as cv
import numpy as np
import os
import pdb

class SpatialFilters(object):
    def __init__(self):
        super(SpatialFilters).__init__()
    
    def imread(self, path):
        img = cv.imread(path)
        img_grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img, img_grayscaled
    
    def imsave(self, path, img):
        cv.imwrite(path, img)

    def aMean4e(self, img, m, n):
        # default assertion:
        # kernel size is odd: 3*3, 5*5, etc
        # channel number: 1
        kernel_size = m
        padding_size = int((kernel_size-1)/2)
        print(padding_size)
        h, w = img.shape[:2]
        img=np.pad(img, ((padding_size,padding_size),(padding_size,padding_size)),'constant', constant_values=(0,0)) 
        # keep the same size of input
        out = np.zeros((h, w))
        print(img.shape)
        for row in range(0, h):
            for col in range(0, w):
                window = img[row: row+kernel_size, col:col+kernel_size]
                #m2 = kernel
                #val = (m1*m2).sum()
                val = np.mean(window)
                out[row,col] = val
        out = np.uint8(out)
        return out

    def geoMean4e(self, img, m, n):
        # default assertion:
        # kernel size is odd: 3*3, 5*5, etc
        # channel number: 1
        kernel_size = m
        padding_size = int((kernel_size-1)/2)
        print(padding_size)
        h, w = img.shape[:2]
        img=np.pad(img, ((padding_size,padding_size),(padding_size,padding_size)),'constant', constant_values=(1,1)) 
        img = img + 1
        # keep the same size of input
        out = np.zeros((h, w))
        print(img.shape)
        #pdb.set_trace()
        for row in range(0, h):
            for col in range(0, w):
                window = img[row: row+kernel_size, col:col+kernel_size]
                val = 1
                #m2 = kernel
                #val = (m1*m2).sum()
                #val = np.mean(window)
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        val *= np.float(window[i][j])
                val = pow(val,1/(kernel_size*kernel_size))
                out[row,col] = val
        out = np.uint8(out)
        return out

    def harMean4e(self, img, m, n):
        # default assertion:
        # kernel size is odd: 3*3, 5*5, etc
        # channel number: 1
        kernel_size = m
        padding_size = int((kernel_size-1)/2)
        print(padding_size)
        h, w = img.shape[:2]
        img=np.pad(img, ((padding_size,padding_size),(padding_size,padding_size)),'constant', constant_values=(1,1)) 
        img = img + 1
        # keep the same size of input
        out = np.zeros((h, w))
        print(img.shape)
        #pdb.set_trace()
        for row in range(0, h):
            for col in range(0, w):
                window = img[row: row+kernel_size, col:col+kernel_size]
                val = 0
                #m2 = kernel
                #val = (m1*m2).sum()
                #val = np.mean(window)
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        val += 1/np.float(window[i][j])
                val = (kernel_size*kernel_size) / val
                out[row,col] = val
        out = np.uint8(out)
        return out

    def ctharMean4e(self, img, m, n, q):
        # default assertion:
        # kernel size is odd: 3*3, 5*5, etc
        # channel number: 1
        kernel_size = m
        out = np.zeros(img.shape)
        padding_size = int((kernel_size-1)/2)
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if row <padding_size or row>(img.shape[0]-padding_size-1) or col < padding_size or col>(img.shape[1]-padding_size-1):
                    out[row][col]=img[row][col]
                else:
                    result_top = 0
                    result_down = 0
                    for n in range(kernel_size):
                        for m in range(kernel_size):
                            # for pepper noise, q should be positive
                            if q>0:
                                result_top +=pow(np.float(img[row-padding_size+n][col-padding_size+m]), q+1)
                                result_down +=pow(np.float(img[row-padding_size+n][col-padding_size+m]), q)
                            # for salt noise, q should be positive
                            else:
                                if img[row-padding_size+n][col-padding_size+m]==0:
                                    out[row][col] = 0
                                    break
                                else:
                                    result_top +=pow(np.float(img[row-padding_size+n][col-padding_size+m]),q+1)
                                    result_down +=pow(np.float(img[row-padding_size+n][col-padding_size+m]),q)
                        else:
                            continue
                        break
                    else:
                        if result_down !=0:
                            out[row][col] = result_top/result_down
        out = np.uint8(out)
        return out

if __name__ == "__main__":
    filters = SpatialFilters()
    # problem (a) & (e)
    
    path_1 = './circuitboard-gaussian.tif'
    img1, img1_gray = filters.imread(path_1)
    out_1 = filters.aMean4e(img1_gray, 3, 3)
    filters.imsave('./result_1.jpg', out_1)
    print("sub problem e is in ./result_1.jpg")

    # problem (b) & (f)
    path_2 = './circuitboard-gaussian.tif'
    img2, img2_gray = filters.imread(path_2)
    out_2 = filters.geoMean4e(img2_gray, 3, 3)
    filters.imsave('./result_2.jpg', out_2)
    print("sub problem f is in ./result_2.jpg")

    # problem (d) & (g)
    path_3 = './circuitboard-pepper.tif'
    img3, img3_gray = filters.imread(path_3)
    out_3 = filters.ctharMean4e(img3_gray, 3, 3, 1.5)
    filters.imsave('./result_3.jpg', out_3)
    print("sub problem g is in ./result_3.jpg")
    
    # problem (d) & (h)
    path_4 = './circuitboard-salt.tif'
    img4, img4_gray = filters.imread(path_4)
    out_4 = filters.ctharMean4e(img4_gray, 3, 3, -1.5)
    filters.imsave('./result_4.jpg', out_4)
    print("sub problem h is in ./result_4.jpg")