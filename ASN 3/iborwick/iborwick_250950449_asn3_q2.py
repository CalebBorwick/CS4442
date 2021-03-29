# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:56:50 2021

@author: Ian Borwick - 250950449
"""

#import statements 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math



#creates a guassian kernel of size
def getGaussKern(size, sigma, display=False):
    gaussian1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        gaussian1D[i] = normalize(gaussian1D[i], 0, sigma)
    gaussian2D = np.outer(gaussian1D.T, gaussian1D.T)
 
    gaussian2D *= 1.0 / gaussian2D.max()
 
    if display:
        plt.imshow(gaussian2D, interpolation='none', cmap='gray')
        plt.title("Kernel of size: ( {}X{} )".format(size, size))
        plt.show()
 
    return gaussian2D


#calculates the univariate normal distribution 
def normalize(x, mu, sd):
    temp =  1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
    return temp


#applies the gaussian blur
def convolution(img, kern, ave = False, display=False):
    if len(img.shape)==3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   
        
    imRow,imCol = img.shape
    kernRow, kernCol = kern.shape
    
    result = np.zeros(img.shape)
    
    padV = int((kernRow -1)/2)
    padH = int((kernCol -1)/2)
    
    padImage = np.zeros((imRow + (2*padV), imCol + (2 * padH)))
    padImage[padV:padImage.shape[0]- padV, padH: padImage.shape[1]-padH] = img
    for row in range(imRow):
        for col in range(imCol):
            temp = padImage[row:row + kernRow, col:col + kernCol]
            result[row,col]= np.sum(kern * temp)
            if ave:
                 result[row,col] /= kern.shape[0]*kern.shape[1]

    if display:
        plt.imshow(img, cmap='gray')
        plt.title('Convolution')
        plt.show()
        
        plt.imshow(result, cmap='gray')
        plt.title('Gaussian Blur Applied')
        plt.show()
    return result

 
#intermediate function to make the gaussian kernel and apply it the the image
def gaussBlur(img, kernSize, sigma, display=False):
    kern = getGaussKern(kernSize, sigma, display= True)
    return convolution(img, kern, ave= True, display=True)


# find the vertical and horizontal edges of the image and return the gradient magnitude and direction
def sobelEdge(img, kern= None, convert=False, display=False):
    if kern ==None:
            kern = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
   
    x = convolution(img, kern)     
    y = convolution(img, np.flip(kern.T, axis=0))

        
    gradMagnitude = np.sqrt(x**2 + y**2)
    gradMagnitude *= 255.0 / gradMagnitude.max()
        
    gradDirection = np.arctan2(y,x)
    
    if convert:
        gradDirection = np.rad2deg(gradDirection)
        gradDirection += 180
        
    if display:
        plt.imshow(x, cmap='gray')
        plt.title('Horizontal Edge')
        plt.show()
        
        plt.imshow(y, cmap='gray')
        plt.title('Verticle Edge')
        plt.show()
        
        plt.imshow(gradMagnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()
        
        plt.imshow(gradDirection, cmap='gray')
        plt.title("Gradient Direction")
        plt.show()
    
    return gradMagnitude, gradDirection


#eliminates pixels that are not local maxima of the magnitude in the direction of the gradient
def suppression(gradMag, gradDir, display = False):
    imRow, imCol = gradMag.shape
    result = np.zeros(gradMag.shape)
        
    for row in range(1,imRow -1):
        for col in range(1, imCol-1):
            direct = gradDir[row,col]
            
            if (0 <= direct < 180 / 8) or (15 * 180 / 8 <= direct <= 2 * 180):
                before = gradMag[row, col - 1]
                after = gradMag[row, col + 1]
     
            elif (180 / 8 <= direct < 3 * 180 / 8) or (9 * 180 / 8 <= direct < 11 * 180 / 8):
                before = gradMag[row + 1, col - 1]
                after = gradMag[row - 1, col + 1]
     
            elif (11 * 180 / 8 <= direct < 13 * 180 / 8) or (3 * 180 / 8 <= direct < 5 * 180 / 8):
                before = gradMag[row - 1, col]
                after = gradMag[row + 1, col]
     
            else:
                before = gradMag[row - 1, col - 1]
                after = gradMag[row + 1, col + 1]
     
            if gradMag[row, col] >= before and gradMag[row, col] >= after:
                result[row, col] = gradMag[row, col]
 
    if display:
        plt.imshow(result, cmap='gray')
        plt.title("Suppression")
        plt.show()   
    return result


#defines edges as either weak or strong to allow for better vizualization and cleaner results
def threshold(img, lowThres, highThresh, weak, display=False):
    result = np.zeros(img.shape)
    
    strong = 255
    
    sRow, sCol = np.where(img>=highThresh)
    wRow, wCol = np.where((img<=highThresh) & (img>=lowThres))
    
    result[sRow, sCol] = strong
    result[wRow, wCol] = weak
    
    if display:
        plt.imshow(result, cmap='gray')
        plt.title("threshold")
        plt.show()
    
    return result 


#helper method to reduce lines in hysteresis
def helper(copy, weak, row, col):
    if copy[row,col]==weak:
        if copy[row,col+1]==255 or copy[
                row, col - 1] == 255 or copy[
                    row - 1, col] == 255 or copy[
                        row + 1, col] == 255 or copy[
                            row - 1, col - 1] == 255 or copy[
                                row + 1, col - 1] == 255 or copy[
                                    row - 1, col + 1] == 255 or copy[
                                        row + 1, col + 1] == 255:
            copy[row, col] = 255
        else:
            copy[row,col] =0
    return copy
    

#selects pixels that are apart of the edges 
def hysteresis(img, weak):
    imRow, imCol = img.shape
    
    #checks top to bottom edges
    tb = img.copy()
    for row in range(1, imRow):
        for col in range(1,imCol):
            tb = helper(tb, weak, row, col)
    
    #checks bottom to top edges
    bt = img.copy()
    for row in range(imRow -1,0,-1):
        for col in range(imCol-1,0,-1):
            bt = helper(bt, weak, row, col)
                    
    #checks right to left edges
    rl = img.copy()
    for row in range(1, imRow):
        for col in range(imCol-1,0,-1):
            rl = helper(rl, weak, row, col)

    #checks left to right edges
    lr = img.copy()
    for row in range(imRow-1,0,-1):
        for col in range(1,imCol):
            lr = helper(lr, weak, row, col)

                    
    #combined results and return
    result = tb+bt+rl+lr
    result[result> 255] = 255
    return result 
    

#main functio nto call with all sub functions within
def canny(image, sigma, lowThresh, highThresh):
    kernalSize = 5 # can be 5+ 10*num
    weak = 100
    gaussianBlur = gaussBlur(image, kernalSize, sigma, display=True)
    gradientMagnitude, gradientDirection = sobelEdge(gaussianBlur,convert =True, display=True)
    nonMaxSuppression = suppression(gradientMagnitude, gradientDirection, display = True)
    thresh = threshold(nonMaxSuppression, lowThresh, highThresh, weak, display=True)
    hyster = hysteresis(thresh, weak)
    
    plt.imshow(hyster, cmap='gray')
    plt.title("Canny Edge")
    plt.show()




image = cv.imread('image1.jfif')
sigma = 3
lowThresh = 2
highThresh = 20

canny(image, sigma, lowThresh, highThresh)



