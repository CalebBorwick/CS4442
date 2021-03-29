# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:56:50 2021

@author: Caleb
"""
#import statements 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def getImageValues(image):
    dy = np.gradient(image)[0]
    dx = np.gradient(image)[1]
    ixx = dx**2
    ixy = dy*dx
    iyy = dy**2
    verticle = image.shape[0]
    horizontal = image.shape[1]
    
    return dy,dx,ixx,ixy,iyy,verticle, horizontal


def checkImageType(image):
    if len(image.shape)==3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  
    if len(image.shape)==4:
        image = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    
    return image


def harrisCorner(img, windSize, k, threshold, display=True):
    dy,dx,ixx,ixy,iyy,verticle, horizontal = getImageValues(img)
    corners = []
   
    offset =windSize //2
    
    img = checkImageType(img)
    image = img.copy()
    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    
    # find corners
    for x in range(offset, horizontal-offset):
        for y in range(offset, verticle-offset):
            windIXX = ixx[y-offset: y+offset+1, x-offset:x+offset+1]
            windIXY = ixy[y-offset: y+offset+1, x-offset:x+offset+1]
            windIYY = iyy[y-offset: y+offset+1, x-offset:x+offset+1]
            sumXX = windIXX.sum()
            sumXY = windIXY.sum()
            sumYY = windIYY.sum()

            responce = ((sumXX *sumYY) - sumXY**2) - k*((sumXX+sumYY)**2)
            
            if responce > threshold:
                corners.append([x,y,responce])
                image.itemset((y,x,0),0)
                image.itemset((y,x,1),0)
                image.itemset((y,x,2),200)
    
    if display:
        plt.imshow(image, cmap='gray')
        plt.title('Harris Corners ')
        plt.show()
        
        for ele in corners:
            print(str(ele[0])+', ' + str(ele[1]) +', '+ str(ele[2]))
    return image, corners

image = cv.imread('image1.jfif')

windowSize = 4
k=0.05
threshold = 1000000000

image, corners = harrisCorner(image, windowSize, k, threshold)