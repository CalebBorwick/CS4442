# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:07:14 2021

@author: Caleb
"""
#import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------
#3a)
#------------------------------------------------------------------------------

#Load in data
imageRaw = np.loadtxt("faces.dat")
imageArray = imageRaw.reshape(400,64,64);

#find the 100th face
face100 = np.rot90(imageArray[99], 3)
plt.imshow(face100, aspect="equal");
plt.title("100th face")
plt.show()


#------------------------------------------------------------------------------
#3b)
#------------------------------------------------------------------------------

#remove mean 
imageArrayMean =imageArray - imageArray.mean(0)
plt.imshow(np.rot90(imageArrayMean[99],3), aspect="equal");
plt.title("100th face mean removed")
plt.show()

#------------------------------------------------------------------------------
#3c)
#------------------------------------------------------------------------------

#reshape data into a 2D array
imageArray2D = imageArrayMean.reshape(400,4096)
covMatrix = np.cov(np.transpose(imageArray2D))

#perform PCA
eigenValues, eigenVectors = np.linalg.eig(covMatrix)
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]

#sort Eigenvalues
eigenPairs.sort(key = lambda x: x[0], reverse =True)

#plot the values
plt.plot(np.arange(4096), eigenValues);
plt.title("eigen vectors vs eigen value")
plt.show()

#------------------------------------------------------------------------------
#3d)
#------------------------------------------------------------------------------

#this means that the nullspace is nontrvial in other terms the image is slightly compressed
#as it is of a lower dimension as well this means the matrix is not inveratble 



#------------------------------------------------------------------------------
#3e)
#------------------------------------------------------------------------------
y = np.zeros((400,1))
x = imageArray2D
xTrain, xTest, yTrain, yTest = train_test_split(x, y)

pca = PCA().fit(xTrain)
plt.title("explained varience")
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.show()

print("\nIn order to keep 95% varience we are reqiured to keep a dimensionality of ~140 meaning we only need to keep ~100 components")

keepVar = 0.95;
requiredVar = keepVar * sum(eigenValues)

requiredDimension = 0;
var = 0
for i in range(len(eigenPairs)):
    var += eigenPairs[i][0]
    if var >= requiredVar :
        requiredDimension = i+1;
        break

print("Required Dimensionality: "+ str(requiredDimension))

#------------------------------------------------------------------------------
#3f)
#------------------------------------------------------------------------------

projMatrix = np.empty(shape = (imageArray2D.shape[1], requiredDimension))

for i in range(requiredDimension):
    eigenVector = eigenPairs[i][1]
    projMatrix[:, i] = eigenVector
    
base = projMatrix.reshape(64,64,requiredDimension)

for i in range(5):
    plt.imshow(np.rot90(base[:,:,i], 3))
    plt.title("Eigen value: "+ str(i+1))
    plt.show()


#------------------------------------------------------------------------------
#3g)
#------------------------------------------------------------------------------

compValues = [10,100,200,399]

for val in compValues:
    projMatrix = np.empty(shape=(x.shape[1], val))
    for j in range(val):
        eigenVector = eigenPairs[j][1]
        projMatrix[:, j] = eigenVector
        
    projData = x.dot(projMatrix)
    
    projImage = np.expand_dims(projData[j],0)
    
    recImage = projImage.dot(np.transpose(projMatrix)).reshape(64,64)

    plt.title("Original Image")
    plt.imshow(np.rot90(imageRaw[j].reshape(64,64),3), aspect="equal");
    plt.show()
    
    plt.title("reconstructed image with " +str(val) + " components")
    plt.imshow(np.rot90(recImage,3))
    plt.show()
