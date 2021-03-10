# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:49:48 2021

@author: Ian Borwick 
"""
#import statements 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold

#-----------------------------------------------------------------------------
#Question 2
#-----------------------------------------------------------------------------

#applies the second degree polynoimial function to the x values
def calcSecondPolyCoeff(x, coeffs):
    fx=[]
    for i in range(len(x)):
        fx.append(coeffs[2] * x[i]**2 + coeffs[1] * x[i] + coeffs[0])
    return fx

#applies the third degree polynoimial function to the x values
def calcThirdPolyCoeff(x, coeffs):
    fx=[]
    for i in range(len(x)):
        fx.append(coeffs[3] * x[i]**3 + coeffs[2] * x[i]**2 + coeffs[1] * x[i] + coeffs[0])
    return fx

#applies the fourth degree polynoimial function to the x values
def calcFourthPolyCoeff(x, coeffs):
    fx=[]
    for i in range(len(x)):
        fx.append(coeffs[4] * x[i]**4 +coeffs[3] * x[i]**3 + coeffs[2] * x[i]**2 + coeffs[1] * x[i] + coeffs[0])
    return fx

#computes the error on the data vs the actual
def computeError(m, h, y):
    error = 0
    for i in range(m[0]):
        out =h[i] - y[i]
        out *= out
        error += out 
    error = error/m
    return error

#makes the scatter plot
def makePlots(x, h, xT, yT, testOrTrain, questionNum, model):
    plt.plot(x, h)
    plt.scatter(xT,yT)
    plt.xlabel('x'+ testOrTrain)
    plt.ylabel('y'+ testOrTrain)
    plt.title(questionNum +" x"+ testOrTrain + " vs. y"+ testOrTrain +" for "+model)
    plt.show()

#-----------------------------------------------------------------------------
#a)
#-----------------------------------------------------------------------------
# load in the data files

xTrain = np.loadtxt("hw1xtr.dat")
yTrain = np.loadtxt("hw1ytr.dat")
xTest = np.loadtxt("hw1xte.dat")
yTest = np.loadtxt("hw1yte.dat")

#ploting train data
makePlots(0, 0, xTrain, yTrain, "Train","2a)", "no additional x")

#plotting test data
makePlots(0, 0, xTest, yTest, "Test","2a)", "no additional x")
print('-------------------------------------')

#-----------------------------------------------------------------------------
#b)
#-----------------------------------------------------------------------------

#adding a column of ones to the features 
n=xTrain.shape
xTrainMod = np.c_[np.ones(n),xTrain]

m=xTest.shape
xTestMod = np.c_[np.ones(m),xTest]

#linear regression forumla to obtain 2 dimensional weight vector 
w= (np.linalg.inv(np.transpose(xTrainMod) @ xTrainMod)) @ ((np.transpose(xTrainMod)) @ yTrain)
h = (xTrainMod @ w)

#plotting linear regression line and training data
makePlots(xTrain, h, xTrain, yTrain, "Train","2b)", "x")

#compute the average error on the training set
trainingError = computeError(n, h, yTrain)
print('2b) The training error for this regression model is:', trainingError)

#-----------------------------------------------------------------------------
#c)
#-----------------------------------------------------------------------------

#applies coefficents to data
h = (xTestMod @ w)

#plotting linear regression line and test data
makePlots(xTest, h, xTest, yTest, "Test","2c)", "x")

#compute the average error on the test set
testError = computeError(m, h, yTest)
print('2c)The test error for this regression model is:', testError)


print('-------------------------------------')

#-----------------------------------------------------------------------------
#d)
#-----------------------------------------------------------------------------

# add x^2 as a feature to the input 
xTrainMod = np.c_[xTrainMod, np.square(xTrain)]
xTestMod = np.c_[xTestMod, np.square(xTest)]

#repeat step b
#linear regression forumla to obtain 2 dimensional weight vector 
w= (np.linalg.inv(np.transpose(xTrainMod) @ xTrainMod)) @ ((np.transpose(xTrainMod)) @ yTrain)

#calculate the polynomial 
h = calcSecondPolyCoeff(xTrain, w)
poly = np.poly1d(np.polyfit(xTrain, h, 3))

#calculate new values
x1 = np.linspace(xTrain.min(), xTrain.max())
y1 = poly(x1)

#plotting linear regression line and training data
makePlots(x1, y1, xTrain, yTrain, "Train","2d)", "x^2")

#compute the average error on the training set
trainingError = computeError(n, h, yTrain)

print('2d) The training error for the x^2  regression model is:', trainingError)

#repeat step c

#calculate the polynomial 
h = calcSecondPolyCoeff(xTest, w)
poly = np.poly1d(np.polyfit(xTest, h, 3))

#calculate new values
x1 = np.linspace(xTest.min(), xTest.max())
y1 = poly(x1)

#plotting linear regression line and test data
makePlots(x1, y1, xTest, yTest, "Test","2d)", "x^2")

#compute the average error on the test set
testError = computeError(m, h, yTest)

print('2d) The test error for the x^2 regression model is:', testError)
print('2d) The x^2 regression is a better fit than linear')

print('-------------------------------------')

#-----------------------------------------------------------------------------
#e)
#-----------------------------------------------------------------------------

# add x^3 as a feature to the input 
xTrainMod = np.c_[xTrainMod, (np.square(xTrain)*xTrain)]
xTestMod = np.c_[xTestMod, (np.square(xTest)*xTest)]

#repeat step b

#regression forumla to obtain 2 dimensional weight vector 
w= (np.linalg.inv(np.transpose(xTrainMod) @ xTrainMod)) @ ((np.transpose(xTrainMod)) @ yTrain)

#calculate the polynomial 
h = calcThirdPolyCoeff(xTrain, w)
poly = np.poly1d(np.polyfit(xTrain, h, 4))

#calculate new values
x1 = np.linspace(xTrain.min(), xTrain.max())
y1 = poly(x1)

#plotting linear regression line and training data
makePlots(x1, y1, xTrain, yTrain, "Train","2e)", "x^3")

#compute the average error on the training set
trainingError = computeError(n, h, yTrain)

print('2e) The training error for the x^3  regression model is:', trainingError)

#repeat step c

#calculate the polynomial 
h = calcThirdPolyCoeff(xTest, w)
poly = np.poly1d(np.polyfit(xTest, h, 4))

#calculate new values
x1 = np.linspace(xTest.min(), xTest.max())
y1 = poly(x1)

#plotting linear regression line and test data
makePlots(x1, y1, xTest, yTest, "Test", "2e)", "x^3")

#compute the average error on the test set
testError = computeError(m, h, yTest)

print('2e) The test error for the x^3 regression model is:', testError)
print('2e) The x^3 regression is a better fit than linear and better than x^2')


print('-------------------------------------')
#-----------------------------------------------------------------------------
#f)
#-----------------------------------------------------------------------------

# add x^4 as a feature to the input 
xTrainMod = np.c_[xTrainMod, (np.square(xTrain)*(np.square(xTrain)))]
xTestMod = np.c_[xTestMod, (np.square(xTest)*(np.square(xTest)))]

#repeat step b

#linear regression forumla to obtain 2 dimensional weight vector 
w= (np.linalg.inv(np.transpose(xTrainMod) @ xTrainMod)) @ ((np.transpose(xTrainMod)) @ yTrain)

#calculate the polynomial 
h = calcFourthPolyCoeff(xTrain, w)
poly = np.poly1d(np.polyfit(xTrain, h, 5))

#calculate new values
x1 = np.linspace(xTrain.min(), xTrain.max())
y1 = poly(x1)

#plotting linear regression line and training data
makePlots(x1, y1, xTrain, yTrain, "Train", "2f)", "x^4")

#compute the average error on the training set
trainingError = computeError(n, h, yTrain)

print('2f) The training error for the x^4  regression model is:', trainingError)

#repeat step c

#calculate the polynomial 
h = calcFourthPolyCoeff(xTest, w)
poly = np.poly1d(np.polyfit(xTest, h, 5))

#calculate new values
x1 = np.linspace(xTest.min(), xTest.max())
y1 = poly(x1)

#plotting linear regression line and test data
makePlots(x1, y1, xTest, yTest, "Test","2f)", "x^4")

#compute the average error on the test set
testError = computeError(m, h, yTest)
print('2f) The test error for the x^4 regression model is:', testError)
print('2f) The x^4 regression is a better fit than linear but worse than the x^2 and x^3')

print('-------------------------------------')


#-----------------------------------------------------------------------------
#Question 3
#-----------------------------------------------------------------------------
 
#-----------------------------------------------------------------------------
#a)
#-----------------------------------------------------------------------------
#creating chart for error function

tempW = np.transpose(xTrainMod) @ xTrainMod
identitySize = int(np.sqrt(tempW.size))
lambdas = [0.01, 0.1, 1, 10, 100, 1000,10000]
errorTrain = plt.scatter(xTrain, yTrain)
plt.xlabel('xTrain')
plt.ylabel('yTrain')
plt.title("3a) xTrain vs. yTrain for error")
print("3a)Train Error values:")

identityMatrix = np.identity(identitySize)
identityMatrix[0][0]=0.0
#for loop to determine the best lambda and plot it

for num in lambdas:
    w= (np.linalg.inv(tempW + (identityMatrix*num)))@ (np.transpose(xTrainMod)@ yTrain)
    #calculate the polynomial 
    h = calcFourthPolyCoeff(xTrain, w)
    poly = np.poly1d(np.polyfit(xTrain, h, 5))
    
    #calculate new values
    x1 = np.linspace(xTrain.min(), xTrain.max())
    y1 = poly(x1)
    plt.plot(x1, y1)
    
    # compute labmbda error
    error = computeError(n, h, yTrain)
    print(num,": ", error)

plt.show()

print('-------------------------------------')

#creating chart for error function
plt.scatter(xTest, yTest)
plt.xlabel('xTest')
plt.ylabel('yTest')
plt.title("3a) xTest vs. yTest for error")
print("3a)Test Error values:")
weightValues = []

#for loop to determine the best lambda and plot it
for num in lambdas:
    w= (np.linalg.inv(tempW + (identityMatrix*num)))@ (np.transpose(xTrainMod)@ yTrain)
    weightValues.append(w)
    #calculate the polynomial 
    h = calcFourthPolyCoeff(xTest, w)
    poly = np.poly1d(np.polyfit(xTest, h, 5))
    
    #calculate new values
    x1 = np.linspace(xTest.min(), xTest.max())
    y1 = poly(x1)
    plt.plot(x1, y1)
    
    # compute labmbda error
    error = computeError(m, h, yTest)
    print(num,": ",error)

plt.show()

print("The best lambda fit is for 0.1")

print('-------------------------------------')

#-----------------------------------------------------------------------------
#b)
#-----------------------------------------------------------------------------

#Plot the weight of each parameter including bias term as a function of log lambda

plt.plot(lambdas, weightValues)
plt.xscale('log')
plt.xlabel('lambdas')
plt.ylabel('weight values')
plt.title('3b) weight values vs lambdas')
plt.show()


#-----------------------------------------------------------------------------
#c)
#-----------------------------------------------------------------------------

#perform 5 fold cross validation and find the best lambda
kf = KFold(n_splits= 5, shuffle=False);
totalError = []
for train, test in kf.split(xTrainMod):
    x_train, x_test = xTrainMod[train], xTrain[test]
    y_train, y_test = yTrain[train], yTrain[test]
    tempW = np.transpose(x_train) @ x_train
    identitySize = int(np.sqrt(tempW.size))
    identityMatrix = np.identity(identitySize)
    identityMatrix[0][0]=0.0
    lamberror = []
    for num in lambdas:
        w= (np.linalg.inv(tempW + (identityMatrix*num)))@ (np.transpose(x_train)@ y_train)
        #calculate the polynomial 
        h = calcFourthPolyCoeff(x_test, w)

        # compute labmbda error
        error = computeError(y_test.shape, h, y_test)
        lamberror.append(error)
    totalError.append(lamberror)

#total all the errors by index/lambda
avgError = totalError[0]
errorCount = 1
for j in range(1,len(totalError)):
    for i in range(len(totalError[j])):
        avgError[i]+= totalError[j][i]
    errorCount += 1
    
#divide by the total number of errors to get the average error per lambda
print("3c)Average Errors")
for i in range(len(avgError)):
    avgError[i]=avgError[i]/errorCount
    print(lambdas[i],": ", avgError[i])
    
#show the average error on the validation set as a function of lambda

plt.plot(lambdas, avgError)
plt.xscale('log')
plt.xlabel("lambdas")
plt.ylabel("average errors")
plt.title("3c) lambdas vs average errors")
plt.show()
print("3c)The best lambda for c) was 0.01 and for a) it was 0.1 so it did changed")


plt.scatter(xTest, yTest)
plt.xlabel('xTest')
plt.ylabel('yTest')
plt.title("3c) xTest vs. yTest for l2-regularized 4th order polynomial regression")

tempW = np.transpose(xTrainMod) @ xTrainMod
identitySize = int(np.sqrt(tempW.size))

w= (np.linalg.inv(tempW + (np.identity(identitySize)*(np.log10(1)))))@ (np.transpose(xTrainMod)@ yTrain)
#calculate the polynomial 
h = calcFourthPolyCoeff(xTest, w)
poly = np.poly1d(np.polyfit(xTest, h, 5))

#calculate new values
x1 = np.linspace(xTest.min(), xTest.max())
y1 = poly(x1)
plt.plot(x1, y1)
plt.show()
