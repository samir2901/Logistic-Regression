'''
Here X is the input which denotes the age of a person and 
y denotes the output which denotes whether a person will take up 
insurance or not. 1 -> will take and 0 -> will not take up insurance

Equation : y_p = sigmoid(theta0 + theta1 * x)
D1 = (1/N) * sum[(y_p[i] - y[i]) * x[i] ]  this is the differentiation of the cost function w.r.t theta1
D0 = (1/N) * sum[(y_p[i] - y[i]) ]  this is the differentiation of the cost function w.r.t theta0
'''
import numpy as np 
import time
#import matplotlib.pyplot as plt 

#training data
X_train = np.array([22,25,47,52,46,56,55,60,62,61,18,28,27,29,49,55,25,58])
y_train = np.array([0,0,1,0,1,1,0,1,1,1,0,0,0,0,1,1,1,1])

#testing data
X_test = np.array([19,18,21,26,40,45,50,54,23])
y_test = np.array([0,0,0,0,1,1,1,1,0])

'''
plt.scatter(X_train,y_train,c='red',marker='o')
plt.scatter(X_test,y_test,c='green',marker='*')
plt.show()
'''

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def LossFunction(y_p, y):
    N = len(y)
    z = 0
    for i in range(N):
        z += -y[i] * np.log(y_p[i]) - (1 - y[i]) * np.log(1 - y_p[i])
    loss = z / N 
    return loss 



def LogisticRegression(X,y,steps=100000,learning_rate=0.01):    
    theta0 = 0
    theta1 = 0
    N = len(X)
    for i in range(steps):
        y_p = sigmoid(theta0 + theta1 * X)
        cost = LossFunction(y_p,y)
        D1 = (1/N) * sum(((y_p - y) * X))
        D0 = (1/N) * sum((y_p - y))
        theta0 = theta0 - learning_rate * D0
        theta1 = theta1 - learning_rate * D1
        print("theta1={}, theta0={}, loss={}, step={}".format(theta1,theta0,cost,i))
    return theta0, theta1
    

def predictProbablility(x, theta1, theta0):
    return sigmoid(theta1 * x + theta0)

def predict(x, theta1, theta0):
    if predictProbablility(x, theta1, theta0) > 0.5:
        return 1
    else:
        return 0

t = time.time()
 
theta = LogisticRegression(X_train,y_train)
print("Coefficients:",theta)

correct = 0
total = 0
for i in range(len(X_test)):
    predictionProbab = predictProbablility(X_test[i], theta[1], theta[0])
    prediction = predict(X_test[i], theta[1], theta[0])
    print("Feature Value:",X_test[i],"Class Probability:",predictionProbab,", Predicted Class:",prediction,", Actual Class:",y_test[i])
    if prediction==y_test[i]:
        correct += 1
    total += 1

print("Accuracy:",correct/total)
print("Time taken:",time.time()-t)










