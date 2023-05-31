
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import csv
import copy
import math
from lab_utils_logistic import sigmoid

##plt.style.use('./deeplearning.mplstyle')

def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):

    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        aux = np.where((f_wb_i) > 0.0000001, (f_wb_i), -10)
        aux2 = np.where((1-f_wb_i) > 0.0000001, (1-f_wb_i), -10)
        cost +=  -y[i]*np.log(aux, out=aux,where=aux>0) - (1-y[i])*np.log(aux2, out=aux2,where=aux2>0)      #scalar
             
    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost    

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

def gradient_descent(X, y, w_in, b_in, alpha, r_lambda, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      r_lambda (float)     : Regularization rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b, r_lambda)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic_reg(X, y, w, b, r_lambda) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history

def y_change(y, cl):
    """
    Creates an independent y vector that only holds 1's for
    the selected class and zero for the rest
    
    Args:
      y (ndarray (m,)) : target values
      cl (scalar)      : The class we are studying.
      
    Returns:
      y_pr (ndarray (n,))   : Array holding only 1's for the 
                              analyzed class.
    """
    y_pr=[]
    for i in range(0, len(y)):
        if y[i] == cl:
            y_pr.append(1)
        else:
            y_pr.append(0)
    return y_pr

def find_param(X, y):
    """
    Creates the w_i vector for the given class.
    
    Args:
      X (ndarray (m,n)    : Data, m examples with n features
      y (ndarray (m,))    : Target values
      
    Returns:
      theta_list (ndarray (n,)) : This is a matrix that will hold a row for the w values
                                  for every i class. 
    """
    w_in = np.random.rand(X.shape[1])
    b_in = 0.5

    alph = 0.1
    r_lambda = 0.7
    iters = 1000

    y_uniq = list(set(y.flatten()))
    theta_list = []
    for i in y_uniq:
        y_tr = pd.Series(y_change(y, i))
        # y_tr = y_tr[:, np.newaxis]
        np.array(y_tr)[:, np.newaxis]
        print(f"\n\nWe will find the weights for class: {i}")
        theta1, _ , _ = gradient_descent(X, y_tr, w_in, b_in, alph, r_lambda, iters) 
        theta_list.append(theta1)
    return theta_list

def predict(theta_list, X, y):
    y_uniq = list(set(y.flatten()))
    y_hat = [0]*len(y)
    for i in range(0, len(y_uniq)):
        y_tr = y_change(y, y_uniq[i])
        # y1 = sigmoid(x, theta_list[i])
        y1 = sigmoid(np.dot(X, theta_list[i]))
        for k in range(0, len(y)):
            if y_tr[k] == 1 and y1[k] >= 0.5:
                y_hat[k] = y_uniq[i]
    return y_hat

def extractFeature(img):
    res = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsvR = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    satR = cv2.mean(hsvR[:,:,0])[0]
    hsvG = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    satG = cv2.mean(hsvG[:,:,1])[0]
    hsvB = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    satB = cv2.mean(hsvB[:,:,2])[0]
    std_dev = cv2.meanStdDev(gray)
    #std_dev = round(std_dev[0][0], 4)

    #Histogram
    histR = cv2.calcHist([img], [2], None, [256], [0, 256])
    histG = cv2.calcHist([img], [1], None, [256], [0, 256])
    histB = cv2.calcHist([img], [0], None, [256], [0, 256])

    #res.append(std_dev)

    for i in histR:
        res.append(float(i))
    for i in histG:
        res.append(float(i))
    for i in histB:
        res.append(float(i))

    
    res.append(satR)
    res.append(satG)
    res.append(satB)

    print(res)

    return res

X = []
Y = []

R = open('featuresHistR.csv', "r")
G = open("featuresHistG.csv", "r")
B = open("featuresHistB.csv", "r")
F = open("features.csv", "r")


R.readline()
G.readline()
B.readline()
F.readline()

for line in F:
    X.append(line.split(",")[2:10])


for i, line in enumerate(R):
    for j in range(2, 258):
        X[i].append(float(line.split(",")[j]))


for i, line in enumerate(G):
    for j in range(2, 258):
        X[i].append(float(line.split(",")[j]))


for i, line in enumerate(B):
    for j in range(2, 258):
        X[i].append(float(line.split(",")[j]))

X = np.array(X, dtype=float)
for x in X:
    x = x.flatten()

F = open("features.csv", "r")
F.readline()
for line in F:
    if line.split(",")[1] == "attackOnTitan_fotos":
        Y.append(1)
    elif line.split(",")[1] == "deathNote_fotos":
        Y.append(2)
    elif line.split(",")[1] == "Evangelion":
        Y.append(3)
    elif line.split(",")[1] == "KimetsuNoYaiba":
        Y.append(4)
    elif line.split(",")[1] == "LOTR":
        Y.append(5)
    elif line.split(",")[1] == "NBA":
        Y.append(6)
    elif line.split(",")[1] == "onePiece_fotos":
        Y.append(7)
    elif line.split(",")[1] == "SNKRS":
        Y.append(8)
    elif line.split(",")[1] == "StarWars":
        Y.append(9)

Y = np.array(Y)
# print("X", X)
# print("Y",Y)

#theta_list = find_param(X, Y)

# w = open("theta_List.csv","w")
# for list in theta_list:
#     for number in list:
#         w.write(str(number) + ",")
#     w.write("\n")

theta_list = []

w = open("./data/theta_List.csv", "r")
for line in w:
    iterationLine = line.replace("\n", "").split(",")
    theta_list.append(iterationLine[:len(iterationLine)-1])

theta_list = np.array(theta_list,  dtype=float)

X = []

image = cv2.imread(
    './fotos/undefined.jpg', cv2.IMREAD_COLOR    
)
X.append(extractFeature(image))
X.append(extractFeature(image))
X.append(extractFeature(image))
X = np.array(X)

Y = [1,2,3,4,5,6,7,8,9]
Y = np.array(Y).flatten()

# Create an 11 by 3 random matrix
# x_train = np.random.rand(11,3)
# y_train = np.array([0,  0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

y_hat = predict(theta_list, X, Y)

if y_hat[0] == 1:
    print("Attack on titan")
elif y_hat[1] == 2:
    print("Death Note")
elif y_hat[2] == 3:
    print("Evangelion")
elif y_hat[3] == 4:
    print("Kimetsu")
elif y_hat[4] == 5:
    print("LOTR")
elif y_hat[5] == 6:
    print("NBA")
elif y_hat[6] == 7:
    print("One Piece")
elif y_hat[7] == 8:
    print("SNKRS")
elif y_hat[8] == 9:
    print("StarWars")

#Plotting the actual and predicted values
f1 = plt.figure()
c = [i for i in range (1,len(Y)+1,1)]
plt.plot(c,Y,color='r',linestyle='-')
plt.plot(c,y_hat,color='b',linestyle='-')
plt.xlabel('Value')
plt.ylabel('Class')
plt.title('Actual vs. Predicted')
plt.show()

# #Plotting the error
f1 = plt.figure()
c = [i for i in range(1,len(Y)+1,1)]
plt.plot(c,Y-y_hat,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()
