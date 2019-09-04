### Import our standard libraries ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Load in the dataset and split it ###

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(n_class = 2, return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 8675309, test_size = 0.5)

### Code for Question 1 ###
avg_one = np.mean(X_train[y_train == 1], axis = 0)
avg_zero = np.mean(X_train[y_train == 0], axis = 0)

plt.imshow(avg_one.reshape(8,8),cmap='Greys')
plt.show()
plt.imshow(avg_zero.reshape(8,8),cmap='Greys')
plt.show()

### Code for Question 2 ###
w, c, score = avg_one - avg_zero, 0, 0
y_hat=[]

#method 1 ------------------------------
y_hat = [1 if x > 0 else 0 for x in X_test.dot(w.T)]
#method 2 ------------------------------
#for i in X_test.dot(w):
#    if i > 0:
#        y_hat.append(1)
#    else:
#        y_hat.append(0)
#-----------------------
for i in (y_hat + y_test):
    if i != 1:
        score += 1

print("Accuracy is " + str(round(score/len(y_hat),4)*100) + " %.")


### Code for Question 3 ###
W = np.column_stack((w, np.ones(len(w))))

# reduce dimention: from 64-d to 2-d so we can visualize
#2-d, first axis/column represent 'decisions',2nd represents brightness,this way we can visulize decision-plane
X_test_2 = X_test.dot(W)

#column,row slicing in matrix
#https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
plt.scatter(X_test_2[:, -2], X_test_2[:,-1], c = y_test)
plt.figure(1), plt.title('scatter: 2-D decision visualization'), plt.ylabel('brightness of the image'), plt.xlabel('decision/weight')
