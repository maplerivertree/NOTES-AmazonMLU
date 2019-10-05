	
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
eider.s3.download('s3://eider-datasets/mlu/Book_Ratings.csv', '/tmp/Book_Ratings.csv')
data = pd.read_csv('tmp/Book_Ratings.csv', dtype = {'User': str, 'ASIN': str, 'Rating': np.int})
train, test = train_test_split(data, random_state = 8675309, stratify = data['ASIN'])
 
binarizer = Binarizer(threshold=0, copy=True)
 
S_train = train.pivot_table(index='User', columns='ASIN', values='Rating', fill_value=0).as_matrix()
R_train = binarizer.fit_transform(S_train)
S_test = test.pivot_table(index='User', columns='ASIN', values='Rating', fill_value=0).as_matrix()
R_test = binarizer.fit_transform(S_test)
 
uniqueUsers = data['User'].unique().tolist()
uniqueASINs = data['ASIN'].unique().tolist()
numUser = len(uniqueUsers)
numASIN = len(uniqueASINs)
num_train = train.shape[0]
num_test = test.shape[0]
 
#For F only use actual ratings, i.e don't calculate mean over the filled in zeros
dumb_F = np.matrix(np.true_divide(S_train.sum(axis = 0),(S_train != 0).sum(axis = 0)))
#really_dumb_F = np.matrix(np.true_divide(
dumb_A = np.matrix(np.ones(R_train.shape[0])).T
dumb_P = dumb_A.dot(dumb_F)
 
def get_RMS(R, S, P):
    n = P.shape[0]
    return np.linalg.norm(np.multiply(R, (P - S)), ord=2)/np.sqrt(float(n))
 
dumb_RMS = get_RMS(R_test, S_test, dumb_P)
print("b) The root mean squared error of the baseline model on the test set was: {:.4f}".format(dumb_RMS))
    
k = 2
F = 2*np.random.rand(k,numASIN)
A = 2*np.random.rand(numUser, k)
P = A.dot(F)
init_RMS = get_RMS(R_test, S_test, P)
print("b) The root mean squared error of x0 on the test set was: {:.4f}".format(init_RMS))



# ==========
	
def get_dydA(k, R, S, A, F):
    grad = (-2.0/k)*(np.multiply(R, (S - A.dot(F))).dot(F.T))
    return grad
    
def get_dydF(k, R, S, A, F):
    grad =  (-2.0/k)*(A.T.dot(np.multiply(R, (S - A.dot(F)))))
    return grad
 
def P_gradient(R_train, R_test, S_train, S_test, A, F, k, n, step):
    train_RMS = [get_RMS(R_train, S_train, P)]
    test_RMS = [get_RMS(R_test, S_test, P)]
    for each in range(n):
        A -= step*get_dydA(k, R_train, S_train, A, F)
        F -= step*get_dydF(k, R_train, S_train, A, F)
        train_RMS.append(get_RMS(R_train, S_train, A.dot(F)))
        test_RMS.append(get_RMS(R_test, S_test, A.dot(F)))
    return {'test_RMS': test_RMS, 
            'train_RMS': train_RMS,
            'A': A, 
            'F': F}
    
k=2
n=400
step=0.0004 #no convergence to solution for larger step functions, seems weird
results = P_gradient(R_train, R_test, S_train, S_test, A, F, k, n, step)

	
print(min(results['test_RMS']), (0.3947-min(results['test_RMS']))*100/0.3947, results['test_RMS'][-1], (0.3947-results['test_RMS'][-1])*100/0.3947)

	
fig, ax = plt.subplots(1,1, figsize=(10, 6))
ax.plot(range(n+1), results['train_RMS'], 
    label=u'RMS (Train): k=2, \u03B7={}'.format(step))
ax.plot(range(n+1), results['test_RMS'], 
    label=u'RMS (Test): k=2, \u03B7={}'.format(step))
ax.legend()
_ = fig.suptitle("Loss function on test and training sets using gradient descent")
