### Import our standard libraries ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Get Some Randomness and Plot It ###

np.random.seed(8675309)
rands = np.random.randn(1000)

plt.scatter(range(1000),rands, s = 1 )


### Code for Question 1 ###
def pdf(x, u):
    fx = np.exp(-1 * np.power(x, 2)/2/np.power(u,2)) / np.sqrt(2*np.pi*np.power(u,2))
    return fx

#multiply each of the pdf from each of the elemnt in rands
D = 1
for i in rands:
    D *= pdf(i,0.5)
print("D = " + str(D))
# (Write Code Here.)

### Code for Question 2 ###
def pdf2(x, u):
    fx = np.power(x,2)/2/np.power(u,2)
    return fx

log_D = 0
for i in rands:
    log_D += pdf2(i, 0.5)

log_D = log_D + 500*np.log(2*np.pi*np.power(0.5,2))
print("-log(D) = " + str(round(log_D,1)))

### Code for Question 3 ###
log_D = 0
list_log_D = []
list_u = []

def pdf2(x, u):
    fx = np.power(x,2)/2/np.power(u,2)
    return fx

#----
for u in np.arange(0.5,2.0,0.001):
    for i in rands:
        log_D += pdf2(i, u)
    log_D = 500*np.log(2*np.pi*np.power(u,2)) + log_D
    
    list_u.append(u)
    list_log_D.append(log_D)
    log_D = 0
    
plt.plot(list_u, list_log_D, color = 'g')
plt.xlabel('u')
plt.ylabel('-log(D)')

#print(list(zip(list_u, list_log_D)))
