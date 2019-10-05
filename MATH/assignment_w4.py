import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x):
    fx = np.exp(x) - 6 * x + 10 / (1 + np.power(( x - 1),2))
    return fx

def d(x):
    a = np.power( (np.power(x - 1, 2) + 1) , 2)
    dx = np.exp(x) - 6 - 20 * (x - 1) / a
    return dx
x = np.arange(-2, 3.5, 0.01)
y = []
dx = []
for i in x:
    y.append(f(i))
    dx.append(d(i))
plt.plot(x, y, color = 'k')
plt.plot(x, dx, color = 'm')

x = 2.273
print(d(x))

### Define hyperparameters
i = np.arange(1, 101, 1) # num of interation
input = np.array([[-0.1, 0.1], [-0.1, 0.2], [-0.1, 0.4], [-0.1, 0.45], [2.2, 0.2], [2.2,0.1], [2.2, 0.05], [2.2, 0.001]])

# define gradient descent function
def des(x_i, a):
    dx = np.exp(x) - 6 - 20 * (x - 1) / np.power( (np.power(x - 1, 2) + 1) , 2)
    x_next = x_i - a*dx
    return x_next

#Generate the list of x (xi,xi+1,...)    
x_list = []

for paras_index in np.arange(1, len(input) + 1, 1):
    x = input[paras_index - 1][0]
    a = input[paras_index - 1][1]
    x_list = [x]
    for interations in i[0:-1]:   #ensure x, y sames shape
        x_list.append(des(x, a))
        x = des(x, a)
    plt.figure(str(paras_index))
    plt.title("x0 = " + str(input[paras_index - 1][0]) + " and learing rate = " + str(input[paras_index - 1][1]) + ".")
    plt.plot(i, x_list, color ='k')  
  # ============
  
  def dd(x):
    ddx = -20*(x-1)/(np.power(np.power(x-1, 2) + 1, 2)) - 6
    return ddx
def d(x):
    dx = np.exp(x) - 6 - 20 * (x - 1) / np.power( (np.power(x - 1, 2) + 1) , 2)
    return dx
#---------------------------------
### Define hyperparameters
i = np.arange(1, 11, 1) # num of interation
input = np.array([[2.2, 0.2], [2.2,0.1], [2.2, 0.001], [2.2, 0.05], ])
#---

# define Newton function
def newt(x_i, a):
    dx = np.exp(x) - 6 - 20 * (x - 1) / np.power( (np.power(x - 1, 2) + 1) , 2)
    x_next = x_i - a*dx
    return x_next

#---------------------

for paras_index in np.arange(1, len(input) + 1, 1):
    x = input[paras_index - 1][0]
    a = input[paras_index - 1][1]
    x_list = [x]
    for interations in i[0:-1]:   #ensure x, y sames shape
        x_list.append(newt(x, a))
        x = newt(x, a)
    plt.figure(str(paras_index))
    plt.title("Newton: x0 = " + str(input[paras_index - 1][0]) + " and learing rate = " + str(input[paras_index - 1][1]) + ".")
    plt.plot(i, x_list, color ='k')  

x = 2.2
a = 0.05
x_list = [x]
for interations in i[0:-1]:   #ensure x, y sames shape
    x_list.append(des(x, a))
    x = des(x, a)
plt.figure(str(paras_index+1))
plt.title("Gradient Descent: x0 = 2.2, a = 0.05")
plt.plot(i, x_list, color ='k')
