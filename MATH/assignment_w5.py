### Define hyperparameters
i = np.arange(1, 101, 1) # num of interation
x0 = 0.8; lr = 0.01 # initial x, and learning rate

#==================================== use tranditional Gradient Descent
def des(x_i, lr):
    dx = np.exp(x_i) - 6 - 20 * (x_i - 1) / np.power( (np.power(x_i - 1, 2) + 1) , 2)
    x_next = x_i - lr*dx
    return x_next

#Generate the list of x (xi,xi+1,...)    
x_list = []; x = x0
for steps in i-1:
    x_list.append(x)
    x = des(x, lr)

print(x_list)
plt.plot(i, x_list, color = "k")

###================================== yse Gradient Descent with Momentum
a = 0.5 #damping factor
def mmtm(x_i, v_previous, lr):
    v = v_previous
    dx = np.exp(x_i) - 6 - 20 * (x_i - 1) / np.power( (np.power(x_i - 1, 2) + 1) , 2)
    v = a * v - lr * dx
    x_next = x_i + v
    return v, x_next

xm_list = []; x = x0; v = 0
for steps in i-1:
    v_previous = v
    xm_list.append(x)
    v, x = mmtm(x, v_previous, lr)

print(xm_list)
plt.plot(i, xm_list, color = "m")
