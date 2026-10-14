import numpy as np
import matplotlib.pyplot as plt


def my_function(x, y):

    A1 =  1.0
    B1 =  1.0
    A2 =  0.0
    B2 = -3.0
    D1 =  2.0
    D2 =  8.0

    result = -2*np.exp(-((x-A1)**2 + (y - B1)**2)/D1) - np.exp(-((x - A2)**2 + (y - B2)**2)/D2)
    return result

def my_derivative(x, y):

    A1 =  1.0
    B1 =  1.0
    A2 =  0.0
    B2 = -3.0
    D1 =  2.0
    D2 =  8.0

    dfdx = 4*(x-A1)/D1*np.exp(-((x-A1)**2 + (y - B1)**2)/D1) + 2*(x-A2)/D2*np.exp(-((x - A2)**2 + (y - B2)**2)/D2)
    dfdy = 4*(y-B1)/D1*np.exp(-((x-A1)**2 + (y - B1)**2)/D1) + 2*(y-B2)/D2*np.exp(-((x - A2)**2 + (y - B2)**2)/D2)
       
    return dfdx, dfdy

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X,Y = np.meshgrid(x,y)

plt.imshow(my_function(X, Y), extent=[-5, 5, -5, 5], origin='lower')

plt.colorbar()

iterations = 1000
x = -3
y = 2
alpha = 0.1

x_list = []
y_list = []

dfdx = 1e4
dfdy = 1e4

ctr = 0

while (np.abs(np.dot(dfdx, dfdy)) > 1e-4) & (ctr < 1e6):

# for i in range(iterations):

    x_list.append(x)
    y_list.append(y)

    dfdx, dfdy = my_derivative(x, y)
    x = x - alpha*dfdx
    y = y - alpha*dfdy 
    ctr += 1

print("Number of iterations: ", ctr)

plt.plot(x_list, y_list, 'x')
plt.show()
