import numpy as np


# from lab1.q1 import relu
def relu(x):
  return np.maximum(0,x)
x=np.random.rand(4)
w=np.random.rand(4)
b=2
z=np.dot(w,x) +b
w1=np.random.rand(3,4)
w2=np.random.rand(2,3)
w3=np.random.rand(1,2)
a=relu(z)
y_hat = a
b1=np.random.rand(3)
b2=np.random.rand(2)
b3=np.random.rand(1)
z1=np.dot(w1,x) + b1
a1=relu(z1)
z2=np.dot(w2,a1) + b2
a2=relu(z2)
z3=np.dot(w3,a2) + b3
a3=relu(z3)
print(z1)
print(z2)
print(z3)
print(a1)
print(a2)
print(a3)
print(y_hat)
print("------------------------------------------------------------------------------------------")
