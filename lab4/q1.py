import numpy as np
import math
# from lab1.q1 import sigmoid
x=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
y=np.array([[0],[1],[1],[0]])
w=np.array([[3],[4],[2]])
b=0
z=x @ w
z=z + b
def sigmoid(x):
    return 1/(1+math.exp(-x))

sigmoid=sigmoid(z)

