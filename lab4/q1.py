import numpy as np

x = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

w = np.array([
    [3],
    [4],
    [2]
])

b = 0

z = x @ w + b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = sigmoid(z)

print("z =")
print(z)

print(f"Sigmoid output =")
print(a)