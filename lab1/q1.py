import matplotlib.pyplot as plt
import math
z=[]
start=-10
end=10
n=100
step=(end-start)/(n-1)
for i in range(n):
    z.append(start+i*step)
def sigmoid(x):
    return 1/(1+math.exp(-x))

def sigmoid_derivative(x):
    s=sigmoid(x)
    return s*(1-s)
sigmoid_output = []
for value in z:
    sigmoid_output.append(sigmoid(value))
sigmoid_derivative_output = []
for value in z:
    sigmoid_derivative_output.append(sigmoid_derivative(value))
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def tanh_derivation(x):
    t=tanh(x)
    return 1-t*t
tanh_output=[]
for value in z:
    tanh_output.append(tanh(value))

tanh_derivative_output=[]
for value in z:
    tanh_derivative_output.append(tanh_derivation(value))

def relu(x):
    if x > 0:
        return x
    else:
        return 0
def relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0

relu_output = []
for value in z:
    relu_output.append(relu(value))

relu_derivative_output = []
for value in z:
    relu_derivative_output.append(relu_derivative(value))
def leaky_relu(x):
    alpha=0.05
    if x> 0:
        return x
    else:
        return alpha*x
def leaky_relu_derivative(x):
    alpha=0.05
    if x> 0:
        return 1
    else:
        return alpha


leaky_relu_output=[]
for value in z:
    leaky_relu_output.append(leaky_relu(value))

leaky_relu_derivative_output = []

for value in z:
    leaky_relu_derivative_output.append(leaky_relu_derivative(value))


def softmax(values):
    exp_values = []

    for value in values:
         exp_values.append(math.exp(value))

    total = sum(exp_values)

    softmax_output = []

    for value in exp_values:
        softmax_output.append(value / total)

    return softmax_output

softmax_output = softmax(z)
print("Softmax Output")
print(softmax_output)

plt.figure(figsize=(12,6))
plt.plot(z,sigmoid_output,label="Sigmoid")
plt.plot(z,sigmoid_derivative_output,label="Sigmoid Derivative")
plt.title("Sigmoid")
plt.xlabel("z")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,7))
plt.plot(z,tanh_output,label="Tanh")
plt.plot(z,tanh_derivative_output,label="Tanh Derivative")
plt.title("Tanh")
plt.xlabel("z")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(17,7))
plt.plot(z,relu_output,label="ReLU")
plt.plot(z,relu_derivative_output,label="ReLU Derivative")
plt.title("ReLU")
plt.xlabel("z")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(13,8))
plt.plot(z,leaky_relu_output,label="Leaky ReLU")
plt.plot(z,leaky_relu_derivative_output,label="Leaky ReLU Derivative")
plt.title("Leaky ReLU")
plt.xlabel("z")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

