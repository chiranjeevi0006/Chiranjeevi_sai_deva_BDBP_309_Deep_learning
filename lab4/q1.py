import numpy as np
import matplotlib.pyplot as plt


x=np.array([[0,1,1],[1,1,1],[1,0,1],[0,0,1]],dtype=float)

y=np.array([[0],[1],[1],[0]],dtype=float)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def binary_cross_entropy(y_true,y_pred):
    epsilon = 1e-8
    loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return loss


np.random.seed(142)
w=np.random.random((3,1)) * 0.1
b=np.random.random((1,1))
print(w)
print(b)

learning_rate=0.1
iterations=1000
loss_history=[]

for i in range(iterations):

    z=np.dot(x,w) + b
    y_pred=sigmoid(z)

    loss=binary_cross_entropy(y_true=y,y_pred=y_pred)
    loss_history.append(loss)

    dz = y_pred-y
    dw=np.dot(x.T,dz)/len(x)
    db = np.mean(dz,axis=0,keepdims=True)
    w=w-learning_rate * dw
    b=b-learning_rate * db

    if i % 100 == 0:
        print(f"itteration{i}: loss = {loss}")


print("training completed")
print(f"w:{w}, b:{b}")

z_final = np.dot(x,w)+b
predictions=sigmoid(z_final)
print(f"predicted probabilities:{predictions}")

predicted_classess = (predictions >= 0.5).astype(int)

print(f"predicted classes:{predicted_classess}")
print(f"actual classes:{y.astype(int)}")

accuracy=np.mean(predicted_classess == y)*100
print(f"accuracy:{accuracy:.2f}%")

plt.figure(figsize=(8,5))

plt.plot(
    range(1,iterations + 1),
    loss_history
)

plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("training loss over 1000 iterations")

plt.grid(True)
plt.show()