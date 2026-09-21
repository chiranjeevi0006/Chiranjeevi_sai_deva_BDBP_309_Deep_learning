import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=128,
    shuffle=False
)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, optimizer, epochs=5):

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss/len(train_loader):.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )


def test_model(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


print(f"========== SGD ==========")

model_sgd = NeuralNetwork().to(device)

optimizer_sgd = optim.SGD(
    model_sgd.parameters(),
    lr=0.01
)

train_model(model_sgd, optimizer_sgd)

print("Test Accuracy:",
      test_model(model_sgd))


print(f"========== SGD + Momentum ==========")

model_momentum = NeuralNetwork().to(device)

optimizer_momentum = optim.SGD(
    model_momentum.parameters(),
    lr=0.01,
    momentum=0.9
)

train_model(model_momentum, optimizer_momentum)

print("Test Accuracy:",
      test_model(model_momentum))


print(f"========== AdaGrad ==========")

model_adagrad = NeuralNetwork().to(device)

optimizer_adagrad = optim.Adagrad(
    model_adagrad.parameters(),
    lr=0.01
)

train_model(model_adagrad, optimizer_adagrad)

print("Test Accuracy:",
      test_model(model_adagrad))


print(f"========== RMSprop ==========")

model_rmsprop = NeuralNetwork().to(device)

optimizer_rmsprop = optim.RMSprop(
    model_rmsprop.parameters(),
    lr=0.001
)

train_model(model_rmsprop, optimizer_rmsprop)

print("Test Accuracy:",
      test_model(model_rmsprop))


print(f"========== Adam ==========")

model_adam = NeuralNetwork().to(device)

optimizer_adam = optim.Adam(
    model_adam.parameters(),
    lr=0.001
)

train_model(model_adam, optimizer_adam)

print("Test Accuracy:",test_model(model_adam))