import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

training_data = datasets.MNIST(
    root="/home/ibab/Documents/Deep_learning_chiranjeevi_sai_deva/data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="/home/ibab/Documents/Deep_learning_chiranjeevi_sai_deva/data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


model = NeuralNetwork().to(device)

print(model)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)


def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if batch % 100 == 0:

            loss_value = loss.item()

            current = (batch + 1) * len(X)

            print(
                f"loss: {loss_value:>7f} "
                f"[{current:>5d}/{size:>5d}]"
            )


def evalu(dataloader, model, loss_fn):

    model.eval()

    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches

    accuracy = correct / size

    print(f"Test Error: "
          f"Accuracy = {accuracy * 100}%, "
          f"Avg loss = {test_loss}"
    )


epochs = 5

for epoch in range(epochs):

    print(f"Epoch {epoch + 1} -------------------------------")

    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer
    )

    evalu(
        test_dataloader,
        model,
        loss_fn
    )

print("Training completed!")