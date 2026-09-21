import os
os.environ["OMP_NUM_THREADS"] = "9"
os.environ["MKL_NUM_THREADS"] = "9"

import torch
torch.set_num_threads(9)
torch.set_num_interop_threads(9)

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=3*3*64, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=num_classes),
        )
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d,nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x=x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

def tr_CNN(model,optimizer,loader,criterion,device):
    model.train()
    total=0
    correct=0
    running_loss=0
    batch_acc=[]
    batch_loss=[]
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        _, predicted = output.max(1)
        acc=(predicted.eq(target).sum().item()/target.size(0))*100
        batch_acc.append(acc)
        running_loss += loss.item()*data.size(0)
        total += data.size(0)
        correct += predicted.eq(target).sum().item()
        print(f"train batch {batch_idx}/{batch_idx+1}    |      loss:{loss.item()}   acc:{acc}")
    epoch_loss=running_loss/len(loader)
    epoch_acc=correct/total
    return epoch_loss,epoch_acc,batch_loss,batch_acc

def eva_CNN(model,loader,criterion,device):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0
    batch_acc = []
    batch_loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            batch_loss.append(loss.item())
            _, predicted = output.max(1)
            acc = (predicted.eq(target).sum().item() / target.size(0)) * 100
            batch_acc.append(acc)
            running_loss += loss.item() * data.size(0)
            total += data.size(0)
            correct += predicted.eq(target).sum().item()
            print(f"train batch {batch_idx}/{batch_idx + 1}    |  loss:{loss.item()}   acc:{acc}")
        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total
        return epoch_loss,epoch_acc,batch_loss,batch_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    model = CNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epoch=2
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,),std=(0.5,))])
    train_dataset=datasets.MNIST(root="./data",download=True,train=True,transform=transform)
    test_dataset=datasets.MNIST(root="./data",download=True,train=False,transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=100,shuffle=False)

    for epoch in range(num_epoch):
        a,b,c,d=tr_CNN(model,optimizer,train_loader,criterion,device)
        e,f,g,h=eva_CNN(model,train_loader,criterion,device)
        print(f"epoch {epoch}/{num_epoch}"
              f"Train loss:{a}   |      Test loss:{e}   |   Train    Accuracy:{b}%   tets Accuracy:{f}%")

if __name__ == "__main__":
    main()