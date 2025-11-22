from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 128×128 → 64×64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                 # 64×64 → 32×32
            nn.AvgPool2d(kernel_size=4, stride=4),           # 32×32 → 8×8
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

def main():
    # Dataset Loader
    parquet_path = Path("../datasets")

    df = pd.read_parquet(os.path.join(parquet_path, 'datasets_reduced.parquet'), engine='pyarrow')
    print(df['Label'].value_counts())

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4
    num_workers = 4

    dataset_path = Path("../dataset_copy")
    save_dir = Path("./is_recaptchav2_safe/pytorch")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set = torchvision.datasets.ImageFolder(dataset_path / "train", transform=transform)
    test_set = torchvision.datasets.ImageFolder(dataset_path / "val", transform=transform)


    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(num_workers, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True
    )

    test_loader  = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True
    )

    classes = df['Label'].unique().tolist()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    model = SimpleCNN(len(classes))
    model.to(device)
    # summary(model, input_size=(128, 3, 32, 32))   # optional

    # Loss & optimiser
    print("Setting up loss & optimiser")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    print("Beggining training")
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    PATH = './is_recaptchav2_safe/pytorch/cnn.pth'
    torch.save(model.state_dict(), PATH)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

if __name__ == "__main__":
    main()