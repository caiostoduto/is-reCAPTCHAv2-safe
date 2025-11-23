from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import h5py
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import confusion_matrix


def convert_imgfolder_to_hdf5(data_dir, output_path, img_size=128):
    print(f"Loading images from: {data_dir}")
    
    # Transform: Resize → ToTensor (produces float32 [C, H, W])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # Load dataset (automatically gets class labels)
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    
    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes")
    
    # Create HDF5 file
    with h5py.File(output_path, "w") as f:
        # Images: [N, C, H, W] as float32
        img_dset = f.create_dataset(
            "images",
            shape=(len(dataset), 3, img_size, img_size),
            dtype="float32",
            chunks=(1, 3, img_size, img_size),  # One image per chunk
            compression="lzf",  # Fast, good for float data
        )
        
        # Labels: [N] as int64
        label_dset = f.create_dataset(
            "labels",
            shape=(len(dataset),),
            dtype="int64",
        )
        
        # Store class names as metadata (optional but useful)
        f.attrs["classes"] = dataset.classes
        
        # Fill datasets
        print("Converting...")
        for i, (img_tensor, label) in tqdm(enumerate(dataset), total=len(dataset)):
            # img_tensor is already float32 [C, H, W] from ToTensor
            img_dset[i] = img_tensor.numpy()
            label_dset[i] = label
    
    print(f"Created {output_path} with {len(dataset)} images")


class HDF5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(self.h5_path, "r") as f:
            self._len = len(f["images"])
        self._h5_file = None

    def _open_file(self):
        if not hasattr(self, "_images"):
            self._h5_file = h5py.File(self.h5_path, "r", swmr=True)
            self._images = self._h5_file["images"]
            self._labels = self._h5_file["labels"]

    def __getitem__(self, idx):
        self._open_file()
        img = torch.from_numpy(self._images[idx])  # Already float32 [C, H, W]
        label = int(self._labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self): return self._len
    def __del__(self):
        if hasattr(self, "_h5_file") and self._h5_file:
            self._h5_file.close()

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
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    batch_size = 32
    num_workers = 4
    epochs = 100

    dataset_path = Path("../dataset_copy")
    save_dir = Path("./is_recaptchav2_safe/pytorch")
    save_dir.mkdir(parents=True, exist_ok=True)

    create_hdf5 = False
    if create_hdf5:
        convert_imgfolder_to_hdf5(dataset_path / "train", "../datasets/train.h5")
        convert_imgfolder_to_hdf5(dataset_path / "val", "../datasets/val.h5")


    train_set = HDF5Dataset("../datasets/train.h5", transform=transform)
    test_set = HDF5Dataset("../datasets/val.h5", transform=transform)

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
    with open(save_dir / "results.csv", 'w') as f:
        f.write("epoch,loss\n")
        print("epoch,loss")

        for epoch in range(epochs):
            running_loss = 0.0
            for data in tqdm(train_loader, total=len(train_loader)):
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

            print(f"{epoch+1}/{epochs},{(running_loss / len(train_loader)):.4f}")
            f.write(f"{epoch},{running_loss / len(train_loader)}\n")
    
    with open(save_dir / "results.txt", 'w') as f:
        all_preds = []
        all_labels = []
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(test_loader, total=len(test_loader)):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        f.write(f"Final accuracy: {(100*accuracy):.2f} %\n")
        print(f"Final accuracy: {(100*accuracy):.2f} %")
        
        cm = confusion_matrix(all_labels, all_preds)

    # Plot Confusion Matrix
    cm = cm.T
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()

    plt.xticks(np.arange(len(classes)), classes, rotation=90)
    plt.yticks(np.arange(len(classes)), classes)

    plt.xlabel("True")
    plt.ylabel("Predicted")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                plt.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')

    cm_norm = cm.astype("float") / cm.sum(axis=0, keepdims=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()

    plt.xticks(np.arange(len(classes)), classes, rotation=90)
    plt.yticks(np.arange(len(classes)), classes)

    plt.xlabel("True")
    plt.ylabel("Predicted")

    thresh = cm_norm.max() / 2
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            if cm_norm[i, j] > 0.009:
                plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black")
                
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()