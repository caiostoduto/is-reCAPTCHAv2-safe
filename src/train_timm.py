#!/usr/bin/env python3
"""
Is reCAPTCHAv2 Safe? - TIMM Training Script
Educational Research Project - UFABC

This script trains a deep learning classifier using PyTorch Image Models (timm)
to analyze reCAPTCHA image classification challenges.

Usage:
    python train_timm.py --model efficientnet_b0 --epochs 100 --batch-size 32
"""

import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Train reCAPTCHA classifier with TIMM')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        help='Model architecture from timm (default: efficientnet_b0)')
    parser.add_argument('--data-dir', type=str, default='../dataset',
                        help='Path to dataset directory')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, mps, cuda, cpu (default: auto)')
    parser.add_argument('--save-dir', type=str, default='is_recaptchav2_safe/timm_experiment',
                        help='Directory to save results (default: is_recaptchav2_safe/timm_experiment)')
    return parser.parse_args()


def get_device(device_arg):
    """Determine the best available device"""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    return device_arg


def get_transforms(img_size):
    """Create data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_dataloaders(data_dir, batch_size, num_workers, train_transform, val_transform):
    """Create training and validation dataloaders"""
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(val_loader), 100. * correct / total


def plot_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot learning rate
    axes[2].plot(history['lr'], label='Learning Rate', marker='o', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Is reCAPTCHAv2 Safe? - TIMM Training")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Patience: {args.patience}")
    print("="*80)
    
    # Data
    train_transform, val_transform = get_transforms(args.img_size)
    train_loader, val_loader, class_names = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, train_transform, val_transform
    )
    
    num_classes = len(class_names)
    print(f"\nDataset loaded:")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Model
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    
    # Training setup
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(args.save_dir, 'best.pt')
    last_model_path = os.path.join(args.save_dir, 'last.pt')
    
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
                'config': vars(args)
            }, best_model_path)
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'class_names': class_names,
            'config': vars(args)
        }, last_model_path)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    
    # Save results
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(args.save_dir, 'training_history.csv'), index=False)
    
    plot_history(history, os.path.join(args.save_dir, 'training_plots.png'))
    
    print("\n" + "="*80)
    print("Training Completed!")
    print("="*80)
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved at: {best_model_path}")
    print(f"Last model saved at: {last_model_path}")
    print(f"Training history saved at: {os.path.join(args.save_dir, 'training_history.csv')}")
    print(f"Training plots saved at: {os.path.join(args.save_dir, 'training_plots.png')}")
    print("="*80)


if __name__ == '__main__':
    main()
