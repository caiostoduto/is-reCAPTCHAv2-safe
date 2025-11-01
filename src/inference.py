#!/usr/bin/env python3
"""
Is reCAPTCHAv2 Safe? - Inference Script
Educational Research Project - UFABC

This script loads a trained model and performs inference on reCAPTCHA images.

Usage:
    python inference.py --model-path best.pt --image-path path/to/image.jpg
    python inference.py --model-path best.pt --image-dir path/to/images/
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import timm


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with trained reCAPTCHA classifier')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--image-path', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Path to directory of images for batch inference')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, mps, cuda, cpu (default: auto)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Show top-k predictions (default: 3)')
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


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Try to get model name from config, otherwise use default
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_name = checkpoint['config']['model']
    else:
        model_name = 'efficientnet_b0'  # Default fallback
    
    # Create model
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, checkpoint


def get_transform(img_size):
    """Create inference transform"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_image(model, image_path, transform, device, class_names, top_k=3):
    """Predict the class of a single image"""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = probabilities[0].topk(min(top_k, len(class_names)))
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            'class': class_names[idx.item()],
            'confidence': prob.item()
        })
    
    return results


def main():
    args = parse_args()
    
    # Validate arguments
    if args.image_path is None and args.image_dir is None:
        print("Error: Must provide either --image-path or --image-dir")
        return
    
    if args.image_path is not None and args.image_dir is not None:
        print("Error: Provide only one of --image-path or --image-dir")
        return
    
    # Setup
    device = get_device(args.device)
    
    print("="*80)
    print("Is reCAPTCHAv2 Safe? - Inference")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model checkpoint: {args.model_path}")
    
    # Load model
    print("\nLoading model...")
    model, class_names, checkpoint = load_model(args.model_path, device)
    
    print(f"Model loaded successfully!")
    print(f"Training epoch: {checkpoint['epoch'] + 1}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"Classes: {class_names}")
    
    # Prepare transform
    transform = get_transform(args.img_size)
    
    # Single image inference
    if args.image_path:
        print("\n" + "="*80)
        print(f"Analyzing: {args.image_path}")
        print("="*80)
        
        results = predict_image(model, args.image_path, transform, device, class_names, args.top_k)
        
        print(f"\nTop {len(results)} predictions:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['class']:20s} - {result['confidence']*100:6.2f}%")
    
    # Batch inference
    else:
        image_dir = Path(args.image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"\nNo images found in {args.image_dir}")
            return
        
        print("\n" + "="*80)
        print(f"Analyzing {len(image_files)} images from {args.image_dir}")
        print("="*80)
        
        for image_file in sorted(image_files):
            results = predict_image(model, image_file, transform, device, class_names, args.top_k)
            
            print(f"\n{image_file.name}:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['class']:20s} - {result['confidence']*100:6.2f}%")


if __name__ == '__main__':
    main()
