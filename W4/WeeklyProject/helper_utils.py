"""
Helper utilities for the Weekly Project on Transfer Learning.
Contains functions for visualizing images, predictions, and training progress.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def imshow(inp, title=None, ax=None):
    """
    Display image for Tensor.
    
    Args:
        inp: Tensor image of shape (C, H, W) or (H, W, C)
        title: Optional title for the image
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Convert tensor to numpy if needed
    if isinstance(inp, torch.Tensor):
        inp = inp.numpy()
    
    # Handle different input shapes
    if inp.shape[0] == 3 or inp.shape[0] == 1:  # (C, H, W)
        inp = inp.transpose((1, 2, 0))
    
    # Denormalize if needed (assuming ImageNet normalization)
    if inp.min() < 0 or inp.max() > 1:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    
    # Handle grayscale images
    if len(inp.shape) == 2 or (len(inp.shape) == 3 and inp.shape[2] == 1):
        ax.imshow(inp.squeeze(), cmap='gray')
    else:
        ax.imshow(inp)
    
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    
    return ax


def visualize_batch(dataloader, class_names, num_images=8, figsize=(12, 6)):
    """
    Visualize a batch of images from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        class_names: List of class names
        num_images: Number of images to display
        figsize: Figure size tuple
    """
    # Get a batch
    inputs, classes = next(iter(dataloader))
    
    # Make a grid
    out = torchvision.utils.make_grid(inputs[:num_images], nrow=4)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Display with titles
    titles = [class_names[x] for x in classes[:num_images]]
    imshow(out, title=', '.join(titles), ax=ax)
    
    plt.tight_layout()
    return fig


def visualize_predictions(model, dataloader, class_names, device, num_images=6):
    """
    Visualize model predictions on validation images.
    
    Args:
        model: Trained PyTorch model
        dataloader: Validation DataLoader
        class_names: List of class names
        device: Device to run inference on
        num_images: Number of images to visualize
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                
                # Get prediction and true label
                pred_class = class_names[preds[j]]
                true_class = class_names[labels[j]]
                title = f'Predicted: {pred_class}\nTrue: {true_class}'
                
                # Color title based on correctness
                if pred_class == true_class:
                    ax.set_title(title, color='green', fontweight='bold')
                else:
                    ax.set_title(title, color='red', fontweight='bold')
                
                imshow(inputs.cpu().data[j], ax=ax)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    return fig
        model.train(mode=was_training)
    
    plt.tight_layout()
    return fig


def visualize_training_history(history):
    """
    Plot training and validation loss and accuracy over epochs.
    
    Args:
        history: Dictionary with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def predict_single_image(model, img_path, data_transforms, class_names, device):
    """
    Make prediction on a single custom image.
    
    Args:
        model: Trained PyTorch model
        img_path: Path to image file
        data_transforms: Transform pipeline (use 'val' transforms)
        class_names: List of class names
        device: Device to run inference on
        
    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score (softmax probability)
    """
    from PIL import Image
    
    was_training = model.training
    model.eval()
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = data_transforms(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        predicted_class = class_names[pred_idx.item()]
        confidence_score = confidence.item()
    
    model.train(mode=was_training)
    
    return predicted_class, confidence_score


def visualize_single_prediction(model, img_path, data_transforms, class_names, device):
    """
    Visualize prediction on a single custom image.
    
    Args:
        model: Trained PyTorch model
        img_path: Path to image file
        data_transforms: Transform pipeline (use 'val' transforms)
        class_names: List of class names
        device: Device to run inference on
    """
    from PIL import Image
    
    was_training = model.training
    model.eval()
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = data_transforms(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
        predicted_class = class_names[pred_idx.item()]
        confidence_score = confidence.item()
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    imshow(img_tensor.cpu().data[0], ax=ax)
    ax.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence_score:.2%}', 
                 fontsize=14, fontweight='bold')
    
    model.train(mode=was_training)
    
    return fig
