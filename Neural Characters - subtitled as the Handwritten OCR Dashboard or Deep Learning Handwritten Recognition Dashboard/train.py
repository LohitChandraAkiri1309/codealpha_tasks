import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import CharCNN

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def train_model(dataset_name, num_classes, model_save_path, subset_size=20000, epochs=2, batch_size=64):
    print(f"\n--- Training {dataset_name} CNN Model ---")
    
    # 1. Define image transforms (MNIST/EMNIST are normalized in standard ways)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 2. Load dataset
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        # MNIST labels are 0-9 (already 0-indexed)
        label_offset = 0
    elif dataset_name == "EMNIST":
        # EMNIST letters has 26 classes, labels are 1-26 (needs to be shifted to 0-25)
        full_dataset = datasets.EMNIST(root="./data", split="letters", train=True, download=True, transform=transform)
        label_offset = -1
    else:
        raise ValueError("Unknown dataset")
        
    # 3. Create a subset for fast training
    indices = torch.randperm(len(full_dataset))[:subset_size]
    dataset_subset = Subset(full_dataset, indices)
    
    loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=True)
    
    # 4. Initialize model, loss, optimizer
    model = CharCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(loader):
            # Apply label offset if needed (for EMNIST 1-26 -> 0-25)
            if label_offset != 0:
                targets = targets + label_offset
                
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                acc = 100. * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(loader)}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
                
    # 6. Save model weights
    torch.save(model.state_dict(), model_save_path)
    print(f"Successfully saved {dataset_name} model to {model_save_path}")

if __name__ == "__main__":
    # Train MNIST Model (10 classes: 0-9)
    train_model(
        dataset_name="MNIST",
        num_classes=10,
        model_save_path="models/mnist_cnn.pth",
        subset_size=20000,
        epochs=2
    )
    
    # Train EMNIST Letters Model (26 classes: A-Z)
    train_model(
        dataset_name="EMNIST",
        num_classes=26,
        model_save_path="models/emnist_cnn.pth",
        subset_size=20000,
        epochs=2
    )
