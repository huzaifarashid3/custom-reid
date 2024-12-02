import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import scipy.io

class Market1501Dataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        """
        Initialize Market-1501 dataset
        
        Args:
            data_path (str): Path to Market-1501 dataset
            mode (str): 'train' or 'test'
            transform (callable): Optional image transformations
        """
        self.data_path = data_path
        self.mode = mode
        self.transform = transform or self._default_transform()
        
        # Load image paths and labels
        self.image_paths, self.labels = self._load_dataset()
    
    def _default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((256, 128)),  # Standard Market-1501 image size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_dataset(self):
        """
        Load image paths and labels for Market-1501
        
        Returns:
            tuple: (image_paths, labels)
        """
        # Paths for training and query images
        if self.mode == 'train':
            img_dir = os.path.join(self.data_path, 'bounding_box_train')
        elif self.mode == 'query':
            img_dir = os.path.join(self.data_path, 'query')
        elif self.mode == 'gallery':
            img_dir = os.path.join(self.data_path, 'bounding_box_test')
        
        image_paths = []
        labels = []

        
        for filename in os.listdir(img_dir):
            # Skip junk images
            if '-1' in filename:
                continue
            
            # Extract person ID from filename
            person_id = int(filename.split('_')[0])
            
            # Full path to image
            full_path = os.path.join(img_dir, filename)
            
            image_paths.append(full_path)
            labels.append(person_id)
        
        unique_labels = sorted(set(labels))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        labels = [label_mapping[label] for label in labels]

        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and transform image
        
        Returns:
            tuple: (transformed image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path)
        
        # Transform image
        if self.transform:
            img = self.transform(img)
        
        return img, label

class PersonReIDModel(nn.Module):
    def __init__(self, num_classes):
        """
        CNN-based Person Re-ID model
        
        Args:
            num_classes (int): Number of unique person identities
        """
        super(PersonReIDModel, self).__init__()
        
        # Feature extraction using ResNet-based architecture
        self.feature_extractor = nn.Sequential(
            # Convolutional layers for feature extraction
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 32 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

def train_market1501_model(data_path, epochs=3, batch_size=16):
    """
    Train person re-identification model on Market-1501 dataset
    
    Args:
        data_path (str): Path to Market-1501 dataset
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
    
    Returns:
        Trained PyTorch model
    """
    # Prepare datasets
    train_dataset = Market1501Dataset(data_path, mode='train')
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    

    # Determine number of unique classes
    num_classes = len(set(train_dataset.labels))
    
    # Initialize model
    model = PersonReIDModel(num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')
        model.train()
        total_loss = 0

        batch_no = 0
        
        for images, labels in train_loader:
            print("batch: ", batch_no)
            batch_no += 1
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    return model

def evaluate_model(model, data_path):
    """
    Evaluate model performance on Market-1501 dataset
    
    Args:
        model (nn.Module): Trained re-identification model
        data_path (str): Path to Market-1501 dataset
    
    Returns:
        Performance metrics
    """
    # Prepare datasets
    query_dataset = Market1501Dataset(data_path, mode='query')
    gallery_dataset = Market1501Dataset(data_path, mode='gallery')
    
    # DataLoaders
    query_loader = DataLoader(query_dataset, batch_size=64, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False)
    
    # Extract features
    model.eval()
    
    def extract_features(dataloader):
        features = []
        labels = []
        with torch.no_grad():
            for images, batch_labels in dataloader:
                outputs = model(images)
                features.append(outputs.numpy())
                labels.append(batch_labels.numpy())
        
        return np.concatenate(features), np.concatenate(labels)
    
    query_features, query_labels = extract_features(query_loader)
    gallery_features, gallery_labels = extract_features(gallery_loader)
    
    # Compute Rank-1 and mAP
    from sklearn.metrics import accuracy_score
    
    # Simple nearest neighbor matching
    predictions = []
    for query_feat in query_features:
        # Find closest match in gallery
        distances = np.linalg.norm(gallery_features - query_feat, axis=1)
        closest_idx = np.argmin(distances)
        predictions.append(gallery_labels[closest_idx])
    
    # Rank-1 accuracy
    rank1_accuracy = accuracy_score(query_labels, predictions)
    
    print(f'Rank-1 Accuracy: {rank1_accuracy:.4f}')
    
    return {'rank1_accuracy': rank1_accuracy}

def main():
    # Path to Market-1501 dataset
    MARKET1501_PATH = './Market-1501-v15.09.15'
    
    # Train the model
    model = train_market1501_model(MARKET1501_PATH)
    
    # Evaluate the model
    evaluate_model(model, MARKET1501_PATH)

if __name__ == "__main__":
    main() 