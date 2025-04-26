import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import FaceRecognitionModel
from dataset import FaceDataset, create_dataloader
from config import batch_size, epochs, learning_rate, checkpoint_path, num_classes

def train_model(model, train_loader, num_epochs=epochs, loss_fn=nn.CrossEntropyLoss(), optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
    # Save the model
    torch.save(model.state_dict(), checkpoint_path)

    # Calculate training accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs, _ = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Training Accuracy: {100 * correct / total}%")
    
    return model

def predict_face(image_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output, embedding = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        print(f"Probabilities: {probabilities}")
        _, predicted = torch.max(output.data, 1)
    return predicted.item(), embedding

if __name__ == "__main__":
    train_dataset = FaceDataset(data_dir="./data")
    print(f"Number of samples in dataset: {len(train_dataset)}")  # Debugging

    # Dynamically determine num_classes
    num_classes = len(train_dataset.label_to_idx)
    print(f"Number of classes: {num_classes}")  # Debugging

    train_loader = create_dataloader(train_dataset, batch_size=batch_size)

    # Pass the dynamically determined num_classes to the model
    model = FaceRecognitionModel(num_classes=num_classes)
    trained_model = train_model(model, train_loader)