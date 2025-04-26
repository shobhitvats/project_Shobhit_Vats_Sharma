import torch
from torchvision import transforms
from PIL import Image
from model import FaceRecognitionModel
from config import resize_x, resize_y, checkpoint_path
import os

def load_model():
    # Load the state dictionary from the checkpoint
    state_dict = torch.load(checkpoint_path)
    
    # Dynamically determine num_classes from the checkpoint
    num_classes = state_dict["fc2.weight"].size(0)  # Get the number of output classes
    
    # Initialize the model with the correct num_classes
    model = FaceRecognitionModel(num_classes=num_classes)
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_face(image_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output, embedding = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item(), embedding

def classify_faces(list_of_face_img_paths):
    model = load_model()
    results = []
    for img_path in list_of_face_img_paths:
        class_id, _ = predict_face(img_path, model)
        results.append(f"Person {class_id}")
    return results

def get_image_paths_from_directory(directory):
    """
    Recursively collects all image file paths from the given directory.

    Args:
        directory (str): Path to the directory containing subdirectories of images.

    Returns:
        list: List of image file paths.
    """
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):  # Add more extensions if needed
                image_paths.append(os.path.join(root, file))
    return image_paths

def classify_faces_from_directory(directory):
    """
    Classifies all images in a directory containing subdirectories of images.

    Args:
        directory (str): Path to the directory containing subdirectories of images.

    Returns:
        list: List of classification results for each image.
    """
    # Get all image paths from the directory
    list_of_face_img_paths = get_image_paths_from_directory(directory)
    
    # Load the model
    model = load_model()
    
    # Classify each image
    results = []
    for img_path in list_of_face_img_paths:
        class_id, _ = predict_face(img_path, model)
        results.append((img_path, f"Person {class_id}"))
    return results

if __name__ == "__main__":
    # Path to the directory containing subdirectories of images
    input_directory = "./data/test_data"
    
    # Classify all images in the directory
    results = classify_faces_from_directory(input_directory)
    
    # Print the results
    for img_path, result in results:
        print(f"Image: {img_path}, Prediction: {result}")