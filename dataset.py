import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import resize_x, resize_y, num_classes

class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((resize_x, resize_y)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Get all images and their labels
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        # Assuming directory structure: data_dir/person_name/*.jpg
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if os.path.isdir(person_dir):
                if person_name not in self.label_to_idx:
                    self.label_to_idx[person_name] = len(self.label_to_idx)
                for img_file in os.listdir(person_dir):
                    if img_file.endswith(('.jpg', '.png')):
                        self.image_paths.append(os.path.join(person_dir, img_file))
                        self.labels.append(self.label_to_idx[person_name])
        
        # Update num_classes in config
        from config import num_classes
        num_classes = len(self.label_to_idx)

        print(f"Labels: {self.labels}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the sample and its label at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image tensor
                   and label is the corresponding class index.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)