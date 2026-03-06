import torch
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import os

class FaceDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Args:
            root_dirs (list): List of root directories for the datasets (A1, A2, B1).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        # Iterate through each root directory and collect image paths and labels
        for root_dir in root_dirs:
            for label in os.listdir(root_dir):
                label_dir = os.path.join(root_dir, label)
                if os.path.isdir(label_dir):
                    for image_name in os.listdir(label_dir):
                        self.image_paths.append(os.path.join(label_dir, image_name))
                        self.labels.append(label)
        
        # Convert labels to unique integers
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom dataset for loading images and labels from a directory.
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load all image paths and their corresponding labels
        for label_idx, label_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, label_name)
            if os.path.isdir(class_dir):
                for image_file in os.listdir(class_dir):
                    if image_file.lower().endswith(('jpg', 'jpeg', 'png')):
                        self.images.append(os.path.join(class_dir, image_file))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format

        if self.transform:
            image = self.transform(image)

        return image, label

def get_student_dataloader(root_folder, transform=None, batch_size=32):
    """
    Creates combined train and test datasets from subfolders A1, A2, and B1 under the root folder.

    Args:
        root_folder (str): Path to the root folder containing A1, A2, and B1 subfolders.
        transform (dict, optional): Dictionary of transforms for train and test datasets.
                                    Example: {'train': train_transform, 'test': test_transform}.
        batch_size (int, optional): Batch size for the DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for the combined train dataset.
        test_loader (DataLoader): DataLoader for the combined test dataset.
    """
    
    train_transform = get_transforms(phase="train")
    test_transform = get_transforms(phase="test")
    
    # Define the paths for A1, A2, and B1 datasets
    a1_train_path = os.path.join(root_folder, 'A1', 'training')
    a1_test_path = os.path.join(root_folder, 'A1', 'testing')
    a2_train_path = os.path.join(root_folder, 'A2', 'training')
    a2_test_path = os.path.join(root_folder, 'A2', 'testing')
    b1_train_path = os.path.join(root_folder, 'B1', 'training')
    b1_test_path = os.path.join(root_folder, 'B1', 'testing')

    # Check if the paths exist
    train_paths = [a1_train_path, a2_train_path, b1_train_path]
    test_paths = [a1_test_path, a2_test_path, b1_test_path]
    for path in train_paths + test_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")

    # Combine all train and test paths
    combined_train_paths = [a1_train_path, a2_train_path, b1_train_path]
    combined_test_paths = [a1_test_path, a2_test_path, b1_test_path]

    # Create combined train and test datasets
    train_datasets = [FaceDataset(combined_train_paths, transform=train_transform)]
    test_datasets = [FaceDataset(combined_test_paths, transform=test_transform)]

    # Combine the train and test datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    # Create DataLoaders for the combined datasets
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_transforms(phase="train"):
    """
    Define data augmentation and preprocessing transforms.
    Args:
        phase (str): Either 'train' or 'test'.
    Returns:
        transform (callable): A composed transform pipeline.
    """
    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif phase == "test":
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError("Phase should be either 'train' or 'test'.")


        
def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training and testing.
    Args:
        data_dir (str): Root directory containing 'train' and 'test' subdirectories.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
    Returns:
        train_loader, test_loader: Data loaders for training and testing.
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Define transforms for training and testing
    train_transform = get_transforms(phase="train")
    test_transform = get_transforms(phase="test")

    # Create datasets
    train_dataset = CustomDataset(train_dir, transform=train_transform)
    test_dataset = CustomDataset(test_dir, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def get_lpfw_dataloaders(batch_size):

    dataset_root = "./lfw_dataset"

    train_transform = get_transforms(phase="train")
    #test_transform = get_transforms(phase="test")

    dataset = datasets.LFWPeople(root=dataset_root, split='train', download=True, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader