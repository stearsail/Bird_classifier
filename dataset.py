
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

CURRENT_DIR = os.getcwd()

class CustomDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    
def load_dataset(file_path):
    return ImageFolder(file_path)

def create_dataloaders(dataset, test_size, val_size, batch_size):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomApply([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(),
            ], p=0.5),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]),
        'test':transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    }

    total_size = len(dataset)
    test_size = int(test_size*total_size)
    val_size = int(val_size*total_size)
    train_size = total_size - (test_size + val_size)

    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

    train_set = CustomDataset(train_subset, transform=data_transforms['train'])
    val_set = CustomDataset(val_subset, transform=data_transforms['valid'])
    test_set = CustomDataset(test_subset, transform=data_transforms['test'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

