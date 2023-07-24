import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

class ImageNet(datasets.ImageNet):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../imagenet_data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

def get_imagenet_train_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform

def get_imagenet_test_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return test_transform


def get_imagenet_dataloaders(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_imagenet_train_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = get_imagenet_test_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_set = ImageNet(
        root=data_folder, split='train', transform=train_transform
    )
    num_data = len(train_set)
    test_set = datasets.ImageNet(
        root=data_folder, split='val', transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return train_loader, test_loader, num_data

if __name__ == "__main__":
    train_loader, test_loader, num_data = get_imagenet_dataloaders(64, 64, 4)
    print(num_data)
    import pdb
    pdb.set_trace()