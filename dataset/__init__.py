from .cifar100 import get_cifar100_dataloaders
from .imagenet import get_imagenet_dataloaders


def get_dataset(dataset_type, batchsize, val_batchsize, num_workers):
    if dataset_type == "cifar100":
        train_loader, val_loader, num_data = get_cifar100_dataloaders(batch_size=batchsize, val_batch_size=val_batchsize, num_workers=num_workers)
        num_classes = 100
    elif dataset_type == "imagenet":
        train_loader, val_loader, num_data = get_imagenet_dataloaders(batch_size=batchsize, val_batch_size=val_batchsize, num_workers=num_workers)
        num_classes = 1000
    return train_loader, val_loader, num_data, num_classes
