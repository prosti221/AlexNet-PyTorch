import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

def get_dataloader(root_dir, batch_size, num_workers, download=True):
    # Define the transforms to apply to the CIFAR100 dataset
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load the CIFAR100 dataset
    train_set = torchvision.datasets.CIFAR100(root=root_dir, train=True,
                                            download=download, transform=transform)

    test_set = torchvision.datasets.CIFAR100(root=root_dir, train=False,
                                           download=download, transform=transform)

    # Define the dataloader for the training set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    # Define the dataloader for the test set
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
