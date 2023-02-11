import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

def cifar100_dataloader(root_dir, batch_size, num_workers, download=True):
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

def imagenet_dataloader(data_path, batch_size, shuffle=True):
    # Define transformations to be applied to the images
    data_transforms = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the data from the directory structure
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.JPEG'):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                image_paths.append(image_path)
                labels.append(label)

    # Convert labels to integers
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    labels = [label_to_int[label] for label in labels]

    # Create a PyTorch dataset and data loader
    dataset = ImageNetDataset(image_paths, labels, data_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label
