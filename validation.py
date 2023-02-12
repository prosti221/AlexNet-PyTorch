import torch
import argparse
from model import AlexNet
from utils.data_loader import *
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='Validation script for a AlexNet')
    parser.add_argument('-model', type=str, default='./models/model.pth', help='path to model')
    parser.add_argument('-dataset', '--dataset', default='./data/ILSVRC2012_img_val', help='path to validation data')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading')

    return parser.parse_args()

def validate(PATH):
    return val_loss, val_acc


if __name__ == '__main__':
    args = get_args()

    
    # ImageNet
    #val_loader = imageNet_dataloader(args.dataset, args.batch_size, args.num_workers)

    #CIFAR-10
    _, val_loader = cifar100_dataloader(args.dataset, args.batch_size, args.num_workers, download=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    torch.load(args.model)
    model.eval()  

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        val_loss += loss.item() * inputs.size(0)

        _, predictions = torch.max(outputs, 1)
        val_correct += (predictions == labels).sum().item()
        val_total += inputs.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

