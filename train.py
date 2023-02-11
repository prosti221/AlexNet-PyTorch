import torch
from model import AlexNet
from utils.data_loader import *

if __name__ == '__main__':
    batch_size = 128
    lr = 0.0001
    num_epoch = 90

    model = AlexNet()
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_loader, val_loader = get_dataloader('./data', 128, 8, download=True) # For CIFAR100

    # For loading ImageNet data set use this
    '''
    train_loader = imagenet_data_loader('data/imagenet-mini/train', batch_size=32)
    val_loader = imagenet_data_loader('data/imagenet-mini/val', batch_size=32)
    '''

    for epoch in range(num_epoch):
        running_loss = 0.0
        last_lost = 0.0
        model.train(True)

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            #inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            print(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
        lr_scheduler.step()


    print('Finished Training')
