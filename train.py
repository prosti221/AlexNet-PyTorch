import torch
from model import AlexNet
from utils.data_loader import *
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    batch_size = 64
    lr = 0.0001
    num_epoch = 90

    model = AlexNet()
    #model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_loader, val_loader = cifar100_dataloader('./data', batch_size, 8, download=True) # For CIFAR100

    # For loading ImageNet data set use this
    '''
    train_loader = imagenet_dataloader('data/imagenet-mini/train', batch_size=batch_size)
    val_loader = imagenet_dataloader('data/imagenet-mini/val', batch_size=batch_size)
    '''
    writer = SummaryWriter()
    steps = 1
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

            loss.backward()
            optimizer.step()

            writer.add_scalar('Training Loss', loss.item(), steps)

            if steps % 100 == 0:
                with torch.no_grad():
                    _, preds = torch.max(outputs, 1)
                    accuracy = torch.sum(preds == labels)

                    writer.add_scalar('Training Accuracy', accuracy.item(), steps)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, steps, loss.item(), accuracy.item()))

            if steps % 1000 == 0:
                checkpoint_path = "./checkpoints/checkpoint_epoch{}_step{}.pth".format(epoch+1, steps)
                torch.save(model.state_dict(), checkpoint_path)
                
            steps += 1
        lr_scheduler.step()
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % 10 == 0:
            writer.add_graph(model, inputs)

    writer.close()

    print('Finished Training')
