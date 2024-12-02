from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from model import ft_net

def main():
    # Define default values
    lr = 0.05
    batchsize = 32 
    weight_decay = 5e-4
    total_epoch = 60 
    stride = 2
    droprate = 0.5
    linear_num = 512
    name = 'ft_ResNet50'
    data_dir = './Market/pytorch'
    num_classes = 751


    # Enable CUDA
    cudnn.enabled = True
    cudnn.benchmark = True

    # Data transformations
    h, w = 256, 128
    transform_train_list = [
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_val_list = [
        transforms.Resize(size=(h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batchsize, shuffle=True,
            num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True
        )
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    use_gpu = torch.cuda.is_available()

    # Training function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        y_loss = {'train': [], 'val': []}
        y_err = {'train': [], 'val': []}

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')

            for phase in ['train', 'val']:
                model.train(phase == 'train')

                pbar = tqdm(total=dataset_sizes[phase], desc=phase)
                running_loss = 0.0
                running_corrects = 0.0

                for data in dataloaders[phase]:
                    inputs, labels = data
                    now_batch_size = inputs.size(0)
                    pbar.update(now_batch_size)

                    # skip last batch as it may not be of fixed size
                    if now_batch_size < batchsize:
                        continue

                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * now_batch_size
                    running_corrects += torch.sum(preds == labels.data).item()
                    # print(running_loss, running_corrects)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                print("this",epoch_loss, epoch_acc)
                
                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0 - epoch_acc)

                pbar.set_postfix({'Loss': epoch_loss, 'Acc': epoch_acc})
                pbar.close()

                if phase == 'train':
                    scheduler.step()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return model

    # Initialize model
    model = ft_net(num_classes, droprate, stride, linear_num=linear_num)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(
        model.parameters(), lr=lr,
        weight_decay=weight_decay, momentum=0.9, nesterov=True
    )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=total_epoch * 2 // 3, gamma=0.1
    )
    model = train_model(
        model, criterion, optimizer_ft, exp_lr_scheduler,
        num_epochs=total_epoch
    )
    # torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()