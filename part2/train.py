import os
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse

def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms['valid']),
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=64, shuffle=False),
    }

    return dataloaders, image_datasets

def build_model(arch, hidden_units, learning_rate, gpu):
    if arch == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    elif arch == 'vgg13':
        model = models.vgg13(weights='VGG13_Weights.DEFAULT')
    else:
        raise ValueError(f'Unsupported architecture: {arch}')

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if gpu and torch.cuda.is_available():
        model.cuda()

    return model, criterion, optimizer

def train_model(model, dataloaders, criterion, optimizer, epochs, gpu, print_every=40):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for images, labels in dataloaders['train']:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images)
                        valid_loss += criterion(log_ps, labels).item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

def save_checkpoint(model, image_datasets, save_dir, arch, hidden_units, learning_rate, epochs):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    dataloaders, image_datasets = load_data(args.data_dir)
    model, criterion, optimizer = build_model(args.arch, args.hidden_units, args.learning_rate, args.gpu)
    train_model(model, dataloaders, criterion, optimizer, args.epochs, args.gpu)
    save_checkpoint(model, image_datasets, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs)

if __name__ == '__main__':
    main()


