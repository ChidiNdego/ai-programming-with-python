import argparse

from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import network

import time



def get_args():
    parser = argparse.ArgumentParser(description="Train a Deep Learning Model for Flower Image Classification")
    parser.add_argument('data_dir', type=str, help="data directory (required)")
    parser.add_argument('--save_dir', default='', type=str, help="directory to save generated checkpoints")
    parser.add_argument('--arch', default='resnet34',
                        help='available neural network architecture: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate for training')
    parser.add_argument('--hidden_dims', default=1024, type=int, help='number of neurons in hidden layer')
    parser.add_argument('--output_dims', default=102, type=int, help='number of output categories')
    parser.add_argument('--drop_prob', default=0.2, type=float, help='dropout probability')
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs for training')
    parser.add_argument('--gpu', default=False, action='store_true', help='available hardware to be used for training')
    return parser.parse_args()



mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def train_transforms():
    return transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])



def test_transforms():
    return transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])



def data_loaders(data_dir="./flowers"):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms())
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms())
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms())
    class_to_idx = train_data.class_to_idx

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    return train_loader, valid_loader, test_loader, class_to_idx



def train(model, train_loader, valid_loader, criterion, optimizer, epochs, print_every, use_gpu):
    if use_gpu and torch.cuda.is_available():
        print("Using GPU hardware...")
        model.cuda()

    steps = 0
    start = time.time()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            if use_gpu and torch.cuda.is_available():    
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
            
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    val_loss, accuracy = validate(model, valid_loader, criterion, use_gpu)
            
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(val_loss/len(valid_loader)),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
            
                running_loss = 0
            
                # Make sure training is back on
                model.train()
            
    end = time.time()
    print('Total time taken for training: {} mins'.format((end-start)/60))



def validate(model, data_loader, criterion, use_gpu):
    if use_gpu and torch.cuda.is_available():
        model.to('cuda')

    val_loss = 0
    accuracy = 0

    for data in data_loader:
        images, labels = data
        if use_gpu and torch.cuda.is_available():
            images, labels = images.to('cuda'), labels.to('cuda')

        with torch.no_grad():
            outputs = model.forward(images)
            val_loss += criterion(outputs, labels).item()
            ps = torch.exp(outputs)

            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    return val_loss, accuracy



def main():
    args = get_args()
    print_training_config(args)
    print("---------------------------------------------")
    print("Loading Data....")
    print("---------------------------------------------")
    train_loader, valid_loader, test_loader, class_to_idx = data_loaders(args.data_dir)
    print("Data Loaded!")
    print("---------------------------------------------")
    model = network.network(args.arch, args.hidden_dims, args.output_dims, args.drop_prob)
    model.class_to_idx = class_to_idx
    criterion = network.loss_function()
    optimizer = network.optimizer(model, args.learning_rate)
    print("Training model...")
    print("---------------------------------------------")
    train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, 32, args.gpu)
    print("Model successfully trained")
    print("---------------------------------------------")
    print("Saving model...")
    print("---------------------------------------------")
    network.save_model(model, args.save_dir, args.arch, args.epochs, args.learning_rate, args.hidden_dims)
    print("Model successfully saved!")



def print_training_config(args):
    print("Training Configuration:")
    print("---------------------------------------------")
    print(f"Architecture: {args.arch}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Hidden Units: {args.hidden_dims}")
    print(f"Output Units: {args.output_dims}")
    print(f"Dropout Probability: {args.drop_prob}")
    print(f"Epochs: {args.epochs}")
    print(f"Use GPU hardware?: {args.gpu}")
    print("---------------------------------------------")


if __name__ == '__main__':
    main()