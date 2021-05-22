from torchvision import datasets, models, transforms
import torch
from torch import nn, optim



def network(arch, hidden_dim, output_dim, drop_prob):
    if arch == 'resnet18':
        print("Using pretrained model: resnet18")
        model = models.resnet18(pretrained=True)
    elif arch == 'resnet34':
        print("Using pretrained model: resnet34")
        model = models.resnet34(pretrained=True)
    elif arch == 'resnet50':
        print("Using pretrained model: resnet50")
        model = models.resnet50(pretrained=True)
    elif arch == 'resnet101':
        print("Using pretrained model: resnet101")
        model = models.resnet101(pretrained=True)
    elif arch == 'resnet101':
        print("Using pretrained model: resnet101")
        model = models.resnet152(pretrained=True)
    else:
        print(f"Oops!! Selected model, {arch}, is not available. Using ResNet34 by default.")
        model = models.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, output_dim),
        nn.LogSoftmax(dim=1)
    )

    model.fc = classifier
    return model



def loss_function():
    return nn.NLLLoss()



def optimizer(model, lr):
    return optim.Adam(model.fc.parameters(), lr)



def save_model(model, save_dir, arch, epochs, lr, hidden_dims):
    if save_dir is '':
        save_path = f'./checkpoint_{arch}.pth'
    else:
        save_path = save_dir + f'/checkpoint_{arch}.pth'

    model.cpu()

    checkpoint = {
        'arch': 'resnet34',
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'epochs': epochs,
        'lr': lr,
        'hidden_dim': hidden_dims,
        'optim_state': optimizer(model,lr)
    }

    torch.save(checkpoint, save_path)



def load_model(checkpoint_path):
    trained_model = torch.load(checkpoint_path)
    model = network(arch=trained_model['arch'], hidden_dim=trained_model['hidden_dim'],
                          output_dim=102, drop_prob=0.1)

    model.class_to_idx = trained_model['class_to_idx']
    model.load_state_dict(trained_model['state_dict'])
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    #optimizer.load_state_dict(checkpoint['optim_state'])

    print(f"Successfully loaded model with {trained_model['arch']} architecture")
    return model, optimizer