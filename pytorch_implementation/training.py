import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from loss_impl import quantize_loss
from module_quantizer import ModuleQuantizer
from config_parse import Configuration, get_args
import csv
import configparser
import os
from tqdm import tqdm


def train_network(cfg):
    # Define a custom dataset and dataloaders (in this case, using CIFAR-10 as an example)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define the ResNet-18 model
    model = resnet18(weights=None)
    num_classes = 10  # Number of classes in CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the last fully connected layer
    model_q = ModuleQuantizer(model, to_train=(False, True, True))  # delta and xmax

    # Define loss function and optimizer
    criterion = torch.nn.functional.cross_entropy
    optimizer = optim.SGD(model_q.network.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_q.to(device)

    num_epochs = 10

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model_q.network.train()
        running_train_loss = 0.0

        # Training
        for i, data in enumerate(tqdm(trainloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model_q.network(inputs)
            loss = quantize_loss(model_q, criterion, outputs, labels, cfg, device)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.detach().item()


        # Validation
        model_q.network.eval()
        running_test_loss = 0.0
        for i, data in enumerate(tqdm(testloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model_q.network(inputs)
            loss = quantize_loss(model_q, criterion, outputs, labels, cfg, device)

            running_test_loss += loss.item()

        l_train = running_train_loss / len(trainloader)
        l_test = running_test_loss / len(testloader)

        print(f"Epoch {epoch + 1}, Train_loss: {l_train}, Test_loss: {l_test}")
        train_losses.append((epoch + 1, l_train))
        test_losses.append((epoch + 1, l_test))

    return train_losses, test_losses


# Type of command:
# python training.py PARAMETRIC_FP_WAQ_DELTA_XMAX_INIT_ADAM --cfg train_resnet_quant_fp.cfg

if __name__ == '__main__':
    # read arguments
    args = get_args()
    print(args)

    cfgs = configparser.ConfigParser()
    cfgs.read(args.cfg)

    cfg = Configuration(**dict(cfgs[args.experiment].items()),
                        experiment=args.experiment)


    train_losses, test_losses = train_network(cfg)

    cfg.params_dir = f"{args.experiment}"
    save_path = f'experiments/{cfg.params_dir}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # write losses into csv file
    # train
    with open(f'{save_path}/train_losses.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss"])
        for loss in train_losses:
            writer.writerow(loss)

    # test
    with open(f'{save_path}/test_losses.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss"])
        for loss in test_losses:
            writer.writerow(loss)
