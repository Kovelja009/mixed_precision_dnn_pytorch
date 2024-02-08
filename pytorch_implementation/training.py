import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchmetrics.classification import MulticlassAccuracy
from loss_impl import quantize_loss
from module_quantizer import ModuleQuantizer
from config_parse import Configuration, get_args
import csv
import configparser
import os
from tqdm import tqdm


def train_network(cfg, should_quantize=True):
    # Define a custom dataset and dataloaders (in this case, using CIFAR-10 as an example)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = 10  # Number of classes in CIFAR-10

    # Define the ResNet-18 model

    if should_quantize:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the last fully connected layer
        # model_q = ModuleQuantizer(model, to_train=(False, True, True))  # delta and xmax
        model_q = ModuleQuantizer(model, to_train=(True, True, False))  # b and delta

    else:
        model_q = resnet18(weights=None)
        model_q.fc = nn.Linear(model_q.fc.in_features, num_classes)  # Replace the last fully connected layer



    # Define loss function and optimizer
    criterion = torch.nn.functional.cross_entropy

    if should_quantize:
        optimizer = optim.SGD(model_q.network.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = optim.SGD(model_q.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_q.to(device)

    num_epochs = 10

    train_losses = []
    test_losses = []

    train_accs = []
    test_accs = []
    accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)

    for epoch in range(num_epochs):
        if should_quantize:
            model_q.network.train()
        else:
            model_q.train()

        running_train_loss = 0.0
        running_batch_acc = 0.0

        ###### Training ######
        for i, data in enumerate(tqdm(trainloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model_q(inputs)
            if should_quantize:
                loss = quantize_loss(model_q, criterion, outputs, labels, cfg, device)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.detach().item()
            running_batch_acc += accuracy(outputs, labels).detach().item()


        ###### Validation ######
        if should_quantize:
            model_q.network.eval()
        else:
            model_q.eval()

        running_test_loss = 0.0
        running_test_acc = 0.0
        for i, data in enumerate(tqdm(testloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model_q(inputs)
            if should_quantize:
                loss = quantize_loss(model_q, criterion, outputs, labels, cfg, device)
            else:
                loss = criterion(outputs, labels)

            running_test_loss += loss.item()
            running_test_acc += accuracy(outputs, labels).detach().item()

        l_train = running_train_loss / len(trainloader)
        l_test = running_test_loss / len(testloader)

        a_train = running_batch_acc / len(trainloader)
        a_test = running_test_acc / len(testloader)

        print(f"Epoch {epoch + 1}, Train_loss: {l_train}, Test_loss: {l_test}, Train_acc: {a_train}, Test_acc: {a_test}")
        train_losses.append((epoch + 1, l_train))
        test_losses.append((epoch + 1, l_test))
        train_accs.append((epoch + 1, a_train))
        test_accs.append((epoch + 1, a_test))

    if should_quantize:
        bitwidths_dict = model_q.get_quantized_bitwidths()
        print('Bithwidths:')
        print(bitwidths_dict)
        print('------------------------------------')
        for name, (quantizer, _) in model_q._quantizers.items():
            print(quantizer)
    return train_losses, test_losses, train_accs, test_accs


def file_writting(path_file, row_names, data):
    with open(path_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_names)
        for row in data:
            writer.writerow(row)

# Type of command:
# python training.py PARAMETRIC_FP_WAQ_DELTA_XMAX_RLR --cfg train_resnet_quant_fp.cfg
if __name__ == '__main__':
    # read arguments
    args = get_args()
    print(args)
    should_quantize = True

    cfgs = configparser.ConfigParser()
    cfgs.read(args.cfg)

    cfg = Configuration(**dict(cfgs[args.experiment].items()),
                        experiment=args.experiment)


    train_losses, test_losses, train_accs, test_accs = train_network(cfg, should_quantize=should_quantize)

    cfg.params_dir = f"{args.experiment}"
    save_path = f'experiments/{cfg.params_dir}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # write losses and accuracies into csv file
    if should_quantize:
        train_losses_file = f'{save_path}/train_losses_q.csv'
        test_losses_file = f'{save_path}/test_losses_q.csv'
        train_accs_file = f'{save_path}/train_accs_q.csv'
        test_accs_file = f'{save_path}/test_accs_q.csv'
    else:
        train_losses_file = f'{save_path}/train_losses.csv'
        test_losses_file = f'{save_path}/test_losses.csv'
        train_accs_file = f'{save_path}/train_accs.csv'
        test_accs_file = f'{save_path}/test_accs.csv'

    file_writting(train_losses_file, ["epoch", "loss"], train_losses)
    file_writting(test_losses_file, ["epoch", "loss"], test_losses)
    file_writting(train_accs_file, ["epoch", "accuracy"], train_accs)
    file_writting(test_accs_file, ["epoch", "accuracy"], test_accs)
