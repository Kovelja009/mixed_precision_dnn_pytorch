import torch
import torch.nn as nn
from module_quantizer import ModuleQuantizer
from config_parse import Configuration, get_args
import numpy as np
#####################
import configparser
import os
#####################
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


# TODO: add support for pow2 quantization
def compute_weights_size(network: ModuleQuantizer, cfg):
    def quantize_pow2(v):
        return 2 ** torch.round(torch.log(v) / np.log(2.0))

    kbytes = None
    num_params = None

    for name, _ in network.network.named_parameters():
        quantizer, param = network._quantizers[name]
        param_per_layer = param.numel()

        # print(param.shape)
        # print(param_per_layer)
        # print(quantizer)
        # print("------------------")

        if cfg.w_quantize is not None:
            if cfg.w_quantize in ['parametric_fp_b_xmax',
                                  'parametric_fp_d_b',
                                  'parametric_pow2_b_xmax',
                                  'parametric_pow2_b_xmin']:
                # parametric quantization
                bitwidth = torch.round(torch.clamp(quantizer.b, cfg.w_bitwidth_min, cfg.w_bitwidth_max))

            elif cfg.w_quantize == 'parametric_fp_d_xmax':
                # this quantization methods do not have bitwidth, so we need to compute it
                xmax = quantizer.xmax

                # ensure that stepsize is in specified range and a power of two
                d_q = quantize_pow2(torch.clamp(quantizer.d, cfg.w_stepsize_min, cfg.w_stepsize_max))

                # ensure that dynamic range is in specified range
                xmax = torch.clamp(xmax, cfg.w_xmax_min, cfg.w_xmax_max)

                # compute real `xmax`
                xmax = torch.round(xmax / d_q) * d_q

                # we do not clip to `cfg.w_bitwidth_max` as xmax/d_q could correspond to more than 8 bit
                log2 = torch.log2(xmax / d_q + 1.0)
                ceil = torch.ceil(log2 + 1.0)
                bitwidth = max(ceil.data, float(cfg.w_bitwidth_min))

            elif cfg.w_quantize == 'parametric_pow2_xmin_xmax':
                raise ValueError("Not implemented yet -> parametric_pow2_xmin_xmax")
            elif cfg.w_quantize == 'fp' or cfg.w_quantize == 'pow2':
                # fixed quantization
                bitwidth = float(cfg.w_bitwidth)
            else:
                raise ValueError(f'Unknown quantization method {cfg.w_quantize}')

        else:
            # float precision
            bitwidth = 32.

        if kbytes is None:
            kbytes = param_per_layer * bitwidth / 8. / 1024.
            num_params = param_per_layer
        else:
            kbytes += param_per_layer * bitwidth / 8. / 1024.
            num_params += param_per_layer

    return num_params, kbytes


def compute_activation_size(network: ModuleQuantizer, cfg):
    def quantize_pow2(v):
        return 2 ** torch.round(torch.log(v) / np.log(2.0))
    kbytes = []
    num_activations = 0

    for module, (quantizer, shape) in network._quantizers_a.items():
        num_activations += np.prod(shape)

        if cfg.a_quantize is not None:
            if cfg.a_quantize in ['fp_relu', 'pow2_relu']:
                # fixed quantization
                bitwidth = float(cfg.a_bitwidth)

            elif cfg.a_quantize in ['parametric_fp_relu',
                                    'parametric_fp_b_xmax_relu',
                                    'parametric_fp_d_b_relu',
                                    'parametric_pow2_b_xmax_relu',
                                    'parametric_pow2_b_xmin_relu']:
                # parametric quantization
                bitwidth = torch.round(torch.clamp(quantizer.b, cfg.a_bitwidth_min, cfg.a_bitwidth_max))

            elif cfg.a_quantize in ['parametric_fp_d_xmax_relu']:
                # these quantization methods do not have bitwidth, so we need to compute it!
                # parametric quantization
                d = quantizer.d
                xmax = quantizer.xmax

                # ensure that stepsize is in specified range and a power of two
                d_q = quantize_pow2(torch.clamp(d, cfg.a_stepsize_min, cfg.a_stepsize_max))

                # ensure that dynamic range is in specified range
                xmax = torch.clamp(xmax, cfg.a_xmax_min, cfg.a_xmax_max)

                # compute real `xmax`
                xmax = torch.round(xmax / d_q) * d_q

                bitwidth = max(torch.ceil(torch.log2(xmax / d_q + 1.0)), cfg.a_bitwidth_min)

            elif cfg.a_quantize in ['parametric_pow2_xmin_xmax_relu']:
                raise ValueError("Not implemented yet -> parametric_pow2_xmin_xmax_relu")

            else:
                raise ValueError("Unknown quantization method {}".format(cfg.a_quantize))

        else:
            # float precision
            bitwidth = 32.

        kbytes.append(np.prod(shape) * bitwidth / 8. / 1024.)

    if cfg.target_activation_type == 'max':
        _kbytes = max(kbytes)
    elif cfg.target_activation_type == 'sum':
        _kbytes = sum(kbytes)

    return num_activations, _kbytes


def quantize_loss(network: ModuleQuantizer, loss_function, outputs, y, cfg, device):
    loss1 = loss_function(outputs, y)
    # print(f"Only network loss: {loss1}")

    cost_lamda2 = torch.Tensor([cfg.initial_cost_lambda2])
    cost_lamda2 = cost_lamda2.to(device)

    cost_lamda3 = torch.Tensor([cfg.initial_cost_lambda3])
    cost_lamda3 = cost_lamda3.to(device)

    if cfg.target_weight_kbytes > 0:
        num_params_w, kbytes_weights = compute_weights_size(network, cfg)
        loss2 = torch.relu(torch.Tensor([kbytes_weights - cfg.target_weight_kbytes])) ** 2
    else:
        loss2 = torch.Tensor([0.0])

    if cfg.target_activation_kbytes > 0:
        num_params_a, kbytes_activations = compute_activation_size(network, cfg)
        loss3 = torch.relu(torch.Tensor([kbytes_activations - cfg.target_activation_kbytes])) ** 2
    else:
        loss3 = torch.Tensor([0.0])

    loss2 = loss2.to(device)

    loss3 = loss3.to(device)

    loss = loss1 + cost_lamda2 * loss2 + cost_lamda3 * loss3
    # print(f"Loss with quantization constraints: {loss}")

    return loss

####################################################################
############################ TESTS #################################
def test_loss(cfg):
    # Load and preprocess a sample image to test the model
    image_path = 'quantizer_test/dog.png'  # Replace with the path to your image
    image = Image.open(image_path).convert('RGB')

    # Define the transformation to preprocess the image for ResNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the preprocessing transformation to the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # load resnet model from torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    quantized_model = ModuleQuantizer(model, to_train=(True, False, True))

    with torch.no_grad():
        output = quantized_model(input_batch)

    #get the predicted class index
    _, predicted_class = output.max(1)

    y = torch.zeros(output.shape)
    y[0,predicted_class] = 1.0

    # calculate loss
    quantized_loss = quantize_loss(quantized_model, torch.nn.functional.cross_entropy, input_batch, y, cfg)
    print(quantized_loss)


def fast_test(cfg):
    # Load and preprocess a sample image to test the model
    image_path = 'quantizer_test/dog.png'  # Replace with the path to your image
    image = Image.open(image_path).convert('RGB')

    # Define the transformation to preprocess the image for ResNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the preprocessing transformation to the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # load resnet model from torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    quantized_model = ModuleQuantizer(model, to_train=(True, False, True))

    num_activations, _kbytes = compute_activation_size(quantized_model, cfg)

    print(f"num_activations: {num_activations}")
    print(f"_kbytes: {_kbytes}")

if __name__ == '__main__':
    # read arguments
    args = get_args()
    print(args)

    cfgs = configparser.ConfigParser()
    cfgs.read(args.cfg)

    cfg = Configuration(**dict(cfgs[args.experiment].items()),
                        experiment=args.experiment)

    cfg.params_dir = f"{args.experiment}"
    if not os.path.exists(cfg.params_dir):
        os.makedirs(cfg.params_dir)

    test_loss(cfg)
