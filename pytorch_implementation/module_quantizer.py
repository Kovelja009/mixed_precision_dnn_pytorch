import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter



# TODO: 1. Add pow2 quantization to the quantizer (not just uniform quantization)
class Quantizer(nn.Module):
    def __init__(
            self,
            to_train: tuple[bool, bool, bool],
            b: tuple[float, float, float] = (8.0, 2.0, 16.0),
            d: tuple[float, float, float] = (2 ** -4, 2 ** -8, 2 ** 8),
            xmax: tuple[float, float, float] = (1.0, 0.001, 10.0),
            sign: bool = True,
    ) -> None:
        super().__init__()
        train_b, train_d, train_xmax = to_train
        self.train_b = train_b
        self.train_d = train_d
        self.train_xmax = train_xmax

        b_init, self.b_min, self.b_max = b
        self.b = Parameter(torch.tensor(b_init), requires_grad=train_b)

        d_init, self.d_min, self.d_max = d
        self.d = Parameter(torch.tensor(d_init), requires_grad=train_d)

        xmax_init, self.xmax_min, self.xmax_max = xmax
        self.xmax = Parameter(torch.tensor(xmax_init), requires_grad=train_xmax)
        self.sign = sign

    def forward(self, x: Tensor):
        def quantize_pow2(v):
            return 2 ** torch.round(torch.log(v) / np.log(2.0))

        # ensure that bitwidth is in specified range and an integer
        b = torch.round(torch.clamp(self.b, self.b_min, self.b_max))
        if self.sign:
            b = b - 1

        # ensure that stepsize is in specified range and a power of two
        d = quantize_pow2(
            torch.clamp(self.d, self.d_min, self.d_max)
            if self.train_d
            else self.xmax / (2 ** b - 1)
        )

        # ensure that dynamic range is in specified range
        xmax = torch.clamp(self.xmax, self.xmax_min, self.xmax_max) if self.train_xmax else d * (2 ** b - 1)

        # compute min/max value that we can represent
        xmin = -xmax if self.sign else torch.tensor(0.0, requires_grad=False)

        # apply fixed-point quantization
        return d * torch.round(torch.clamp(x, xmin, xmax) / d)

    def __str__(self):
        return f"Quantizer(b={self.b.data}   (is_trained={self.train_b}),        d={self.d.data}   (is_trained={self.train_d}),       xmax={self.xmax.data}   (is_trained={self.train_xmax}))"


class ModuleQuantizer(nn.Module):
    def __init__(self, network: nn.Module, input_shape=(1, 3, 224, 224), **q_kwargs) -> None:
        super().__init__()
        self._quantizers: dict[str, tuple[Quantizer, Parameter]] = {}
        for name, parameter in network.named_parameters():
            self._quantizers[name] = Quantizer(**q_kwargs), parameter
        self.network = network
        self._activations = count_and_record_activation_layers(network, input_shape)
        self._quantizers_a: dict[nn.Module, tuple[Quantizer, Tensor]] = {}
        for module, shape in self._activations.items():
            self._quantizers_a[module] = Quantizer(**q_kwargs), shape

    def forward(self, inputs, unquantize_after: bool = False):
        _old_param_data = {}
        for name, (quantizer, parameter) in self._quantizers.items():
            _old_param_data[name] = parameter.data
            parameter.data = quantizer(parameter.data)
        outputs = self.network(inputs)
        # Restore original parameter values
        if unquantize_after:
            for name, (_, parameter) in self._quantizers.items():
                parameter.data = _old_param_data[name]
        return outputs




# it says that it has 17 activation layers, but in the list thers is only 9 (half of them), maybe
# it is because of residual blocks (it is said that two modules are the same if they have the same type, shape and parameters)
# TODO: discuss with Yifan
# Function to count the number of activation layers and record their shapes
def count_and_record_activation_layers(model, input_shape):
    activation_layers = [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU]
    num_activation_layers = 0
    activation_shapes = {}

    def hook_fn(module, input, output):
        nonlocal num_activation_layers
        if any(isinstance(module, act_fn) for act_fn in activation_layers):
            num_activation_layers += 1
            activation_shapes[module] = output.shape

    for layer in model.modules():
        layer.register_forward_hook(hook_fn)

    # Pass a dummy input through the model to trigger forward pass
    dummy_input = torch.randn(input_shape)  # Adjust the shape to match your input data
    with torch.no_grad():
        model(dummy_input)

    return activation_shapes