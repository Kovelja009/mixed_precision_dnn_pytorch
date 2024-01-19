import numpy as np
import nnabla as nn
import nnabla.functions as F
import torch
from nnabla.parametric_functions import parametric_function_api
from nnabla.parameter import get_parameter_or_create
from nnabla.initializer import ConstantInitializer
from ..module_quantizer import Quantizer


# CASE B: PARAMETRIZATION BY D AND XMAX
@parametric_function_api("parametric_fp_d_xmax", [
    ('d', 'step size (float)', '()', True),
    ('xmax', 'dynamic range (float)', '()', True),
])
def parametric_fixed_point_quantize_d_xmax(x, sign=True,
                                           d_init=2 ** -4, d_min=2 ** -8, d_max=2 ** 8,
                                           xmax_init=1, xmax_min=0.001, xmax_max=10,
                                           fix_parameters=False):
    """Parametric version of `fixed_point_quantize` where the
    stepsize `d` and dynamic range `xmax` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """

    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(v) / np.log(2.))

    d = get_parameter_or_create("d", (),
                                ConstantInitializer(d_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    xmax = get_parameter_or_create("xmax", (),
                                   ConstantInitializer(xmax_init),
                                   need_grad=True,
                                   as_need_grad=not fix_parameters)

    # ensure that stepsize is in specified range and a power of two
    d = quantize_pow2(clip_scalar(d, d_min, d_max))

    # ensure that dynamic range is in specified range
    xmax = clip_scalar(xmax, xmax_min, xmax_max)

    # compute min/max value that we can represent
    if sign:
        xmin = -xmax
    else:
        xmin = nn.Variable((1,), need_grad=False)
        xmin.d = 0.

    # broadcast variables to correct size
    d = broadcast_scalar(d, shape=x.shape)
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # apply fixed-point quantization
    return d * F.round(F.clip_by_value(x, xmin, xmax) / d)


@parametric_function_api("parametric_fp_d_b", [
    ('n', 'bitwidth (float)', '()', True),
    ('d', 'step size (float)', '()', True),
])
def parametric_fixed_point_quantize_d_b(x, sign=True,
                                        n_init=8, n_min=2, n_max=16,
                                        d_init=2 ** -4, d_min=2 ** -8, d_max=2 ** 8,
                                        fix_parameters=False):
    """Parametric version of `fixed_point_quantize` where the
    bitwidth `b` and stepsize `d` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """

    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(v) / np.log(2.))

    n = get_parameter_or_create("n", (),
                                ConstantInitializer(n_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    d = get_parameter_or_create("d", (),
                                ConstantInitializer(d_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)

    # ensure that bitwidth is in specified range and an integer
    n = F.round(clip_scalar(n, n_min, n_max))
    if sign:
        n = n - 1

    # ensure that stepsize is in specified range and a power of two
    d = quantize_pow2(clip_scalar(d, d_min, d_max))

    # ensure that dynamic range is in specified range
    xmax = d * (2 ** n - 1)

    # compute min/max value that we can represent
    if sign:
        xmin = -xmax
    else:
        xmin = nn.Variable((1,), need_grad=False)
        xmin.d = 0.

    # broadcast variables to correct size
    d = broadcast_scalar(d, shape=x.shape)
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # apply fixed-point quantization
    return d * F.round(F.clip_by_value(x, xmin, xmax) / d)


# CASE A: PARAMETRIZATION BY B AND XMAX
@parametric_function_api("parametric_fp_b_xmax", [
    ('n', 'bitwidth (float)', '()', True),
    ('xmax', 'dynamic range (float)', '()', True),
])
def parametric_fixed_point_quantize_b_xmax(x, sign=True,
                                           n_init=8, n_min=2, n_max=16,
                                           xmax_init=1, xmax_min=0.001, xmax_max=10,
                                           fix_parameters=False):
    """Parametric version of `fixed_point_quantize` where the
    bitwidth `b` and dynamic range `xmax` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """

    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(v) / np.log(2.))

    n = get_parameter_or_create("n", (),
                                ConstantInitializer(n_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    xmax = get_parameter_or_create("xmax", (),
                                   ConstantInitializer(xmax_init),
                                   need_grad=True,
                                   as_need_grad=not fix_parameters)

    # ensure that bitwidth is in specified range and an integer
    n = F.round(clip_scalar(n, n_min, n_max))
    if sign:
        n = n - 1

    # ensure that dynamic range is in specified range
    xmax = clip_scalar(xmax, xmax_min, xmax_max)

    # compute step size from dynamic range and make sure that it is a pow2
    d = quantize_pow2(xmax / (2 ** n - 1))

    # compute min/max value that we can represent
    if sign:
        xmin = -xmax
    else:
        xmin = nn.Variable((1,), need_grad=False)
        xmin.d = 0.

    # broadcast variables to correct size
    d = broadcast_scalar(d, shape=x.shape)
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # apply fixed-point quantization
    return d * F.round(F.clip_by_value(x, xmin, xmax) / d)


def get_quantization_results(vector, sign, quantization_mode):
    if quantization_mode == (True, True, False):
        return parametric_fixed_point_quantize_d_b(vector, sign=sign)
    elif quantization_mode == (True, False, True):
        return parametric_fixed_point_quantize_b_xmax(vector, sign=sign)
    else:
        return parametric_fixed_point_quantize_d_xmax(vector, sign=sign)


def compare_two_quantizers(vector=[2.544, 12.345, 1.233], sign: bool = True,
                           quantization_mode: tuple[bool, bool, bool] = (False, True, True)):


    print(f"unquantized values: {vector}")
    # old quantizer
    x = nn.Variable((len(vector),))
    x.d = np.array(vector)
    y = get_quantization_results(x, sign, quantization_mode)
    y.forward()
    old_results = np.array(y.d)
    print(f'old quantizer results: {old_results}')

    # new quantizer
    quantizer = Quantizer(quantization_mode)
    x = torch.Tensor(vector)
    y = quantizer(x)
    new_results = y.detach().numpy()
    print(f'new quantizer results: {new_results}')

    # compare results
    print(f'Are the results the same: {np.allclose(old_results, new_results)}')

if __name__ == '__main__':
    compare_two_quantizers(vector=[2.544, 0.345, 1.233, 0.0011], sign=True, quantization_mode=(True, False, True))
