import argparse
import configparser
import attr

import sys
import os


@attr.s
class Configuration(object):
    experiment = attr.ib()
    num_layers = attr.ib(converter=int)
    shortcut_type = attr.ib()
    initial_learning_rate = attr.ib(converter=float)
    optimizer = attr.ib(default=None)
    weightfile = attr.ib(default=None)

    w_quantize = attr.ib(default=None)
    a_quantize = attr.ib(default=None)

    # Uniform quantization (sign=True):
    #   xmax = stepsize * ( 2**(bitwidth-1) - 1 )
    #      -> xmax_min = stepsize_min * ( 2**(bitwidth_min-1) - 1)
    #      -> xmax_max = stepsize_max * ( 2**(bitwdith_max-1) - 1)
    # Pow2 quantization (sign=True, zero=True):
    #   xmax = xmin * 2**(2**(bitwidth-2) - 1)
    w_stepsize = attr.ib(converter=float, default=2**-3)
    w_stepsize_min = attr.ib(converter=float, default=2**-8)
    w_stepsize_max = attr.ib(converter=float, default=1)
    w_xmin_min = attr.ib(converter=float, default=2**-16)
    w_xmin_max = attr.ib(converter=float, default=127)
    w_xmax_min = attr.ib(converter=float, default=2**-8)
    w_xmax_max = attr.ib(converter=float, default=127)
    w_bitwidth = attr.ib(converter=int, default=4)
    w_bitwidth_min = attr.ib(converter=int, default=2)  # one bit for sign
    w_bitwidth_max = attr.ib(converter=int, default=8)

    # Uniform quantization (sign=False):
    #   xmax = stepsize * ( 2**bitwidth - 1 )
    # Pow2 quantization (sign=False, zero=True)
    #   xmax = xmin * 2**(2**(bitwidth-1) - 1)
    a_stepsize = attr.ib(converter=float, default=2**-3)
    a_stepsize_min = attr.ib(converter=float, default=2**-8)
    a_stepsize_max = attr.ib(converter=float, default=1)
    a_xmin_min = attr.ib(converter=float, default=2**-14)
    a_xmin_max = attr.ib(converter=float, default=255)
    a_xmax_min = attr.ib(converter=float, default=2**-8)
    a_xmax_max = attr.ib(converter=float, default=255)
    a_bitwidth = attr.ib(converter=int, default=4)
    a_bitwidth_min = attr.ib(converter=int, default=1)
    a_bitwidth_max = attr.ib(converter=int, default=8)

    target_weight_kbytes = attr.ib(converter=float, default=-1.)
    target_activation_kbytes = attr.ib(converter=float, default=-1.)
    target_activation_type = attr.ib(default='max')

    initial_cost_lambda2 = attr.ib(converter=float, default=0.1)
    initial_cost_lambda3 = attr.ib(converter=float, default=0.1)

    scale_layer = attr.ib(converter=bool, default=False)


def get_args():
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    description = "Getting params from cfg file"
    parser = argparse.ArgumentParser(description)
    parser.add_argument('experiment')
    parser.add_argument('--gpu', metavar='NUMBER', type=int,
                        help="use the (n+1)'th GPU, thus counting from 0",
                        default=0)
    parser.add_argument('--cfg', metavar='STRING',
                        help="experiment configuration file",
                        default=f"{os.path.splitext(sys.argv[0])[0]}.cfg")
    args = parser.parse_args()
    return args