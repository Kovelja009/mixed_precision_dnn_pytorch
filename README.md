# MIXED PRECISION DNNs

This is the pytorch implementation of the paper:  All you need is a good parametrization, ICLR 2020.

> [**Mixed Precision DNNs: All you need is a good parametrization**](https://openreview.net/forum?id=Hyx0slrFvH&noteId=Hyx0slrFvH&invitationId=ICLR.cc/2020/Conference/Paper2519),            
> Uhlich, Stefan and Mauch, Lukas and Cardinaux, Fabien and Yoshiyama, Kazuki and Garcia, Javier Alonso and Tiedemann, Stephen and Kemp, Thomas and Nakamura, Akira. 
> ICLR 2020 
> *arXiv technical report ([arXiv 1905.11452]( https://arxiv.org/abs/1905.11452))*

![](imgs/bitwidth.png)

## Abstract 
Efficient deep neural network (DNN) inference on mobile or embedded devices typically involves quantization of the network parameters and activations. In particular, mixed precision networks achieve better performance than networks with homogeneous bitwidth for the same size constraint. Since choosing the optimal bitwidths is not straight forward, training methods, which can learn them, are desirable. Differentiable quantization with straight-through gradients allows to learn the quantizer's parameters using gradient methods. We show that a suited parametrization of the quantizer is the key to achieve a stable training and a good final performance. Specifically, we propose to parametrize the quantizer with the step size and dynamic range. The bitwidth can then be inferred from them. Other parametrizations, which explicitly use the bitwidth, consistently perform worse. We confirm our findings with experiments on CIFAR-10 and ImageNet and we obtain mixed precision DNNs with learned quantization parameters, achieving state-of-the-art performance. 

## Dependencies 
All needed dependencies are listed in the requirements.txt file.

To load them you can use virtual environments:
```
python3 -m venv venv
```
Then activate the virtual environment:
```
source venv/bin/activate
```
And install the dependencies:
```
pip install -r requirements.txt
```

## Running mixed precision training 

This repository provides the code to train ResNet18 with mixed precision. 

The training setup can be adjusted in the config files (.cfg). The setup used in the paper results are included in the provided configuration files. 

For uniform and power-of-two
quantization, there are three different parametrizations possible.
Depending on what we want to learn, the following parametrization
should be used:

* Fixed bitwidth: Use `b_xmax`
* Learnable bitwidth: Use `delta_xmax` (uniform) and `xmin_xmax`
  (power-of-two)

Example: In order to learn a network both **weights** and **activations** where we optimize
for step size (`delta`) and the maximum value (`xmax`) you can use the following command, but first need to go to `pytorch_implementation` folder:

```
python training.py PARAMETRIC_FP_WAQ_DELTA_XMAX_RLR --cfg train_resnet_quant_fp.cfg
```

## Seeing the results
If you want to see the Training and Validation Loss after training, you can run the following command:
```
python training_plot.py experiments/<name of the experiment>
```
Where the name of the experiment for the example above is `PARAMETRIC_FP_WAQ_DELTA_XMAX_RLR`.