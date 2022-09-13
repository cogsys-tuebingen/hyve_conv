from hyve.hyve_convolution import HyVEConv
import torch.nn as nn


def get_trainable_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params


if __name__ == '__main__':
    in_channels = 200
    out_channels = 25
    kernel_size = 3
    num_of_wrois = 5
    wavelength_range = (450, 1000)
    bias = False

    hyveConv_pp = HyVEConv(out_channels=out_channels,
                           num_of_wrois=num_of_wrois,
                           kernel_size=kernel_size,
                           wavelength_range=wavelength_range,
                           enable_extension=True,
                           bias=bias)

    hyveConv = HyVEConv(out_channels=out_channels,
                        num_of_wrois=num_of_wrois,
                        kernel_size=kernel_size,
                        wavelength_range=wavelength_range,
                        enable_extension=False,
                        bias=bias)

    conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)

    print("List the trainable parameters for each model:")
    print(f"[Conv2d] Trainable parameters {get_trainable_parameters(conv2d)}")
    print(f"[HyVEConv] Trainable parameters {get_trainable_parameters(hyveConv)}")
    print(f"[HyVEConv++] Trainable parameters {get_trainable_parameters(hyveConv_pp)}")

