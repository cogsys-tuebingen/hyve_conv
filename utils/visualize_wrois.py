import numpy as np
from statistics import NormalDist
import argparse
import matplotlib.pyplot as plt

from hyve.hyve_convolution import HyVEConv
from model.deephs_net_with_hyveconv import DeepHSNet_with_HyVEConv
import torch


def get_init_gaussian(gauss_num, wavelength_range):
    from hyve.gaussian import GaussDistributionModule
    gauss_variance_factor = (wavelength_range[1] - wavelength_range[0])
    gaussian = GaussDistributionModule(gauss_num,
                                       wavelength_range[0],
                                       (wavelength_range[1] - wavelength_range[0]),
                                       gauss_variance_factor
                                       )
    return np.stack([g.detach().numpy() for g in gaussian.scaled_params()]).transpose(1, 0)


def gauss(x, mean, variance):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-np.power(x - mean, 2)
                                                      / (2 * variance))


def calc_overlap(mean_1, variance_1, mean_2, variance_2):
    return NormalDist(mu=mean_1, sigma=np.sqrt(variance_1)).overlap(
        NormalDist(mu=mean_2, sigma=np.sqrt(variance_2)))


def calc_overlap_matrix(means, variances):
    assert len(means) == len(variances)
    matrix = np.zeros((len(means), len(means)))

    for x, (mean_1, variance_1) in enumerate(zip(means, variances)):
        for y, (mean_2, variance_2) in enumerate(zip(means, variances)):
            matrix[x, y] = calc_overlap(mean_1, variance_1, mean_2, variance_2)
            if np.isnan(matrix[x, y]):
                print("Break.")

    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visualize the underlying WROIs")
    parser.add_argument("--checkpoint", default=None, type=str)
    opt = parser.parse_args()

    conv = None
    if opt.checkpoint is not None:
        model = torch.load(opt.checkpoint)
        model = model.eval()
        conv = model.get_hyve_conv()
    else:
        print("No checkpoint given. Initialize new HyVEConv++")
        out_channels = 25
        kernel_size = 3
        num_of_wrois = 5
        wavelength_range = (450, 1000)
        bias = False

        conv = HyVEConv(out_channels=out_channels,
                        num_of_wrois=num_of_wrois,
                        kernel_size=kernel_size,
                        wavelength_range=wavelength_range,
                        enable_extension=True,
                        bias=bias)

    assert conv is not None and isinstance(conv, HyVEConv)

    means, variances = conv.get_gauss().scaled_params()
    means, variances = means.detach(), variances.detach()
    wavelengths = np.arange(conv.wavelength_range[0],
                            conv.wavelength_range[1],
                            (conv.wavelength_range[1] - conv.wavelength_range[0]) / 1000)

    print(f"Means: {means.tolist()}")
    print(f"Variances: {variances.tolist()}")
    print()
    print(f"Overlap matrix:")
    overlap_matrix = np.round(calc_overlap_matrix(means, variances), 2)
    max_overlap = overlap_matrix[~np.eye(
        overlap_matrix.shape[0], dtype=bool)].max()
    print(overlap_matrix)
    print(f"-> Max overlap: {max_overlap}")
    print()

    if conv.share_features:
        alpha, beta = conv.get_kernel_prototype_share_factors()
        print(f"HyVEConv++ factor alpha = {round(alpha.item(),2)}")
        print(f"HyVEConv++ factor beta = {round(beta.item(),2)}")

    plt.figure()
    plt.title(f"Wavelength ranges of Interest (WROIs)")
    for gauss_id, (mean, variance) in enumerate(zip(means, variances)):
        plt.plot(wavelengths, [gauss(w, mean, variance) for w in wavelengths], label=f'Gaussian {gauss_id}')

    plt.legend()
    plt.tight_layout()
    plt.xlabel("Wavelength (nm)")
    plt.show()




