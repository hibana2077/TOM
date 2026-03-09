import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add src to path so we can import sDTW and uDTW
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from sDTW import sDTW
from uDTW import uDTW


def power_norm_for_vis(arr, power=0.1):
    arr = np.maximum(arr, 0.0)
    vmax = float(arr.max())
    if vmax > 0:
        arr = arr / (vmax + 1e-12)
    arr = np.power(arr + 1e-12, power)
    vmax = float(arr.max())
    if vmax > 0:
        arr = arr / (vmax + 1e-12)
    return arr


def get_soft_alignment_from_cost(cost_matrix, x, y, gamma, use_cuda):
    cost_matrix = cost_matrix.detach().requires_grad_(True)
    sdtw = sDTW(use_cuda=use_cuda, gamma=gamma, normalize=False)
    func_dtw = sdtw._get_func_dtw(x, y)
    loss = func_dtw(cost_matrix, sdtw.gamma, sdtw.bandwidth)
    align = torch.autograd.grad(loss.sum(), cost_matrix)[0]
    return align


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class Sigmoid(nn.Module):
    def forward(self, a, b, input_tensor):
        return a * torch.sigmoid(input_tensor) + b


class SimpleSigmaNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = max(2 * input_dim, 8)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = Sigmoid()

    def forward(self, x, a, b):
        batch_size, length, _ = x.shape
        x = x.view(batch_size * length, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batch_size, length, -1).mean(2, keepdim=True)
        return self.sigmoid(a, b, x)


class sDTWvis:
    def __init__(self, use_cuda=None):
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda

    def _cost(self, x, y):
        sdtw = sDTW(use_cuda=self.use_cuda, gamma=1.0, normalize=False)
        return sdtw._calc_distance_matrix(x, y).detach()

    def alignments(self, x, y, gamma_values):
        cost = self._cost(x, y)
        out = []
        for gamma in gamma_values:
            A = get_soft_alignment_from_cost(cost, x, y, gamma, self.use_cuda)[0].detach().cpu().numpy()
            out.append((gamma, A))
        return out

    def plot(self, x, y, gamma_values, save_path="sdtw_vis.png"):
        aligns = self.alignments(x, y, gamma_values)
        fig, axes = plt.subplots(1, len(gamma_values), figsize=(4.2 * len(gamma_values), 3.7))
        if len(gamma_values) == 1:
            axes = [axes]
        for i, (gamma, A) in enumerate(aligns):
            im = axes[i].imshow(power_norm_for_vis(A), cmap="gray", origin="lower")
            axes[i].set_title(f"sDTW (gamma={gamma:g})")
            axes[i].set_xlabel("Y")
            axes[i].set_ylabel("X")
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        print(f"Saved: {save_path}")


class uDTWvis:
    def __init__(self, a=1.5, b=0.5, beta=1.0, use_cuda=None):
        self.a = a
        self.b = b
        self.beta = beta
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda

    def _effective_cost(self, x, y):
        sigmanet = SimpleSigmaNet(input_dim=x.shape[-1]).to(x.device)
        sigmanet.apply(weight_init)
        sigma_x = sigmanet(x, self.a, self.b).detach()
        sigma_y = sigmanet(y, self.a, self.b).detach()
        udtw = uDTW(use_cuda=self.use_cuda, gamma=1.0, normalize=False)
        cost, _ = udtw._calc_distance_matrix(x, y, sigma_x, sigma_y, beta=self.beta)
        return cost.detach()

    def alignments(self, x, y, gamma_values):
        cost = self._effective_cost(x, y)
        out = []
        for gamma in gamma_values:
            A = get_soft_alignment_from_cost(cost, x, y, gamma, self.use_cuda)[0].detach().cpu().numpy()
            out.append((gamma, A))
        return out

    def plot(self, x, y, gamma_values, save_path="udtw_vis.png"):
        aligns = self.alignments(x, y, gamma_values)
        fig, axes = plt.subplots(1, len(gamma_values), figsize=(4.2 * len(gamma_values), 3.7))
        if len(gamma_values) == 1:
            axes = [axes]
        for i, (gamma, A) in enumerate(aligns):
            im = axes[i].imshow(power_norm_for_vis(A), cmap="gray", origin="lower")
            axes[i].set_title(f"uDTW (gamma={gamma:g})")
            axes[i].set_xlabel("Y")
            axes[i].set_ylabel("X")
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        print(f"Saved: {save_path}")


def main():
    torch.manual_seed(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x = torch.rand((1, 30, 10), device=device)
    y = torch.rand((1, 30, 10), device=device)
    gammas = [1.0, 0.1, 0.01, 0.001, 0.0001]

    sDTWvis(use_cuda=use_cuda).plot(x, y, gammas, save_path="sdtw_vis.png")
    uDTWvis(use_cuda=use_cuda).plot(x, y, gammas, save_path="udtw_vis.png")

    plt.show()


if __name__ == "__main__":
    main()
