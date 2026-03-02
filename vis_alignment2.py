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


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, input_tensor):
        return a * torch.sigmoid(input_tensor) + b


class SimpleSigmaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.sigmoid = Sigmoid()

    def forward(self, x, a, b):
        batch_size = x.shape[0]
        length = x.shape[1]
        x = x.view(batch_size * length, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batch_size, length, -1).mean(2, keepdim=True)
        return self.sigmoid(a, b, x)


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


def build_sigma_matrix(sigma_x, sigma_y):
    # Sigma[m, n] = 0.5 * (sigma_x[m]^2 + sigma_y[n]^2)
    # sigma_x, sigma_y shape: [B, T, 1]
    sigma_x2 = sigma_x.pow(2)
    sigma_y2 = sigma_y.pow(2)
    sigma_xy = 0.5 * (
        sigma_x2.expand(-1, -1, sigma_y.shape[1])
        + sigma_y2.transpose(1, 2).expand(-1, sigma_x.shape[1], -1)
    )
    return sigma_xy


def main():
    torch.manual_seed(42)

    batch_size, len_x, len_y, dims = 1, 30, 30, 10
    a, b = 1.5, 0.5
    gamma_small, gamma_large = 0.01, 0.1
    beta_vis = 1.0
    bin_ratio = 0.05

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x = torch.rand((batch_size, len_x, dims), device=device)
    y = torch.rand((batch_size, len_y, dims), device=device)

    sigmanet = SimpleSigmaNet().to(device)
    sigmanet.apply(weight_init)
    sigma_x = sigmanet(x, a, b).detach()
    sigma_y = sigmanet(y, a, b).detach()

    # Base sDTW cost D
    sdtw_ref = sDTW(use_cuda=use_cuda, gamma=gamma_small, normalize=False)
    D_xy = sdtw_ref._calc_distance_matrix(x, y).detach()

    # uDTW effective cost C = D \odot Sigma^{-1} (in this codebase this is D_udtw)
    # and uncertainty regularizer term S_udtw = 0.5 * beta * log(sigma_xy)
    udtw_ref = uDTW(use_cuda=use_cuda, gamma=gamma_small, normalize=False)
    D_udtw, S_udtw = udtw_ref._calc_distance_matrix(x, y, sigma_x, sigma_y, beta=beta_vis)

    # Build Sigma explicitly for panel (e)
    Sigma_xy = build_sigma_matrix(sigma_x, sigma_y).detach()

    print(f"D_xy shape: {D_xy.shape}, mean={D_xy.mean().item():.4f}, std={D_xy.std().item():.4f}")
    print(f"D_udtw shape: {D_udtw.shape}, mean={D_udtw.mean().item():.4f}, std={D_udtw.std().item():.4f}")
    print(f"S_udtw shape: {S_udtw.shape}, mean={S_udtw.mean().item():.4f}, std={S_udtw.std().item():.4f}")
    print(f"Sigma_xy shape: {Sigma_xy.shape}, mean={Sigma_xy.mean().item():.4f}, std={Sigma_xy.std().item():.4f}")

    # (a) sDTW, gamma=0.01
    A_sdtw_001 = get_soft_alignment_from_cost(D_xy, x, y, gamma_small, use_cuda)[0].detach().cpu().numpy()
    # (b) sDTW, gamma=0.1
    A_sdtw_01 = get_soft_alignment_from_cost(D_xy, x, y, gamma_large, use_cuda)[0].detach().cpu().numpy()
    # (c) uDTW, gamma=0.01, path from effective cost C
    A_udtw_001 = get_soft_alignment_from_cost(D_udtw, x, y, gamma_small, use_cuda)[0].detach().cpu().numpy()
    # (d) uDTW, gamma=0.1, path from effective cost C
    A_udtw_01 = get_soft_alignment_from_cost(D_udtw, x, y, gamma_large, use_cuda)[0].detach().cpu().numpy()

    # (e) uncertainty-on-path = binarize(c) * Sigma
    thr = bin_ratio * float(A_udtw_001.max())
    A_udtw_001_bin = (A_udtw_001 > thr).astype(np.float32)
    U_on_path = A_udtw_001_bin * Sigma_xy[0].detach().cpu().numpy()
    U_on_path = U_on_path / (U_on_path.max() + 1e-12)

    panels = [
        ("(a) sDTW (gamma=0.01)", power_norm_for_vis(A_sdtw_001)),
        ("(b) sDTW (gamma=0.1)", power_norm_for_vis(A_sdtw_01)),
        ("(c) uDTW (gamma=0.01)", power_norm_for_vis(A_udtw_001)),
        ("(d) uDTW (gamma=0.1)", power_norm_for_vis(A_udtw_01)),
        ("(e) uDTW bin(path) * Sigma", U_on_path),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle("Fig.2-style Soft Paths and Uncertainty-on-Path", fontsize=14)

    for ax, (title, mat) in zip(axes, panels):
        im = ax.imshow(mat, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Sequence Y")
        ax.set_ylabel("Sequence X")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = "soft_alignment_matrices_fig2_style.png"
    plt.savefig(save_path, dpi=160)
    print(f"Saved visualization to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
