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
    gamma_values = [1.0, 0.1, 0.01, 0.001, 0.0001]
    gamma_ref = gamma_values[0]

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
    sdtw_ref = sDTW(use_cuda=use_cuda, gamma=gamma_ref, normalize=False)
    D_xy = sdtw_ref._calc_distance_matrix(x, y).detach()

    # uDTW effective cost C = D \odot Sigma^{-1} (in this codebase this is D_udtw)
    # and uncertainty regularizer term S_udtw = 0.5 * beta * log(sigma_xy)
    udtw_ref = uDTW(use_cuda=use_cuda, gamma=gamma_ref, normalize=False)
    D_udtw, S_udtw = udtw_ref._calc_distance_matrix(x, y, sigma_x, sigma_y, beta=beta_vis)

    # Build Sigma explicitly for panel (e)
    Sigma_xy = build_sigma_matrix(sigma_x, sigma_y).detach()

    print(f"D_xy shape: {D_xy.shape}, mean={D_xy.mean().item():.4f}, std={D_xy.std().item():.4f}")
    print(f"D_udtw shape: {D_udtw.shape}, mean={D_udtw.mean().item():.4f}, std={D_udtw.std().item():.4f}")
    print(f"S_udtw shape: {S_udtw.shape}, mean={S_udtw.mean().item():.4f}, std={S_udtw.std().item():.4f}")
    print(f"Sigma_xy shape: {Sigma_xy.shape}, mean={Sigma_xy.mean().item():.4f}, std={Sigma_xy.std().item():.4f}")

    sdtw_alignments = []
    udtw_alignments = []
    uncertainty_maps = []
    sigma_np = Sigma_xy[0].detach().cpu().numpy()

    print(f"Testing gammas: {gamma_values}")
    for gamma in gamma_values:
        A_sdtw = get_soft_alignment_from_cost(D_xy, x, y, gamma, use_cuda)[0].detach().cpu().numpy()
        A_udtw = get_soft_alignment_from_cost(D_udtw, x, y, gamma, use_cuda)[0].detach().cpu().numpy()

        thr = bin_ratio * float(A_udtw.max())
        A_udtw_bin = (A_udtw > thr).astype(np.float32)
        U_on_path = A_udtw_bin * sigma_np
        U_on_path = U_on_path / (U_on_path.max() + 1e-12)

        sdtw_alignments.append((gamma, A_sdtw))
        udtw_alignments.append((gamma, A_udtw))
        uncertainty_maps.append((gamma, U_on_path))

        print(
            f"gamma={gamma:.6f} | "
            f"sDTW sum={A_sdtw.sum():.4f}, max={A_sdtw.max():.4f} | "
            f"uDTW sum={A_udtw.sum():.4f}, max={A_udtw.max():.4f}"
        )

    num_cols = len(gamma_values)
    fig, axes = plt.subplots(3, num_cols, figsize=(4.2 * num_cols, 10))
    fig.suptitle("Multi-gamma Soft Paths and Uncertainty-on-Path", fontsize=14)

    if num_cols == 1:
        axes = np.array(axes).reshape(3, 1)

    for col, ((gamma_s, A_sdtw), (gamma_u, A_udtw), (gamma_unc, U_on_path)) in enumerate(
        zip(sdtw_alignments, udtw_alignments, uncertainty_maps)
    ):
        im1 = axes[0, col].imshow(power_norm_for_vis(A_sdtw), cmap="gray", origin="lower")
        axes[0, col].set_title(f"sDTW (gamma={gamma_s:g})")
        axes[0, col].set_xlabel("Sequence Y")
        axes[0, col].set_ylabel("Sequence X")
        fig.colorbar(im1, ax=axes[0, col], fraction=0.046, pad=0.04)

        im2 = axes[1, col].imshow(power_norm_for_vis(A_udtw), cmap="gray", origin="lower")
        axes[1, col].set_title(f"uDTW (gamma={gamma_u:g})")
        axes[1, col].set_xlabel("Sequence Y")
        axes[1, col].set_ylabel("Sequence X")
        fig.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

        im3 = axes[2, col].imshow(U_on_path, cmap="gray", origin="lower")
        axes[2, col].set_title(f"uDTW bin(path)*Sigma (gamma={gamma_unc:g})")
        axes[2, col].set_xlabel("Sequence Y")
        axes[2, col].set_ylabel("Sequence X")
        fig.colorbar(im3, ax=axes[2, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = "soft_alignment_multi_gamma.png"
    plt.savefig(save_path, dpi=160)
    print(f"Saved visualization to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
