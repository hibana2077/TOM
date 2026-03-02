import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys

# Add src to path so we can import sDTW and uDTW
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from sDTW import sDTW
from uDTW import uDTW


def _normalize_nonnegative(mat, eps=1e-12):
    mat = torch.clamp(mat, min=0.0)
    denom = mat.sum(dim=(-2, -1), keepdim=True)
    return mat / (denom + eps)


def _extract_udtw_alignment(D_xy, S_xy, func_dtw, gamma, bandwidth, mode='hybrid'):
    out_xy, outs_xy = func_dtw(D_xy, S_xy, gamma, bandwidth)

    grad_d = torch.autograd.grad(out_xy.sum(), D_xy, retain_graph=True)[0]
    grad_s = torch.autograd.grad(outs_xy.sum(), S_xy)[0]

    if mode == 'raw':
        return grad_d
    if mode == 'positive':
        return _normalize_nonnegative(grad_d)
    if mode == 'sigma':
        return _normalize_nonnegative(grad_s)
    if mode == 'hybrid':
        d_norm = _normalize_nonnegative(grad_d)
        s_norm = _normalize_nonnegative(grad_s)
        return _normalize_nonnegative(0.7 * d_norm + 0.3 * s_norm)

    raise ValueError(f"Unknown uDTW path mode: {mode}")

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, input):
        return a * torch.sigmoid(input) + b

class SimpleSigmaNet(nn.Module):
    def __init__(self):
        super(SimpleSigmaNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.sigmoid = Sigmoid()

    def forward(self, x, a, b):
        batch_size = x.shape[0]
        length = x.shape[1]

        x = x.view(batch_size*length, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(batch_size, length, -1).mean(2, keepdim = True)
        sigma = self.sigmoid(a, b, x)
        return sigma

def main():
    torch.manual_seed(42)  # set seed for reproducibility

    # Use slightly longer sequences to make the alignment matrix visualization clear
    batch_size, len_x, len_y, dims = 1, 30, 30, 10
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    x = torch.rand((batch_size, len_x, dims)).to(device)
    y = torch.rand((batch_size, len_y, dims)).to(device)

    # define parameters for scaled sigmoid function
    a = 1.5
    b = 0.5

    sigmanet = SimpleSigmaNet().to(device)
    sigmanet.apply(weight_init)

    # Obtain sigma for uDTW
    sigma_x_udtw = sigmanet(x, a, b).detach()
    sigma_y_udtw = sigmanet(y, a, b).detach()

    gammas = [1.0, 0.1, 0.01, 0.001, 0.0001]
    
    # Try different path extraction modes for uDTW.
    # options: 'raw', 'positive', 'sigma', 'hybrid'
    udtw_path_mode = 'hybrid'

    # Create the figure
    fig, axes = plt.subplots(2, len(gammas), figsize=(4 * len(gammas), 8))
    fig.suptitle(f'Soft Alignment Matrices for sDTW and uDTW (uDTW path={udtw_path_mode})', fontsize=16)

    # ==========================
    # 1. Visualize sDTW
    # ==========================
    for i, g in enumerate(gammas):
        sdtw = sDTW(use_cuda=use_cuda, gamma=g, normalize=False)
        
        # We need gradients with respect to the distance matrix D_xy
        # So we manually compute D_xy and set requires_grad
        D_xy = sdtw._calc_distance_matrix(x, y).detach().requires_grad_(True)
        
        func_dtw = sdtw._get_func_dtw(x, y)
        loss = func_dtw(D_xy, sdtw.gamma, sdtw.bandwidth)
        
        # Gradient wrt D_xy is the soft alignment matrix
        align_sdtw = torch.autograd.grad(loss.sum(), D_xy)[0][0].detach().cpu().numpy()
        
        ax = axes[0, i]
        im = ax.imshow(align_sdtw, cmap='gray', origin='lower')
        ax.set_title(f'sDTW (gamma={g})')
        ax.set_xlabel('Sequence Y')
        ax.set_ylabel('Sequence X')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ==========================
    # 2. Visualize uDTW
    # ==========================
    for i, g in enumerate(gammas):
        udtw = uDTW(use_cuda=use_cuda, gamma=g, normalize=False)
        
        # Manually compute D_xy and S_xy
        D_xy_raw, S_xy_raw = udtw._calc_distance_matrix(x, y, sigma_x_udtw, sigma_y_udtw, beta=0.)
        D_xy = D_xy_raw.detach().requires_grad_(True)
        S_xy = S_xy_raw.detach().requires_grad_(True)
        
        func_dtw = udtw._get_func_dtw(x, y)
        align_udtw_t = _extract_udtw_alignment(
            D_xy,
            S_xy,
            func_dtw,
            udtw.gamma,
            udtw.bandwidth,
            mode=udtw_path_mode,
        )

        align_udtw = align_udtw_t[0].detach().cpu().numpy()

        neg_ratio = float((align_udtw < 0).mean())
        tiny_ratio = float((abs(align_udtw) < 1e-8).mean())
        print(f"uDTW gamma={g:.4g} | mode={udtw_path_mode} | neg_ratio={neg_ratio:.3f} | tiny_ratio={tiny_ratio:.3f}")
        
        ax = axes[1, i]
        im = ax.imshow(align_udtw, cmap='gray', origin='lower')
        ax.set_title(f'uDTW (gamma={g})')
        ax.set_xlabel('Sequence Y')
        ax.set_ylabel('Sequence X')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    # Save the result
    save_path = 'soft_alignment_matrices.png'
    plt.savefig(save_path)
    print(f"Saved alignment matrix visualization to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
