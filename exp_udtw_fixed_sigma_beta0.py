import os
import sys
import types
import torch


try:
    import numba  # noqa: F401
except ImportError:
    def _identity_jit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    fake_cuda = types.SimpleNamespace(
        jit=_identity_jit,
        as_cuda_array=lambda x: x,
    )
    fake_numba = types.ModuleType("numba")
    fake_numba.jit = _identity_jit
    fake_numba.cuda = fake_cuda
    sys.modules["numba"] = fake_numba


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from uDTW import uDTW


def summarize_tensor(name, value):
    flat = value.detach().reshape(-1)
    print(
        f"{name}: shape={tuple(value.shape)} "
        f"min={flat.min().item():.8f} "
        f"max={flat.max().item():.8f} "
        f"mean={flat.mean().item():.8f}"
    )


def run_case(case_name, x, y, gamma, normalize, use_cuda):
    sigma_x = torch.ones((x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype)
    sigma_y = torch.ones((y.shape[0], y.shape[1], 1), device=y.device, dtype=y.dtype)
    beta = 0.0

    udtw = uDTW(use_cuda=use_cuda, gamma=gamma, normalize=normalize)

    with torch.no_grad():
        dist, reg = udtw(x, y, sigma_x, sigma_y, beta=beta)
        d_xy, s_xy = udtw._calc_distance_matrix(x, y, sigma_x, sigma_y, beta=beta)

        expected_d_xy = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(dim=3) / (2.0 * x.shape[2])
        max_cost_diff = (d_xy - expected_d_xy).abs().max().item()

    print(f"\n=== {case_name} | normalize={normalize} | gamma={gamma} ===")
    summarize_tensor("uDTW distance", dist)
    summarize_tensor("uDTW regularizer", reg)
    summarize_tensor("D_xy", d_xy)
    summarize_tensor("S_xy", s_xy)
    print(f"regularizer_is_zero: {torch.allclose(reg, torch.zeros_like(reg), atol=1e-7)}")
    print(f"S_xy_is_zero: {torch.allclose(s_xy, torch.zeros_like(s_xy), atol=1e-7)}")
    print(f"max_abs(D_xy - squared_l2/(2d)): {max_cost_diff:.8e}")


def main():
    torch.manual_seed(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 2
    len_x = 6
    len_y = 7
    dims = 10
    gamma = 0.01

    x = torch.rand((batch_size, len_x, dims), device=device)
    y = torch.rand((batch_size, len_y, dims), device=device)

    print("Test setting: fixed sigma=1, beta=0")
    print(f"device={device}, batch_size={batch_size}, len_x={len_x}, len_y={len_y}, dims={dims}")

    run_case("random x vs random y", x, y, gamma=gamma, normalize=False, use_cuda=use_cuda)
    run_case("random x vs random y", x, y, gamma=gamma, normalize=True, use_cuda=use_cuda)
    run_case("identical x vs x", x, x, gamma=gamma, normalize=True, use_cuda=use_cuda)

    print("\nInterpretation:")
    print("- beta=0 makes the uncertainty regularizer exactly zero.")
    print("- fixed sigma=1 makes uDTW use a non-zero cost matrix: squared_l2 / (2 * dims).")
    print("- normalize=True and identical inputs can return ~0 because of divergence normalization.")


if __name__ == "__main__":
    main()
