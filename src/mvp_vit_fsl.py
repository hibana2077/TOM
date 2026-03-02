import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

try:
    import sDTW as sdtw_mod
    import uDTW as udtw_mod
    from sDTW import sDTW
    from uDTW import uDTW
except ImportError:
    from src import sDTW as sdtw_mod
    from src import uDTW as udtw_mod
    from src.sDTW import sDTW
    from src.uDTW import uDTW


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SplitImageFolder(Dataset):
    def __init__(self, root: str, transform: T.Compose):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Split path not found: {self.root}")

        self.transform = transform
        self.class_names = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        self.class_to_indices: Dict[int, List[int]] = {i: [] for i in range(len(self.class_names))}

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for cname in self.class_names:
            cdir = self.root / cname
            for p in sorted(cdir.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    cid = self.class_to_id[cname]
                    idx = len(self.samples)
                    self.samples.append((p, cid))
                    self.class_to_indices[cid].append(idx)

        valid = {k: v for k, v in self.class_to_indices.items() if len(v) > 0}
        self.class_to_indices = valid
        if len(self.class_to_indices) == 0:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, str(path)


@dataclass
class Episode:
    support_imgs: torch.Tensor
    support_labels: torch.Tensor
    query_imgs: torch.Tensor
    query_labels: torch.Tensor


def sample_episode(
    dataset: SplitImageFolder,
    n_way: int,
    n_shot: int,
    n_query: int,
    device: torch.device,
) -> Episode:
    class_ids = random.sample(list(dataset.class_to_indices.keys()), n_way)

    support_imgs = []
    support_labels = []
    query_imgs = []
    query_labels = []

    for epi_label, class_id in enumerate(class_ids):
        idx_pool = dataset.class_to_indices[class_id]
        pick = random.sample(idx_pool, n_shot + n_query)
        support_pick = pick[:n_shot]
        query_pick = pick[n_shot:]

        for idx in support_pick:
            img, _, _ = dataset[idx]
            support_imgs.append(img)
            support_labels.append(epi_label)
        for idx in query_pick:
            img, _, _ = dataset[idx]
            query_imgs.append(img)
            query_labels.append(epi_label)

    return Episode(
        support_imgs=torch.stack(support_imgs).to(device),
        support_labels=torch.tensor(support_labels, device=device, dtype=torch.long),
        query_imgs=torch.stack(query_imgs).to(device),
        query_labels=torch.tensor(query_labels, device=device, dtype=torch.long),
    )


class SharedProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim + 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.Tensor, a: float, b: float):
        out = self.fc(tokens)
        feat = out[..., :-1]
        feat = torch.nn.functional.layer_norm(feat, (feat.shape[-1],))
        sigma = a * torch.sigmoid(out[..., -1:]) + b
        return feat, sigma


def extract_patch_tokens(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = backbone.forward_features(x)

    if isinstance(feats, (list, tuple)):
        feats = feats[0]

    if feats.ndim != 3:
        raise RuntimeError(
            f"Expected token tensor with 3 dims, got shape={tuple(feats.shape)}. "
            "Pick a ViT model where forward_features returns tokens."
        )

    if feats.shape[1] == 197:
        feats = feats[:, 1:, :]
    elif feats.shape[1] != 196:
        raise RuntimeError(f"Expected 196 or 197 tokens, got {feats.shape[1]}")

    return feats


def sigma_pair_matrix(sigma_x: torch.Tensor, sigma_y: torch.Tensor) -> torch.Tensor:
    sx = sigma_x.squeeze(-1)
    sy = sigma_y.squeeze(-1)
    return 0.5 * (sx.pow(2).unsqueeze(1) + sy.pow(2).unsqueeze(0))


def binary_path(path: torch.Tensor, q: float = 0.85) -> torch.Tensor:
    thr = torch.quantile(path.flatten(), q)
    return (path >= thr).float()


def normalize_for_vis(matrix: torch.Tensor, power: float = 0.1) -> torch.Tensor:
    matrix = matrix.clamp(min=0)
    matrix = matrix / (matrix.max() + 1e-8)
    if power is not None:
        matrix = matrix.pow(power)
    return matrix


def save_heatmap(matrix: torch.Tensor, out_file: Path, title: str, power: float = 0.1) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    arr = normalize_for_vis(matrix, power=power).detach().cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.imshow(arr, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def alignment_from_cost_matrix(cost: torch.Tensor, gamma: float, bandwidth: float, use_cuda: bool) -> torch.Tensor:
    func = sdtw_mod._SoftDTWCUDA.apply if use_cuda else sdtw_mod._SoftDTW.apply
    cost = cost.clone().requires_grad_(True)
    out = func(cost, gamma, bandwidth)
    (A,) = torch.autograd.grad(out.sum(), cost, create_graph=False)
    return A.detach()


def score_query_against_support(
    q_feat: torch.Tensor,
    q_sigma: torch.Tensor,
    s_feat: torch.Tensor,
    s_sigma: torch.Tensor,
    sdtw_obj: sDTW,
    udtw_obj: uDTW,
    beta: float,
    n_way: int,
    n_shot: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_support = s_feat.shape[0]
    q_rep = q_feat.unsqueeze(0).expand(num_support, -1, -1)
    q_sig_rep = q_sigma.unsqueeze(0).expand(num_support, -1, -1)

    s_scores = sdtw_obj(q_rep, s_feat)
    ud_d, ud_s = udtw_obj(q_rep, s_feat, q_sig_rep, s_sigma, beta=beta)
    u_scores = ud_d + ud_s

    s_scores = s_scores.view(n_way, n_shot).mean(dim=1)
    u_scores = u_scores.view(n_way, n_shot).mean(dim=1)
    return s_scores, u_scores


def train_projection(
    backbone: nn.Module,
    proj: SharedProjection,
    train_data: SplitImageFolder,
    n_way: int,
    n_shot: int,
    gamma: float,
    beta: float,
    bandwidth: float,
    episodes: int,
    lr: float,
    sigma_a: float,
    sigma_b: float,
    device: torch.device,
) -> None:
    if episodes <= 0:
        return

    backbone.eval()
    proj.train()

    udtw_obj = uDTW(use_cuda=(device.type == "cuda"), gamma=gamma, normalize=True, bandwidth=bandwidth)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=lr)

    for ep in range(1, episodes + 1):
        episode = sample_episode(train_data, n_way=n_way, n_shot=n_shot + 1, n_query=1, device=device)
        all_imgs = torch.cat([episode.support_imgs, episode.query_imgs], dim=0)

        with torch.no_grad():
            tokens = extract_patch_tokens(backbone, all_imgs)
        feat, sigma = proj(tokens, sigma_a, sigma_b)

        num_support = n_way * (n_shot + 1)
        support_feat = feat[:num_support]
        support_sigma = sigma[:num_support]
        query_feat = feat[num_support:]
        query_sigma = sigma[num_support:]

        support_labels = episode.support_labels
        query_labels = episode.query_labels

        pos_q, pos_s, pos_sq, pos_ss = [], [], [], []
        neg_q, neg_s, neg_sq, neg_ss = [], [], [], []

        for i in range(query_feat.shape[0]):
            y = query_labels[i].item()
            pos_pool = torch.where(support_labels == y)[0]
            neg_pool = torch.where(support_labels != y)[0]
            pos_idx = pos_pool[random.randrange(len(pos_pool))]
            neg_idx = neg_pool[random.randrange(len(neg_pool))]

            pos_q.append(query_feat[i])
            pos_sq.append(query_sigma[i])
            pos_s.append(support_feat[pos_idx])
            pos_ss.append(support_sigma[pos_idx])

            neg_q.append(query_feat[i])
            neg_sq.append(query_sigma[i])
            neg_s.append(support_feat[neg_idx])
            neg_ss.append(support_sigma[neg_idx])

        pos_q = torch.stack(pos_q)
        pos_s = torch.stack(pos_s)
        pos_sq = torch.stack(pos_sq)
        pos_ss = torch.stack(pos_ss)
        neg_q = torch.stack(neg_q)
        neg_s = torch.stack(neg_s)
        neg_sq = torch.stack(neg_sq)
        neg_ss = torch.stack(neg_ss)

        pd, ps = udtw_obj(pos_q, pos_s, pos_sq, pos_ss, beta=beta)
        nd, ns = udtw_obj(neg_q, neg_s, neg_sq, neg_ss, beta=beta)

        L = pos_q.shape[1] * pos_s.shape[1]
        pos_score = (pd + ps) / L
        neg_score = (nd + ns) / L

        loss = ((pos_score - 0.0) ** 2).mean() + ((neg_score - 1.0) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % max(1, episodes // 10) == 0:
            print(f"[train] episode={ep}/{episodes} loss={loss.item():.6f}")


def run_eval(
    backbone: nn.Module,
    proj: SharedProjection,
    eval_data: SplitImageFolder,
    n_way: int,
    n_shot: int,
    n_query: int,
    eval_episodes: int,
    gamma: float,
    beta: float,
    bandwidth: float,
    sigma_a: float,
    sigma_b: float,
    vis_episodes: int,
    vis_out: Path,
    device: torch.device,
) -> None:
    sdtw_obj = sDTW(use_cuda=(device.type == "cuda"), gamma=gamma, normalize=True, bandwidth=bandwidth)
    udtw_obj = uDTW(use_cuda=(device.type == "cuda"), gamma=gamma, normalize=True, bandwidth=bandwidth)

    backbone.eval()
    proj.eval()

    sdtw_correct = 0
    udtw_correct = 0
    total = 0

    with torch.no_grad():
        for epi in range(1, eval_episodes + 1):
            episode = sample_episode(eval_data, n_way=n_way, n_shot=n_shot, n_query=n_query, device=device)

            all_imgs = torch.cat([episode.support_imgs, episode.query_imgs], dim=0)
            tokens = extract_patch_tokens(backbone, all_imgs)
            feat, sigma = proj(tokens, sigma_a, sigma_b)

            num_support = n_way * n_shot
            support_feat = feat[:num_support]
            support_sigma = sigma[:num_support]
            query_feat = feat[num_support:]
            query_sigma = sigma[num_support:]

            for q_idx in range(query_feat.shape[0]):
                s_scores, u_scores = score_query_against_support(
                    q_feat=query_feat[q_idx],
                    q_sigma=query_sigma[q_idx],
                    s_feat=support_feat,
                    s_sigma=support_sigma,
                    sdtw_obj=sdtw_obj,
                    udtw_obj=udtw_obj,
                    beta=beta,
                    n_way=n_way,
                    n_shot=n_shot,
                )

                gt = episode.query_labels[q_idx].item()
                s_pred = int(torch.argmin(s_scores).item())
                u_pred = int(torch.argmin(u_scores).item())
                sdtw_correct += int(s_pred == gt)
                udtw_correct += int(u_pred == gt)
                total += 1

            if epi <= vis_episodes:
                qf = query_feat[0]
                qs = query_sigma[0]
                sf = support_feat[0]
                ss = support_sigma[0]

                with torch.enable_grad():
                    d_base = sdtw_obj._calc_distance_matrix(qf.unsqueeze(0), sf.unsqueeze(0))[0]
                    sigma_xy = sigma_pair_matrix(qs, ss).clamp_min(1e-8)
                    sigma_inv = sigma_xy.reciprocal()
                    c_udtw = d_base * sigma_inv

                    a_s_001 = alignment_from_cost_matrix(
                        cost=d_base.unsqueeze(0),
                        gamma=0.01,
                        bandwidth=sdtw_obj.bandwidth,
                        use_cuda=(device.type == "cuda"),
                    )[0]
                    a_s_01 = alignment_from_cost_matrix(
                        cost=d_base.unsqueeze(0),
                        gamma=0.1,
                        bandwidth=sdtw_obj.bandwidth,
                        use_cuda=(device.type == "cuda"),
                    )[0]
                    a_u_001 = alignment_from_cost_matrix(
                        cost=c_udtw.unsqueeze(0),
                        gamma=0.01,
                        bandwidth=sdtw_obj.bandwidth,
                        use_cuda=(device.type == "cuda"),
                    )[0]
                    a_u_01 = alignment_from_cost_matrix(
                        cost=c_udtw.unsqueeze(0),
                        gamma=0.1,
                        bandwidth=udtw_obj.bandwidth,
                        use_cuda=(device.type == "cuda"),
                    )[0]
                    a_u_bin = binary_path(a_u_001)
                    unc_on_path = a_u_bin * sigma_xy

                epi_dir = vis_out / f"episode_{epi:04d}"
                save_heatmap(a_s_001, epi_dir / "A_sdtw_g001.png", f"Episode {epi} - sDTW gamma=0.01")
                save_heatmap(a_s_01, epi_dir / "A_sdtw_g01.png", f"Episode {epi} - sDTW gamma=0.1")
                save_heatmap(a_u_001, epi_dir / "A_udtw_g001.png", f"Episode {epi} - uDTW gamma=0.01")
                save_heatmap(a_u_01, epi_dir / "A_udtw_g01.png", f"Episode {epi} - uDTW gamma=0.1")
                save_heatmap(
                    unc_on_path,
                    epi_dir / "A_udtw_g001_bin_mul_uncertainty.png",
                    f"Episode {epi} - bin(A_udtw gamma=0.01) * Sigma",
                    power=None,
                )

            if epi % max(1, eval_episodes // 10) == 0:
                print(
                    f"[eval] episode={epi}/{eval_episodes} "
                    f"sDTW_acc={sdtw_correct / max(1, total):.4f} "
                    f"uDTW_acc={udtw_correct / max(1, total):.4f}"
                )

    print("\n========== Final ==========")
    print(f"Total queries: {total}")
    print(f"sDTW accuracy: {sdtw_correct / max(1, total):.4f}")
    print(f"uDTW accuracy: {udtw_correct / max(1, total):.4f}")
    print(f"Path visualizations saved to: {vis_out}")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ViT + sDTW/uDTW CIFAR-FS MVP")
    parser.add_argument("--data-root", type=str, default="CIFARFS")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--n-shot", type=int, default=1)
    parser.add_argument("--n-query", type=int, default=15)
    parser.add_argument("--eval-episodes", type=int, default=600)
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--bandwidth", type=float, default=0.0)
    parser.add_argument("--sigma-a", type=float, default=1.5)
    parser.add_argument("--sigma-b", type=float, default=0.5)
    parser.add_argument("--vis-episodes", type=int, default=10)
    parser.add_argument("--vis-out", type=str, default="outputs/path_viz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-proj", type=str, default="")
    parser.add_argument("--load-proj", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = build_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_path = os.path.join(args.data_root, args.train_split)
    eval_path = os.path.join(args.data_root, args.eval_split)
    train_data = SplitImageFolder(train_path, transform=transform)
    eval_data = SplitImageFolder(eval_path, transform=transform)

    backbone = timm.create_model(
        args.model_name,
        pretrained=(not args.no_pretrained),
        num_classes=0,
        global_pool="",
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    with torch.no_grad():
        probe = torch.zeros(1, 3, 224, 224, device=device)
        tokens = extract_patch_tokens(backbone, probe)
    in_dim = tokens.shape[-1]

    proj = SharedProjection(in_dim=in_dim, out_dim=args.proj_dim).to(device)
    if args.load_proj:
        proj.load_state_dict(torch.load(args.load_proj, map_location=device))
        print(f"Loaded projection from: {args.load_proj}")

    train_projection(
        backbone=backbone,
        proj=proj,
        train_data=train_data,
        n_way=args.n_way,
        n_shot=args.n_shot,
        gamma=args.gamma,
        beta=args.beta,
        bandwidth=args.bandwidth,
        episodes=args.train_episodes,
        lr=args.train_lr,
        sigma_a=args.sigma_a,
        sigma_b=args.sigma_b,
        device=device,
    )

    if args.save_proj:
        Path(args.save_proj).parent.mkdir(parents=True, exist_ok=True)
        torch.save(proj.state_dict(), args.save_proj)
        print(f"Saved projection to: {args.save_proj}")

    run_eval(
        backbone=backbone,
        proj=proj,
        eval_data=eval_data,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        eval_episodes=args.eval_episodes,
        gamma=args.gamma,
        beta=args.beta,
        bandwidth=args.bandwidth,
        sigma_a=args.sigma_a,
        sigma_b=args.sigma_b,
        vis_episodes=args.vis_episodes,
        vis_out=Path(args.vis_out),
        device=device,
    )


if __name__ == "__main__":
    main()
