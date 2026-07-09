"""
Mobile ALOHA 升降维度 (dim_26) 训练脚本.
基于 act-plus-plus-main 的 ACTPolicy 架构, 仅训练升降维度.
"""

import os, sys, h5py, argparse, time, fnmatch
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# add act-plus-plus-main paths (BEFORE any imports from it)
_ACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "act-plus-plus-main")
_DETR_DIR = os.path.join(_ACT_DIR, "detr")
for p in [_ACT_DIR, _DETR_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- build ACT model from detr (no robomimic dependency!) ---
from detr.main import get_args_parser
from detr.models import build_ACT_model

# --- minimal ACTPolicy (copy of act-plus-plus's, no DiffusionPolicy/CNNMLP) ---
def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    return total_kld, klds.mean(0), klds.mean(1).mean(0, True)


class LiftPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        # Build model WITHOUT triggering parse_args on sys.argv
        parser = ap.ArgumentParser(parents=[get_args_parser()], add_help=False)
        args = parser.parse_args([])
        for k, v in args_override.items():
            setattr(args, k, v)
        self.model = build_ACT_model(args)
        self.model.cuda()
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        self.kl_weight = args_override.get("kl_weight", 10)
        self.vq = args_override.get("vq", False)
    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar), probs, binaries =                 self.model(qpos, image, env_state, actions, is_pad)
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0, device=mu.device)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            return {"loss": l1 + total_kld[0] * self.kl_weight,
                    "l1": l1, "kl": total_kld[0]}
        else:
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


# ============================================================
#  paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "act_real")
CKPT_DIR_DEFAULT = os.path.join(ROOT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "act_real_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIFT_DIM = 26
CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


# ============================================================
#  HDF5 file cache
# ============================================================
class HDF5Cache:
    def __init__(self, dataset_paths):
        self._handles = {}
        self._paths = dict(enumerate(dataset_paths))

    def get(self, episode_id):
        if episode_id not in self._handles:
            self._handles[episode_id] = h5py.File(self._paths[episode_id], "r")
        return self._handles[episode_id]

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()


# ============================================================
#  LiftEpisodicDataset
# ============================================================
class LiftEpisodicDataset(Dataset):
    def __init__(self, hdf5_cache, episode_ids, episode_len, camera_names,
                 norm_stats, chunk_size):
        super().__init__()
        self._cache = hdf5_cache
        self.episode_ids = episode_ids
        self.episode_len = episode_len
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)

    def _locate_transition(self, index):
        episode_index = int(np.argmax(self.cumulative_len > index))
        start_ts = index - int(self.cumulative_len[episode_index] -
                               self.episode_len[episode_index])
        return self.episode_ids[episode_index], start_ts

    def __len__(self):
        return int(self.cumulative_len[-1])

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        root = self._cache.get(episode_id)

        qpos = root["/observations/qpos"][start_ts][:14].astype(np.float32)

        all_cam_images = []
        for cam_name in self.camera_names:
            img = root[f"/observations/images/{cam_name}"][start_ts]
            all_cam_images.append(img)
        all_cam_images = np.stack(all_cam_images, axis=0)
        all_cam_images = np.einsum("k h w c -> k c h w", all_cam_images)

        raw_action = root["/action"][()]
        episode_len = raw_action.shape[0]
        start_idx = max(0, start_ts - 1)
        action = raw_action[start_idx:, LIFT_DIM:LIFT_DIM+1].astype(np.float32)
        action_len = episode_len - start_idx

        padded_action = np.zeros((self.chunk_size, 1), dtype=np.float32)
        padded_action[:min(action_len, self.chunk_size)] = (
            action[:min(action_len, self.chunk_size)])
        is_pad = np.zeros(self.chunk_size, dtype=np.float32)
        is_pad[min(action_len, self.chunk_size):] = 1.0

        image_data = torch.from_numpy(all_cam_images).float() / 255.0
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        qpos_data = ((qpos_data - self.norm_stats["qpos_mean"]) /
                     self.norm_stats["qpos_std"])
        action_data = ((action_data - self.norm_stats["action_mean"]) /
                       self.norm_stats["action_std"])

        return image_data, qpos_data, action_data, is_pad


# ============================================================
#  BatchSampler
# ============================================================
def _batch_sampler(batch_size, episode_len_l, sample_weights=None):
    sample_probs = (np.array(sample_weights) / np.sum(sample_weights)
                    if sample_weights is not None else None)
    sum_lens = np.cumsum([0] + [np.sum(ep_len) for ep_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            ep_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_lens[ep_idx], sum_lens[ep_idx + 1])
            batch.append(step_idx)
        yield batch


# ============================================================
#  data loading
# ============================================================
def find_all_hdf5(dataset_dir):
    matched = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if fnmatch.fnmatch(f, "*.hdf5"):
                matched.append(os.path.join(root, f))
    return sorted(matched)


def get_norm_stats_lift(dataset_paths):
    all_qpos = []
    all_action = []
    all_ep_len = []
    for path in dataset_paths:
        with h5py.File(path, "r") as root:
            qpos = root["/observations/qpos"][()][:, :14]
            action = root["/action"][()][:, LIFT_DIM:LIFT_DIM+1]
            all_qpos.append(qpos)
            all_action.append(action)
            all_ep_len.append(action.shape[0])
    all_qpos = np.concatenate(all_qpos, axis=0)
    all_action = np.concatenate(all_action, axis=0)
    norm_stats = {
        "qpos_mean": torch.from_numpy(np.mean(all_qpos, axis=0)).float(),
        "qpos_std": torch.from_numpy(np.std(all_qpos, axis=0)).float().clamp(1e-2),
        "action_mean": torch.from_numpy(np.mean(all_action, axis=0)).float(),
        "action_std": torch.from_numpy(np.std(all_action, axis=0)).float().clamp(1e-2),
    }
    return norm_stats, all_ep_len


def load_data_lift(dataset_dir, camera_names, batch_size_train,
                   batch_size_val, chunk_size, train_ratio=0.9):
    dataset_paths = find_all_hdf5(dataset_dir)
    print(f"Found {len(dataset_paths)} episodes")
    norm_stats, all_ep_len = get_norm_stats_lift(dataset_paths)
    print(f"action_mean={norm_stats['action_mean'].item():.4f}, "
          f"action_std={norm_stats['action_std'].item():.4f}")

    num_eps = len(dataset_paths)
    num_train = max(1, int(train_ratio * num_eps))
    rng = np.random.RandomState(0)
    ids = rng.permutation(num_eps)
    train_ids = ids[:num_train].tolist()
    val_ids = ids[num_train:].tolist()
    train_ep_len = [all_ep_len[i] for i in train_ids]
    val_ep_len = [all_ep_len[i] for i in val_ids]
    print(f"Train: {len(train_ids)} ep, Val: {len(val_ids)} ep")

    cache = HDF5Cache(dataset_paths)
    train_dataset = LiftEpisodicDataset(
        cache, train_ids, train_ep_len, camera_names, norm_stats, chunk_size)
    val_dataset = LiftEpisodicDataset(
        cache, val_ids, val_ep_len, camera_names, norm_stats, chunk_size)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=_batch_sampler(batch_size_train, [train_ep_len]),
        pin_memory=True, num_workers=4, prefetch_factor=2,
        persistent_workers=True)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=_batch_sampler(batch_size_val, [val_ep_len]),
        pin_memory=True, num_workers=4, prefetch_factor=2,
        persistent_workers=True)
    return train_loader, val_loader, norm_stats, cache


# ============================================================
#  forward pass
# ============================================================
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda(non_blocking=True)
    qpos_data = qpos_data.cuda(non_blocking=True)
    action_data = action_data.cuda(non_blocking=True)
    is_pad = is_pad.cuda(non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad)


# ============================================================
#  train
# ============================================================
def train(args):
    torch.manual_seed(1)
    np.random.seed(1)

    ckpt_dir = args["ckpt_dir"]
    batch_size = args["batch_size"]
    chunk_size = args["chunk_size"]
    num_steps = args["num_steps"]
    lr = args["lr"]
    kl_weight = args["kl_weight"]
    hidden_dim = args["hidden_dim"]
    dim_feedforward = args["dim_feedforward"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, norm_stats, hdf5_cache = load_data_lift(
        dataset_dir=DATA_DIR, camera_names=CAMERA_NAMES,
        batch_size_train=batch_size, batch_size_val=batch_size,
        chunk_size=chunk_size, train_ratio=0.9)

    policy_config = {
        "lr": lr, "num_queries": chunk_size, "kl_weight": kl_weight,
        "hidden_dim": hidden_dim, "dim_feedforward": dim_feedforward,
        "lr_backbone": 1e-5, "backbone": "resnet18",
        "enc_layers": 4, "dec_layers": 7, "nheads": 8,
        "camera_names": CAMERA_NAMES, "vq": False,
        "vq_class": None, "vq_dim": None, "action_dim": 1,
        "no_encoder": False,
    }

    policy = LiftPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Training {num_steps} steps, batch={batch_size}, chunk={chunk_size}...")

    min_val_loss = float("inf")
    best_ckpt_info = None
    train_history = []
    val_history = []
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in tqdm(range(num_steps)):
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        policy.train()
        optimizer.zero_grad()
        forward_dict = forward_pass(data, policy)
        loss = forward_dict["loss"]
        loss.backward()
        optimizer.step()

        train_history.append({
            "loss": loss.item(),
            "l1": forward_dict["l1"].item(),
            "kl": forward_dict["kl"].item(),
        })

        if step % 200 == 0 and step > 0:
            mean_loss = np.mean([h["loss"] for h in train_history[-200:]])
            mean_l1 = np.mean([h["l1"] for h in train_history[-200:]])
            mean_kl = np.mean([h["kl"] for h in train_history[-200:]])
            sps = step / (time.time() - t0)
            print(f"Step {step:6d} | loss={mean_loss:.4f} "
                  f"l1={mean_l1:.4f} kl={mean_kl:.4f} | {sps:.1f} steps/s")

        if step % validate_every == 0 and step > 0:
            policy.eval()
            val_dicts = []
            with torch.no_grad():
                for val_data in val_loader:
                    fwd = forward_pass(val_data, policy)
                    val_dicts.append({k: v.item() for k, v in fwd.items()})
            val_summary = {k: np.mean([d[k] for d in val_dicts])
                          for k in val_dicts[0]}
            val_history.append({"step": step, **val_summary})
            if val_summary["loss"] < min_val_loss:
                min_val_loss = val_summary["loss"]
                best_ckpt_info = (step, min_val_loss,
                                  deepcopy(policy.serialize()))
            print(f"  ==> Val loss={val_summary['loss']:.5f} "
                  f"(best={min_val_loss:.5f})")

        if step % save_every == 0 and step > 0:
            torch.save(policy.serialize(),
                       os.path.join(ckpt_dir, f"policy_step_{step}.ckpt"))

    torch.save(policy.serialize(), os.path.join(ckpt_dir, "policy_last.ckpt"))
    if best_ckpt_info:
        best_step, best_loss, best_state = best_ckpt_info
        torch.save(best_state,
                   os.path.join(ckpt_dir, f"policy_best_step_{best_step}.ckpt"))
        print(f"\nDone! Best val loss = {best_loss:.6f} at step {best_step}")
    else:
        print("\nDone!")

    hdf5_cache.close()

    # plot loss
    steps_a = range(len(train_history))
    losses = [h["loss"] for h in train_history]
    l1s = [h["l1"] for h in train_history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(steps_a, losses, alpha=0.2, color="blue")
    if len(losses) > 100:
        s = np.convolve(losses, np.ones(100)/100, mode="valid")
        ax1.plot(range(99, 99+len(s)), s, color="blue", lw=2, label="Smoothed")
    if val_history:
        vsteps = [h["step"] for h in val_history]
        ax1.plot(vsteps, [h["loss"] for h in val_history], "ro-", ms=4)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.grid(True, alpha=0.3); ax1.set_title("Training Loss")
    ax2.plot(steps_a, l1s, alpha=0.2, color="green")
    if len(l1s) > 100:
        s = np.convolve(l1s, np.ones(100)/100, mode="valid")
        ax2.plot(range(99, 99+len(s)), s, color="green", lw=2)
    ax2.set_xlabel("Step"); ax2.set_ylabel("L1 Loss")
    ax2.grid(True, alpha=0.3); ax2.set_title("L1 Regression Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lift_training_loss.png"), dpi=100)
    plt.close()
    print(f"Loss chart: {os.path.join(OUTPUT_DIR, 'lift_training_loss.png')}")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--validate_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2000)
    args = parser.parse_args()
    train(vars(args))
