"""
Mobile ALOHA 升降维度 (dim_26) 训练脚本.

基于 act-plus-plus-main 的 ACTPolicy 架构, 仅训练升降维度.
与 mobile-aloha-main 相比仅修改了 action 维度 (16 -> 1) 和数据提取逻辑.

用法:
  python aloha_scripts/train_lift.py --ckpt_dir ../checkpoints --num_steps 50000 --batch_size 64
"""

import os, sys, h5py, argparse, time, fnmatch
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# add act-plus-plus-main to PYTHONPATH
_ACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "act-plus-plus-main")
if _ACT_DIR not in sys.path:
    sys.path.insert(0, _ACT_DIR)

from policy import ACTPolicy
from detr.main import build_ACT_model_and_optimizer
from utils import compute_dict_mean, set_seed

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
#  HDF5 file cache - solves the per-__getitem__ open/close bottleneck
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
#  LiftEpisodicDataset - like act-plus-plus EpisodicDataset but action_dim=1
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

        # qpos: first 14 dims (arm joints)
        qpos = root["/observations/qpos"][start_ts][:14].astype(np.float32)

        # images: 3 cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            img = root[f"/observations/images/{cam_name}"][start_ts]
            all_cam_images.append(img)
        all_cam_images = np.stack(all_cam_images, axis=0)
        all_cam_images = np.einsum("k h w c -> k c h w", all_cam_images)

        # action: only dim_26 (lift)
        raw_action = root["/action"][()]
        episode_len = raw_action.shape[0]
        start_idx = max(0, start_ts - 1)
        action = raw_action[start_idx:, LIFT_DIM:LIFT_DIM+1].astype(np.float32)
        action_len = episode_len - start_idx

        # pad to chunk_size
        padded_action = np.zeros((self.chunk_size, 1), dtype=np.float32)
        padded_action[:min(action_len, self.chunk_size)] = (
            action[:min(action_len, self.chunk_size)])
        is_pad = np.zeros(self.chunk_size, dtype=np.float32)
        is_pad[min(action_len, self.chunk_size):] = 1.0

        # to torch
        image_data = torch.from_numpy(all_cam_images).float() / 255.0
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # normalize
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
        "qpos_std": (torch.from_numpy(np.std(all_qpos, axis=0)).float()
                     .clamp(1e-2)),
        "action_mean": torch.from_numpy(np.mean(all_action, axis=0)).float(),
        "action_std": (torch.from_numpy(np.std(all_action, axis=0)).float()
                       .clamp(1e-2)),
    }
    return norm_stats, all_ep_len


def load_data_lift(dataset_dir, camera_names, batch_size_train,
                   batch_size_val, chunk_size, train_ratio=0.9):
    dataset_paths = find_all_hdf5(dataset_dir)
    print(f"Found {len(dataset_paths)} episodes in {dataset_dir}")

    norm_stats, all_ep_len = get_norm_stats_lift(dataset_paths)
    print(f"action_mean={norm_stats['action_mean'].item():.4f}, "
          f"action_std={norm_stats['action_std'].item():.4f}")

    # train/val split
    num_eps = len(dataset_paths)
    num_train = max(1, int(train_ratio * num_eps))
    rng = np.random.RandomState(0)
    ids = rng.permutation(num_eps)
    train_ids = ids[:num_train].tolist()
    val_ids = ids[num_train:].tolist()

    train_ep_len = [all_ep_len[i] for i in train_ids]
    val_ep_len = [all_ep_len[i] for i in val_ids]
    print(f"Train: {len(train_ids)} episodes, Val: {len(val_ids)} episodes")

    # HDF5 cache - all files opened once
    cache = HDF5Cache(dataset_paths)

    train_dataset = LiftEpisodicDataset(
        cache, train_ids, train_ep_len, camera_names, norm_stats, chunk_size)
    val_dataset = LiftEpisodicDataset(
        cache, val_ids, val_ep_len, camera_names, norm_stats, chunk_size)

    num_workers = 4
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=_batch_sampler(batch_size_train, [train_ep_len]),
        pin_memory=True, num_workers=num_workers, prefetch_factor=2,
        persistent_workers=True)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=_batch_sampler(batch_size_val, [val_ep_len]),
        pin_memory=True, num_workers=num_workers, prefetch_factor=2,
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
    set_seed(1)

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

    # load data
    print("Loading data...")
    train_loader, val_loader, norm_stats, hdf5_cache = load_data_lift(
        dataset_dir=DATA_DIR,
        camera_names=CAMERA_NAMES,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        chunk_size=chunk_size,
        train_ratio=0.9,
    )

    # build ACT policy (action_dim=1: lift only)
    policy_config = {
        "lr": lr,
        "num_queries": chunk_size,
        "kl_weight": kl_weight,
        "hidden_dim": hidden_dim,
        "dim_feedforward": dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": CAMERA_NAMES,
        "vq": False,
        "vq_class": None,
        "vq_dim": None,
        "action_dim": 1,
        "no_encoder": False,
    }

    model, optimizer = build_ACT_model_and_optimizer(policy_config)
    model = model.cuda()
    policy = ACTPolicy(policy_config)
    policy.model = model
    policy.optimizer = optimizer
    policy.cuda()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Training {num_steps} steps, batch_size={batch_size}, "
          f"chunk_size={chunk_size}...")

    # training loop
    min_val_loss = float("inf")
    best_ckpt_info = None
    train_history = []
    val_history = []

    train_iter = iter(train_loader)
    t0 = time.time()

    for step in tqdm(range(num_steps)):
        # fetch data
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        # forward + backward
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

        # log progress
        if step % 200 == 0 and step > 0:
            summary = compute_dict_mean(train_history[-200:])
            elapsed = time.time() - t0
            sps = step / elapsed
            print(f"Step {step:6d} | loss={summary['loss']:.4f} "
                  f"l1={summary['l1']:.4f} kl={summary['kl']:.4f} "
                  f"| {sps:.1f} steps/s")

        # validate
        if step % validate_every == 0 and step > 0:
            policy.eval()
            val_dicts = []
            with torch.no_grad():
                for val_data in val_loader:
                    fwd = forward_pass(val_data, policy)
                    val_dicts.append({k: v.item() for k, v in fwd.items()})
            val_summary = compute_dict_mean(val_dicts)
            val_history.append({"step": step, **val_summary})

            if val_summary["loss"] < min_val_loss:
                min_val_loss = val_summary["loss"]
                best_ckpt_info = (step, min_val_loss,
                                  deepcopy(policy.serialize()))
            print(f"  ==> Val loss={val_summary['loss']:.5f} "
                  f"(best={min_val_loss:.5f})")

        # save checkpoint
        if step % save_every == 0 and step > 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}.ckpt")
            torch.save(policy.serialize(), ckpt_path)

    # final save
    torch.save(policy.serialize(), os.path.join(ckpt_dir, "policy_last.ckpt"))

    if best_ckpt_info:
        best_step, best_loss, best_state = best_ckpt_info
        torch.save(best_state,
                   os.path.join(ckpt_dir, f"policy_best_step_{best_step}.ckpt"))
        print(f"\nTraining done! Best val loss = {best_loss:.6f} "
              f"at step {best_step}")
    else:
        print("\nTraining done!")

    # cleanup
    hdf5_cache.close()

    # plot loss
    steps_axis = range(len(train_history))
    losses = [h["loss"] for h in train_history]
    l1s = [h["l1"] for h in train_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(steps_axis, losses, alpha=0.2, color="blue")
    if len(losses) > 100:
        smoothed = np.convolve(losses, np.ones(100) / 100, mode="valid")
        ax1.plot(range(99, 99 + len(smoothed)), smoothed, color="blue",
                 linewidth=2, label="Smoothed")
    if val_history:
        vsteps = [h["step"] for h in val_history]
        vlosses = [h["loss"] for h in val_history]
        ax1.plot(vsteps, vlosses, "ro-", label="Val", markersize=4)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Total Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_title("Training Loss")

    ax2.plot(steps_axis, l1s, alpha=0.2, color="green")
    if len(l1s) > 100:
        smoothed_l1 = np.convolve(l1s, np.ones(100) / 100, mode="valid")
        ax2.plot(range(99, 99 + len(smoothed_l1)), smoothed_l1,
                 color="green", linewidth=2)
    ax2.set_xlabel("Step"); ax2.set_ylabel("L1 Loss")
    ax2.grid(True, alpha=0.3); ax2.set_title("L1 Regression Loss")

    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, "lift_training_loss.png")
    plt.savefig(loss_path, dpi=100)
    plt.close()
    print(f"Loss chart saved to: {loss_path}")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mobile ALOHA lift dimension")
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default 64 for GPU utilization)")
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
