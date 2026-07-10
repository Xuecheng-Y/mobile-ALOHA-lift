"""
Deploy a trained act-lift ACT policy on the real robot.
Loads the ACT checkpoint and runs inference in a control loop.

Usage:
    python deploy_act.py \
        --ckpt_dir /path/to/act-lift/checkpoints \
        --policy_class ACT \
        --chunk_size 100 \
        --kl_weight 10 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --max_timesteps 500 \
        --temporal_agg

Prerequisites:
    - act-lift must be on PYTHONPATH (for policy.py and detr/)
    - Trained checkpoint (policy_best.ckpt) and stats (dataset_stats.pkl) in ckpt_dir
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

import torch

from constants import DT, LIFT_DIM, TASK_CONFIGS
from real_env import make_real_env

try:
    import IPython
    e = IPython.embed
except ImportError:
    pass


# ---------- Model loading (from act-lift) ----------

def import_act_lift(act_lift_path=None):
    """Import act-lift modules. Add act-lift path to sys.path if provided."""
    if act_lift_path is not None:
        sys.path.insert(0, act_lift_path)
    from policy import ACTPolicy, CNNMLPPolicy
    return ACTPolicy, CNNMLPPolicy


def make_policy(policy_class, policy_config, act_lift_path=None):
    ACTPolicy, CNNMLPPolicy = import_act_lift(act_lift_path)
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def load_policy_and_stats(ckpt_dir, policy_class, policy_config, act_lift_path=None):
    """Load trained model weights and normalization statistics."""
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f'Stats not found: {stats_path}')

    policy = make_policy(policy_class, policy_config, act_lift_path)
    state_dict = torch.load(ckpt_path, map_location='cuda')
    policy.load_state_dict(state_dict)
    policy.cuda()
    policy.eval()
    print(f'Loaded checkpoint from {ckpt_path}')

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    print(f"Loaded stats: qpos_mean={stats['qpos_mean']:.4f}, action_mean={stats['action_mean']:.4f}")

    return policy, stats


# ---------- Inference ----------

def do_inference(policy, stats, images, qpos_lift):
    """
    Run one inference pass.

    Args:
        policy: ACTPolicy instance
        stats: dict with 'qpos_mean', 'qpos_std', 'action_mean', 'action_std'
        images: numpy array (num_cams, H, W, 3) uint8
        qpos_lift: numpy array (1,) float32, the current lift joint position

    Returns:
        raw_actions: numpy array (chunk_size, 1), denormalized actions
    """
    # Normalize image: /255 + channels-first
    image_data = torch.from_numpy(images / 255.0).float().cuda().unsqueeze(0)  # (1, Cams, H, W, 3)
    image_data = torch.einsum('b k h w c -> b k c h w', image_data)  # -> (B, Cams, C, H, W)

    # Normalize qpos
    qpos = (qpos_lift - stats['qpos_mean']) / stats['qpos_std']
    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # (1, 1)

    # Inference (normalization to ImageNet stats done inside policy.__call__)
    with torch.inference_mode():
        all_actions = policy(qpos, image_data)  # (1, chunk_size, 1)
    all_actions = all_actions.squeeze(0).cpu().numpy()  # (chunk_size, 1)

    # Denormalize
    raw_actions = all_actions * stats['action_std'] + stats['action_mean']
    return raw_actions


# ---------- Main deployment loop ----------

def run_deployment(args):
    # ---- Configuration ----
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    state_dim = 1  # lift-only

    policy_config = {
        'lr': 1e-5,
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'state_dim': state_dim,
    }

    # ---- Load model ----
    policy, stats = load_policy_and_stats(
        args.ckpt_dir, args.policy_class, policy_config, args.act_lift_path
    )

    # ---- Setup environment ----
    env = make_real_env(init_node=True, robot_namespace=args.robot_namespace)
    ts = env.reset(fake=True)
    print(f'Environment ready. qpos dim: {len(ts.observation["qpos"])}')

    # ---- Warmup ----
    print('Warming up model...')
    dummy_qpos = np.array([0.0], dtype=np.float32)
    h, w = 480, 640
    dummy_images = np.zeros((3, h, w, 3), dtype=np.uint8)
    _ = do_inference(policy, stats, dummy_images, dummy_qpos)
    print('Warmup done. Starting control loop...')

    # ---- Control loop ----
    chunk_size = args.chunk_size
    max_timesteps = args.max_timesteps
    temporal_agg = args.temporal_agg
    state_dim_val = 1

    if temporal_agg:
        all_time_actions = np.zeros(
            (max_timesteps, max_timesteps + chunk_size, state_dim_val), dtype=np.float32
        )
        query_frequency = 1
    else:
        query_frequency = chunk_size

    print(f'Timesteps: {max_timesteps} | Chunk size: {chunk_size}')
    print(f'Temporal aggregation: {temporal_agg} | Control freq: {1/DT:.0f} Hz')
    print('Press Ctrl+C to stop.\n')

    try:
        for t in range(max_timesteps):
            loop_start = time.time()

            # 1. Get observation
            obs = env.get_observation()
            qpos_full = obs['qpos']
            qpos_lift = qpos_full[LIFT_DIM:LIFT_DIM + 1].astype(np.float32)
            images = np.stack([obs['images'][cam] for cam in camera_names], axis=0)

            # 2. Run inference
            if temporal_agg:
                raw_action = do_inference(policy, stats, images, qpos_lift)
                all_time_actions[t, t:t + chunk_size] = raw_action.squeeze(-1)

                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = np.any(all_time_actions[:, t] != 0.0, axis=1)
                if actions_populated.any():
                    lift_action = actions_for_curr_step[actions_populated].mean()
                else:
                    lift_action = 0.0
            else:
                if t % query_frequency == 0:
                    raw_action = do_inference(policy, stats, images, qpos_lift)
                lift_action = raw_action[t % query_frequency, 0]

            # 3. Construct full action (zeros for non-lift dimensions)
            full_action = np.array(qpos_full)  # default: hold current position
            full_action[LIFT_DIM] = lift_action

            # 4. Step the environment
            env.step(full_action)

            # 5. Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                if t % 50 == 0:
                    print(f'[WARN] Step {t}: overrun by {-sleep_time*1000:.1f}ms')

            if t % 50 == 0:
                print(f'Step {t}/{max_timesteps} | lift_qpos={qpos_lift[0]:.4f} | action={lift_action:.4f}')

    except KeyboardInterrupt:
        print('\nInterrupted by user.')
    finally:
        print('Deployment finished.')


# ---------- Entry point ----------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy act-lift on real robot')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to act-lift checkpoint directory')
    parser.add_argument('--act_lift_path', type=str, default=None,
                        help='Path to act-lift source directory (if not on PYTHONPATH)')
    parser.add_argument('--policy_class', type=str, default='ACT',
                        help='ACT or CNNMLP')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Action chunk size (must match training)')
    parser.add_argument('--kl_weight', type=int, default=10,
                        help='KL weight (must match training)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Transformer hidden dim (must match training)')
    parser.add_argument('--dim_feedforward', type=int, default=3200,
                        help='Feedforward dim (must match training)')
    parser.add_argument('--max_timesteps', type=int, default=500,
                        help='Max timesteps per episode')
    parser.add_argument('--temporal_agg', action='store_true',
                        help='Enable temporal aggregation for smoother actions')
    parser.add_argument('--robot_namespace', type=str, default='puppet',
                        help='ROS namespace for the robot')
    args = parser.parse_args()

    run_deployment(args)
