"""
数据预处理: 从 HDF5 提取升降维度数据 + 下采样图像, 保存为 .npz 文件.
运行本脚本一次即可, 训练时直接加载 .npz 文件, 速度极快.

用法: python aloha_scripts/preprocess_lift.py
"""

import os, sys, time
import numpy as np
import h5py
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "act_real")       # HDF5 输入
PROC_DIR = os.path.join(ROOT_DIR, "act_real_proc")  # 预处理输出
os.makedirs(PROC_DIR, exist_ok=True)

LIFT_DIM = 26
IMG_SIZE = 224
CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

def preprocess_one(fp, proc_path):
    """预处理单个 HDF5 文件."""
    with h5py.File(fp, "r") as f:
        n_steps = f["action"].shape[0]

        # 提取 lift action 和 qpos
        lift_action = f["action"][:, LIFT_DIM].astype(np.float32)       # (T,)
        lift_qpos = f["observations/qpos"][:, LIFT_DIM].astype(np.float32)  # (T,)

        # 下采样图像
        images = {}
        for cn in CAMERA_NAMES:
            raw = f[f"observations/images/{cn}"][:]                        # (T, H, W, 3)
            T, H, W, C = raw.shape
            resized = np.empty((T, IMG_SIZE, IMG_SIZE, C), dtype=np.uint8)
            for t in range(T):
                resized[t] = cv2.resize(raw[t], (IMG_SIZE, IMG_SIZE),
                                        interpolation=cv2.INTER_LINEAR)
            images[cn] = resized

    # 保存为 .npz (压缩)
    np.savez_compressed(
        proc_path,
        lift_action=lift_action,
        lift_qpos=lift_qpos,
        cam_high=images["cam_high"],
        cam_left_wrist=images["cam_left_wrist"],
        cam_right_wrist=images["cam_right_wrist"],
    )
    return n_steps


def main():
    hdf5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5")])
    if not hdf5_files:
        print(f"[错误] {DATA_DIR} 中没有 .hdf5 文件")
        sys.exit(1)

    print(f"数据源: {DATA_DIR}")
    print(f"输出目录: {PROC_DIR}")
    print(f"找到 {len(hdf5_files)} 个 episode\n")

    t0 = time.time()
    total_frames = 0

    for fi, fn in enumerate(hdf5_files):
        fp = os.path.join(DATA_DIR, fn)
        proc_path = os.path.join(PROC_DIR, fn.replace(".hdf5", "_proc.npz"))

        if os.path.exists(proc_path):
            print(f"[{fi+1}/{len(hdf5_files)}] {fn} → 已存在, 跳过")
            with np.load(proc_path) as d:
                total_frames += d["lift_action"].shape[0]
            continue

        t1 = time.time()
        n = preprocess_one(fp, proc_path)
        total_frames += n
        sz_mb = os.path.getsize(proc_path) / 1024 / 1024
        print(f"[{fi+1}/{len(hdf5_files)}] {fn} → {n} 帧, {sz_mb:.0f}MB, {time.time()-t1:.1f}s")

    print(f"\n完成! 总帧数: {total_frames}, 总耗时: {time.time()-t0:.1f}s")
    print(f"预处理数据: {PROC_DIR}/")


if __name__ == "__main__":
    main()
