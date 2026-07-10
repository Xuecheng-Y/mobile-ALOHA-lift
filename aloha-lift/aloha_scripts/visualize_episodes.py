import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

try:
    import IPython
    e = IPython.embed
except ImportError:
    pass

JOINT_NAMES = [f"joint_{i}" for i in range(29)]


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at\n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, effort, action, image_dict


def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx:06d}'

    qpos, qvel, effort, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))


def save_videos(video, dt, video_path=None):
    if isinstance(video, dict):
        cam_names = list(video.keys())
        # First frame determines dimensions
        first_frame = video[cam_names[0]]
        if first_frame.ndim == 3:
            n_frames, h, w, _ = first_frame.shape
            fps = int(1 / dt)
            all_frames = []
            for t in range(n_frames):
                frame_list = []
                for cam_name in cam_names:
                    img = video[cam_name][t]
                    img = img[:, :, [2, 1, 0]]  # BGR
                    frame_list.append(img)
                all_frames.append(np.concatenate(frame_list, axis=1))
            total_w = all_frames[0].shape[1]
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_w, h))
            for frame in all_frames:
                out.write(frame)
            out.release()
        else:
            # Time-last format
            frames, h, w, _ = first_frame.shape
            all_cam_videos = [video[name] for name in cam_names]
            all_cam_videos = np.concatenate(all_cam_videos, axis=2)
            n_frames, h, w, _ = all_cam_videos.shape
            fps = int(1 / dt)
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for t in range(n_frames):
                img = all_cam_videos[t][:, :, [2, 1, 0]]
                out.write(img)
            out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None):
    qpos = np.array(qpos_list)
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape

    # Plot lift dimension and a few neighbors for context
    dims_to_plot = list(range(max(0, 23), min(num_dim, 28)))  # show dims around lift (dim 26)
    num_plots = len(dims_to_plot)

    fig, axs = plt.subplots(num_plots, 1, figsize=(8, 2 * num_plots))
    if num_plots == 1:
        axs = [axs]
    for idx, dim_idx in enumerate(dims_to_plot):
        ax = axs[idx]
        ax.plot(qpos[:, dim_idx], label='State')
        ax.plot(command[:, dim_idx], label='Command')
        marker = ' *** LIFT ***' if dim_idx == 26 else ''
        ax.set_title(f'Joint {dim_idx}{marker}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))
