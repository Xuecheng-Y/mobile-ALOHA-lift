import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from constants import DT, TASK_CONFIGS
from robot_utils import Recorder, ImageRecorder
from real_env import make_real_env

try:
    import IPython
    e = IPython.embed
except ImportError:
    pass


def get_action_from_topic(master_recorder):
    """
    Read teleoperation action from the master device's joint states.
    Returns a 29-dim numpy array.
    
    Override this to match your teleoperation hardware.
    Default: reads joint positions from a ROS topic via Recorder.
    """
    return np.array(master_recorder.qpos)


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name,
                         overwrite, master_recorder):
    print(f'Dataset name: {dataset_name}')

    env = make_real_env(init_node=False)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path + '.hdf5') and not overwrite:
        print(f'Dataset already exist at\n{dataset_path}.hdf5\nHint: set overwrite to True.')
        exit()

    print('Press Ctrl+C to start recording, then operate the master device...')
    try:
        input('Press Enter when ready...')
    except (KeyboardInterrupt, EOFError):
        pass

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []

    for t in tqdm(range(max_timesteps)):
        t0 = time.time()
        action = get_action_from_topic(master_recorder)
        t1 = time.time()
        ts = env.step(action)
        t2 = time.time()
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 42:
        print(f'WARNING: control frequency low ({freq_mean:.1f} Hz), consider re-recording')
        return False

    qpos_dim = len(actions[0])

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    # Write HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3))
        _ = obs.create_dataset('qpos', (max_timesteps, qpos_dim))
        _ = obs.create_dataset('qvel', (max_timesteps, qpos_dim))
        _ = obs.create_dataset('effort', (max_timesteps, qpos_dim))
        _ = root.create_dataset('action', (max_timesteps, qpos_dim))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    # Setup master recorder to read teleoperation commands
    master_recorder = Recorder(robot_namespace='master', init_node=True)

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)

    dataset_name = f'episode_{episode_idx:06d}'
    print(dataset_name + '\n')

    while True:
        is_healthy = capture_one_episode(
            DT, max_timesteps, camera_names, dataset_dir, dataset_name,
            overwrite=True, master_recorder=master_recorder
        )
        if is_healthy:
            break


def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i:06d}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args()))
