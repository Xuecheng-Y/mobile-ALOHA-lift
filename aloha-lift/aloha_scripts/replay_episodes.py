import os
import h5py
import argparse
from real_env import make_real_env

try:
    import IPython
    e = IPython.embed
except ImportError:
    pass


def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx:06d}'

    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at\n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]

    env = make_real_env(init_node=True)
    env.reset()
    for action in actions:
        env.step(action)

    print(f'Replayed {len(actions)} steps from {dataset_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))
