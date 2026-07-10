### Task parameters

DATA_DIR = '<put your data dir here>'
TASK_CONFIGS = {
    'act_real_lift': {
        'dataset_dir': DATA_DIR + '/act_real',
        'num_episodes': 10,
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        'qpos_dim': 29,
    },
}

### Lift robot fixed constants
DT = 0.02
LIFT_DIM = 26  # lift dimension in the 29-dim qpos/action array (0-indexed)

# Joint names for visualization (29-dim custom robot)
JOINT_NAMES = [f"joint_{i}" for i in range(29)]

############################ Helper functions ############################
