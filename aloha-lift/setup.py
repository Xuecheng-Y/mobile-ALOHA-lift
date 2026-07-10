from setuptools import setup

setup(
    name='aloha_lift',
    version='0.1.0',
    packages=['aloha_scripts'],
    install_requires=[
        'numpy',
        'h5py',
        'opencv-python',
        'matplotlib',
        'dm_env',
        'tqdm',
    ],
)
