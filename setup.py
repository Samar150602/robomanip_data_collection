import os

from glob import glob
from setuptools import find_packages, setup

package_name = 'robomanip_data_collection'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='samar',
    maintainer_email='samarislam6954@gmail.com',
    description="Data collection for dual-arm manipulation.",
    license="Apache-2.0",
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "fairino_state_node = "
            "robomanip_data_collection.fairino_state_node:main",

            "mock_fairino_node = "
            "robomanip_data_collection.mock_fairino_node:main",

            "episode_recorder = "
            "robomanip_data_collection.episode_recorder:main"
        ],
    },
)
