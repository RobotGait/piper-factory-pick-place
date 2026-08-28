from setuptools import setup
import os
from glob import glob

package_name = 'piper_factory_demo'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='GaitOne',
    maintainer_email='321972015+RobotGait@users.noreply.github.com',
    description='AgileX PiPER factory pick-and-place demo (ROS2 + MoveIt).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pick_place_demo = piper_factory_demo.pick_place_demo:main',
        ],
    },
)
