from setuptools import setup

package_name = 'so_arm100_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Blue cube perception for SO-ARM100',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'blue_cube_detector = so_arm100_perception.blue_cube_detector:main',
        ],
    },
)
