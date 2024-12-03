from setuptools import find_packages, setup

package_name = 'lidar_distance_calculator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/distance_calculator_launch.py']),
        ('share/' + package_name + '/launch', ['launch/filters_launch.py']),
        ('share/' + package_name + '/config',
         ['config/median_filter_example.yaml']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'laser_filters',
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib'],
    zip_safe=True,
    maintainer='ter',
    maintainer_email='taweeporn1947@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'distance_calculator = lidar_distance_calculator.distance_calculator:main',
        ],
    },
)
