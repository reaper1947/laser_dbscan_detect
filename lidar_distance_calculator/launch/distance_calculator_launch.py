
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lidar_distance_calculator',
            executable='distance_calculator',
            name='distance_calculator',
            parameters=[{'proximity_threshold': 0.03}],
            output='screen',
        ),
    ])
