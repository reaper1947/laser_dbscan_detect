import enum
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import Point
import tf_transformations
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import os
from visualization_msgs.msg import Marker


class States(enum.Enum):
    IDLE = 1
    DETECTING = 2
    BROADCASTING = 3
    NO_OBJECT_DETECTED = 4


class function:
    """Class to categorize groups into width and length based on distance thresholds"""

    # def __init__(self, width_range=(0.60, 0.63), length_range=(0.63, 0.70)):
    def __init__(self, width_range=(0.60, 0.65), length_range=(0.60, 0.65)):
        self.width_range = width_range
        self.length_range = length_range

    def categorize_distance(self, group1, group2):
        """Categorize the distance between two groups as width or length based on predefined ranges"""
        distance = self.calculate_distance(group1, group2)
        is_width = self.width_range[0] <= distance <= self.width_range[1]
        is_length = self.length_range[0] <= distance <= self.length_range[1]

        if is_width and is_length:
            return ['width', 'length'], distance
        elif is_width:
            return ['width'], distance
        elif is_length:
            return ['length'], distance
        else:
            return None, None

    def calculate_distance(self, group1, group2):
        """Calculate distance between centroids of two groups"""
        centroid1 = self.calculate_centroid(group1)
        centroid2 = self.calculate_centroid(group2)
        return math.dist(centroid1, centroid2)

    def calculate_centroid(self, group):
        """Calculate the centroid of a group of points"""
        x_avg = sum(p[0] for p in group) / len(group)
        y_avg = sum(p[1] for p in group) / len(group)
        return x_avg, y_avg


class DistanceCalculator(Node):
    def __init__(self):
        super().__init__('distance_calculator')
        self.current_state = States.IDLE
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_filtered',
            self.lidar_callback,
            10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_object_time = None
        self.timer = self.create_timer(0.1, self.state_machine)

        # Parameters
        self.declare_parameter('proximity_threshold', 0.03)
        self.declare_parameter('offset_length', -0.68)
        self.time_threshold_ns = 0.5 * 1e9

        # Real-time plotting setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Directory to save plots
        self.plot_dir = 'plots'
        os.makedirs(self.plot_dir, exist_ok=True)

        # Initialize function categorization class
        self.function = function()

    def state_machine(self):
        """State machine logic"""
        if self.current_state == States.IDLE:
            self.get_logger().info("State: IDLE. Waiting for data...", throttle_duration_sec=10.0)
            if self.last_object_time:
                self.current_state = States.DETECTING

        elif self.current_state == States.DETECTING:
            if not self.last_object_time or \
                    (self.get_clock().now() - self.last_object_time).nanoseconds > self.time_threshold_ns:
                self.current_state = States.NO_OBJECT_DETECTED
            else:
                self.current_state = States.BROADCASTING

        elif self.current_state == States.BROADCASTING:
            self.current_state = States.IDLE

        elif self.current_state == States.NO_OBJECT_DETECTED:
            self.get_logger().warn("State: NO_OBJECT_DETECTED. No objects detected!")
            self.current_state = States.IDLE

    def get_proximity_threshold(self):
        """Fetch the current value of the proximity_threshold parameter"""
        return self.get_parameter('proximity_threshold').get_parameter_value().double_value

    def get_offset_length(self):
        """Fetch the current value of the offset_length parameter"""
        return self.get_parameter('offset_length').get_parameter_value().double_value

    def lidar_callback(self, msg):
        """Lidar data processing"""
        proximity_threshold = self.get_proximity_threshold()
        offset_length = self.get_offset_length()
        detected_points = self.filter_points(msg)
        grouped_points = self.group_points(detected_points, proximity_threshold=proximity_threshold)

        self.last_object_time = self.get_clock().now() if detected_points else None

        if self.current_state == States.DETECTING:
            self.plot_points_realtime(detected_points, grouped_points)
            # save plot
            # self.save_plot()

            if len(grouped_points) >= 2:

                centroids = [self.calculate_centroid(group) for group in grouped_points[:4]]
                # width = self.calculate_distance_between_groups(centroids[3], centroids[0])
                # length = self.calculate_distance_between_groups(centroids[0], centroids[1])
                x_centroid, y_centroid = self.calculate_centroid(centroids)
                angle = math.atan2(y_centroid, x_centroid)
                quaternion = tf_transformations.quaternion_from_euler(0, 0, angle)

                # self.get_logger().info(f"Outer_Width : {width:.2f} meters")
                # self.get_logger().info(f"Outer_Length : {length:.2f} meters")
                # self.get_logger().info(f"Angle: {math.degrees(angle):.2f}°")
                if len(grouped_points) >= 4:
                    self.send_transform(x_centroid, y_centroid, "centroid", quaternion)
                # Check distances between groups and categorize them
                for i in range(len(centroids) - 1):
                    categories, distance = self.function.categorize_distance(
                        grouped_points[i], grouped_points[i + 1])
                    if categories is None:
                        continue
                    # if category:
                    #     self.get_logger().info(
                    #         f"Group {i} to Group {i+1}: {category} - Distance: {distance:.2f}meters")
                    #     x_centroid, y_centroid = self.calculate_centroid(
                    #         [centroids[i], centroids[i + 1]])
                    #     self.send_transform(
                    #         x_centroid, y_centroid, f"{category}_{i}_{i+1}", quaternion)
                    #     self.broadcast_group_transforms(grouped_points, quaternion)

                    if 'width' in categories and len(grouped_points) != 2:
                        # self.get_logger().info(
                        #     f"Group {i} to Group {i+1}: width - Distance: {distance:.2f}meters")
                        x_centroid, y_centroid = self.calculate_centroid(
                            [centroids[i], centroids[i + 1]])
                        self.send_transform(
                            x_centroid, y_centroid, f"width_{i}_{i+1}", quaternion)

                        self.broadcast_group_transforms(grouped_points, quaternion)

                    if 'length' in categories and len(grouped_points) != 2:
                        # self.get_logger().info(
                        #     f"Group {i} to Group {i+1}: length - Distance: {distance:.2f}meters")
                        x_centroid, y_centroid = self.calculate_centroid(
                            [centroids[i], centroids[i + 1]])
                        self.send_transform(
                            x_centroid, y_centroid, f"length_{i}_{i+1}", quaternion)
                        self.broadcast_group_transforms(grouped_points, quaternion)

                    if 'length' in categories or 'width' in categories:
                        self.get_logger().info("Detect SHELF!!!!!!!!!", throttle_duration_sec=2.0)

                    if len(grouped_points) == 2 and 'width' in categories:
                        x_avg1 = sum(p[0] for p in grouped_points[0]) / len(grouped_points[0])
                        y_avg1 = sum(p[1] for p in grouped_points[0]) / len(grouped_points[0])

                        x_avg2 = sum(p[0] for p in grouped_points[1]) / len(grouped_points[1])
                        y_avg2 = sum(p[1] for p in grouped_points[1]) / len(grouped_points[1])
                        angle2 = math.atan2(y_avg2 - y_avg1, x_avg2 - x_avg1)
                        perpendicular_angle = angle2 + math.pi / 2
                        offset_x = offset_length * math.cos(perpendicular_angle)
                        offset_y = offset_length * math.sin(perpendicular_angle)

                        self.x_outer1 = x_avg1 + offset_x
                        self.y_outer1 = y_avg1 + offset_y
                        self.x_outer2 = x_avg2 + offset_x
                        self.y_outer2 = y_avg2 + offset_y

                        x_centroid, y_centroid = self.calculate_centroid(
                            [centroids[i], centroids[i + 1]])
                        self.send_transform(
                            x_centroid, y_centroid, f"width_IN", quaternion)
                        self.broadcast_group_transforms(grouped_points, quaternion)
                        self.send_transform(
                            self.x_outer1, self.y_outer1, "outer_corner_1", quaternion)
                        self.send_transform(
                            self.x_outer2, self.y_outer2, "outer_corner_2", quaternion)

    def publish_polygon_marker(self, grouped_points):
        """Publish a polygon marker based on grouped points."""
        if len(grouped_points) < 4:
            return

        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "group_polygon"
        marker.id = 1
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8

        # Define vertices for polygon (assuming 4 groups form a rectangle)
        order = [0, 1, 3, 2]  # Define the clockwise order for the vertices
        vertices = [grouped_points[i][0] for i in order]

        # Create triangle pairs for the polygon
        triangles = [
            [vertices[0], vertices[1], vertices[2]],
            [vertices[2], vertices[3], vertices[0]],
        ]

        for triangle in triangles:
            for vertex in triangle:
                point = geometry_msgs.msg.Point()
                point.x = vertex[0]
                point.y = vertex[1]
                point.z = 0.0
                marker.points.append(point)

        self.marker_pub.publish(marker)

    # def publish_line_strip_marker(self, grouped_points):
    #     """Publish a LINE_STRIP marker connecting each consecutive group."""
    #     if len(grouped_points) < 4:
    #         return

    #     marker = Marker()
    #     marker.header.frame_id = "laser"
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.ns = "group_line_strip"
    #     marker.id = 0
    #     marker.type = Marker.LINE_STRIP
    #     marker.action = Marker.ADD
    #     marker.scale.x = 0.02
    #     marker.color.r = 0.0
    #     marker.color.g = 1.0
    #     marker.color.b = 0.0
    #     marker.color.a = 1.0

    #     order = [0, 1, 3, 2, 0]
    #     for i in range(len(order) - 1):
    #         start_index = order[i]
    #         end_index = order[i + 1]

    #         start_point = geometry_msgs.msg.Point()
    #         start_point.x = grouped_points[start_index][0][0]
    #         start_point.y = grouped_points[start_index][0][1]
    #         start_point.z = 0.0
    #         marker.points.append(start_point)

    #         end_point = geometry_msgs.msg.Point()
    #         end_point.x = grouped_points[end_index][0][0]
    #         end_point.y = grouped_points[end_index][0][1]
    #         end_point.z = 0.0
    #         marker.points.append(end_point)

    #     self.marker_pub.publish(marker)

    def broadcast_group_transforms(self, grouped_points, quaternion):
        """Broadcast transforms for each group, creating the rectangular shape."""
        for i in range(len(grouped_points)):
            x_avg, y_avg = self.calculate_centroid(grouped_points[i])
            self.send_transform(x_avg, y_avg, f"group_{i}", quaternion)
        self.publish_polygon_marker(grouped_points)
        self.detect_perpendicularity(grouped_points)

    def calculate_angle(self, start_point, end_point):
        vector_x = end_point.x - start_point.x
        vector_y = end_point.y - start_point.y
        angle_rad = math.atan2(vector_y, vector_x)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def detect_perpendicularity(self, grouped_points):
        order = [0, 1, 3, 2, 0]
        for i in range(len(order) - 1):
            start_index = order[i]
            end_index = order[i + 1]
            if 0 <= start_index < len(grouped_points):
                start_point = Point()
                start_point.x = grouped_points[start_index][0][0]
                start_point.y = grouped_points[start_index][0][1]
            if 0 <= end_index < len(grouped_points):
                end_point = Point()
                end_point.x = grouped_points[end_index][0][0]
                end_point.y = grouped_points[end_index][0][1]

            angle = self.calculate_angle(start_point, end_point)

            if abs(angle - 90) < 7:
                self.get_logger().info("***********SHELF___NO.1***********", throttle_duration_sec=2.0)
                return

    def calculate_distance_between_groups(self, centroid1, centroid2):
        """Calculate distance between two centroids"""
        distance = math.dist(centroid1, centroid2)
        return distance

    def filter_points(self, msg):
        """Filter points based on range only and prioritize nearest object"""
        detected_points = [
            (
                msg.ranges[i] * math.cos(msg.angle_min + i * msg.angle_increment),
                msg.ranges[i] * math.sin(msg.angle_min + i * msg.angle_increment),
                msg.ranges[i]  # Adding range as the 3rd element to track distance
            )
            for i in range(len(msg.ranges))
            if msg.ranges[i] < 1.5  # Only filter by range (no intensity check)
        ]

        # Sort points by their distance from the LIDAR (nearest points first)
        detected_points.sort(key=lambda p: p[2])

        return detected_points

    def group_points(self, detected_points, method='dbscan', proximity_threshold=0.03):
        """Group points by proximity and prioritize the closest object first"""
        if not detected_points:
            return []

        # Prioritize the nearest point first
        nearest_point = detected_points[0]  # The first point is the closest one
        remaining_points = detected_points[1:]  # The rest of the points

        # Group the remaining points using KNN or DBSCAN
        if method == 'dbscan':
            return self.group_points_dbscan(remaining_points, proximity_threshold)
        elif method == 'knn':
            return self.group_points_knn(remaining_points, proximity_threshold)
        else:
            raise ValueError(f"Unknown grouping method: {method}")

    # def group_points_dbscan(self, detected_points):
    #     """Group points using DBSCAN"""
    #     if not detected_points:
    #         return []
    #     detected_points_np = np.array(detected_points)
    #     db = DBSCAN(eps=self.proximity_threshold, min_samples=10).fit(detected_points_np)
    #     grouped_points = [
    #         detected_points_np[db.labels_ == label].tolist()
    #         for label in set(db.labels_) if label != -1
    #     ]
    #     return grouped_points
    def group_points_dbscan(self, detected_points, proximity_threshold):
        """Group points using DBSCAN with additional distance filtering"""
        if not detected_points:
            return []

        detected_points_np = np.array(detected_points)
        db = DBSCAN(eps=proximity_threshold, min_samples=6).fit(detected_points_np)

        grouped_points = []
        for label in set(db.labels_):
            if label == -1:  # Skip noise points
                continue

            # Extract points in the current cluster
            cluster_points = detected_points_np[db.labels_ == label]

            # Check if the cluster's maximum pairwise distance exceeds the threshold
            max_distance = np.max(pdist(cluster_points)) if len(cluster_points) > 1 else 0
            if 0.01 <= max_distance <= 0.1:  # Keep only clusters within the allowed max distance
                grouped_points.append(cluster_points.tolist())
        return grouped_points

    def group_points_knn(self, detected_points, proximity_threshold):
        if not detected_points:
            return []

        detected_points_np = np.array(detected_points)
        nbrs = NearestNeighbors(radius=proximity_threshold).fit(detected_points_np)
        distances, indices = nbrs.radius_neighbors(detected_points_np)

        grouped_points = []
        visited = set()

        for idx, neighbors in enumerate(indices):
            if idx not in visited:
                group = [tuple(detected_points_np[n]) for n in neighbors]
                grouped_points.append(group)
                visited.update(neighbors)

        return grouped_points

    def calculate_centroid(self, points):
        """Calculate centroid"""
        x_avg = sum(p[0] for p in points) / len(points)
        y_avg = sum(p[1] for p in points) / len(points)
        return x_avg, y_avg

    def send_transform(self, x, y, child_frame_id, quaternion):
        """Send TF with quaternion"""
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'laser'
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]

        self.tf_broadcaster.sendTransform(transform)

    def plot_points_realtime(self, detected_points, grouped_points):
        """Plot points in real time"""
        self.ax.cla()
        detected_points_np = np.array(detected_points)
        self.ax.scatter(detected_points_np[:, 0],
                        detected_points_np[:, 1], c='gray', label='Detected Points')
        centroids = []
        for i, group in enumerate(grouped_points):
            group_np = np.array(group)
            self.ax.scatter(group_np[:, 0], group_np[:, 1], label=f'Group {i}')

            centroid_x = np.mean(group_np[:, 0])
            centroid_y = np.mean(group_np[:, 1])
            centroids.append((centroid_x, centroid_y))

            self.ax.scatter(
                centroid_x,
                centroid_y,
                c='red',
                marker='x',
                label=f'Group {i} Centroid')
            # self.ax.text(
            #     centroid_x,
            #     centroid_y,
            #     f'({centroid_x:.2f}, {centroid_y:.2f})',
            #     color='red')

            y_values = group_np[:, 1]
            y_min, y_max = np.min(y_values), np.max(y_values)
            y_range = y_max - y_min
            # self.get_logger().info(f"Group {i}: Y-length = {y_range:.2f} meters")

            mid_x = centroid_x
            mid_y = (y_min + y_max) / 2
            self.ax.text(mid_x, mid_y, f"LEG = {y_range:.2f} m", color='green')

        for i in range(len(centroids) - 2):
            x_distance = abs(centroids[i][0] - centroids[i + 2][0])
            y_distance = abs(centroids[i][1] - centroids[i + 1][1])
            self.get_logger().info(
                f"Y-distance between Group {i} and Group {i+1}: {y_distance:.2f} meters", throttle_duration_sec=3.0)
            self.get_logger().info(
                f"X-distance between Group {i} and Group {i+2}: {x_distance:.2f} meters", throttle_duration_sec=3.0)

            mid_x = (centroids[i][0] + centroids[i + 1][0]) / 2
            mid_y = (centroids[i][1] + centroids[i + 1][1]) / 2
            self.ax.text(mid_x, mid_y, f"{y_distance:.2f} m", color='blue')
            mid_x2 = (centroids[i][0] + centroids[i + 2][0]) / 2
            mid_y2 = (centroids[i][1] + centroids[i + 2][1]) / 2
            self.ax.text(mid_x2, mid_y2, f"{x_distance:.2f} m", color='red')

        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_title("Real-time Lidar Detection Graph")
        plt.pause(0.001)

    def save_plot(self):
        timestamp = self.get_clock().now().to_msg().sec
        plot_filename = os.path.join(self.plot_dir, f"lidar_plot_{timestamp}.png")
        self.fig.savefig(plot_filename)
        self.get_logger().info(f"Plot saved to {plot_filename}")


def main(args=None):
    rclpy.init(args=args)
    node = DistanceCalculator()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
