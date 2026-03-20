import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class BlueCubeDetector(Node):
    def __init__(self):
        super().__init__('blue_cube_detector')
        self.bridge = CvBridge()
        self.camera_info = None
        self.camera_height = 1.2
        self.table_height  = 0.4

        self.image_sub = self.create_subscription(
            Image, '/overhead_camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/overhead_camera/camera_info', self.info_callback, 10)
        self.position_pub = self.create_publisher(
            PointStamped, '/blue_cube_position', 10)
        self.debug_pub = self.create_publisher(
            Image, '/blue_cube_debug', 10)

        self.get_logger().info('Blue Cube Detector started!')

    def info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Wide HSV range for blue in Gazebo lighting
        lower_blue = np.array([90,  50,  30])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_frame = frame.copy()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            self.get_logger().info(f'Largest blue contour area: {area:.1f}')

            if area > 100:
                x, y, w, h = cv2.boundingRect(largest)
                cx = x + w // 2
                cy = y + h // 2

                world_x, world_y = self.pixel_to_world(cx, cy, frame.shape)

                if world_x is not None:
                    point = PointStamped()
                    point.header.stamp = self.get_clock().now().to_msg()
                    point.header.frame_id = 'world'
                    point.point.x = world_x
                    point.point.y = world_y
                    point.point.z = self.table_height
                    self.position_pub.publish(point)
                    self.get_logger().info(
                        f'Blue cube at world: x={world_x:.3f}, y={world_y:.3f}')

                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(debug_frame, f'BLUE ({cx},{cy})',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            self.get_logger().warn('No blue detected - check HSV range')
            cv2.putText(debug_frame, 'No blue cube detected',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Also show the mask for debugging
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([debug_frame, mask_bgr])
        self.debug_pub.publish(
            self.bridge.cv2_to_imgmsg(combined, encoding='bgr8'))

    def pixel_to_world(self, cx, cy, shape):
        if self.camera_info is None:
            return None, None
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        px = self.camera_info.k[2]
        py = self.camera_info.k[5]
        dz = self.camera_height - self.table_height
        cam_x = (cx - px) * dz / fx
        cam_y = (cy - py) * dz / fy
        world_x = 0.3 + cam_x
        world_y = 0.0 + cam_y
        return world_x, world_y

def main(args=None):
    rclpy.init(args=args)
    node = BlueCubeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
