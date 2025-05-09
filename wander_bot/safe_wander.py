import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import random
import time

class SafeWander(Node):
    def __init__(self):
        super().__init__('safe_wander')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.last_turn_time = time.time()

    def scan_callback(self, msg):
        ranges = msg.ranges
        front = ranges[0:10] + ranges[-10:]
        left = ranges[80:100]
        right = ranges[260:280]

        def min_valid(r): return min([d for d in r if d > 0.05], default=1.0)

        min_front = min_valid(front)
        min_left = min_valid(left)
        min_right = min_valid(right)

        twist = Twist()

        if min_front < 0.35:
            # If stuck, turn randomly or back up occasionally
            if time.time() - self.last_turn_time > 2.0:
                twist.linear.x = -0.05
                twist.angular.z = random.uniform(-1.5, 1.5)
                self.last_turn_time = time.time()
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.3
        elif min_left < 0.25:
            twist.linear.x = 0.1
            twist.angular.z = -0.5  # turn away from left
        elif min_right < 0.25:
            twist.linear.x = 0.1
            twist.angular.z = 0.5  # turn away from right
        else:
            twist.linear.x = 0.15
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SafeWander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
