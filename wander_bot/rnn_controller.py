import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import pickle
import os
import math
from tf_transformations import euler_from_quaternion
from std_msgs.msg import Header
import csv

INPUT_FEATURES = ['x', 'y', 'lin_vel', 'ang_vel', 'cmd_lin_x', 'cmd_ang_z']
SEQ_LEN = 20

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n

class Decoder(nn.Module):
    def __init__(self, hidden_size=64, output_size=2):
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, h):
        return self.fc(h[-1])

class Seq2One(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, output_size=2)
    def forward(self, x):
        h_n, c_n = self.encoder(x)
        return self.decoder(h_n)

class RNNController(Node):
    def __init__(self):
        super().__init__('rnn_controller_node')

        self.safety_enabled = True  # Toggle obstacle avoidance

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pred_pub = self.create_publisher(PoseStamped, '/predicted_pose', 10)
        self.pred_path_pub = self.create_publisher(Path, '/predicted_path', 10)
        self.actual_path_pub = self.create_publisher(Path, '/actual_path', 10)

        base_path = os.path.expanduser('~/ros2_ws')
        with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        self.model = Seq2One(input_size=len(INPUT_FEATURES))
        self.model.load_state_dict(torch.load(os.path.join(base_path, 'encoder_decoder_model.pt')))
        self.model.eval()

        self.latest_state = {key: 0.0 for key in INPUT_FEATURES}
        self.current_position = (0.0, 0.0)
        self.current_yaw = 0.0
        self.buffer = deque(maxlen=SEQ_LEN)
        self.pred_path = Path()
        self.actual_path = Path()
        self.obstacle_near = False

        self.log_path = os.path.join(base_path, 'rnn_control_log.csv')
        with open(self.log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Actual_x', 'Actual_y', 'Predicted_x', 'Predicted_y', 'Linear_x', 'Angular_z'])

    def scan_callback(self, msg):
        front = msg.ranges[len(msg.ranges)//2 - 10 : len(msg.ranges)//2 + 10]
        self.obstacle_near = any(r < 0.2 for r in front if r > 0.0)
        if self.safety_enabled and self.obstacle_near:
            self.get_logger().warn("🚧 Obstacle detected! Stopping.")

    def odom_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.latest_state['x'], self.latest_state['y'] = self.current_position
        self.latest_state['lin_vel'] = msg.twist.twist.linear.x
        self.latest_state['ang_vel'] = msg.twist.twist.angular.z

        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.append_actual_path()
        self.predict_and_control()

    def cmd_callback(self, msg):
        self.latest_state['cmd_lin_x'] = msg.linear.x
        self.latest_state['cmd_ang_z'] = msg.angular.z

    def predict_and_control(self):
        input_vector = [self.latest_state[key] for key in INPUT_FEATURES]
        self.buffer.append(input_vector)

        if len(self.buffer) < SEQ_LEN:
            return

        input_array = np.array(self.buffer).reshape(1, SEQ_LEN, len(INPUT_FEATURES))
        input_scaled = self.scaler.transform(input_array[0])
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(input_tensor).numpy()[0]

        pred_x, pred_y = prediction
        self.get_logger().info(f'🧠 Predicted next position: x = {pred_x:.2f}, y = {pred_y:.2f}')

        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(pred_x)
        pose.pose.position.y = float(pred_y)
        pose.pose.orientation.w = 1.0
        self.pred_pub.publish(pose)
        self.append_predicted_path(pose)

        dx = pred_x - self.current_position[0]
        dy = pred_y - self.current_position[1]
        target_theta = math.atan2(dy, dx)
        angle_error = (target_theta - self.current_yaw + math.pi) % (2 * math.pi) - math.pi
        distance = math.hypot(dx, dy)

        twist = Twist()
        if self.safety_enabled and self.obstacle_near:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            twist.linear.x = min(0.2, distance)
            twist.angular.z = 1.5 * angle_error

        self.cmd_pub.publish(twist)

        with open(self.log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.get_clock().now().nanoseconds,
                self.current_position[0],
                self.current_position[1],
                pred_x,
                pred_y,
                twist.linear.x,
                twist.angular.z
            ])

    def append_actual_path(self):
        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.current_position[0]
        pose.pose.position.y = self.current_position[1]
        pose.pose.orientation.w = 1.0
        self.actual_path.header = pose.header
        self.actual_path.poses.append(pose)
        self.actual_path_pub.publish(self.actual_path)

    def append_predicted_path(self, pose):
        self.pred_path.header = pose.header
        self.pred_path.poses.append(pose)
        self.pred_path_pub.publish(self.pred_path)

def main(args=None):
    rclpy.init(args=args)
    node = RNNController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
