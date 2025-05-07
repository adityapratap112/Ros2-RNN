import sqlite3
import pandas as pd
import rclpy.serialization
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rosidl_runtime_py.utilities import get_message

bag_path = 'data_0.db3'

conn = sqlite3.connect(bag_path)
cursor = conn.cursor()

cursor.execute("SELECT id, name, type FROM topics")
topics = cursor.fetchall()

type_map = {}
for id, name, type_str in topics:
    type_map[id] = (name, type_str)

cursor.execute("SELECT topic_id, timestamp, data FROM messages")
rows = cursor.fetchall()

odom_data = []
cmd_vel_data = []

for topic_id, timestamp, data in rows:
    topic_name, type_str = type_map[topic_id]
    msg_type = get_message(type_str)
    msg = rclpy.serialization.deserialize_message(data, msg_type)

    time_sec = timestamp / 1e9

    if topic_name == '/odom':
        odom_data.append([time_sec, msg.pose.pose.position.x, msg.pose.pose.position.y,
                          msg.twist.twist.linear.x, msg.twist.twist.angular.z])
    elif topic_name == '/cmd_vel':
        cmd_vel_data.append([time_sec, msg.linear.x, msg.angular.z])

df_odom = pd.DataFrame(odom_data, columns=['time', 'x', 'y', 'lin_vel', 'ang_vel'])
df_cmd = pd.DataFrame(cmd_vel_data, columns=['time', 'cmd_lin_x', 'cmd_ang_z'])

df_odom.to_csv('odom.csv', index=False)
df_cmd.to_csv('cmd_vel.csv', index=False)

print("✅ CSV files saved: odom.csv, cmd_vel.csv")
