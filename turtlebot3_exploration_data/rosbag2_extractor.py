import os
import rclpy
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader
from rosbag2_py import StorageOptions, ConverterOptions

storage_options = StorageOptions(uri='.', storage_id='sqlite3')
converter_options = ConverterOptions('', '')

reader = SequentialReader()
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

msgs = {'/odom': [], '/cmd_vel': []}

rclpy.init()

while reader.has_next():
    topic, data, t = reader.read_next()
    if topic in msgs:
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        if topic == '/odom':
            msgs[topic].append({
                'time': t / 1e9,
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'lin_vel': msg.twist.twist.linear.x,
                'ang_vel': msg.twist.twist.angular.z
            })
        elif topic == '/cmd_vel':
            msgs[topic].append({
                'time': t / 1e9,
                'cmd_lin_x': msg.linear.x,
                'cmd_ang_z': msg.angular.z
            })

rclpy.shutdown()

pd.DataFrame(msgs['/odom']).to_csv('odom.csv', index=False)
pd.DataFrame(msgs['/cmd_vel']).to_csv('cmd_vel.csv', index=False)

print("✅ Saved: odom.csv, cmd_vel.csv")
