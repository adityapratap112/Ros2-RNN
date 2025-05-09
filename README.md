
---

```markdown
# 🧠 TurtleBot3 ROS2 RNN Project

This project demonstrates how to simulate a TurtleBot3 in ROS2 Humble using Gazebo,
enable autonomous LiDAR-based navigation, record its trajectory data, and prepare for
 RNN-based trajectory prediction.

---




---

 💻 Terminal Commands Used

🔧 Environment Setup

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
````

---

🏗️ Workspace & Package Creation

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python wander_bot
mkdir wander_bot
touch wander_bot/__init__.py
```

---

 ✍️ Add Autonomous Navigation Script

Create safe\_wander.py in wander\_bot/.
* Add LiDAR-based obstacle avoidance logic.

In setup.py, under `entry_points`, add:

```python
'console_scripts': [
    'safe_wander = wander_bot.safe_wander:main',
],
```

Then build the workspace:

```bash
cd ~/ros2_ws***
colcon build***
source install/setup.bash***
```

---

🧪 Run Simulation + Robot Movement

 🧱 Launch Gazebo

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

 🚦 In another terminal, run the navigation node:

```bash
ros2 run wander_bot safe_wander
```

---

🧾 Record ROS2 Bag Data

To record trajectory and velocity commands:

```bash
ros2 bag record /odom /cmd_vel -o turtlebot3_exploration_data
```

Stop with `Ctrl + C`.

---

 📤 Extract CSV from Bag File

 📂 Navigate to bag folder:

```bash
cd ~/turtlebot3_exploration_data
```

 📜 Create `rosbag2_extractor.py` and run:

```bash
python3 rosbag2_extractor.py
```

This generates:

* odom.csv
* cmd\_vel.csv

---

## ⏭️ Coming Soon: RNN Training & Deployment

Next steps (to be added):

* Normalize CSV data
* Sequence generation for RNN input
* Train LSTM/GRU in PyTorch
* Deploy model back into ROS2 to predict trajectories

---

## 👤 Author

* **Aditya Pratap** – [GitHub](https://github.com/adityapratap112)

---

