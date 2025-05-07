import pandas as pd

# === Step 1: Load the CSVs ===
# odom.csv contains robot position and actual velocity
# cmd_vel.csv contains velocity commands sent to the robot

odom_df = pd.read_csv('odom.csv')
cmd_df = pd.read_csv('cmd_vel.csv')

# === Step 2: Convert timestamps to float ===
# Some ROS logs may be in scientific notation — make sure it's float
odom_df['time'] = odom_df['time'].astype(float)
cmd_df['time'] = cmd_df['time'].astype(float)

# === Step 3: Set 'time' as index for interpolation ===
odom_df.set_index('time', inplace=True)
cmd_df.set_index('time', inplace=True)

# === Step 4: Interpolate cmd_vel to match odom timestamps ===
# This assumes odom timestamps are more consistent
cmd_df_interp = cmd_df.reindex(odom_df.index, method='nearest', tolerance=0.05)
# Drop rows with no matching cmd_vel data
merged_df = odom_df.copy()
merged_df[['cmd_lin_x', 'cmd_ang_z']] = cmd_df_interp

# Drop any rows where interpolation failed (NaNs)
merged_df.dropna(inplace=True)

# === Step 5: Reset index to have 'time' as a column again ===
merged_df.reset_index(inplace=True)

# === Step 6: Save the aligned dataset ===
merged_df.to_csv('aligned_data.csv', index=False)

print("✅ Aligned dataset saved as 'aligned_data.csv'")
print("Columns in output:", list(merged_df.columns))
