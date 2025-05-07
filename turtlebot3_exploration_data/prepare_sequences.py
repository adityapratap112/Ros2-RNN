import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# === CONFIGURATION ===
sequence_length = 20  # Number of past timesteps for RNN input
prediction_target = ['x', 'y']  # What to predict
input_features = ['x', 'y', 'lin_vel', 'ang_vel', 'cmd_lin_x', 'cmd_ang_z']

# === Step 1: Load aligned dataset ===
df = pd.read_csv('aligned_data.csv')

# === Step 2: Normalize all features ===
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[input_features + prediction_target])
normalized_df = pd.DataFrame(normalized_data, columns=input_features + prediction_target)

# === Step 3: Create sequences ===
X = []
Y = []

for i in range(len(normalized_df) - sequence_length):
    input_seq = normalized_df[input_features].iloc[i:i+sequence_length].values
    target = normalized_df[prediction_target].iloc[i+sequence_length].values
    X.append(input_seq)
    Y.append(target)

X = np.array(X)  # shape: (samples, timesteps, features)
Y = np.array(Y)  # shape: (samples, prediction_targets)

print(f"✅ Generated {X.shape[0]} sequences.")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# === Step 4: Save as .npy for PyTorch training ===
np.save('X_rnn.npy', X)
np.save('Y_rnn.npy', Y)
print("✅ Saved X_rnn.npy and Y_rnn.npy")
