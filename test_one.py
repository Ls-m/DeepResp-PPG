import torch
import numpy as np
import matplotlib.pyplot as plt

import helperfunctions
from models.model2 import Correncoder_model  # your model class
from helperfunctions import *


import numpy as np
import torch

y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.1, 2.1, 3.1])
mae, rmse, corr = evaluate_metrics(y_true, y_pred)
print(mae, rmse, corr)




exit()
# Assume testX_norm and testy_norm are already loaded/prepared and normalized
data = np.load('test_data.npz')
testX_norm = data['testX']
testy_norm = data['testy']
device = torch.device("mps")
print(f"Using device: {device}")
# Pick a single segment (first segment from first subject)
x = testX_norm[0, 0, :]  # shape: (segment_length,)
y = testy_norm[0, 0, :]  # shape: (segment_length,)

# Reshape for batch and channel
x = x[np.newaxis, np.newaxis, :]  # (1, 1, segment_length)
y = y[np.newaxis, :]              # (1, segment_length)

# Convert to torch tensors and move to device
xT = torch.from_numpy(x.astype(np.float32)).to(device)
yT = torch.from_numpy(y.astype(np.float32)).to(device)

# Build the model (set dropout to 0 for this test!)
model = Correncoder_model().to(device)
for m in model.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

num_epochs = 500  # Overfit deliberately
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(xT)
    loss = criterion(output.squeeze(1), yT)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if (epoch+1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# Plot the training loss curve
plt.figure(figsize=(8,4))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting Loss on Single Sample")
plt.show()

# Plot prediction vs ground truth after overfitting
model.eval()
with torch.no_grad():
    y_pred = model(xT).cpu().numpy().squeeze()

plt.figure(figsize=(10, 4))
plt.plot(y[0], label='Ground Truth', color='red', linestyle='--')
plt.plot(y_pred, label='Model Prediction', color='blue')
plt.legend()
plt.title("Model Overfitting a Single Sample")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Amplitude")
plt.show()
