import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Step 1: Load and Save Dataset ===
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
csv_path = "diabetes.csv"
df.to_csv(csv_path, index=False)
print(f"Dataset saved as {csv_path}")

# === Step 2: Preprocessing ===
data = pd.read_csv(csv_path)
X = data.drop(columns='target').values
y = data['target'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === Step 3: Dataset Class ===
class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(DiabetesDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(DiabetesDataset(X_test, y_test), batch_size=32)

# === Step 4: Define Model ===
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = RegressionModel().to(device)

# === Step 5: Train Model ===
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

epochs = 1500
best_loss = float('inf')
patience = 50
counter = 0
best_model_path = "best_diabetes_model.pth"

for epoch in range(epochs):
    # Training
    model.train()
    total_train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for X_batch, y_batch in loop:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(test_loader)
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# === Step 6: Load Best Model for Evaluation and Inference ===
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print(f"Best model loaded from {best_model_path}")

# Evaluate on test set
test_loss = 0.0
all_preds = []
all_actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        test_loss += criterion(output, y_batch).item()
        preds_scaled = output.cpu().numpy()
        preds = scaler_y.inverse_transform(preds_scaled).flatten()
        actuals = scaler_y.inverse_transform(y_batch.cpu().numpy()).flatten()
        all_preds.extend(preds)
        all_actuals.extend(actuals)

avg_test_loss = test_loss / len(test_loader)
r2 = r2_score(all_actuals, all_preds)
print(f"Test Loss: {avg_test_loss:.4f}, R² Score: {r2:.3f}")

# === Step 7: Inference on 1 Sample ===
sample = X[0]  # original (not scaled)
sample_scaled = scaler_X.transform([sample])
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    pred_scaled = model(sample_tensor).cpu().numpy()[0][0]
    prediction = scaler_y.inverse_transform([[pred_scaled]])[0][0]
print(f"\n🎯 Predicted progression for sample 0: {prediction:.2f}")

# === Step 8: Inference on 10 Samples ===
samples = torch.tensor(scaler_X.transform(X[:10]), dtype=torch.float32).to(device)
with torch.no_grad():
    preds_scaled = model(samples).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled)
print("\n📊 Predictions for first 10 samples:")
print(np.round(preds.flatten(), 2))

# === Step 9: Improved Visualization ===
plt.figure(figsize=(8, 8))
plt.scatter(all_actuals, all_preds, alpha=0.6, color='blue', edgecolors='w', s=100)
plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], 'r--', lw=2)
plt.xlabel("Actual Disease Progression", fontsize=12)
plt.ylabel("Predicted Disease Progression", fontsize=12)
plt.title(f"Predicted vs. Actual Disease Progression (Test Set, R² = {r2:.3f})", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()