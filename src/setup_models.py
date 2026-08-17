import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DRIVE_DIR = '/content/drive/MyDrive/Bachelor/new coding/CMAPSSData/'
DATA_DIR = '/content/drive/MyDrive/Bachelor/new coding/NEW_DATA/'
MODEL_DIR = '/content/drive/MyDrive/Bachelor/new coding/NEW_MODELS/'

class CNN(nn.Module):
  def __init__(self, input_channels=1, sequence_length=30, feature_length=14):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=(10, 1))
    self.conv2 = nn.Conv2d(10, 10, kernel_size=(10, 1))
    self.conv3 = nn.Conv2d(10, 10, kernel_size=(10, 1))
    self.conv4 = nn.Conv2d(10, 10, kernel_size=(10, 1))
    self.conv5 = nn.Conv2d(10, 1, kernel_size=(3, 1))
    self.dropout = nn.Dropout(p=0.5)
    self.fc1 = nn.Linear((sequence_length - 2) * feature_length, 100)
    self.fc2 = nn.Linear(100, 1)
  def forward(self, x):
    x = F.pad(x, (0, 0, 4, 5)) # basically: (zero) padding='same'
    x = torch.tanh(self.conv1(x))
    x = F.pad(x, (0, 0, 4, 5))
    x = torch.tanh(self.conv2(x))
    x = F.pad(x, (0, 0, 4, 5))
    x = torch.tanh(self.conv3(x))
    x = F.pad(x, (0, 0, 4, 5))
    x = torch.tanh(self.conv4(x))
    x = torch.tanh(self.conv5(x))
    x = torch.flatten(x, start_dim=1)
    x = torch.tanh(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x

class TimeWindowDataset(Dataset):
  def __init__(self, data, is_train=True):
    self.data = torch.tensor(data, dtype=torch.float32)
    self.is_train = is_train
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    sample = self.data[idx]
    features = sample[:, :-1].unsqueeze(0)
    labels = sample[-1, -1].unsqueeze(0)
    return features, labels

def create_dataloaders(train_data, test_data, batch_size=512):
  train_loaders = [DataLoader(TimeWindowDataset(data, is_train=True), batch_size=batch_size, shuffle=True) for data in train_data]
  test_loaders = [DataLoader(TimeWindowDataset(data, is_train=False), batch_size=batch_size, shuffle=False) for data in test_data]
  return train_loaders, test_loaders

def setup_data():
  train_path = os.path.join(DATA_DIR, "train_timed.pt")
  test_path = os.path.join(DATA_DIR, "test_timed.pt")

  if os.path.exists(train_path) and os.path.exists(test_path):
    train_timed = torch.load(train_path, weights_only=False)
    test_timed = torch.load(test_path, weights_only=False)

  train_loaders, test_loaders = create_dataloaders(train_timed, test_timed)
  return train_loaders, test_loaders

train_loaders, test_loaders = setup_data()

# Test shapes
index, sample = next(enumerate(train_loaders[0])) # iter if no index needed
features, labels = sample
print(features.shape)
print(labels.shape)
print(labels[7])

index, sample = next(enumerate(train_loaders[3]))
features, labels = sample
print(features.shape)
print(labels.shape)
print(labels[7])

index, sample = next(enumerate(train_loaders[3]))
features, labels = sample
print(features.shape)
print(labels.shape)
print(labels[7])

features = features[0][0]
labels = labels[0]

plt.figure(figsize=(10, 6))
plt.title('Sensor Data for Sample')
plt.xlabel('Time')
plt.ylabel('Feature Value')
plt.grid(True)

num_features = features.shape[1]
for i in range(num_features):
  plt.plot(range(features.shape[0]), features[:, i], label=f'Feature {i+1}')

plt.ylim(-1, 1)  # Adjust y-axis range
plt.legend()

def xavier_init(model):
  for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
      nn.init.xavier_normal_(module.weight)

def rmse_loss(predictions, targets):
  criterion = nn.MSELoss()
  return torch.sqrt(criterion(predictions, targets))

def scoring_function(rul_estimated, rul_actual):
  if not isinstance(rul_estimated, torch.Tensor):
    rul_estimated = torch.tensor(rul_estimated)
  if not isinstance(rul_actual, torch.Tensor):
    rul_actual = torch.tensor(rul_actual)

  rul_estimated = rul_estimated.clone().detach()
  rul_actual = rul_actual.clone().detach()
  di = rul_estimated - rul_actual

  si = torch.where(
    di < 0,
    torch.exp(-di / 13) - 1,
    torch.exp(di / 10) - 1
  )

  score = torch.sum(si)
  return score.item()

def lr_lambda(epoch):
  if epoch < 200:
    return 1.0
  else:
    return 0.1

def train_model(train_dl, test_dl, sequence_length, file_name, learning_rate=0.001, num_epochs=250):
  model = CNN(input_channels=1, sequence_length=sequence_length, feature_length=14)
  model.to(DEVICE)
  xavier_init(model)
  optimizer = optim.Adam(model.named_parameters(), lr=learning_rate)
  scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
  rmse_values = []
  scoring_values = []
  best_val_loss = float('inf')
  best_epoch = None
  best_state = None

  print(f"Training model for {num_epochs} epochs...")
  for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_inputs, batch_targets in train_dl:
      batch_inputs, batch_targets = batch_inputs.to(DEVICE), batch_targets.to(DEVICE)
      optimizer.zero_grad()
      outputs = model(batch_inputs)

      loss = rmse_loss(outputs, batch_targets)
      loss.backward()
      optimizer.step()

      # de-normalize
      outputs = (outputs + 1) / 2 * 125
      batch_targets = (batch_targets + 1) / 2 * 125

      loss = rmse_loss(outputs, batch_targets)
      train_loss += loss.item() * batch_inputs.size(0)

    train_loss /= len(train_dl.dataset)

    model.eval()
    val_loss = 0.0
    score_loss = 0.0
    with torch.no_grad():
      for val_batch_inputs, val_batch_targets in test_dl:
        val_batch_inputs, val_batch_targets = val_batch_inputs.to(DEVICE), val_batch_targets.to(DEVICE)
        val_outputs = model(val_batch_inputs)

        # de-norm
        val_outputs = (val_outputs + 1) / 2 * 125
        val_batch_targets = (val_batch_targets + 1) / 2 * 125

        val_loss += rmse_loss(val_outputs, val_batch_targets).item() * val_batch_inputs.size(0)
        score_loss += scoring_function(val_outputs, val_batch_targets) * val_batch_inputs.size(0)

    val_loss /= len(test_dl.dataset)
    score_loss /= len(test_dl.dataset)
    rmse_values.append(val_loss)
    scoring_values.append(score_loss)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_epoch = epoch + 1
      best_state = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 10 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Scoring: {score_loss:.4f}')

    scheduler.step()

  mean_rmse = sum(rmse_values) / len(rmse_values)
  mean_scoring = sum(scoring_values) / len(scoring_values)
  print(f'\nMean RMSE after {num_epochs} epochs: {mean_rmse:.4f}, Mean Scoring: {mean_scoring:.4f}')

  if best_state is not None:
    print(f'Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}, restoring that checkpoint')
    model.load_state_dict(best_state)

  model_path = os.path.join(MODEL_DIR, f"{file_name}.pth")
  torch.save(model.state_dict(), model_path)
  print(f"Model saved to {model_path}")

  return model

sequences = [30, 20, 30, 15]
names = [f"model{i}" for i in range(len(train_loaders))]

for train_loader, test_loader, sequence_length, file_name in zip(train_loaders, test_loaders, sequences, names):
  if file_name == "model1":
    continue
  if file_name == "model2":
    continue
  train_model(train_loader, test_loader, sequence_length, file_name)