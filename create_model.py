import pandas as pd
import joblib
import numpy as np
import torch
import random
import albumentations
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import time
import cv2

from modules.asl_neural_network import AslNeuralNet
from modules.image_preprocessing import ImagePreprocessor, Mapper
from modules.relative_paths import Path

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


ImagePreprocessor().preprocess_images(2000)
Mapper().one_hot_encoding()


def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

SEED=42
seed_everything(SEED=SEED)

# set computation device
device = 'cpu'
# device = ('cƒuda:0' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(Path().DATAFRAME)
X = df.image_path.values
y = df.target.values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"Training on {len(xtrain)} images")
print(f"Validation on {len(xtest)} images")


class ImageDataset(Dataset):
    def __init__(self, path, labels):
        self.X = path
        self.y = labels
        self.aug = albumentations.Compose([albumentations.Resize(224, 224, always_apply=True), ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        image = cv2.imread(self.X[i])
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)


train_data = ImageDataset(xtrain, ytrain)
test_data = ImageDataset(xtest, ytest)

# dataloaders
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=False)

model = AslNeuralNet().to(device)
print(model)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss function
criterion = nn.CrossEntropyLoss()


def fit(model, dataloader):
    print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(dataloader.dataset)
    train_accuracy = 100. * running_correct / len(dataloader.dataset)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

    return train_loss, train_accuracy


# validation function
def validate(model, dataloader):
    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data) / dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == target).sum().item()

        val_loss = running_loss / len(dataloader.dataset)
        val_accuracy = 100. * running_correct / len(dataloader.dataset)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')

        return val_loss, val_accuracy


num_steps = 5
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for step in range(num_steps):
    print(f"step {step+1} of {num_steps}")
    train_step_loss, train_step_accuracy = fit(model, trainloader)
    val_step_loss, val_step_accuracy = validate(model, testloader)
    train_loss.append(train_step_loss)
    train_accuracy.append(train_step_accuracy)
    val_loss.append(val_step_loss)
    val_accuracy.append(val_step_accuracy)
end = time.time()

plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validation accuracy')
plt.xlabel('steps')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('resources/accuracy.png')
plt.show()

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig('resources/loss.png')
plt.show()

print('Saving model...')
torch.save(model.state_dict(), Path().MODEL)

print(f'train_accuracy: {train_accuracy}')
print(f'train_loss: {train_loss}')
print(f'val_accuracy: {val_accuracy}')
print(f'val_accuracy: {val_loss}')


