import os

import numpy as np
from torchvision import datasets, transforms

# Save directory for the dataset
save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)

# Download the dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(
    root ="./data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Convert to Numpy arrays and Normalize
X_train = train_data.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = train_data.targets.numpy().astype(np.int32)
X_test = test_data.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test = test_data.targets.numpy().astype(np.int32)

# Save as binary files
X_train.tofile(os.path.join(save_dir, "X_train.bin"))
y_train.tofile(os.path.join(save_dir, "y_train.bin"))
X_test.tofile(os.path.join(save_dir, "X_test.bin"))
y_test.tofile(os.path.join(save_dir, "y_test.bin"))

# Save metadata
with open(os.path.join(save_dir, "metadata.txt"), "w") as f:
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Test samples: {X_test.shape[0]}\n")
    f.write(f"Input dimensions: {X_train.shape[1]}\n")
    f.write(f"Number of classes: {len(np.unique(y_train))}\n")

print("MNIST dataset has been downloaded and saved in binary format.")