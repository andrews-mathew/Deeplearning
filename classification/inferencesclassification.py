import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model class
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the trained model
model = CNNClassifier().to(device)
model.load_state_dict(torch.load("classification/deeper_cnn_cifar10.pth", map_location=device))
model.eval()

# Transform (same as during test time)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Inference function with image display
def predict_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        class_index = predicted.item()
        class_name = classes[class_index]

    # Prepare image for display (undo normalization and convert to numpy)
    image_display = image_tensor.squeeze(0).cpu().numpy()
    image_display = image_display * 0.5 + 0.5  # Undo normalization
    image_display = np.transpose(image_display, (1, 2, 0))  # Change to HWC format
    image_display = np.clip(image_display, 0, 1)  # Ensure values are in valid range

    # Display image with prediction
    plt.imshow(image_display)
    plt.title(f"Predicted class: {class_name} ({class_index})")
    plt.axis('off')
    plt.show()

    print(f"Predicted class: {class_name} ({class_index})")

# Example usage
predict_image("classification/airplane.jpeg")