import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\book college\semester 5\annpp001\gg\convnext_tiny_best.pth"
IMAGE_PATH = r"D:\book college\semester 5\annpp001\Cattle-Buffalo-breeds.v1i.folder\valid\Murrah\Murrah_17_jpg.rf.68f7cfc8fb842d70e6747c1c1bf32fbd.jpg"
SAVE_DIR = r"C:\Users\shash\Downloads"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------


# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]
num_classes = len(class_names)

# Rebuild model
model = models.convnext_tiny(pretrained=False)
num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# Inference transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probs, 1)

predicted_breed = class_names[predicted_idx.item()]
confidence_percent = confidence.item() * 100

# Save result image
os.makedirs(SAVE_DIR, exist_ok=True)
output_path = os.path.join(
    SAVE_DIR,
    f"prediction_{predicted_breed}_{confidence_percent:.2f}.png"
)

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.title(f"Predicted Breed: {predicted_breed}\nConfidence: {confidence_percent:.2f}%")
plt.savefig(output_path, bbox_inches="tight")
plt.close()

print(f"Prediction saved at: {output_path}")
print(f"Breed: {predicted_breed}, Confidence: {confidence_percent:.2f}%")
