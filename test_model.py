import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATASET = "/workspace/Music-gesture-Control/dataset"
MODEL = "/workspace/Music-gesture-Control/models/gesture_resnet18.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATASET, "test"),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

classes = test_dataset.classes

print("Classes:", classes)
print("Test images:", len(test_dataset))

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

checkpoint = torch.load(MODEL, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

correct = 0
total = 0

class_correct = [0] * len(classes)
class_total = [0] * len(classes)

confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        for label, prediction in zip(labels, predictions):

            label = label.item()
            prediction = prediction.item()

            class_total[label] += 1

            if label == prediction:
                class_correct[label] += 1

            confusion[label][prediction] += 1

accuracy = 100 * correct / total

print("\n==============================")
print(f"OVERALL TEST ACCURACY: {accuracy:.2f}%")
print("==============================")

print("\nPER-CLASS ACCURACY:")

for i, class_name in enumerate(classes):

    class_accuracy = (
        100 * class_correct[i] / class_total[i]
        if class_total[i] > 0 else 0
    )

    print(
        f"{class_name:10s}: "
        f"{class_accuracy:.2f}% "
        f"({class_correct[i]}/{class_total[i]})"
    )

print("\nCONFUSION MATRIX")
print("Rows = actual")
print("Columns = predicted")

print("          " + " ".join(f"{c:>10s}" for c in classes))

for i, class_name in enumerate(classes):

    row = " ".join(
        f"{confusion[i][j].item():10d}"
        for j in range(len(classes))
    )

    print(f"{class_name:10s} {row}")
