import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models/gesture_resnet18.pth"

classes = ['NEXT', 'NONE', 'PAUSE', 'PLAY', 'PREVIOUS']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Create ResNet-18
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# Load trained model
checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

if not cap.isOpened():
    print("ERROR: Could not open /dev/video0")
    exit()

print("Camera opened.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame")
        break

    # Convert OpenCV BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    image = Image.fromarray(rgb)

    # Prepare for ResNet
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    gesture = classes[prediction.item()]
    confidence = confidence.item() * 100

    text = f"{gesture}: {confidence:.1f}%"

    cv2.putText(
        frame,
        text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Music Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
