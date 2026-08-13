import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import socket
import time

MODEL_PATH = "models/gesture_resnet18.pth"

classes = ['NEXT', 'NONE', 'PAUSE', 'PLAY', 'PREVIOUS']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------------
# Load ResNet-18
# -----------------------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# VLC control
# -----------------------------

VLC_HOST = "127.0.0.1"
VLC_PORT = 4212

def vlc_command(command):
    try:
        s = socket.socket()
        s.settimeout(1)
        s.connect((VLC_HOST, VLC_PORT))
        s.sendall((command + "\n").encode())
        s.close()
        print("VLC:", command)
    except Exception as e:
        print("VLC error:", e)

# -----------------------------
# Gesture settings
# -----------------------------

CONFIDENCE_THRESHOLD = 80.0

# Don't repeatedly trigger the same gesture
COOLDOWN = 1.5
last_command = None
last_command_time = 0

# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

if not cap.isOpened():
    print("ERROR: Could not open /dev/video0")
    exit()

print("Camera opened.")
print("Music Gesture Control started.")
print("Q = quit")

# -----------------------------
# Main loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame")
        break

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    image = Image.fromarray(rgb)

    # Prepare image
    tensor = transform(image).unsqueeze(0).to(device)

    # Neural network prediction
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    gesture = classes[prediction.item()]
    confidence = confidence.item() * 100

    # -----------------------------
    # Gesture -> VLC command
    # -----------------------------

    current_time = time.time()

    if confidence >= CONFIDENCE_THRESHOLD:

        command = None

        if gesture == "PLAY":
            command = "play"

        elif gesture == "PAUSE":
            command = "pause"

        elif gesture == "NEXT":
            command = "next"

        elif gesture == "PREVIOUS":
            command = "prev"

        # NONE does nothing

        if command is not None:
            if (
                command != last_command
                or current_time - last_command_time >= COOLDOWN
            ):
                vlc_command(command)
                last_command = command
                last_command_time = current_time

    # -----------------------------
    # Display
    # -----------------------------

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
