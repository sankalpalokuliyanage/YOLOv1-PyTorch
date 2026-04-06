import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from model import YOLOv1
import numpy as np

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
S, B, C = 7, 2, 20
CHECKPOINT_PATH = "yolov1_epoch_120.pth" # දැනට තියෙන අලුත්ම checkpoint එක දෙන්න
IMAGE_PATH = "test_image.jpg" # පරීක්ෂා කිරීමට අවශ්‍ය පින්තූරයේ නම

# VOC Classes (YOLO paper standard)
CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def predict():
    # 1. Load Model
    model = YOLOv1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Preprocess Image
    img = Image.open(IMAGE_PATH).convert("RGB")
    orig_w, orig_h = img.size
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 3. Forward Pass
    with torch.no_grad():
        prediction = model(img_tensor) # Shape: [1, 7*7*30]
        prediction = prediction.reshape(-1, S, S, C + B * 5)

    # 4. Simple Visualization (Simplified for testing)
    draw = ImageDraw.Draw(img)
    for i in range(S):
        for j in range(S):
            conf = prediction[0, i, j, C].item()
            if conf > 0.1: # Confidence threshold
                # Get class
                class_idx = torch.argmax(prediction[0, i, j, :C]).item()
                
                # Get box coordinates (x, y relative to cell, w, h relative to image)
                box = prediction[0, i, j, C+1:C+5]
                x_cell, y_cell, w, h = box[0], box[1], box[2], box[3]
                
                # Convert to image coordinates
                x = (j + x_cell) / S * orig_w
                y = (i + y_cell) / S * orig_h
                width = w * orig_w
                height = h * orig_h
                
                # Draw box
                left, top = x - width/2, y - height/2
                right, bottom = x + width/2, y + height/2
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                draw.text((left, top), f"{CLASSES[class_idx]} {conf:.2f}", fill="red")

    img.show()
    img.save("result.jpg")
    print("Testing finished! Check result.jpg")
    print(f"Max confidence in prediction: {torch.max(prediction[0, :, :, C]).item():.4f}")

if __name__ == "__main__":
    predict()