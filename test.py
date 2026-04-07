import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from model import YOLOv1
import os

# Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
S, B, C = 7, 2, 20
CHECKPOINT_PATH = "yolov1_epoch_30.pth"
IMAGE_PATH = "test_image.jpg"

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def predict():
    # 1. Load Model
    model = YOLOv1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found!")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Preprocess Image
    img = Image.open(IMAGE_PATH).convert("RGB")
    orig_w, orig_h = img.size
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 3. Forward Pass
    with torch.no_grad():
        prediction = model(img_tensor)
        prediction = prediction.reshape(-1, S, S, C + B * 5)
        
        # Debugging print
        max_conf = torch.max(prediction[0, :, :, C]).item()
        print(f"DEBUG: Max Confidence found in grid: {max_conf:.4f}")

        flat_conf = prediction[0, :, :, C]
        max_idx = torch.argmax(flat_conf)
        row, col = max_idx // S, max_idx % S
        best_class = torch.argmax(prediction[0, row, col, :C]).item()

        print(f"DEBUG: Best Cell at [{row},{col}]")
        print(f"DEBUG: Confidence: {prediction[0, row, col, C].item():.4f}")
        print(f"DEBUG: Predicted Class: {CLASSES[best_class]} (Index: {best_class})")

    draw = ImageDraw.Draw(img)
    boxes_found = 0

    # 4. Process Grid
    for i in range(S): # row
        for j in range(S): # col
            # YOLO predicts 2 boxes per cell. Get confidence for both.
            conf1 = prediction[0, i, j, C].item()
            conf2 = prediction[0, i, j, C+5].item()
            
            # Select the box with higher confidence
            if conf1 > conf2:
                best_conf = conf1
                box_data = prediction[0, i, j, C+1:C+5]
            else:
                best_conf = conf2
                box_data = prediction[0, i, j, C+6:C+10]

            # Threshold check (Best confidence * max class probability)
            class_probs = prediction[0, i, j, :C]
            max_prob, class_idx = torch.max(class_probs, dim=0)
            
            # Final Score as per YOLO paper
            final_score = best_conf * max_prob.item()

            if final_score > 0.10: # adjusted threshold
                boxes_found += 1
                x_cell, y_cell, w, h = box_data[0], box_data[1], box_data[2], box_data[3]
                
                # Cell to Image coordinates
                x = (j + x_cell) / S * orig_w
                y = (i + y_cell) / S * orig_h
                width = w * orig_w
                height = h * orig_h
                
                left, top = x - width/2, y - height/2
                right, bottom = x + width/2, y + height/2
                
                # Drawing
                draw.rectangle([left, top, right, bottom], outline="lime", width=4)
                label = f"{CLASSES[class_idx]} {final_score:.2f}"
                draw.text((left, top - 15), label, fill="lime")
                print(f"Detected: {label} at grid [{i},{j}]")

    if boxes_found == 0:
        print("No objects detected. Try lowering the threshold or check if model is trained enough.")
    else:
        img.save("result.jpg")
        print(f"Testing finished! {boxes_found} boxes found. Check result.jpg")
        img.show()

if __name__ == "__main__":
    predict()