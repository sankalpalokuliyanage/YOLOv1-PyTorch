import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from model import YOLOv1
from loss import YoloLoss

# Hyperparameters
LEARNING_RATE = 1e-6 # Fine-tuning learning rate as requested
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 
WEIGHT_DECAY = 0.0005 
EPOCHS = 135 
S = 7 
B = 2 
C = 20 
NUM_WORKERS = 20 
PIN_MEMORY = True 

# Image transformations
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

def voc_to_yolo_collate(batch):
    images = []
    targets = []
    for img, anno in batch:
        images.append(img)
        label_matrix = torch.zeros((S, S, C + 5 * B))
        objs = anno['annotation']['object']
        if not isinstance(objs, list): objs = [objs]
        for obj in objs:
            class_label = 0 
            bbox = obj['bndbox']
            xmin, ymin, xmax, ymax = float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])
            orig_w, orig_h = float(anno['annotation']['size']['width']), float(anno['annotation']['size']['height'])
            x, y = (xmin + xmax) / (2 * orig_w), (ymin + ymax) / (2 * orig_h)
            w, h = (xmax - xmin) / orig_w, (ymax - ymin) / orig_h
            i, j = int(S * y), int(S * x)
            x_cell, y_cell = S * x - j, S * y - i
            if label_matrix[i, j, C] == 0:
                label_matrix[i, j, C] = 1 
                label_matrix[i, j, C+1:C+5] = torch.tensor([x_cell, y_cell, w, h])
                label_matrix[i, j, class_label] = 1 
        targets.append(label_matrix)
    return torch.stack(images), torch.stack(targets)

def train_fn(train_loader, model, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Using modern torch.amp.autocast
        with torch.amp.autocast('cuda'):
            predictions = model(x)
            loss = loss_fn(predictions, y)

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward() 
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch finished. Average Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    model = YOLOv1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()
    
    # Using modern torch.amp.GradScaler
    scaler = torch.amp.GradScaler('cuda') 
    
    best_loss = float('inf') # Initialize best loss as infinity

    # Check if dataset exists
    dataset_path = "./data/VOCdevkit/VOC2007"
    if os.path.exists(dataset_path):
        print("Dataset found in the directory. Skipping download...")
        download_required = False
    else:
        print("Dataset not found. Starting download...")
        download_required = True

    train_dataset = torchvision.datasets.VOCDetection(
        root="./data", year="2007", image_set="trainval", 
        download=download_required, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, collate_fn=voc_to_yolo_collate,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY 
    )

    print(f"Starting training on {DEVICE} with High Performance settings...")

    for epoch in range(EPOCHS):
        print(f"\n--- Starting Epoch {epoch+1}/{EPOCHS} ---")
        current_avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # SAVE BEST MODEL: If current epoch loss is the lowest so far
        if current_avg_loss < best_loss:
            best_loss = current_avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, "yolo_best_model.pth")
            print(f"==> New Best Model Saved with Loss: {best_loss:.4f}")

        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_name = f"yolov1_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_avg_loss,
            }, checkpoint_name)
            print(f"==> Checkpoint saved as {checkpoint_name}")

if __name__ == "__main__":
    main()