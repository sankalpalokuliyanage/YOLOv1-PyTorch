import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from model import YOLOv1
from loss import YoloLoss
from huggingface_hub import snapshot_download 

# 1. Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 
WEIGHT_DECAY = 0.0001 
EPOCHS = 50 
S = 7 
B = 2 
C = 2 
NUM_WORKERS = 20 
PIN_MEMORY = True 

LEARNING_RATE = 5e-4 
CLASSES = ["dog", "cat"]

def download_dataset():
    repo_id = "sankalpa1998/cat-dog-yolo-dataset"
    local_dir = "data"
    
    if not os.path.exists(os.path.join(local_dir, "labels")):
        print(f"Downloading dataset from Hugging Face: {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print("Download complete.")
    else:
        print("Dataset already exists locally. Skipping download.")

# 2. Custom YOLO Dataset Class
class CustomYOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, S=7, B=2, C=2, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_files[index])
        image = Image.open(img_path).convert("RGB")
        
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[index])[0] + ".txt")
        boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    class_label, x, y, w, h = [float(val) for val in line.split()]
                    boxes.append([class_label, x, y, w, h])

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, w, h = box
            class_label = int(class_label)
            
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1 
                label_matrix[i, j, self.C+1:self.C+5] = torch.tensor([x_cell, y_cell, w, h])
                label_matrix[i, j, class_label] = 1 

        if self.transform:
            image = self.transform(image)

        return image, label_matrix

# Image transformations
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(),
])

def train_fn(train_loader, model, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(x)
            loss = loss_fn(predictions, y)

        if torch.isnan(loss):
            print("Warning: NaN loss detected! Skipping batch.")
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward() 
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Average Loss for Epoch: {avg_loss:.4f}")
    return avg_loss

def main():
    download_dataset()

    print(f"Initializing model on {DEVICE}...")
    model = YOLOv1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss(S=S, B=B, C=C)
    scaler = torch.amp.GradScaler('cuda') 

   
    IMG_DIR = "data/images" 
    LABEL_DIR = "data/labels"

    train_dataset = CustomYOLODataset(
        img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transform, C=C
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY 
    )

    print(f"Starting Training for {EPOCHS} epochs with LR: {LEARNING_RATE}")

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        if (epoch + 1) % 10 == 0:
            checkpoint_name = "final_yolo_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, checkpoint_name)
            print(f"==> Checkpoint saved: {checkpoint_name}")

    print("\nInitial training complete. Saving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
    }, "final_yolo_model.pth")
    print("==> Final Model Saved. You can now start fine-tuning with a lower LR.")

if __name__ == "__main__":
    main()