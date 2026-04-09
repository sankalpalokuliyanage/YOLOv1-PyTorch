import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageTk, ImageFont
from model import YOLOv1
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
S, B, C = 7, 2, 2
CHECKPOINT_PATH = "fine_tuned_yolo_model.pth"
CLASSES = ["Dog", "Cat"]

class YOLOv1PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv1 Dog & Cat Detector")
        self.root.geometry("700x850")
        self.root.configure(bg="#2c3e50")

        # 1. Model Loading
        self.model = YOLOv1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            messagebox.showwarning("Warning", "Weight file not found! Please train first.")

        # 2. UI Setup
        self.header = tk.Label(root, text="Object Detection with YOLOv1", font=("Helvetica", 18, "bold"), bg="#2c3e50", fg="#ecf0f1")
        self.header.pack(pady=15)

        self.btn_select = tk.Button(root, text="Upload Image", command=self.process_image, font=("Helvetica", 12, "bold"), bg="#3498db", fg="white", padx=20, pady=10)
        self.btn_select.pack(pady=10)

        self.image_label = tk.Label(root, bg="#34495e") # Frame for image
        self.image_label.pack(pady=10, padx=20)

        self.result_text = tk.Label(root, text="Prediction: None", font=("Helvetica", 14), bg="#2c3e50", fg="#f1c40f")
        self.result_text.pack(pady=5)

        self.conf_text = tk.Label(root, text="Confidence: 0%", font=("Helvetica", 14), bg="#2c3e50", fg="#f1c40f")
        self.conf_text.pack(pady=5)

    def process_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return

        # Preprocessing
        img = Image.open(file_path).convert("RGB")
        orig_w, orig_h = img.size
        
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            prediction = self.model(img_tensor)
            prediction = prediction.reshape(-1, S, S, C + B * 5)

        draw = ImageDraw.Draw(img)
        best_overall_score = 0
        best_label = "None"
        found = False

        # Post-processing
        for i in range(S):
            for j in range(S):
                conf1 = prediction[0, i, j, C].item()
                conf2 = prediction[0, i, j, C+5].item()
                
                # Select better box
                if conf1 > conf2:
                    score = conf1
                    box = prediction[0, i, j, C+1:C+5]
                else:
                    score = conf2
                    box = prediction[0, i, j, C+6:C+10]

                class_probs = prediction[0, i, j, :C]
                max_prob, class_idx = torch.max(class_probs, dim=0)
                final_score = score * max_prob.item()

                if final_score > 0.25: # Threshold
                    found = True
                    if final_score > best_overall_score:
                        best_overall_score = final_score
                        best_label = CLASSES[class_idx]

                    # Map coordinates
                    x = (j + box[0]) / S * orig_w
                    y = (i + box[1]) / S * orig_h
                    w = box[2] * orig_w
                    h = box[3] * orig_h
                    
                    l, t, r, b = x-w/2, y-h/2, x+w/2, y+h/2
                    draw.rectangle([l, t, r, b], outline="#00FF00", width=6)
                    draw.text((l+5, t+5), f"{CLASSES[class_idx]} {final_score:.2%}", fill="#00FF00")

        # Update GUI
        if found:
            self.result_text.config(text=f"Prediction: {best_label}", fg="#2ecc71")
            self.conf_text.config(text=f"Confidence: {best_overall_score:.2%}", fg="#2ecc71")
        else:
            self.result_text.config(text="Prediction: No Object", fg="#e74c3c")
            self.conf_text.config(text="Confidence: < 25%", fg="#e74c3c")

        # Resize for display
        img.thumbnail((500, 500))
        tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=tk_img)
        self.image_label.image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv1PredictorGUI(root)
    root.mainloop()