# YOLOv1 Implementation from Scratch with PyTorch
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

A modern, high-performance implementation of the original YOLO (You Only Look Once) real-time object detection algorithm, optimized for NVIDIA GPUs using Automatic Mixed Precision (AMP).

## 🚀 Features
* From-Scratch Implementation: Built based on the original 2016 research paper.
* Modern PyTorch Syntax: Uses the latest torch.amp for high-speed training.
* Hardware Optimized: Specifically tuned for high VRAM GPUs (RTX 4500) and large memory systems (256GB RAM).
* Automatic Mixed Precision (AMP): Faster training with lower memory footprint using FP16.
* Custom Loss Function: Full implementation of the multi-part YOLO sum-squared error loss.
* Automated Pipeline: Automatic PASCAL VOC dataset downloading and formatting.


## 🛠 Hardware Environment
* This implementation was developed and tested on a high-performance research workstation at Kyungpook National University:
* GPU: NVIDIA RTX 4500 (Used with torch.amp for Tensor Core acceleration)
* RAM: 256GB System Memory
* OS: Windows/Linux (Optimized for CUDA)

---

## 📁 Project Structure
.
* ├── model.py           # YOLOv1 Architecture (24 Conv layers + 2 FC layers)
* ├── loss.py            # Multi-part YOLO Loss Function
* ├── train.py           # High-performance training script
* ├── dataset.py         # VOC Dataset loading and SxSx(B*5+C) encoding
* └── yolov1_epoch_10.pth # Saved model checkpoints (ignored by git)

## ⚙️ Hyperparameters
Based on the research paper and hardware optimization:
* | Parameter | Value | Description |
* | :--- | :--- | :--- |
* | Learning Rate | 2e-5 | Stabilized initial rate |
* | Batch Size | 64 | Optimized for RTX 4500 VRAM |
* | Epochs | 135 | Total training iterations |
* | Optimizer | SGD | Momentum 0.9, Weight Decay 0.0005 |
* | Image Size | 448 x 448 | Standard YOLO resolution |

## 📦 Installation & Usage
Clone the repository:
```bash
git clone https://github.com/sankalpalokuliyanage/YOLOv1-PyTorch.git
cd YOLOv1-PyTorch
```

Install dependencies:
```bash
pip install torch torchvision numpy pillow pandas
```

Start Training:
```bash
python train.py
```

## 📊 Training Progress
Training is currently ongoing. Initial results show stable loss convergence:
* Epoch 1: Avg Loss ~700
* Epoch 3: Avg Loss ~649 (Current progress)

## 📝 Acknowledgments
* Redmon et al. for the original YOLO paper.
* Kyungpook National University (KNU) for providing the computational resources.
