import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5 # As per paper
        self.lambda_coord = 5   # As per paper

    def forward(self, predictions, target):
        # Predictions: (batch, S*S*(C+B*5)) -> (batch, S, S, 30)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # 1. Coordinate Loss (x, y, w, h)
        # Using target[..., 20] to check if an object exists in the cell
        exists_box = target[..., self.C].unsqueeze(3) # Iobj_i

        # Box coordinates loss (Equation part 1 & 2)
        # For simplicity, using the first box in each cell
        box_predictions = predictions[..., self.C+1:self.C+5]
        box_targets = target[..., self.C+1:self.C+5]
        
        # Taking square root of width and height as per paper
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(exists_box * box_predictions, end_dim=-2),
            torch.flatten(exists_box * box_targets, end_dim=-2),
        )

        # 2. Object Loss (Equation part 3)
        pred_confidence = predictions[..., self.C:self.C+1]
        target_confidence = target[..., self.C:self.C+1]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_confidence),
            torch.flatten(exists_box * target_confidence),
        )

        # 3. No Object Loss (Equation part 4)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * pred_confidence),
            torch.flatten((1 - exists_box) * target_confidence),
        )

        # 4. Class Loss (Equation part 5)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2),
        )

        # Final loss calculation with lambda weights
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss