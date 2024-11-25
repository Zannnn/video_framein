import torch
import torch.nn as nn
from transformers import ViTModel

class ForegroundModel(nn.Module):
    def __init__(self):
        super(ForegroundModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(768, 3)  # Convert Transformer output to RGB values

    def forward(self, frame1, frame3):
        batch_size, _, h, w = frame1.size()
        frame1_flat = frame1.view(batch_size, 3, -1).permute(0, 2, 1)  # Flatten to patches
        frame3_flat = frame3.view(batch_size, 3, -1).permute(0, 2, 1)
        vit_input = torch.cat([frame1_flat, frame3_flat], dim=1)
        vit_output = self.vit(inputs_embeds=vit_input)
        middle_frame_flat = self.fc(vit_output.last_hidden_state)
        return middle_frame_flat.permute(0, 2, 1).view(batch_size, 3, h, w)
