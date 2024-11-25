import torch
import torch.nn as nn

class BackgroundModel(nn.Module):
    def __init__(self):
        super(BackgroundModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, frame1, frame3):
        x = torch.cat([frame1, frame3], dim=1)  # Concatenate along channel dimension
        encoded = self.encoder(x)
        return self.decoder(encoded)

