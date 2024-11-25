import torch
import torchvision.models as models

class SemanticSegmentationModel:
    def __init__(self, num_classes=2):
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)  # Add batch dimension
            output = self.model(image.squeeze(0))['out']
            return torch.argmax(output, dim=1).squeeze(0)  # Return segmentation mask
