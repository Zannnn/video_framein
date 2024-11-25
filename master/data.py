# import os
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torchvision.transforms as transforms

# class VideoFrameDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.sub_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
#         self.transform = transform

#     def __len__(self):
#         return len(self.sub_dirs)

#     def __getitem__(self, idx):
#         folder = self.sub_dirs[idx]
#         frames = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
#         frame1 = Image.open(frames[0])
#         frame3 = Image.open(frames[2])
#         if self.transform:
#             frame1 = self.transform(frame1)
#             frame3 = self.transform(frame3)
#         return frame1, frame3, frames[1]  # Return the ground truth middle frame for evaluation

# def get_dataloader(root_dir, batch_size=8, transform=None):
#     dataset = VideoFrameDataset(root_dir, transform=transform)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VideoInterpolationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_dirs = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        sequence_dir = os.path.join(self.root_dir, self.sequence_dirs[idx])
        frame1_path = os.path.join(sequence_dir, "im1.png")
        frame3_path = os.path.join(sequence_dir, "im3.png")
        gt_middle_path = os.path.join(sequence_dir, "im2.png")

        # 加载图像
        frame1 = Image.open(frame1_path).convert("RGB")
        frame3 = Image.open(frame3_path).convert("RGB")
        gt_middle = Image.open(gt_middle_path).convert("RGB")

        # 转换为张量
        if self.transform:
            frame1 = self.transform(frame1)
            frame3 = self.transform(frame3)
            gt_middle = self.transform(gt_middle)

        return frame1, frame3, gt_middle,gt_middle_path 

def get_dataloader(root_dir, batch_size=8, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),   # 调整图像大小
        transforms.ToTensor()           # 转换为张量
    ])
    dataset = VideoInterpolationDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

