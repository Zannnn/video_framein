# import torch
# from data import get_dataloader
# from segmentation import SemanticSegmentationModel
# from background_model import BackgroundModel
# from foreground_model import ForegroundModel
# from fusion import fuse_images

# def infer(model_paths, root_dir):
#     seg_model = SemanticSegmentationModel()
#     bg_model = torch.load(model_paths['background'])
#     fg_model = torch.load(model_paths['foreground'])
#     dataloader = get_dataloader(root_dir, batch_size=1)

#     for frame1, frame3, _ in dataloader:
#         mask1 = seg_model.predict(frame1)
#         mask3 = seg_model.predict(frame3)
#         mask_middle = (mask1 + mask3) / 2

#         pred_bg = bg_model(frame1, frame3)
#         pred_fg = fg_model(frame1, frame3)
#         pred_middle = fuse_images(pred_fg, pred_bg, mask_middle)

#         yield pred_middle


import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from segmentation import SemanticSegmentationModel
from background_model import BackgroundModel
from foreground_model import ForegroundModel
from fusion import fuse_images
from data import VideoFrameDataset

# 推理函数
def infer(model_paths, root_dir, output_dir):
    # 加载模型
    seg_model = SemanticSegmentationModel()
    bg_model = torch.load(model_paths['background'])
    fg_model = torch.load(model_paths['foreground'])
    
    # 转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))  # 假设模型输入尺寸
    ])
    dataset = VideoFrameDataset(root_dir, transform=transform)

    os.makedirs(output_dir, exist_ok=True)
    bg_model.eval()
    fg_model.eval()

    with torch.no_grad():
        for i, (frame1, frame3, gt_path) in enumerate(dataset):
            mask1 = seg_model.predict(frame1)
            mask3 = seg_model.predict(frame3)
            mask_middle = (mask1 + mask3) / 2  # 平均化前景背景概率

            # 背景与前景插值
            pred_bg = bg_model(frame1.unsqueeze(0), frame3.unsqueeze(0)).squeeze(0)
            pred_fg = fg_model(frame1.unsqueeze(0), frame3.unsqueeze(0)).squeeze(0)

            # 融合结果
            pred_middle = fuse_images(pred_fg, pred_bg, mask_middle)

            # 保存结果
            pred_middle_image = transforms.ToPILImage()(pred_middle)
            output_path = os.path.join(output_dir, f"output_{i + 1}.jpg")
            pred_middle_image.save(output_path)

            print(f"Saved interpolated frame to {output_path}")

if __name__ == "__main__":
    model_paths = {
        "background": "/pub/data/lz/video_frame_in/master/checkpoints/background_model.pth",
        "foreground": "/pub/data/lz/video_frame_in/master/checkpoints/foreground_model.pth"
    }
    root_dir = "/pub/data/lijm/data/vimeo_triplet/vimeo_triplet/sequences/00078/0651/"
    output_dir = "/pub/data/lz/video_frame_in/test_image"
    infer(model_paths, root_dir, output_dir)
