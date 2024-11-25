import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from segmentation import SemanticSegmentationModel
from background_model import BackgroundModel
from foreground_model import ForegroundModel
from fusion import fuse_images
from data import VideoInterpolationDataset

# 推理函数
def infer(model_paths, root_dir, output_dir):
    # 加载分割模型
    seg_model = SemanticSegmentationModel()

    # 创建模型实例
    bg_model = BackgroundModel()
    fg_model = ForegroundModel()

    # 加载权重
    bg_model.load_state_dict(torch.load(model_paths['background']))
    fg_model.load_state_dict(torch.load(model_paths['foreground']))

    # 将模型设置为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bg_model = bg_model.to(device).eval()
    fg_model = fg_model.to(device).eval()

    # 定义图像转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))  # 假设模型输入尺寸
    ])
    dataset = VideoInterpolationDataset(root_dir, transform=transform)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 推理过程
    with torch.no_grad():
        for i, (frame1, frame3, gt, gt_path) in enumerate(dataset):
            if i == 27:
                # 数据转移到设备
                frame1 = frame1.to(device)
                frame3 = frame3.to(device)

                # 语义分割生成掩码
                mask1 = seg_model.predict(frame1.unsqueeze(0))
                mask3 = seg_model.predict(frame3.unsqueeze(0))
                mask_middle = (mask1 + mask3) / 2  # 平均化前景背景概率

                # 背景与前景插值
                pred_bg = bg_model(frame1.unsqueeze(0), frame3.unsqueeze(0)).squeeze(0)
                pred_fg = fg_model(frame1.unsqueeze(0), frame3.unsqueeze(0)).squeeze(0)

                # 融合结果
                pred_middle = fuse_images(pred_fg, pred_bg, mask_middle)

                # 保存结果
                pred_middle_image = transforms.ToPILImage()(pred_middle.cpu())
                folder_name = os.path.basename(os.path.dirname(gt_path))
                output_path = os.path.join(output_dir, "output_"+folder_name+".jpg")
                pred_middle_image.save(output_path)

                print(f"Saved interpolated frame to {output_path}")

if __name__ == "__main__":
    model_paths = {
        "background": "/pub/data/lz/video_frame_in/master/checkpoints/background_model_epoch5.pth",
        "foreground": "/pub/data/lz/video_frame_in/master/checkpoints/foreground_model_epoch5.pth"
    }
    root_dir = "/pub/data/lijm/data/vimeo_triplet/vimeo_triplet/sequences/00078/"
    output_dir = "/pub/data/lz/video_frame_in/test_image"
    infer(model_paths, root_dir, output_dir)
