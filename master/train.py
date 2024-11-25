# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from data import get_dataloader
# from segmentation import SemanticSegmentationModel
# from background_model import BackgroundModel
# from foreground_model import ForegroundModel
# from fusion import fuse_images

# def train(root_dir, epochs=10, batch_size=8, lr=1e-3):
#     dataloader = get_dataloader(root_dir, batch_size=batch_size)
#     seg_model = SemanticSegmentationModel()
#     bg_model = BackgroundModel()
#     fg_model = ForegroundModel()
#     criterion = nn.MSELoss()
#     optimizer_bg = Adam(bg_model.parameters(), lr=lr)
#     optimizer_fg = Adam(fg_model.parameters(), lr=lr)

#     for epoch in range(epochs):
#         for frame1, frame3, gt_middle in dataloader:
#             # Step 1: Semantic Segmentation
#             mask1 = seg_model.predict(frame1)
#             mask3 = seg_model.predict(frame3)
#             mask_middle = (mask1 + mask3) / 2

#             # Step 2: Background Prediction
#             pred_bg = bg_model(frame1, frame3)

#             # Step 3: Foreground Prediction
#             pred_fg = fg_model(frame1, frame3)

#             # Step 4: Fuse Results
#             pred_middle = fuse_images(pred_fg, pred_bg, mask_middle)

#             # Step 5: Compute Loss
#             loss = criterion(pred_middle, gt_middle)

#             # Step 6: Backpropagation
#             optimizer_bg.zero_grad()
#             optimizer_fg.zero_grad()
#             loss.backward()
#             optimizer_bg.step()
#             optimizer_fg.step()

#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from data import get_dataloader
from segmentation import SemanticSegmentationModel
from background_model import BackgroundModel
from foreground_model import ForegroundModel
from fusion import fuse_images
from tqdm import tqdm

def train(root_dir, epochs=10, batch_size=8, lr=1e-3, save_dir="checkpoints"):
    # 数据加载器
    dataloader = get_dataloader(root_dir, batch_size=batch_size)

    # 初始化模型
    seg_model = SemanticSegmentationModel()
    bg_model = BackgroundModel()
    fg_model = ForegroundModel()

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer_bg = Adam(bg_model.parameters(), lr=lr)
    optimizer_fg = Adam(fg_model.parameters(), lr=lr)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 训练循环
    for epoch in range(epochs):
        epoch_loss = 0.0
        # 使用 tqdm 包装数据加载器
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch")
        
        for frame1, frame3, gt_middle in progress_bar:
            # 1. 前处理：语义分割
            mask1 = seg_model.predict(frame1)
            mask3 = seg_model.predict(frame3)
            mask_middle = (mask1 + mask3) / 2

            # 2. 插帧预测
            pred_bg = bg_model(frame1, frame3)  # 背景插帧
            pred_fg = fg_model(frame1, frame3)  # 前景插帧

            # 3. 融合预测
            pred_middle = fuse_images(pred_fg, pred_bg, mask_middle)

            # 4. 计算损失
            gt_middle = gt_middle.to(pred_middle.device)  # 确保ground truth在同一设备
            loss = criterion(pred_middle, gt_middle)

            # 5. 反向传播
            optimizer_bg.zero_grad()
            optimizer_fg.zero_grad()
            loss.backward()
            optimizer_bg.step()
            optimizer_fg.step()

            # 记录损失
            epoch_loss += loss.item()

            # 更新进度条的描述信息
            progress_bar.set_postfix(loss=loss.item())

        # 打印每个epoch的平均损失
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

        # 保存模型
        bg_model_path = os.path.join(save_dir, f"background_model_epoch{epoch + 1}.pth")
        fg_model_path = os.path.join(save_dir, f"foreground_model_epoch{epoch + 1}.pth")
        torch.save(bg_model.state_dict(), bg_model_path)
        torch.save(fg_model.state_dict(), fg_model_path)

        print(f"Saved background model to {bg_model_path}")
        print(f"Saved foreground model to {fg_model_path}")


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train video interpolation model.")
    parser.add_argument("--root_dir", type=str,default="/pub/data/lijm/data/vimeo_triplet/vimeo_triplet/sequences/00078/", required=False, help="Path to the training dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--save_dir", type=str, default="/pub/data/lz/video_frame_in/master/checkpoints", help="Directory to save model checkpoints.")

    args = parser.parse_args()

    # 打印训练参数
    print(f"Training with the following parameters:")
    print(f"Dataset Path: {args.root_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Save Directory: {args.save_dir}")

    # 调用训练函数
    train(
        root_dir=args.root_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir
    )

