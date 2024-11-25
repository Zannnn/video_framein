import torch

def fuse_images(mask, foreground, background):
    device = foreground.device  # 获取 foreground 的设备
    mask = mask.to(device)
    background = background.to(device)
    # 确保 mask 的形状为 [batch_size, 1, H, W]，与 foreground 和 background 的通道对齐
    # print("mask shape:", mask.shape)
    # print("foreground shape:", foreground.shape)
    # print("background shape:", background.shape)
    # if mask.size(1) != foreground.size(1):
    #     mask = mask.repeat(1, foreground.size(1), 1, 1)  # 将单通道扩展到多通道

    if background.size(1) != foreground.size(1):
        background = background.unsqueeze(1).repeat(1, foreground.size(1), 1, 1)

    return mask * foreground + (1 - mask) * background
