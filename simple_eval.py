import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser

def load_images(folder_path, image_size=(128, 128)):
    """按文件名排序加载图像，返回 [0,1] 范围的 Tensor (N, C, H, W)"""
    image_files = sorted([f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = []
    for f in image_files:
        img = Image.open(os.path.join(folder_path, f)).convert('RGB')
        img = img.resize(image_size, Image.LANCZOS)
        img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0  # [0,1]
        images.append(img_t)
    return torch.stack(images, dim=0), image_files

def psnr(img1, img2, max_val=1.0):
    """计算 PSNR (dB)"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / torch.sqrt(mse).item())

def ssim(img1, img2, window_size=11, sigma=1.5, max_val=1.0):
    """简化的 SSIM (基于高斯窗口，单通道或RGB平均)"""
    # 将图像转为 [0,1] 浮点，假设输入已归一化
    # 如果RGB，计算每通道的SSIM再平均
    if img1.dim() == 4:  # (B, C, H, W) 取第一个batch
        img1 = img1[0]
        img2 = img2[0]
    C = img1.shape[0]
    if C == 3:
        ssims = []
        for c in range(3):
            ssims.append(_ssim_single_channel(img1[c:c+1], img2[c:c+1], window_size, sigma, max_val))
        return sum(ssims) / 3.0
    else:
        return _ssim_single_channel(img1, img2, window_size, sigma, max_val)

def _ssim_single_channel(img1, img2, window_size, sigma, max_val):
    # 创建高斯窗口
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device) - window_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    window = g[:, None] * g[None, :]  # (window_size, window_size)
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size)
    
    # 计算均值、方差、协方差
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1**2, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=1) - mu1_mu2
    
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def l1_distance(img1, img2):
    """L1 平均绝对误差 (范围 [0,1])"""
    return F.l1_loss(img1, img2).item()

def main():
    parser = ArgumentParser()
    parser.add_argument('--real_dir', type=str, default='./test_data/test_2_1')
    parser.add_argument('--fake_dir', type=str, default='./test_generated_200_epoch_0416/test_without_latest_8')
    parser.add_argument('--img_size', type=int, default=128)
    args = parser.parse_args()
    
    real_imgs, real_files = load_images(args.real_dir, (args.img_size, args.img_size))
    fake_imgs, fake_files = load_images(args.fake_dir, (args.img_size, args.img_size))
    
    assert len(real_imgs) == len(fake_imgs), f"数量不匹配: {len(real_imgs)} vs {len(fake_imgs)}"
    print(f"评估 {len(real_imgs)} 对图像\n")
    
    psnr_vals, ssim_vals, l1_vals = [], [], []
    for i in range(len(real_imgs)):
        r = real_imgs[i:i+1].clone()
        f = fake_imgs[i:i+1].clone()
        psnr_vals.append(psnr(r, f))
        ssim_vals.append(ssim(r, f))
        l1_vals.append(l1_distance(r, f))
    
    print("========== 逐图指标 ==========")
    for i, (rf, ff) in enumerate(zip(real_files, fake_files)):
        print(f"{rf} vs {ff}: PSNR={psnr_vals[i]:.2f} dB, SSIM={ssim_vals[i]:.4f}, L1={l1_vals[i]:.4f}")
    
    print("\n========== 平均指标 ==========")
    print(f"PSNR (mean): {np.mean(psnr_vals):.2f} dB")
    print(f"SSIM (mean): {np.mean(ssim_vals):.4f}")
    print(f"L1 (mean): {np.mean(l1_vals):.4f}")

if __name__ == '__main__':
    main()