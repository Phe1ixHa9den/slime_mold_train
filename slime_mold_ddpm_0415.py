import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# -------------------- 1. 数据加载 --------------------
class ImageSequenceDataset(Dataset):
    """从文件夹读取有序图片序列，构造训练对：过去cond_len帧 -> 未来1帧"""
    def __init__(self, folder_path, cond_len=4, img_size=(64,64), transform=None):
        """
        Args:
            folder_path: 图片文件夹路径
            cond_len: 条件帧数（历史帧数）
            img_size: 图片缩放尺寸
            transform: 额外的图像变换（可选）
        """
        self.folder_path = folder_path
        self.cond_len = cond_len
        self.img_size = img_size
        self.transform = transform
        
        # 获取所有图片文件并按文件名排序
        self.image_files = sorted([f for f in os.listdir(folder_path) 
                                   if f.endswith(('.png','.jpg','.jpeg'))])
        if len(self.image_files) < cond_len + 1:
            raise ValueError("图片数量不足，无法构造训练对")
        
        # 预定义变换：转为Tensor，归一化到[-1,1]（利于扩散模型）
        self.default_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) if img_size[-1]==3 else transforms.Normalize([0.5],[0.5])
        ])
        
    def __len__(self):
        return len(self.image_files) - self.cond_len  # 每cond_len+1帧构成一个样本
        
    def __getitem__(self, idx):
        # 取连续 cond_len+1 张图片，前cond_len张作为条件，最后一张作为目标
        imgs = []
        for i in range(self.cond_len + 1):
            img_path = os.path.join(self.folder_path, self.image_files[idx + i])
            img = Image.open(img_path).convert('RGB')  # 统一转为RGB
            if self.transform:
                img = self.transform(img)
            else:
                img = self.default_transform(img)
            imgs.append(img)
        cond = torch.stack(imgs[:-1], dim=0)  # (cond_len, C, H, W)
        target = imgs[-1]                     # (C, H, W)
        return cond, target

# -------------------- 2. 扩散模型组件 --------------------
class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的 sinusoidal 嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """基本的残差块，用于U-Net"""
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.activation = nn.SiLU()
        
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch)
            )
            
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        if t_emb is not None and self.time_mlp is not None:
            t_emb = self.time_mlp(t_emb)
            # 时间嵌入加到特征图上（广播）
            h = h + t_emb[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        return h + self.residual(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNet(nn.Module):
    """条件U-Net，输入包括：噪声图像 + 条件帧拼接，以及时间步嵌入"""
    def __init__(self, in_channels, cond_channels, img_size, time_dim=256):
        super().__init__()
        """
        Args:
            in_channels: 噪声图像的通道数（通常为3）
            cond_channels: 条件帧的通道数（cond_len * 3）
            img_size: 图像大小（H,W），用于自动计算下采样次数
            time_dim: 时间嵌入维度
        """
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.img_size = img_size
        self.time_dim = time_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 编码器（增加一层下采样）
        self.enc1 = Block(in_channels + cond_channels, 64, time_dim)
        self.down1 = Downsample(64)      # 64 -> 32 (若原图128，这里变为64)
        self.enc2 = Block(64, 128, time_dim)
        self.down2 = Downsample(128)     # 128 -> 64 (64->32)
        self.enc3 = Block(128, 256, time_dim)
        self.down3 = Downsample(256)     # 256 -> 128 (32->16)
        self.enc4 = Block(256, 512, time_dim)   # 新增编码块
        self.down4 = Downsample(512)     # 512 -> 256 (16->8)
        
        # 中间层（保持原中间层维度）
        self.mid = Block(512, 512, time_dim)   # 注意输入输出通道改为512
        
        # 解码器（对称增加上采样）
        self.up4 = Upsample(512)          # 新增上采样
        self.dec4 = Block(512 + 512, 256, time_dim)  # 跳跃连接来自enc4
        self.up3 = Upsample(256)
        self.dec3 = Block(256 + 256, 128, time_dim)  # 跳跃连接来自enc3
        self.up2 = Upsample(128)
        self.dec2 = Block(128 + 128, 64, time_dim)
        self.up1 = Upsample(64)
        self.dec1 = Block(64 + 64, 64, time_dim)
        
        # 输出层不变
        self.out_conv = nn.Conv2d(64, in_channels, 1)
        
    def forward(self, x, cond, t):
        # 拼接输入
        x = torch.cat([x, cond], dim=1)  # (B, C+cond_channels, H, W)
    
        # 计算时间嵌入（必须添加这一行）
        t_emb = self.time_mlp(t)  # (B, time_dim)
    
        # 编码
        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb)
        d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb)   # 如果增加了第4层
        d4 = self.down4(e4)         # 如果增加了第4层
    
        # 中间
        m = self.mid(d4, t_emb) if hasattr(self, 'down4') else self.mid(d3, t_emb)
    
        # 解码（根据层数调整跳跃连接）
        if hasattr(self, 'down4'):
            u4 = self.up4(m)
            u4 = torch.cat([u4, e4], dim=1)
            d4_out = self.dec4(u4, t_emb)
            u3 = self.up3(d4_out)
            u3 = torch.cat([u3, e3], dim=1)
            d3_out = self.dec3(u3, t_emb)
        else:
            u3 = self.up3(m)
            u3 = torch.cat([u3, e3], dim=1)
            d3_out = self.dec3(u3, t_emb)
    
        u2 = self.up2(d3_out)
        u2 = torch.cat([u2, e2], dim=1)
        d2_out = self.dec2(u2, t_emb)
        u1 = self.up1(d2_out)
        u1 = torch.cat([u1, e1], dim=1)
        d1_out = self.dec1(u1, t_emb)
    
        out = self.out_conv(d1_out)
        return out

# -------------------- 3. 扩散过程 --------------------
class GaussianDiffusion:
    """DDPM 扩散过程，支持条件生成"""
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.)
        # 前向扩散所需系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # 后验分布系数
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x0, t, noise=None):
        """前向扩散：从x0加噪得到xt"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    @torch.no_grad()
    def p_sample(self, model, x, cond, t):
        """单步去噪：从xt预测xt-1"""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = 1. / torch.sqrt(self.alphas[t])[:, None, None, None]
        
        # 预测噪声
        predicted_noise = model(x, cond, t)
        # 计算均值
        mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        # 计算方差
        posterior_variance_t = self.posterior_variance[t][:, None, None, None]
        if t[0] == 0:
            # 最后一步不加噪声
            return mean
        else:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model, cond, img_shape):
        """从噪声生成图像，给定条件cond"""
        device = next(model.parameters()).device
        batch_size = cond.shape[0]
        x = torch.randn((batch_size, *img_shape), device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, cond, t)
        return x

# -------------------- 4. 训练与评估 --------------------
def train_diffusion(model, diffusion, dataloader, optimizer, epochs, device):
    model.train()
    epoch_losses = []     # 用于记录每个epoch的平均loss
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for cond, target in pbar:
            cond = cond.to(device)       # (B, cond_len, C, H, W)
            target = target.to(device)   # (B, C, H, W)
            batch_size = target.shape[0]
            
            # 将条件帧展平为单通道维度（便于拼接）
            B, T, C, H, W = cond.shape
            cond_flat = cond.view(B, T*C, H, W)  # (B, cond_len*C, H, W)
            
            # 随机采样时间步
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device).long()
            # 加噪
            noise = torch.randn_like(target)
            x_t, _ = diffusion.q_sample(target, t, noise)
            # 预测噪声
            predicted_noise = model(x_t, cond_flat, t)
            # 损失
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")
        return epoch_losses

def evaluate_and_visualize(model, diffusion, dataset, device, cond_len, img_shape, num_samples=4):
    model.eval()
    idx = np.random.randint(0, len(dataset), num_samples)
    fig, axes = plt.subplots(num_samples, cond_len+2, figsize=(15, 3*num_samples))
    with torch.no_grad():
        for i, sample_idx in enumerate(idx):
            cond_seq, target = dataset[sample_idx]  # cond_seq: (cond_len, C, H, W), target: (C, H, W)
            cond_seq = cond_seq.unsqueeze(0).to(device)  # (1, cond_len, C, H, W)
            B, T, C, H, W = cond_seq.shape
            cond_flat = cond_seq.view(B, T*C, H, W)
            # 生成预测
            generated = diffusion.sample(model, cond_flat, img_shape)  # (1, C, H, W)
            # 显示
            for j in range(cond_len):
                img = cond_seq[0, j].cpu().numpy().transpose(1,2,0)
                img = (img + 1) / 2  # 从[-1,1]到[0,1]
                axes[i, j].imshow(np.clip(img, 0, 1))
                axes[i, j].set_title(f"Input {j+1}")
                axes[i, j].axis('off')
            # 显示真实目标
            target_img = target.cpu().numpy().transpose(1,2,0)
            target_img = (target_img + 1) / 2
            axes[i, cond_len].imshow(np.clip(target_img, 0, 1))
            axes[i, cond_len].set_title("Ground Truth")
            axes[i, cond_len].axis('off')
            # 显示生成
            gen_img = generated[0].cpu().numpy().transpose(1,2,0)
            gen_img = (gen_img + 1) / 2
            axes[i, cond_len+1].imshow(np.clip(gen_img, 0, 1))
            axes[i, cond_len+1].set_title("Generated")
            axes[i, cond_len+1].axis('off')
    plt.tight_layout()
    plt.savefig('prediction_example.png')
    plt.show()

# -------------------- 5. 主函数 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./processed_data/color_processed_images_final", help='图片序列文件夹路径')
    parser.add_argument('--cond_len', type=int, default=4, help='历史帧数')
    parser.add_argument('--img_size', type=int, default=128, help='图像缩放尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_model', type=str, default='./generate_model/diffusion_model_128_0415.pth', help='模型保存路径')
    args = parser.parse_args()
    
    # 数据集
    dataset = ImageSequenceDataset(args.data_dir, cond_len=args.cond_len, img_size=(args.img_size, args.img_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # 模型参数
    in_channels = 3  # RGB图像
    cond_channels = args.cond_len * in_channels
    img_shape = (in_channels, args.img_size, args.img_size)
    
    model = UNet(in_channels, cond_channels, img_size=args.img_size, time_dim=256).to(args.device)
    diffusion = GaussianDiffusion(num_timesteps=args.num_timesteps, device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练
    loss_history = train_diffusion(model, diffusion, dataloader, optimizer, args.epochs, args.device)
    
    # 将损失写入文本文件
    loss_file = "loss_log_128_0415.txt"
    with open(loss_file, "w") as f:
        f.write("epoch\tloss\n")
        for epoch, loss in enumerate(loss_history, 1):
            f.write(f"{epoch}\t{loss:.6f}\n")
    print(f"Loss log saved to {loss_file}")

    # 保存模型
    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")
    
    # 可视化评估
    evaluate_and_visualize(model, diffusion, dataset, args.device, args.cond_len, img_shape, num_samples=4)

if __name__ == '__main__':
    main()
