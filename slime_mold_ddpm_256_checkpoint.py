import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------- 1. 数据集 --------------------
class ImageSequenceDataset(Dataset):
    """从文件夹读取按文件名排序的图像序列，构造 (条件帧序列, 目标帧) 样本"""
    def __init__(self, folder_path, context_frames=3, image_size=256, transform=None):
        self.folder_path = folder_path
        self.context_frames = context_frames
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
        ])
        # 获取所有图片文件并排序
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")) +
                                  glob.glob(os.path.join(folder_path, "*.jpg")) +
                                  glob.glob(os.path.join(folder_path, "*.jpeg")))
        if len(self.image_paths) < context_frames + 1:
            raise ValueError(f"需要至少 {context_frames+1} 张图片，当前只有 {len(self.image_paths)} 张")

    def __len__(self):
        return len(self.image_paths) - self.context_frames

    def __getitem__(self, idx):
        # 条件帧: idx 到 idx+context_frames-1
        context_paths = self.image_paths[idx:idx+self.context_frames]
        target_path = self.image_paths[idx+self.context_frames]
        context_imgs = [self.transform(Image.open(p).convert('RGB')) for p in context_paths]
        target_img = self.transform(Image.open(target_path).convert('RGB'))
        # 将条件帧沿通道维度拼接: (C*context, H, W)
        context = torch.cat(context_imgs, dim=0)
        return context, target_img

# -------------------- 2. 扩散过程 (DDPM) --------------------
class DiffusionProcess:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """前向加噪: x_t = sqrt(alpha_cumprod) * x0 + sqrt(1-alpha_cumprod) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_t * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t, context, cond_weight=1.0):
        """单步去噪，无条件引导（可调强度）"""
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        pred_noise = model(x_t, t_tensor, context)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        pred_x0 = (x_t - beta_t / torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
        else:
            noise = 0
            sigma_t = 0
        x_prev = pred_x0 + sigma_t * noise
        return x_prev

    @torch.no_grad()
    def sample(self, model, context, shape, cond_weight=1.0):
        """从纯噪声生成图像"""
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t, context, cond_weight)
        return x

# -------------------- 3. 条件 UNet 模型 (混合卷积 + 扩散迭代块) --------------------
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class MixConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv9 = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, time_emb):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out9 = self.conv9(x)
        out = out3 + out5 + out9
        out = self.norm(out)
        time_emb_out = self.time_mlp(time_emb)[:, :, None, None]
        out = out + time_emb_out
        out = self.act(out)
        return out

class IterativeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        time_emb_out = self.time_mlp(time_emb)[:, :, None, None]
        out = out + time_emb_out
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = out + time_emb_out
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = out + time_emb_out
        out = self.act(out)

        out = out + identity
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block = MixConvBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, time_emb):
        x = self.block(x, time_emb)
        skip = x
        x = self.pool(x)
        return x, skip

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = IterativeBlock(in_channels * 2, out_channels, time_emb_dim)

    def forward(self, x, skip, time_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, time_emb)
        return x

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, context_channels=9, time_dim=256, base_channels=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        total_in_channels = in_channels + context_channels
        self.inc = nn.Conv2d(total_in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownSample(base_channels, base_channels*2, time_dim)
        self.down2 = DownSample(base_channels*2, base_channels*4, time_dim)
        self.down3 = DownSample(base_channels*4, base_channels*8, time_dim)
        self.down4 = DownSample(base_channels*8, base_channels*16, time_dim)

        self.mid = MixConvBlock(base_channels*16, base_channels*16, time_dim)

        self.up1 = UpSample(base_channels*16, base_channels*8, time_dim)
        self.up2 = UpSample(base_channels*8, base_channels*4, time_dim)
        self.up3 = UpSample(base_channels*4, base_channels*2, time_dim)
        self.up4 = UpSample(base_channels*2, base_channels, time_dim)

        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, t, context):
        time_emb = self.time_mlp(t)
        h = torch.cat([x, context], dim=1)
        h = self.inc(h)

        h, s1 = self.down1(h, time_emb)
        h, s2 = self.down2(h, time_emb)
        h, s3 = self.down3(h, time_emb)
        h, s4 = self.down4(h, time_emb)

        h = self.mid(h, time_emb)

        h = self.up1(h, s4, time_emb)
        h = self.up2(h, s3, time_emb)
        h = self.up3(h, s2, time_emb)
        h = self.up4(h, s1, time_emb)

        out = self.outc(h)
        return out

# -------------------- 4. 训练和采样函数 --------------------
def train_one_epoch(model, diffusion, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for context, target in pbar:
        context = context.to(device)
        target = target.to(device)
        batch_size = target.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        x_t, noise = diffusion.q_sample(target, t)
        pred_noise = model(x_t, t, context)
        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

@torch.no_grad()
def sample_next_frame(model, diffusion, context_frames, device, shape=(1,3,256,256)):
    model.eval()
    context = context_frames.to(device)
    generated = diffusion.sample(model, context, shape)
    generated = (generated + 1) / 2
    return generated

def save_image(tensor, path):
    img = tensor.squeeze(0).cpu().clamp(0,1)
    img = transforms.ToPILImage()(img)
    img.save(path)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_path):
    """保存完整的检查点，包含模型、优化器、调度器状态和训练进度"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """加载检查点，返回起始 epoch 和 loss"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1   # 从下一轮开始
    loss = checkpoint['loss']
    print(f"Resumed from checkpoint {checkpoint_path}, starting epoch {start_epoch}")
    return start_epoch, loss

# -------------------- 5. 主程序（支持断点续训） --------------------
def main():
    parser = argparse.ArgumentParser(description='Train DDPM with conditional UNet for image sequence prediction')
    parser.add_argument('--data_folder', type=str, default='./color_processed_images',
                        help='Folder containing image sequence')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--context_frames', type=int, default=3,
                        help='Number of context frames')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channels for UNet')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集和加载器
    dataset = ImageSequenceDataset(args.data_folder, context_frames=args.context_frames, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 模型、扩散过程、优化器、调度器
    model = ConditionalUNet(in_channels=3, context_channels=3*args.context_frames, base_channels=args.base_channels).to(device)
    diffusion = DiffusionProcess(num_timesteps=args.num_timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    start_epoch = 1
    if args.resume is not None:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_one_epoch(model, diffusion, dataloader, optimizer, device, epoch)
        scheduler.step(loss)
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

        # 每10个epoch保存一次完整检查点 + 预测样例
        if epoch % 10 == 0:
            # 保存检查点（用于后续恢复）
            checkpoint_path = f"checkpoint_epoch{epoch}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_path)

            # 保存模型权重（兼容旧格式）
            torch.save(model.state_dict(), f"unet_diffusion_epoch{epoch}.pth")

            # 生成预测样例
            model.eval()
            with torch.no_grad():
                context_sample, target_sample = dataset[0]
                context_sample = context_sample.unsqueeze(0).to(device)
                target_sample = target_sample.unsqueeze(0).to(device)
                pred = sample_next_frame(model, diffusion, context_sample, device, shape=(1,3,args.image_size, args.image_size))
                target_vis = (target_sample + 1) / 2
                save_image(pred, f"pred_epoch{epoch}.png")
                save_image(target_vis, f"target_epoch{epoch}.png")
                print(f"Saved prediction and target at epoch {epoch}")

    # 最终保存模型权重和完整检查点
    torch.save(model.state_dict(), "final_unet_diffusion.pth")
    save_checkpoint(model, optimizer, scheduler, args.epochs, loss, "final_checkpoint.pth")
    print("Training completed.")

if __name__ == "__main__":
    main()