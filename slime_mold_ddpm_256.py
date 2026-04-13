import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# -------------------- 1. 数据加载 --------------------
class ImageSequenceDataset(Dataset):
    """从文件夹读取有序图片序列，构造训练对：过去cond_len帧 -> 未来1帧"""
    def __init__(self, folder_path, cond_len=4, img_size=(256,256), transform=None):
        self.folder_path = folder_path
        self.cond_len = cond_len
        self.img_size = img_size
        self.transform = transform
        
        self.image_files = sorted([f for f in os.listdir(folder_path) 
                                   if f.endswith(('.png','.jpg','.jpeg'))])
        if len(self.image_files) < cond_len + 1:
            raise ValueError("图片数量不足，无法构造训练对")
        
        self.default_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        
    def __len__(self):
        return len(self.image_files) - self.cond_len
        
    def __getitem__(self, idx):
        imgs = []
        for i in range(self.cond_len + 1):
            img_path = os.path.join(self.folder_path, self.image_files[idx + i])
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = self.default_transform(img)
            imgs.append(img)
        cond = torch.stack(imgs[:-1], dim=0)
        target = imgs[-1]
        return cond, target

# -------------------- 2. 扩散模型组件 --------------------
class SinusoidalPositionEmbeddings(nn.Module):
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

# -------------------- 适配256x256的U-Net --------------------
class UNet(nn.Module):
    """针对256x256图像的U-Net（4次下采样，通道配置轻量化）"""
    def __init__(self, in_channels, cond_channels, img_size=256, time_dim=256,
                 down_channels=[64, 128, 256, 512, 512]):
        """
        Args:
            down_channels: 各下采样阶段的输出通道数，长度=下采样次数+1
        """
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.time_dim = time_dim
        self.down_channels = down_channels
        self.num_levels = len(down_channels) - 1   # 下采样次数 = 4
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 编码器
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        in_ch = in_channels + cond_channels
        for i, out_ch in enumerate(down_channels):
            self.enc_blocks.append(Block(in_ch, out_ch, time_dim))
            in_ch = out_ch
            if i < self.num_levels:
                self.down_blocks.append(Downsample(out_ch))
        
        # 中间层
        self.mid_block = Block(down_channels[-1], down_channels[-1], time_dim)
        
        # 解码器
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(self.num_levels)):
            up_in_ch = down_channels[i+1]
            self.up_blocks.append(Upsample(up_in_ch))
            dec_in_ch = up_in_ch + down_channels[i]
            dec_out_ch = down_channels[i]
            self.dec_blocks.append(Block(dec_in_ch, dec_out_ch, time_dim))
        
        # 输出层
        self.out_conv = nn.Conv2d(down_channels[0], in_channels, 1)
        
    def forward(self, x, cond, t):
        x = torch.cat([x, cond], dim=1)
        t_emb = self.time_mlp(t)
        
        skip_features = []
        for i, enc in enumerate(self.enc_blocks):
            x = enc(x, t_emb)
            if i < self.num_levels:
                skip_features.append(x)
                x = self.down_blocks[i](x)
        
        x = self.mid_block(x, t_emb)
        
        for i, (up, dec) in enumerate(zip(self.up_blocks, self.dec_blocks)):
            x = up(x)
            skip = skip_features[-i-1]
            x = torch.cat([x, skip], dim=1)
            x = dec(x, t_emb)
        
        return self.out_conv(x)

# -------------------- 3. 扩散过程 --------------------
class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    @torch.no_grad()
    def p_sample(self, model, x, cond, t):
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = 1. / torch.sqrt(self.alphas[t])[:, None, None, None]
        
        predicted_noise = model(x, cond, t)
        mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = self.posterior_variance[t][:, None, None, None]
        if t[0] == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model, cond, img_shape):
        device = next(model.parameters()).device
        batch_size = cond.shape[0]
        x = torch.randn((batch_size, *img_shape), device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, cond, t)
        return x

# -------------------- 4. 训练与评估 --------------------
def train_diffusion(model, diffusion, dataloader, optimizer, epochs, device, start_epoch=0, save_checkpoint_func=None):
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for cond, target in pbar:
            cond = cond.to(device)
            target = target.to(device)
            B, T, C, H, W = cond.shape
            cond_flat = cond.view(B, T*C, H, W)
            
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()
            noise = torch.randn_like(target)
            x_t, _ = diffusion.q_sample(target, t, noise)
            predicted_noise = model(x_t, cond_flat, t)
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")
        
        # 每轮结束后保存检查点（可选）
        if save_checkpoint_func is not None:
            save_checkpoint_func(epoch, model, optimizer, avg_loss)

def evaluate_and_visualize(model, diffusion, dataset, device, cond_len, img_shape, num_samples=4):
    model.eval()
    idx = np.random.randint(0, len(dataset), num_samples)
    fig, axes = plt.subplots(num_samples, cond_len+2, figsize=(15, 3*num_samples))
    with torch.no_grad():
        for i, sample_idx in enumerate(idx):
            cond_seq, target = dataset[sample_idx]
            cond_seq = cond_seq.unsqueeze(0).to(device)
            B, T, C, H, W = cond_seq.shape
            cond_flat = cond_seq.view(B, T*C, H, W)
            generated = diffusion.sample(model, cond_flat, img_shape)
            for j in range(cond_len):
                img = cond_seq[0, j].cpu().numpy().transpose(1,2,0)
                img = (img + 1) / 2
                axes[i, j].imshow(np.clip(img, 0, 1))
                axes[i, j].set_title(f"Input {j+1}")
                axes[i, j].axis('off')
            target_img = target.cpu().numpy().transpose(1,2,0)
            target_img = (target_img + 1) / 2
            axes[i, cond_len].imshow(np.clip(target_img, 0, 1))
            axes[i, cond_len].set_title("Ground Truth")
            axes[i, cond_len].axis('off')
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
    parser.add_argument('--data_dir', type=str, default="./balanced2", help='图片序列文件夹路径')
    parser.add_argument('--cond_len', type=int, default=4, help='历史帧数')
    parser.add_argument('--img_size', type=int, default=256, help='图像缩放尺寸（256x256）')
    parser.add_argument('--batch_size', type=int, default=4, help='批大小（显存不足可降至2或1）')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_timesteps', type=int, default=2000, help='扩散步数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_model', type=str, default='diffusion_model_256.pth', help='模型保存路径')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='从之前的检查点恢复训练（.pth文件）')
    args = parser.parse_args()
    
    dataset = ImageSequenceDataset(args.data_dir, cond_len=args.cond_len, 
                                   img_size=(args.img_size, args.img_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=2, drop_last=True)
    
    in_channels = 3
    cond_channels = args.cond_len * in_channels
    img_shape = (in_channels, args.img_size, args.img_size)
    
    # 256x256 配置：4次下采样，通道 [64,128,256,512,512]
    model = UNet(in_channels, cond_channels, img_size=args.img_size, time_dim=256,
                 down_channels=[64, 128, 256, 512, 512]).to(args.device)
    diffusion = GaussianDiffusion(num_timesteps=args.num_timesteps, device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    # 处理恢复训练
    if args.resume_checkpoint is not None:
        print(f"Loading checkpoint from {args.resume_checkpoint} ...")
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device)
        # 检查是否为完整检查点（包含epoch、optimizer等）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1   # 从下一轮开始
            print(f"Resumed from epoch {start_epoch}, last loss: {checkpoint.get('loss', 'N/A')}")
        else:
            # 兼容旧格式：仅保存模型权重
            model.load_state_dict(checkpoint)
            print("Loaded only model weights (no optimizer/epoch info). Training from scratch.")
            start_epoch = 0
    
    # 定义检查点保存函数（在train_diffusion的每轮结束后调用）
    def save_checkpoint(epoch, model, optimizer, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, args.save_model)
        print(f"Checkpoint saved to {args.save_model} (epoch {epoch+1})")
    
    train_diffusion(model, diffusion, dataloader, optimizer, args.epochs, args.device,
                    start_epoch=start_epoch, save_checkpoint_func=save_checkpoint)
    
    # 最终保存（覆盖检查点文件，也保留最终模型）
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': None,
    }
    torch.save(final_checkpoint, args.save_model)
    print(f"Final model saved to {args.save_model}")
    
    evaluate_and_visualize(model, diffusion, dataset, args.device, args.cond_len, img_shape, num_samples=4)

if __name__ == '__main__':
    main()