import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from glob import glob

# -------------------- 模型定义（与训练时完全一致） --------------------
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

class UNet(nn.Module):
    def __init__(self, in_channels, cond_channels, img_size=256, time_dim=256,
                 down_channels=[64, 128, 256, 512, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.time_dim = time_dim
        self.down_channels = down_channels
        self.num_levels = len(down_channels) - 1
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        in_ch = in_channels + cond_channels
        for i, out_ch in enumerate(down_channels):
            self.enc_blocks.append(Block(in_ch, out_ch, time_dim))
            in_ch = out_ch
            if i < self.num_levels:
                self.down_blocks.append(Downsample(out_ch))
        
        self.mid_block = Block(down_channels[-1], down_channels[-1], time_dim)
        
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(self.num_levels)):
            up_in_ch = down_channels[i+1]
            self.up_blocks.append(Upsample(up_in_ch))
            dec_in_ch = up_in_ch + down_channels[i]
            dec_out_ch = down_channels[i]
            self.dec_blocks.append(Block(dec_in_ch, dec_out_ch, time_dim))
        
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

# -------------------- 扩散过程 --------------------
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

# -------------------- 图像预处理/后处理 --------------------
def get_transform(img_size=(256,256)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

def denormalize(tensor):
    """将[-1,1]范围的tensor转为[0,1]的numpy图像(H,W,C)"""
    img = tensor.detach().cpu().numpy().transpose(1,2,0)
    img = (img + 1) / 2
    return np.clip(img, 0, 1)

def load_cond_frames(paths, transform, cond_len):
    """加载条件帧，如果提供的帧数少于cond_len则报错；如果多于cond_len则取最后cond_len帧"""
    if len(paths) < cond_len:
        raise ValueError(f"至少需要 {cond_len} 张条件图像，但只提供了 {len(paths)} 张")
    # 只使用最后 cond_len 帧（滑动窗口起始）
    selected = paths[-cond_len:]
    cond_tensors = []
    for p in selected:
        img = Image.open(p).convert('RGB')
        cond_tensors.append(transform(img))
    cond_seq = torch.stack(cond_tensors, dim=0)  # [T, C, H, W]
    return cond_seq

def save_image(tensor, save_path):
    """保存单张图像（tensor格式[C,H,W]，范围[-1,1]）"""
    img_np = denormalize(tensor)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_pil.save(save_path)

# -------------------- 生成主逻辑 --------------------
def generate_sequence(model, diffusion, cond_seq, num_frames, img_shape, device):
    """
    迭代生成后续帧
    cond_seq: [T, C, H, W] 初始条件序列，T = cond_len
    num_frames: 要生成的帧数
    返回: 生成的图像列表，每个元素为 [C,H,W] tensor (范围[-1,1])
    """
    model.eval()
    cond_len = cond_seq.shape[0]
    current_cond = cond_seq.to(device)  # [cond_len, C, H, W]
    generated_frames = []
    
    with torch.no_grad():
        for _ in range(num_frames):
            # 准备模型输入：将条件帧展平为 [1, cond_len*C, H, W]
            B = 1
            cond_flat = current_cond.view(1, cond_len * img_shape[0], img_shape[1], img_shape[2])
            # 采样生成下一帧
            x_t = torch.randn((B, img_shape[0], img_shape[1], img_shape[2]), device=device)
            # 逐步去噪
            for t in reversed(range(diffusion.num_timesteps)):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                x_t = diffusion.p_sample(model, x_t, cond_flat, t_tensor)
            new_frame = x_t[0]  # [C, H, W]
            generated_frames.append(new_frame.cpu())
            # 更新条件序列：滑动窗口，丢弃最早帧，加入新帧
            current_cond = torch.cat([current_cond[1:], new_frame.unsqueeze(0)], dim=0)
    return generated_frames

# -------------------- 主程序 --------------------
def main():
    parser = argparse.ArgumentParser(description="使用训练好的扩散模型生成图像序列")
    parser.add_argument('--checkpoint', type=str, default="diffusion_model_256.pth", help='模型检查点路径 (.pth)')
    parser.add_argument('--cond_dir', type=str, default="./test_data/test_1", help='条件帧所在文件夹（按文件名排序）')
    parser.add_argument('--cond_files', type=str, nargs='+', default=None, help='直接指定条件帧图像文件路径列表')
    parser.add_argument('--cond_len', type=int, default=4, help='模型需要的条件帧数量')
    parser.add_argument('--img_size', type=int, default=128, help='图像尺寸 (正方形)')
    parser.add_argument('--num_frames', type=int, default=10, help='要生成的帧数')
    parser.add_argument('--output_dir', type=str, default='./test_generated_1_200_epoch_0415', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='扩散步数（需与训练一致）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # 1. 加载条件帧路径
    if args.cond_files is not None:
        paths = args.cond_files
    elif args.cond_dir is not None:
        paths = sorted(glob(os.path.join(args.cond_dir, '*.*')))
        paths = [p for p in paths if p.lower().endswith(('.png','.jpg','.jpeg'))]
    else:
        raise ValueError("请指定 --cond_dir 或 --cond_files")
    if len(paths) == 0:
        raise ValueError("未找到任何条件帧图像")
    print(f"找到 {len(paths)} 张条件图像，需要 {args.cond_len} 张")
    
    # 2. 图像预处理
    transform = get_transform((args.img_size, args.img_size))
    cond_seq = load_cond_frames(paths, transform, args.cond_len)
    print(f"条件序列形状: {cond_seq.shape}")
    
    # 3. 构建模型并加载权重
    in_channels = 3
    cond_channels = args.cond_len * in_channels
    img_shape = (in_channels, args.img_size, args.img_size)
    
    model = UNet(in_channels, cond_channels, img_size=args.img_size, time_dim=256,
                 down_channels=[64, 128, 256, 512, 512]).to(args.device)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载检查点 (epoch {checkpoint.get('epoch', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("加载模型权重")
    
    # 4. 扩散过程
    diffusion = GaussianDiffusion(num_timesteps=args.num_timesteps, device=args.device)
    
    # 5. 生成序列
    print(f"开始生成 {args.num_frames} 帧...")
    generated = generate_sequence(model, diffusion, cond_seq, args.num_frames, img_shape, args.device)
    print("生成完成")
    
    # 6. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    # 保存条件帧（便于对照）
    for i in range(args.cond_len):
        save_image(cond_seq[i], os.path.join(args.output_dir, f"cond_{i+1:04d}.png"))
    # 保存生成的帧
    for idx, frame in enumerate(generated):
        save_image(frame, os.path.join(args.output_dir, f"gen_{idx+1:04d}.png"))
    print(f"已保存至 {args.output_dir}")

if __name__ == '__main__':
    main()