import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# -------------------- 模型结构定义（与训练脚本完全一致，已修复 UpSample）--------------------
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
        # 修复：skip 的通道数与 in_channels 相同，因此拼接后为 in_channels * 2
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

# -------------------- 扩散过程 --------------------
class DiffusionProcess:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @torch.no_grad()
    def p_sample(self, model, x_t, t, context):
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
    def sample(self, model, context, shape):
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(model, x, t, context)
        return x

# -------------------- 图像预处理与后处理 --------------------
def load_and_preprocess(image_path, image_size=256, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor

def save_image(tensor, path):
    img = (tensor + 1) / 2
    img = img.squeeze(0).cpu().clamp(0, 1)
    img = transforms.ToPILImage()(img)
    img.save(path)

def build_context_from_paths(context_paths, image_size, device):
    tensors = [load_and_preprocess(p, image_size, device) for p in context_paths]
    context = torch.cat(tensors, dim=1)
    return context

# -------------------- 序列生成主函数 --------------------
def generate_sequence(model, diffusion, init_context_paths, num_frames, output_dir,
                      image_size=256, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    context_frames = len(init_context_paths)
    context = build_context_from_paths(init_context_paths, image_size, device)

    # 保存初始条件帧
    for i, path in enumerate(init_context_paths):
        img = Image.open(path).convert('RGB')
        img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))

    current_context = context
    for step in range(num_frames):
        print(f"  Generating frame {step+1}/{num_frames}...")
        shape = (1, 3, image_size, image_size)
        pred = diffusion.sample(model, current_context, shape)
        out_path = os.path.join(output_dir, f"frame_{context_frames + step:04d}.png")
        save_image(pred, out_path)

        # 滑动窗口更新条件
        context_list = torch.split(current_context, 3, dim=1)
        new_context_list = list(context_list[1:]) + [pred]
        current_context = torch.cat(new_context_list, dim=1)

    print(f"  Generated sequence saved to {output_dir}")

# -------------------- 模型加载辅助函数 --------------------
def load_model_from_file(model_path, device, context_frames=3, base_channels=64):
    """加载模型权重，自动兼容纯权重文件和检查点文件"""
    checkpoint = torch.load(model_path, map_location=device)

    # 判断是检查点字典还是纯 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        state_dict = checkpoint
        print("Loaded raw state_dict")

    model = ConditionalUNet(in_channels=3, context_channels=3*context_frames,
                            base_channels=base_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -------------------- 批量评估逻辑 --------------------
def batch_evaluate(args):
    """遍历所有模型文件并生成序列"""
    device = torch.device(args.device)
    diffusion = DiffusionProcess(num_timesteps=args.num_timesteps, device=device)

    # 查找所有模型文件
    if args.model_pattern:
        model_files = sorted(glob.glob(args.model_pattern))
    else:
        # 默认匹配常见模式
        model_files = sorted(glob.glob("unet_diffusion_epoch*.pth") +
                             glob.glob("final_unet_diffusion.pth"))

    if not model_files:
        print("No model files found. Please specify --model_pattern or place models in current directory.")
        return

    print(f"Found {len(model_files)} model files.")

    for model_path in model_files:
        print(f"\n--- Processing {model_path} ---")
        try:
            model = load_model_from_file(model_path, device,
                                         context_frames=args.context_frames,
                                         base_channels=args.base_channels)
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            continue

        # 为每个模型创建独立的输出目录
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.join(args.output_base_dir, model_name)

        generate_sequence(
            model=model,
            diffusion=diffusion,
            init_context_paths=args.init_images,
            num_frames=args.num_frames,
            output_dir=output_dir,
            image_size=args.image_size,
            device=device
        )

# -------------------- 主程序入口 --------------------
def main():
    parser = argparse.ArgumentParser(description="Generate image sequence with conditional DDPM")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a single trained model weights (.pth). If not provided, batch mode is activated.")
    parser.add_argument("--model_pattern", type=str, default=None,
                        help="Glob pattern for batch evaluation (e.g., 'unet_diffusion_epoch*.pth').")
    parser.add_argument("--output_base_dir", type=str, default="./generated_sequences",
                        help="Base output directory for batch evaluation (each model gets a subfolder).")
    parser.add_argument("--init_images", type=str, nargs='+', required=True,
                        help="List of initial condition image paths (exactly context_frames images, order matters)")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--output_dir", type=str, default="./generated_sequence",
                        help="Output directory (only used in single model mode)")
    parser.add_argument("--image_size", type=int, default=256, help="Image size (square)")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Diffusion steps")
    parser.add_argument("--context_frames", type=int, default=3,
                        help="Number of condition frames (must match training)")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channels for UNet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    args = parser.parse_args()

    if len(args.init_images) != args.context_frames:
        raise ValueError(f"Expected {args.context_frames} initial images, got {len(args.init_images)}")

    # 判断是单模型模式还是批量模式
    if args.model_path is not None:
        # 单模型模式（兼容原用法）
        device = torch.device(args.device)
        model = load_model_from_file(args.model_path, device,
                                     context_frames=args.context_frames,
                                     base_channels=args.base_channels)
        diffusion = DiffusionProcess(num_timesteps=args.num_timesteps, device=device)
        generate_sequence(
            model=model,
            diffusion=diffusion,
            init_context_paths=args.init_images,
            num_frames=args.num_frames,
            output_dir=args.output_dir,
            image_size=args.image_size,
            device=device
        )
    else:
        # 批量模式
        batch_evaluate(args)

if __name__ == "__main__":
    main()