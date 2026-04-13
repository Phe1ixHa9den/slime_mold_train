#!/usr/bin/env python3
"""
高光平衡 + 整体亮度调整 图像批量处理程序
功能：
  1. 压缩图像中特别亮的部分，同时适度提升暗部区域（高光平衡）。
  2. 可选整体降低亮度（暗化处理），使画面更柔和或营造暗调风格。
原理：
  - 高光平衡：在HSV的V通道应用非线性映射曲线 V_out = V_in / (1 + strength * V_in) * (1 + strength)
  - 整体亮度调整：对平衡后的V通道乘以系数 (1 - darken)
使用方法：
  python script.py --input_dir ./input --output_dir ./output [--strength 1.5] [--auto] [--darken 0.2] [--recursive]
"""

import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def balance_highlights(v_channel, strength=1.0):
    """
    对亮度通道应用高光平衡映射
    v_channel: float32, 范围[0,1]
    返回平衡后的v通道（范围[0,1]）
    """
    denominator = 1.0 + strength * v_channel
    v_balanced = v_channel / denominator * (1.0 + strength)
    return np.clip(v_balanced, 0.0, 1.0)

def apply_darken(v_channel, darken=0.0):
    """
    整体降低亮度
    v_channel: float32, 范围[0,1]
    darken: 降低比例，0~1，如0.2表示亮度降低20%
    返回调整后的v通道
    """
    if darken == 0.0:
        return v_channel
    factor = 1.0 - darken
    # 确保不低于0
    return np.clip(v_channel * factor, 0.0, 1.0)

def process_image_rgb(image, strength=1.0, auto_strength=False, darken=0.0):
    """
    处理单张图像（BGR格式）
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    
    if is_color:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
    else:
        v_channel = image.astype(np.float32) / 255.0
    
    # 高光平衡
    if auto_strength:
        # 注意：auto_strength 需要原始图像来计算，这里简单处理：
        # 如果 auto_strength 为 True，调用外部传入的 strength 实际上会被覆盖
        # 为简化，auto_strength 在主流程中单独计算 strength 值后传进来
        pass
    cur_strength = strength
    v_balanced = balance_highlights(v_channel, cur_strength)
    
    # 整体亮度调整（降低）
    v_darkened = apply_darken(v_balanced, darken)
    
    v_uint8 = (v_darkened * 255).astype(np.uint8)
    
    if is_color:
        hsv[:, :, 2] = v_uint8
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        result = v_uint8
    return result

def auto_strength_from_histogram(image, percentile=95):
    """根据图像亮度直方图自动估算高光平衡强度"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    high_thresh = np.percentile(gray, percentile)
    over_exposed_ratio = np.mean(gray > high_thresh)
    strength = 0.5 + over_exposed_ratio * 3.0
    return np.clip(strength, 0.5, 2.0)

def imread_unicode(path):
    """支持中文路径的图片读取"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法解码图像: {path}")
    return img

def imwrite_unicode(path, img):
    """支持中文路径的图片保存"""
    ext = os.path.splitext(path)[1].lower()
    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"编码图像失败: {path}")
    with open(path, 'wb') as f:
        f.write(encoded.tobytes())
    return True

def process_image(input_path, output_path, strength, auto_strength, darken):
    """处理单张图像（包含自动强度逻辑）"""
    try:
        img = imread_unicode(input_path)
    except Exception as e:
        print(f"  读取失败: {e}")
        return False
    
    # 确定高光平衡强度
    cur_strength = strength
    if auto_strength:
        cur_strength = auto_strength_from_histogram(img)
    
    try:
        result = process_image_rgb(img, cur_strength, auto_strength=False, darken=darken)
    except Exception as e:
        print(f"  处理失败: {e}")
        return False
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imwrite_unicode(output_path, result)
    except Exception as e:
        print(f"  保存失败: {e}")
        return False
    
    print(f"  已处理: {input_path} -> {output_path} (高光强度={cur_strength:.2f}, 暗化={darken:.2f})")
    return True

def collect_images(input_dir, extensions, recursive):
    """收集输入目录下所有符合条件的图像文件路径"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    images = []
    if recursive:
        for ext in extensions:
            for file_path in input_dir.rglob(f"*{ext}"):
                if file_path.is_file():
                    images.append(file_path)
    else:
        for ext in extensions:
            for file_path in input_dir.glob(f"*{ext}"):
                if file_path.is_file():
                    images.append(file_path)
    return images

def main():
    parser = argparse.ArgumentParser(
        description="批量平衡图像高光并可选整体降低亮度",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_dir", default="output_resized2", help="输入文件夹路径")
    parser.add_argument("--output_dir", default="balanced2", help="输出文件夹路径")
    parser.add_argument("--strength", type=float, default=1.5,
                        help="高光压缩强度 (0.5~2.5)，默认1.0。若启用--auto则此参数被忽略")
    parser.add_argument("--auto", action="store_true",
                        help="自动根据每张图像的高光占比估算强度（覆盖--strength）")
    parser.add_argument("--darken", type=float, default=0.2,
                    help="整体亮度降低比例，范围0~1，例如0.2表示亮度降低20%%。默认0（不降低）。"
                         "也可使用负值（如-0.1）提高整体亮度。")
    parser.add_argument("--ext", nargs="+", default=None,
                        help="要处理的文件扩展名，例如 --ext .jpg .png （默认支持常见图像格式）")
    parser.add_argument("--recursive", action="store_true",
                        help="递归处理子文件夹中的图像，并在输出目录中保持相同的相对路径")
    
    args = parser.parse_args()
    
    # 参数有效性检查
    if args.darken > 1.0 or args.darken < -0.5:
        print(f"警告: --darken 值 {args.darken} 超出推荐范围[-0.5, 1.0]，效果可能不自然", file=sys.stderr)
    
    # 扩展名处理
    if args.ext:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in args.ext]
    else:
        extensions = SUPPORTED_EXT
    
    # 收集图像
    try:
        image_paths = collect_images(args.input_dir, extensions, args.recursive)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not image_paths:
        print(f"在目录 '{args.input_dir}' 中未找到任何支持的图像文件。", file=sys.stderr)
        sys.exit(1)
    
    print(f"找到 {len(image_paths)} 张图像，开始批量处理...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.auto:
        print("自动强度模式: 每张图独立估算高光压缩强度")
    else:
        print(f"固定高光压缩强度: {args.strength}")
    print(f"整体亮度调整: darken = {args.darken} (负值表示提亮)")
    print("-" * 50)
    
    success_count = 0
    for img_path in image_paths:
        rel_path = img_path.relative_to(args.input_dir) if args.recursive else img_path.name
        output_path = Path(args.output_dir) / rel_path
        if process_image(str(img_path), str(output_path), args.strength, args.auto, args.darken):
            success_count += 1
    
    print("-" * 50)
    print(f"处理完成！成功: {success_count}/{len(image_paths)}")
    if success_count < len(image_paths):
        sys.exit(1)

if __name__ == "__main__":
    main()
