import cv2
import numpy as np
import os
from pathlib import Path

def process_image_single(image_path, output_path, params):
    """
    处理单张图片，参数通过字典传入
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"  读取失败，跳过: {image_path}")
        return False
    
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # ---- 1. 处理白色亮斑 ----
    white_spot_mask = (v > params['white_spot_v_thresh']) & (s < params['white_spot_s_thresh'])
    
    v_new = v.astype(np.float32)
    v_new[white_spot_mask] = v_new[white_spot_mask] * params['white_spot_v_reduce']
    v_new = np.clip(v_new, 0, 255).astype(np.uint8)
    
    s_new = s.astype(np.float32)
    s_new[white_spot_mask] = s_new[white_spot_mask] * params['white_spot_s_boost']
    s_new = np.clip(s_new, 0, 255).astype(np.uint8)
    
    hsv_after_white = cv2.merge([h, s_new, v_new])
    
    # ---- 2. 增强黄色区域 ----
    h2, s2, v2 = cv2.split(hsv_after_white)
    
    yellow_mask = (h2 >= params['yellow_hue_range'][0]) & (h2 <= params['yellow_hue_range'][1]) & (s2 > 30)
    
    s2_float = s2.astype(np.float32)
    s2_float[yellow_mask] = s2_float[yellow_mask] * params['yellow_sat_boost']
    s2 = np.clip(s2_float, 0, 255).astype(np.uint8)
    
    v2_float = v2.astype(np.float32)
    v2_float[yellow_mask] = v2_float[yellow_mask] * params['yellow_val_boost']
    v2 = np.clip(v2_float, 0, 255).astype(np.uint8)
    
    hsv_final = cv2.merge([h2, s2, v2])
    result = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    return True

def batch_process(input_folder, output_folder, params=None):
    """
    批量处理文件夹中的所有图片
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径（自动创建）
    :param params: 处理参数字典，为None时使用默认参数
    """
    # 默认参数
    default_params = {
        'yellow_hue_range': (20, 40),
        'yellow_sat_boost': 1.4,
        'yellow_val_boost': 1.1,
        'white_spot_v_thresh': 210,
        'white_spot_s_thresh': 40,
        'white_spot_v_reduce': 0.65,
        'white_spot_s_boost': 1.4
    }
    
    if params is None:
        params = default_params
    else:
        # 合并用户参数（未提供的使用默认值）
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
    
    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 确保输出文件夹存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"输入文件夹不存在: {input_folder}")
        return
    
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"在 {input_folder} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始处理...")
    
    success_count = 0
    for img_file in image_files:
        output_path = Path(output_folder) / img_file.name
        print(f"处理: {img_file.name} -> {output_path}")
        try:
            if process_image_single(str(img_file), str(output_path), params):
                success_count += 1
        except Exception as e:
            print(f"  处理出错: {e}")
    
    print(f"完成！成功处理 {success_count}/{len(image_files)} 张图片")

if __name__ == "__main__":
    # ========== 配置区域 ==========
    INPUT_DIR = "./processed_data_1/balanced"      # 输入文件夹路径
    OUTPUT_DIR = "./processed_data_1/color_processed_images_final"    # 输出文件夹路径
    
    # 可选：自定义参数（不设置则使用默认值）
    custom_params = {
        'yellow_hue_range': (20, 40),   # 黄色色调范围
        'yellow_sat_boost': 1.4,        # 黄色饱和度增强
        'yellow_val_boost': 1.1,        # 黄色亮度增强
        'white_spot_v_thresh': 210,     # 亮斑亮度阈值
        'white_spot_s_thresh': 40,      # 亮斑饱和度阈值
        'white_spot_v_reduce': 0.65,    # 亮斑亮度削弱系数
        'white_spot_s_boost': 1.4       # 亮斑饱和度提升系数
    }
    # =============================
    
    batch_process(INPUT_DIR, OUTPUT_DIR, custom_params)