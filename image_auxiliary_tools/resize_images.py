import cv2
import os
import sys

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def resize_with_letterbox(img, target_size=(256, 256), color=(0, 0, 0)):
    """
    保持宽高比缩放，不足部分填充指定颜色（默认黑色）
    :param img:          OpenCV 图像 (BGR)
    :param target_size:  目标尺寸 (width, height)
    :param color:        填充颜色 (B, G, R)
    :return:             缩放并填充后的图像
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建画布并填充背景色
    canvas = cv2.copyMakeBorder(resized, 0, target_h - new_h, 0, target_w - new_w,
                                cv2.BORDER_CONSTANT, value=color)
    # 如果需要居中，可先计算偏移量，此处直接放置在左上角；若要居中，可修改为：
    # top = (target_h - new_h) // 2
    # bottom = target_h - new_h - top
    # left = (target_w - new_w) // 2
    # right = target_w - new_w - left
    # canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return canvas

def resize_images(input_dir, output_dir, target_size=(256, 256), mode='stretch'):
    """
    批量缩放图片
    :param input_dir:    输入文件夹路径
    :param output_dir:   输出文件夹路径
    :param target_size:  目标尺寸 (width, height)
    :param mode:         'stretch' 或 'letterbox'
    """
    if not os.path.exists(input_dir):
        print(f"错误：输入文件夹 '{input_dir}' 不存在")
        return

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(SUPPORTED_EXT):
            continue

        input_path = os.path.join(input_dir, filename)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_resized{ext}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"警告：无法读取图片 '{filename}'，已跳过")
                continue

            if mode == 'stretch':
                # 直接拉伸到目标尺寸
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            elif mode == 'letterbox':
                resized = resize_with_letterbox(img, target_size)
            else:
                raise ValueError("mode 必须是 'stretch' 或 'letterbox'")

            cv2.imwrite(output_path, resized)
            print(f"已处理：{filename} -> {output_filename} (尺寸: {target_size[0]}x{target_size[1]})")

        except Exception as e:
            print(f"处理 '{filename}' 时发生错误：{e}")

    print("批量缩放完成！")

if __name__ == "__main__":
    # ========== 用户配置区域 ==========
    INPUT_DIR = "./processed_data_1/output_cropped"    # 输入文件夹
    OUTPUT_DIR = "./processed_data_1/output_resized" # 输出文件夹
    TARGET_SIZE = (128, 128)        # 目标尺寸 (宽, 高)
    MODE = 'stretch'                # 'stretch' 或 'letterbox'
    # =================================

    # 支持命令行参数覆盖输入输出路径
    if len(sys.argv) >= 3:
        INPUT_DIR = sys.argv[1]
        OUTPUT_DIR = sys.argv[2]
        print(f"使用命令行参数：输入='{INPUT_DIR}'，输出='{OUTPUT_DIR}'")
    else:
        print(f"使用默认路径：输入='{INPUT_DIR}'，输出='{OUTPUT_DIR}'")

    resize_images(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE, MODE)