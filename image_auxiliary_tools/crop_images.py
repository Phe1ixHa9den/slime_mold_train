import cv2
import numpy as np
import os
import sys

# 支持的图片扩展名
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def crop_by_rectangle(img, points):
    """
    矩形裁剪：根据四个点计算最小外接矩形并裁剪
    :param img:     OpenCV 图像 (BGR)
    :param points:  四个点的列表，格式为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    :return:        裁剪后的图像（可能不是严格的四边形区域，而是轴对齐矩形）
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    # 确保坐标在图像范围内
    x_min, x_max = max(0, x_min), min(img.shape[1], x_max)
    y_min, y_max = max(0, y_min), min(img.shape[0], y_max)
    cropped = img[y_min:y_max, x_min:x_max]
    return cropped

def crop_by_perspective(img, points, output_size=None):
    """
    透视变换裁剪：将四个点围成的四边形区域矫正为矩形
    :param img:          OpenCV 图像 (BGR)
    :param points:       四个点的列表，顺序应为 [左上, 右上, 右下, 左下]（顺时针或逆时针）
    :param output_size:  输出图像的尺寸 (width, height)，若为 None 则根据四边形边长自动计算
    :return:             矫正后的矩形图像
    """
    pts_src = np.array(points, dtype=np.float32)

    # 计算输出尺寸（默认取四边形四条边的平均长度作为宽高）
    if output_size is None:
        # 计算四边形的四条边长
        def distance(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))
        width_top = distance(points[0], points[1])
        width_bottom = distance(points[3], points[2])
        height_left = distance(points[0], points[3])
        height_right = distance(points[1], points[2])
        width = int((width_top + width_bottom) / 2)
        height = int((height_left + height_right) / 2)
        output_size = (width, height)
    else:
        width, height = output_size

    # 目标矩形的四个角点（左上、右上、右下、左下）
    pts_dst = np.float32([[0, 0],
                          [width, 0],
                          [width, height],
                          [0, height]])

    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    cropped = cv2.warpPerspective(img, M, (width, height))
    return cropped

def batch_crop_images(input_dir, output_dir, crop_points, crop_mode='rectangle', output_size=None):
    """
    批量裁剪图片
    :param input_dir:    输入文件夹路径
    :param output_dir:   输出文件夹路径
    :param crop_points:  四点坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    :param crop_mode:    裁剪模式 'rectangle' 或 'perspective'
    :param output_size:  透视模式下的输出尺寸 (width, height)，None 则自动
    """
    if not os.path.exists(input_dir):
        print(f"错误：输入文件夹 '{input_dir}' 不存在")
        return

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(SUPPORTED_EXT):
            continue

        input_path = os.path.join(input_dir, filename)
        # 输出文件名（添加后缀，保留原扩展名）
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_cropped{ext}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"警告：无法读取图片 '{filename}'，已跳过")
                continue

            if crop_mode == 'rectangle':
                cropped = crop_by_rectangle(img, crop_points)
            elif crop_mode == 'perspective':
                cropped = crop_by_perspective(img, crop_points, output_size)
            else:
                raise ValueError("crop_mode 必须是 'rectangle' 或 'perspective'")

            cv2.imwrite(output_path, cropped)
            print(f"已处理：{filename} -> {output_filename} (尺寸: {cropped.shape[1]}x{cropped.shape[0]})")

        except Exception as e:
            print(f"处理 '{filename}' 时发生错误：{e}")

    print("批量裁剪完成！")

if __name__ == "__main__":
    # ========== 用户配置区域 ==========
    # 1. 设置输入和输出文件夹路径
    INPUT_DIR = "./data1"
    OUTPUT_DIR = "./processed_data_1/output_cropped"

    # 2. 设置四点坐标 (x, y) —— 请根据你的图片实际坐标修改
    #    顺序：[左上, 右上, 右下, 左下] （适用于透视变换）
    #    对于矩形裁剪，顺序任意，程序会自动计算包围盒
    CROP_POINTS = [
        (640, 370),   # 左上
        (1234, 370),   # 右上
        (1235, 950),   # 右下
        (640, 950)    # 左下
    ]

    # 3. 选择裁剪模式：'rectangle' 或 'perspective'
    CROP_MODE = 'perspective'   # 透视变换（矫正四边形）

    # 4. 透视模式下的输出尺寸（宽, 高），若为 None 则自动计算
    OUTPUT_SIZE = None   # 例如 (600, 400)

    # =================================

    # 如果通过命令行参数传入输入输出文件夹，优先使用（可选）
    if len(sys.argv) >= 3:
        INPUT_DIR = sys.argv[1]
        OUTPUT_DIR = sys.argv[2]
        print(f"使用命令行参数：输入='{INPUT_DIR}'，输出='{OUTPUT_DIR}'")
    else:
        print(f"使用默认路径：输入='{INPUT_DIR}'，输出='{OUTPUT_DIR}'")

    batch_crop_images(INPUT_DIR, OUTPUT_DIR, CROP_POINTS, CROP_MODE, OUTPUT_SIZE)
