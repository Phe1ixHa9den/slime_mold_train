import matplotlib
matplotlib.use('Agg')   # 非交互式后端，不弹出窗口
import matplotlib.pyplot as plt

# 读取数据文件
epochs = []
losses = []

with open('loss_log_128_0415.txt', 'r') as f:
    lines = f.readlines()
    # 跳过第一行标题（epoch	loss）
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        # 按制表符分割（也可用 split() 自动处理空格/制表符）
        parts = line.split()
        if len(parts) == 2:
            epoch = int(parts[0])
            loss = float(parts[1])
            epochs.append(epoch)
            losses.append(loss)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, 'b-', linewidth=1.5, label='Training Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Curve for 128x128 Diffusion Model', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# 保存图片（可选）
plt.savefig('loss_curve_128.png', dpi=150)

# 显示图像（在 Jupyter 或本地 Python 环境中会弹出窗口）
plt.show()