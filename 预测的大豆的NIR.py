import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']=['sans-serif']
plt.rcParams['font.sans-serif']=['SimHei']

# ======================
# 读取文件
# ======================
file_path = r"C:\Users\19917\Downloads\p.csv"
df = pd.read_csv(file_path)

# ======================
# 自动提取波长
# ======================
wave_cols = [col for col in df.columns if str(col).isdigit()]
wavelengths = np.array(wave_cols, dtype=int)

# ======================
# 【美化版】画图
# ======================
plt.figure(figsize=(12, 6))  # 变得更宽，彻底不挤

# 漂亮颜色列表
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i in range(len(df)):
    spectrum = df.iloc[i][wave_cols].values
    plt.plot(
        wavelengths,
        spectrum,
        label=f"大豆样本 {i+1}",
        color=colors[i],
        linewidth=1,      # 线条粗细适中
        alpha=0.85        # 透明度，更柔和
    )

# 样式美化
plt.xlabel("波长 (nm)", fontsize=10, fontweight='bold')
plt.ylabel("光谱反射值", fontsize=10, fontweight='bold')
plt.title("大豆近红外光谱曲线图", fontsize=13, fontweight='bold', pad=20)

plt.grid(True, alpha=0.3, linestyle='--')  # 柔和网格
plt.legend(loc='upper right', frameon=True, fontsize=13)  # 图例不挤

plt.tight_layout()
plt.show()