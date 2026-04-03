import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import streamlit as st

# ==========================
# Streamlit 页面初始化
# ==========================
plt.rcParams['font.family']=['sans-serif']
plt.rcParams['font.sans-serif']=['SimHei']
st.title("大豆蛋白含量预测")
st.subheader("基于光谱数据 + 深度学习（1D-CNN）模型")

# =====================
# 固定随机种子
# =====================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =====================
# 1. 读取并清洗数据
# =====================
df = pd.read_csv("LegumesCombinedNitrogenV2.csv")
df = df[df["Species"] == "Soybean"].copy()

wavelength_cols = [col for col in df.columns if str(col).isdigit()]
for c in wavelength_cols:
    df[c] = df[c].replace('.', np.nan)
    df[c] = pd.to_numeric(df[c], errors='coerce')

df["CP"] = pd.to_numeric(df["CP"], errors='coerce')
df = df.dropna()

# =====================
# 2. 数据准备
# =====================
X = df[wavelength_cols].values.astype(np.float32)
y = df["CP"].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 1, X_train.shape[1])
X_test = X_test.reshape(-1, 1, X_test.shape[1])

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# =====================
# 模型定义
# =====================
class SpectraNet(nn.Module):
    def __init__(self, input_len):
        super(SpectraNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(0.1)

        with torch.no_grad():
            dummy = torch.randn(1, 1, input_len)
            dummy = self.pool(torch.relu(self.conv1(dummy)))
            dummy = self.pool(torch.relu(self.conv2(dummy)))
            self.fc_in = dummy.numel()

        self.fc1 = nn.Linear(self.fc_in, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =====================
# 训练模型
# =====================
model = SpectraNet(X_train.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

train_losses = []
test_losses = []

for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    pred_train = model(X_train)
    loss_train = criterion(pred_train, y_train)
    loss_train.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(X_test)
        loss_test = criterion(pred_test, y_test)

    train_losses.append(loss_train.item())
    test_losses.append(loss_test.item())

# =====================
# 训练误差曲线
# =====================
st.markdown("### 训练误差曲线")
fig1, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(train_losses, label='训练误差')
ax1.plot(test_losses, label='测试误差')
ax1.set_title("训练误差曲线")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# =====================
# 模型评估
# =====================
model.eval()
with torch.no_grad():
    yp_train = model(X_train).numpy()
    yp_test = model(X_test).numpy()

r2 = r2_score(y_test.numpy(), yp_test)
rmse = np.sqrt(mean_squared_error(y_test.numpy(), yp_test))

# =====================
# 散点拟合图（你要的图！）
# =====================
st.markdown("### 训练集 vs 测试集 拟合曲线图")
fig2, ax2 = plt.subplots(figsize=(6, 6))

sns.scatterplot(x=y_train.numpy().ravel(), y=yp_train.ravel(), label='训练集', alpha=0.7, s=50, ax=ax2)
sns.scatterplot(x=y_test.numpy().ravel(), y=yp_test.ravel(), label='测试集', alpha=0.9, s=50, ax=ax2)

# 测试拟合线（黄色）
sns.regplot(x=y_test.numpy().ravel(), y=yp_test.ravel(),
            scatter=False, color='gold', line_kws={'lw':2}, label='测试拟合线', ax=ax2)

# 训练拟合线（蓝色）
sns.regplot(x=y_train.numpy().ravel(), y=yp_train.ravel(),
            scatter=False, color='blue', line_kws={'lw':2}, label='训练拟合线', ax=ax2)

ax2.set_xlabel('True')
ax2.set_ylabel('Predict')
ax2.set_title(f'$R^2$={r2:.3f}  RMSE={rmse:.3f}')
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# =====================
# 测试集拟合曲线
# =====================
st.markdown("### 测试集真实值曲线 vs 测试集预测值曲线")
fig3, ax3 = plt.subplots(figsize=(10, 4))
x_idx = np.arange(len(y_test))
y_real = y_test.numpy().ravel()
y_pred = yp_test.ravel()

sns.lineplot(x=x_idx, y=y_real, linewidth=2.5, label='真实值', color='black', ax=ax3)
sns.lineplot(x=x_idx, y=y_pred, linewidth=2.5, label='预测值', color='red', ax=ax3)
ax3.set_xlabel('测试集样本数')
ax3.set_ylabel('蛋白质含量')
ax3.set_title(f'测试集拟合曲线  $R^2$={r2:.3f}  RMSE={rmse:.3f}')
ax3.legend()
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

# =====================
# 最终预测结果展示
# =====================
st.markdown("---")
st.markdown("## 模型预测结果")

uploaded = st.file_uploader("上传大豆NIR的CSV文件", type="csv")
if uploaded is not None:
    try:
        df_new = pd.read_csv(uploaded, encoding="utf-8")
    except:
        df_new = pd.read_csv(uploaded, encoding="gbk")

    # 自动提取波长列
    wave_cols = [c for c in df_new.columns if str(c).isdigit()]
    X_new = df_new[wave_cols].values.astype(np.float32)

    # 标准化（和训练一致）
    X_new = scaler.transform(X_new)
    X_new = X_new.reshape(-1, 1, X_new.shape[1])
    X_new = torch.tensor(X_new, dtype=torch.float32)

    # 预测
    model.eval()
    with torch.no_grad():
        pred = model(X_new).cpu().numpy().ravel()

    # 只显示最终结果
    for i, p in enumerate(pred):
        st.success(f"样本 {i+1} 蛋白质含量：{p:.2f} %")