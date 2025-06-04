import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# 设置 Matplotlib 样式
sns.set_style("whitegrid")  # 改为更清晰的网格背景
plt.rcParams.update({'font.size': 20, 'axes.labelpad': 15})  # 统一字体大小和标签间距

# 定义保存路径
output_folder = r"D:\matlab工具箱\代码\Github\SHAP 2d"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载数据
X = np.load(r'X_py_all2.npy')
print("Feature matrix shape:", X.shape)
feature_names = ['ABL', 'CVS', 'CR', 'UCPA', 'TE', 'ONA', 'TNE', 'CCN', 'BG']
X_py = pd.DataFrame(X, columns=feature_names)

# 加载 SHAP 值
shap_values = np.load(r'shap_values_all.npy')
shap_values_selected = shap_values[:, :]
shap_values_df = pd.DataFrame(shap_values_selected, columns=feature_names)  # 定义 shap_values_df

# 创建 Explanation 对象
shap_explanation = shap.Explanation(
    values=shap_values_selected,
    base_values=np.zeros(shap_values_selected.shape[0]),
    data=X_py.values
)

def plot_shap_interaction(shap_explanation, X_py, shap_values_df, feature_names):
    # 生成所有特征对
    feature_pairs = list(itertools.combinations(range(len(feature_names)), 2))
    for i, j in feature_pairs:
        feature_i = feature_names[i]
        feature_j = feature_names[j]

        # 数据清洗
        x_i = X_py.iloc[:, i].replace([np.inf, -np.inf], np.nan).dropna()
        x_j = X_py.iloc[:, j].replace([np.inf, -np.inf], np.nan).dropna()
        shap_i = shap_values_df.iloc[:, i].replace([np.inf, -np.inf], np.nan).dropna()
        shap_j = shap_values_df.iloc[:, j].replace([np.inf, -np.inf], np.nan).dropna()

        # 确保数据对齐
        valid_indices = x_i.index.intersection(shap_i.index).intersection(x_j.index).intersection(shap_j.index)
        x_i = x_i[valid_indices]
        x_j = x_j[valid_indices]
        shap_i = shap_i[valid_indices]
        shap_j = shap_j[valid_indices]

        # 计算交互 SHAP 值
        interactive_shap = shap_i * shap_j  # 直接计算交互值，保留正负符号

        # 如果数据长度不匹配，则跳过
        if len(x_i) < 3 or len(interactive_shap) < 3:
            print(f"Skipping {feature_i}-{feature_j} due to insufficient data.")
            continue

        # 创建画布
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)  # 增大图形大小

        # 绘制散点图，点的大小和颜色由特征 j 决定
        cmapper = plt.get_cmap('viridis')  # 使用 viridis 颜色映射
        norm = plt.Normalize(x_j.min(), x_j.max())
        colors = cmapper(norm(x_j.values))

        # 增大点的大小
        sc = ax.scatter(x_i, interactive_shap, c=x_j, cmap='viridis', alpha=0.7, edgecolor='white', s=220, zorder=3)

        # 添加基线
        ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=5)  # 增大基线线宽

        # 设置坐标轴和标题
        ax.set_xlabel(feature_i, fontsize=32, labelpad=20)  # 增大字体和标签间距
        ax.set_ylabel(f'SHAP Interaction for {feature_i} and {feature_j}', fontsize=26, labelpad=20)  # 增大字体和标签间距
        #ax.set_title(f'{feature_i} vs. {feature_j} SHAP Interaction', fontsize=22, pad=20)  # 增大标题字体

        # 美化坐标轴
        ax.tick_params(axis='both', which='major', labelsize=26, width=2, length=8)  # 增大字体和线条粗细、长度

        # 关闭网格线
        ax.grid(False)  # 去掉背景网格线（可选）

        # 添加颜色条
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, aspect=20)
        cbar.set_label(feature_j, rotation=270, labelpad=25, fontsize=26)  # 增大字体
        cbar.ax.tick_params(labelsize=14, width=2, length=8)  # 增大颜色条刻度字体和线条粗细
        cbar.outline.set_visible(False)  # 去掉颜色条的边框

        # 保存图片为 PNG 格式
        output_path = os.path.join(output_folder, f'shap_interaction_{feature_i}_{feature_j}.png')
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Generated {output_path}")

# 调用函数
plot_shap_interaction(shap_explanation, X_py, shap_values_df, feature_names)