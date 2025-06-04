import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from pygam import LinearGAM, s  # 导入GAM库

# 设置Matplotlib样式
sns.set_style("whitegrid")  # 改为更清晰的网格背景
plt.rcParams.update({'font.size': 16, 'axes.labelpad': 10})  # 统一字体大小和标签间距

# 定义保存路径
output_folder = r"D:\matlab工具箱\代码\Github\SHAP 1d"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载数据
X = np.load(r'X_py_all2.npy')
print(X.shape)
feature_names = ['ABL', 'CVS', 'CR', 'UCPA', 'TE', 'ONA', 'TNE', 'CCN', 'BG']
X_py = pd.DataFrame(X, columns=feature_names)

# 加载SHAP值
shap_values = np.load(r'shap_values_all.npy')
shap_values_selected = shap_values[:, :]
shap_values_df = pd.DataFrame(shap_values_selected, columns=feature_names)

# 创建Explanation对象
shap_explanation = shap.Explanation(
    values=shap_values_selected,
    base_values=np.zeros(shap_values_selected.shape[0]),
    data=X_py.values
)


def plot_shap_dependence(shap_explanation, X_py):
    for feature in X_py.columns:
        # 数据清洗
        x = X_py[feature].replace([np.inf, -np.inf], np.nan).dropna()
        y = shap_values_df[feature].replace([np.inf, -np.inf], np.nan).dropna()

        if len(x) < 3 or len(y) < 3:  # 至少需要3个点才能绘制回归线
            print(f"Skipping {feature} due to insufficient data.")
            continue

        # 创建画布
        fig = plt.figure(figsize=(8, 6), dpi=300)
        grid = gridspec.GridSpec(4, 4, hspace=0.3, wspace=0.3)

        # 主绘图区
        ax = fig.add_subplot(grid[1:, :-1])

        # 使用GAM拟合SHAP值与特征参数的关系
        gam = LinearGAM(s(0, lam=0.1)).fit(x.values.reshape(-1, 1), y.values)  # 调整lam值
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = gam.predict(x_fit.reshape(-1, 1))
        y_conf = gam.confidence_intervals(x_fit.reshape(-1, 1), width=0.95)

        # 找到蓝色曲线与纵坐标0的所有交点作为tipping points
        tipping_points = []
        for i in range(len(y_fit) - 1):
            if (y_fit[i] < 0 and y_fit[i + 1] >= 0) or (y_fit[i] > 0 and y_fit[i + 1] <= 0):
                # 线性插值找到交点
                x_interp = x_fit[i] + (x_fit[i + 1] - x_fit[i]) * (-y_fit[i]) / (y_fit[i + 1] - y_fit[i])
                tipping_points.append(x_interp)

        # 区域着色
        # 添加背景颜色区域
        for i in range(len(tipping_points) + 1):
            if i == 0:
                left = x.min()
                right = tipping_points[0] if len(tipping_points) > 0 else x.max()
            elif i == len(tipping_points):
                left = tipping_points[-1]
                right = x.max()
            else:
                left = tipping_points[i - 1]
                right = tipping_points[i]

            # 判断左侧区域的SHAP值正负
            if y_fit[(x_fit >= left) & (x_fit <= right)].mean() > 0:
                color = '#6497B1'  # 正的SHAP值用蓝色
            else:
                color = '#FFA07A'  # 负的SHAP值用橙色

            ax.axvspan(left, right, color=color, alpha=0.3, zorder=1)  # 背景颜色区域
        # 确保区域覆盖整个图
        ax.set_xlim(x.min(), x.max())  # 确保x轴范围覆盖整个数据范围

        # 绘制散点图
        #ax.scatter(x, y, s=12, alpha=0.7, color="#1F77B4", edgecolor='white', linewidth=0.5, zorder=3)

        # 绘制GAM拟合曲线和置信区间
        ax.plot(x_fit, y_fit, color="#1F77B4", linewidth=2, label="GAM Fit", zorder=4)
        ax.fill_between(x_fit, y_conf[:, 0], y_conf[:, 1], color="#1F77B4", alpha=0.2, label="95% CI", zorder=2)

        # 添加基线（SHAP基值）
        ax.axhline(0, color='black', linestyle='--', linewidth=1, zorder=5)

        # 添加所有tipping points的标注
        for tipping_point in tipping_points:
            ax.axvline(tipping_point, color='gray', linestyle='--', linewidth=1, zorder=5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=5)
            ax.scatter(tipping_point, 0, color='darkred', s=100, zorder=5)

        # 设置坐标轴和标题
        ax.set_xlabel(feature, fontsize=16, labelpad=10)
        ax.set_ylabel('SHAP Value', fontsize=16, labelpad=10)
        # ax.set_title(f'SHAP Dependence for {feature}', fontsize=14, pad=20)

        # 添加统计信息
        r2 = r2_score(y, gam.predict(x.values.reshape(-1, 1)))
        p_val = pearsonr(x, y)[1]
        p_text = f"p={p_val:.3g}" if p_val >= 0.01 else f"p<0.01"
        ax.text(
            0.95, 0.05,
            f'R²={r2:.2f}\n{p_text}',
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=None  # 去掉白色背景框
        )

        # 美化坐标轴
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(False)  # 去掉背景网格线

        # 保存图片
        output_path = os.path.join(output_folder, f'shap_dependence_{feature}.png')
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Generated {output_path}")

plot_shap_dependence(shap_explanation, X_py)