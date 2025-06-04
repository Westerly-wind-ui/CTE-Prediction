import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def shap_force_plot(shap_values, base_value, sample_data, feature_names, save_path):
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})  # 继续增大字体
    plt.figure(figsize=(12, 5))  # 继续加宽画布
    plt.tight_layout(pad=4)  # 进一步增加边距

    # 创建Explanation对象（自动格式化数值）
    explainer = shap.Explanation(
        values=np.round(shap_values, 2),  # SHAP值保留两位小数
        base_values=np.round(base_value, 2),
        data=np.round(sample_data, 2),  # 特征值保留两位小数
        feature_names=feature_names
    )

    # 生成力图（调整图像尺寸）
    plt.figure(figsize=(10, 4), dpi=200)  # 加宽画布，提高分辨率
    shap.plots.force(explainer, matplotlib=True, show=False)

    # 获取当前图中的所有标注
    ax = plt.gca()
    texts = ax.texts

    # 手动调整 f(x) 和 base value 的位置
    for text in texts:
        if text.get_text() == "f(x)":  # 找到 f(x) 标注
            current_x, current_y = text.get_position()
            new_y = current_y - 0.7  # 向下移动
            text.set_position((current_x, new_y))
            text.set_bbox(dict(facecolor='none', edgecolor='none', boxstyle='square'))
        elif text.get_text() == "base value":  # 找到 base value 标注
            current_x, current_y = text.get_position()
            new_y = current_y - 0.7  # 向下移动
            text.set_position((current_x, new_y))
            text.set_bbox(dict(facecolor='none', edgecolor='none', boxstyle='square'))


    # 手动设置坐标轴范围防止截断
    ax = plt.gca()
    ax.set_xlim(left=explainer.base_values - 1, right=explainer.base_values + np.sum(explainer.values) + 1)

    # 调整布局并保存
    plt.tight_layout(pad=3)  # 增加边距
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == '__main__':
    # 加载数据（示例路径）
    test_data = np.load(r'D:\matlab工具箱\代码\Github\X_py_all2.npy')
    base_value = np.load(r'D:\matlab工具箱\代码\Github\expected_value_all_py.npy').item()
    shap_values = np.load(r'D:\matlab工具箱\代码\Github\shap_values_all.npy')

    # 参数设置
    feature_names = ['CVS', 'ABL', 'CR', 'UCPA', 'CCN', 'ONA', 'TNE', 'TE', 'BG']
    sample_idx = 11  #第一个样本
    sample_data = test_data[sample_idx]

    # 格式化数据
    sample_shap = np.round(shap_values[sample_idx], 2) if len(shap_values.shape) == 2 else np.round(shap_values, 2)
    sample_data = np.round(sample_data, 2)
    base_value = np.round(base_value, 2)

    # 生成图像的保存路径（为每个样本生成一个文件，包含样本索引）
    save_path = fr'D:\matlab工具箱\代码\Github\SHAP heatmap\shap_force_sample_{sample_idx}.png'

    # 生成并保存图像
    shap_force_plot(sample_shap, base_value, sample_data, feature_names, save_path)

    print(f"Sample {sample_idx} processed and saved to {save_path}")