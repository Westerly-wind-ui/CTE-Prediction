import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def shap_heatmap(shap_values,base_values,test_data,feature_names,save_path):
    shap_explanation = shap.Explanation(values=shap_values,base_values=base_values,data=test_data,feature_names = feature_names)
    shap.plots.heatmap(shap_explanation, show=False)
    plt.savefig(save_path, bbox_inches='tight')  # 添加bbox_inches='tight'以确保图像完整

if __name__ == '__main__':
    test_data = np.load(r'D:\matlab工具箱\代码\thermal expansion coefficient\X_py_all2.npy')
    base_values = np.load(r'D:\matlab工具箱\代码\thermal expansion coefficient\expected_value_all_py.npy')
    base_values = np.array([base_values])
    shap_values = np.load(r'D:\matlab工具箱\代码\thermal expansion coefficient\shap_values_all.npy')
    feature_names = ['CVS','ABL','CR','UCPA','CCN','ONA','TNE','TE','BG']
    test_py = pd.DataFrame( test_data, columns=feature_names)
    save_path = r'D:\matlab工具箱\代码\thermal expansion coefficient\SHAP heatmap\shap_heatmap.png'  # 设置保存路径
    shap_heatmap(shap_values, base_values, test_py , feature_names, save_path)