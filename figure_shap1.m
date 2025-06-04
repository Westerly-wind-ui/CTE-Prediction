function model_py = figure_shap1(i, x_featur_train, y_train, num_trees, min_leaf_size)  
    % 导入 Python 模块  
    shap_fig = py.importlib.import_module('shap');  
    sklearn = py.importlib.import_module('sklearn.ensemble');  
  
    % 将数据转换为 numpy 格式  
    X_py = py.numpy.array(x_featur_train);  
    y_py = py.numpy.array(y_train);  
    % 注意：X_test_py 和 Y_test_py 在后续代码中未使用，如果不需要可以移除  
    % X_test_py = py.numpy.array(x_test);  
    % Y_test_py = py.numpy.array(y_test);  
  
    % 在 Python 中定义并训练模型，对齐 MATLAB 的 TreeBagger 参数  
    model_py = sklearn.RandomForestRegressor(pyargs(...  
        'n_estimators', int32(num_trees), ...        % 树的数量  
        'min_samples_leaf', int32(min_leaf_size) ... % 最小叶子大小  
    ));  
    model_py.fit(X_py, y_py); % 在 Python 中训练模型  
  
    % 使用 shap 计算 SHAP 值（这部分代码在原始函数中未完全使用，但保留以供参考）  
    explainer = shap_fig.TreeExplainer(model_py);  
    shap_values = explainer.shap_values(X_py);  
  
    % 保存为 .npy 文件（这部分代码在原始函数中未使用，但可以根据需要保留或移除）  
    X_py_name = strcat('X_py_', num2str(i), '.npy');  
    shap_values_name = strcat('shap_values_', num2str(i), '.npy');  
    py.numpy.save(X_py_name, X_py);  
    py.numpy.save(shap_values_name, shap_values);  
  
    % 注意：以下与绘图相关的代码已被注释掉，因为原始函数没有要求返回图像  
    % 生成并保存 SHAP 总结图（这部分代码可以根据需要取消注释）  
    % py.shap.summary_plot(shap_values, X_py, 'show', py.False);  
    % output_path = strcat(pwd, '\shap_summary_plot_', num2str(i), '.png');  
    % py.matplotlib.pyplot.savefig(output_path);  
  
    % 注意：以下 MATLAB 绘图代码与 Python 生成的图像无关，已被注释掉  
    % figure('Name', 'SHAP Summary Plot');  
    % imshow 不能直接用于显示 .png 文件，应使用 imread 和 imagesc 或其他方法  
    % imshow(output_path); % 这行代码是错误的，因为 imshow 不接受文件路径作为输入  
end
