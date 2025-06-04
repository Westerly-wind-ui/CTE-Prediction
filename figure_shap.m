%function [] = figure_shap(i,x_featur_train,model)

%shap_fig = py.importlib.import_module('shap');
%X_py = py.numpy.array(x_featur_train); % 将 MATLAB 数据转换为 numpy 数组
%shap_values = py.shap.TreeExplainer(model).shap_values(X_py);
%py.shap_fig.summary_plot(shap_values, X_py,'show', py.False);
%py.matplotlib.pyplot.savefig(strcat(pwd,'\shap_summary_plot_',num2str(i),'.png'));  % 保存图像

% 在 MATLAB 中加载并显示图像
%figure('Name', 'SHAP Summary Plot'); % 自定义图窗口名称
%imshow('shap_summary_plot.png');
%end

function [] = figure_shap(i, x_feature_train, model)
     % 将 model 设置为全局变量，使其在 MATLAB 会话中可访问
    assignin('base', 'your_model', model);  % 将模型传递到 MATLAB 基础工作区

    % 导入 Python 模块
    shap_module = py.importlib.import_module('shap_predict_with_matlab');
    
    % 将输入特征转换为 numpy 格式
    X_py = py.numpy.array(x_feature_train); 

    % 通过 compute_shap_values 函数计算 SHAP 值
    shap_values = shap_module.compute_shap_values(model,X_py);

    % 生成并保存 SHAP 总结图
    py.shap.summary_plot(shap_values, X_py, 'show', py.False);
    output_path = strcat(pwd, '\shap_summary_plot_', num2str(i), '.png'); % 设置输出路径
    py.matplotlib.pyplot.savefig(output_path);  % 保存图像


    % 在 MATLAB 中加载并显示图像
    % figure('Name', 'SHAP Summary Plot'); % 自定义图窗口名称
    % imshow(output_path);
end





%function [] = figure_shap(i, x_featur_train, y_train)

% 导入 Python 模块
%shap_fig = py.importlib.import_module('shap');
%sklearn = py.importlib.import_module('sklearn.ensemble');

% 将数据转换为 numpy 格式
%X_py = py.numpy.array(x_featur_train);
%y_py = py.numpy.array(y_train);

% 在 Python 中定义并训练模型
%model_py = sklearn.RandomForestRegressor(pyargs('n_estimators', 100)); % 创建随机森林模型
%model_py.fit(X_py, y_py); % 在 Python 中训练模型

% 使用 shap 计算 SHAP 值
%explainer = shap_fig.TreeExplainer(model_py);
%shap_values = explainer.shap_values(X_py);

% 生成并保存 SHAP 总结图
%py.shap.summary_plot(shap_values, X_py, 'show', py.False);
%output_path = strcat(pwd, '\shap_summary_plot_', num2str(i), '.png'); % 设置输出路径
%py.matplotlib.pyplot.savefig(output_path);  % 保存图像

% 在 MATLAB 中加载并显示图像
% figure('Name', 'SHAP Summary Plot'); % 自定义图窗口名称
% imshow(output_path);

%end
