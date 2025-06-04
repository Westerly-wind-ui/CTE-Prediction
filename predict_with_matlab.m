function predictions = predict_with_matlab(model, data)
    % 使用 MATLAB 中的 TreeBagger 模型对数据进行预测
    % model: MATLAB 的 TreeBagger 模型
    % data: 输入数据，MATLAB double 格式
    predictions = predict(model, data); % 得到预测结果
    %predictions = str2double(predictions); % 将结果转换为 MATLAB 数组
end
