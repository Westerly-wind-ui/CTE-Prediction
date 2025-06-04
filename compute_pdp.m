% 假设 rf_model 是已经训练好的随机森林模型
% val_X 是 174x8 的验证数据集，其中每一列是一个特征
 
% 定义部分依赖值计算的辅助函数
function pdp_values = compute_pdp(model, dataset, feature1_idx, feature2_idx, grid_points)
    % model: 训练好的模型
    % dataset: 验证数据集
    % feature1_idx, feature2_idx: 要计算部分依赖的特征索引
    % grid_points: 每个特征的网格点数
    
    % 获取数据集的维度
    [num_samples, num_features] = size(dataset);
    
    % 初始化部分依赖值矩阵
    pdp_values = zeros(grid_points, grid_points);
    
    % 生成特征网格
    [feature1_grid, feature2_grid] = ndgrid(linspace(min(dataset(:, feature1_idx)), max(dataset(:, feature1_idx)), grid_points), ...
                                             linspace(min(dataset(:, feature2_idx)), max(dataset(:, feature2_idx)), grid_points));
    
    % 遍历网格点，计算部分依赖值
    for i = 1:grid_points
        for j = 1:grid_points
            % 固定特征值，其他特征使用验证数据集中的值（这里简单处理为使用对应列的中位数填充）
            fixed_features = dataset;
            median_features = median(dataset); % 计算中位数
            fixed_features(:, feature1_idx) = feature1_grid(i, j);
            fixed_features(:, feature2_idx) = feature2_grid(i, j);
            
            % 为了保持其他特征不变，我们可以将除了这两个特征之外的所有特征都设置为它们的中位数
            % 注意：这种方法可能不是最优的，因为它忽略了特征之间的相关性。在实际应用中，可能需要更复杂的策略。
            for k = 1:num_features
                if k ~= feature1_idx && k ~= feature2_idx
                    fixed_features(:, k) = median_features(k);
                end
            end
            
            % 使用模型进行预测
            predictions = predict(model, fixed_features);
            
            % 计算平均预测值（这里假设是回归问题，如果是分类问题，可能需要不同的处理）
            pdp_values(i, j) = mean(predictions);
        end
    end
end