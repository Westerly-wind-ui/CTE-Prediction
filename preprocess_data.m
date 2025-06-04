function [data_all_last, label_all_last, data_biao] = preprocess_data(data1, data2)
    % 获取数据中的特征名
    test_data = table2cell(data1(1, 1:end));
    
    % 判断每列数据类型（字符类型、数值类型等）
    index_la = zeros(1, length(test_data));
    for i = 1:length(test_data)    
        if ischar(test_data{1, i})
            index_la(i) = 1;  % char 类型
        elseif isnumeric(test_data{1, i})
            index_la(i) = 2;  % double 类型
        else
            index_la(i) = 0;  % 其他类型
        end
    end
    
    % 获取字符型和数值型数据的列索引
    index_char = find(index_la == 1);
    index_double = find(index_la == 2);
    
    % 数值类型数据处理
    if length(index_double) >= 1
        data_numshuju = table2array(data1(:, index_double));    
        index_double1 = index_double;
        
        index_double1_index = 1:size(data_numshuju, 2);    
        data_NAN = isnan(data_numshuju);    % 找列的缺失值    
        num_NAN_ROW = sum(data_NAN);    
        index_NAN = num_NAN_ROW > round(0.2 * size(data1, 1));    
        index_double1(index_NAN == 1) = [];
        index_double1_index(index_NAN == 1) = [];    
        
        data_numshuju1 = data_numshuju(:, index_double1_index);    
        data_NAN1 = isnan(data_numshuju1);  % 找行的缺失值    
        num_NAN_COL = sum(data_NAN1');    
        index_NAN1 = num_NAN_COL > 0;    
        index_double2_index = 1:size(data_numshuju, 1);    
        index_double2_index(index_NAN1 == 1) = [];    
        
        data_numshuju2 = data_numshuju1(index_double2_index, :);    
        index_need_last = index_double1;    
    else    
        index_need_last = [];    
        data_numshuju2 = [];    
    end
    
    % 文本类型数据处理
    data_shuju = [];
    if length(index_char) >= 1    
        for j = 1:length(index_char)    
            data_get = table2array(data1(index_double2_index, index_char(j)));    
            data_label = unique(data_get);    
            if j == length(index_char)    
                data_label_str = data_label;    
            end    
            
            for NN = 1:length(data_label)    
                idx = find(ismember(data_get, data_label{NN, 1}));    
                data_shuju(idx, j) = NN;    
            end    
        end    
    end
    
    % 合并数值型和文本型数据
    label_all_last = [index_char, index_need_last];
    [~, label_max] = max(label_all_last);    
    if label_max == length(label_all_last)    
        str_label = 0; % 标记输出是否字符类型    
        data_all_last = [data_shuju, data_numshuju2];    
        label_all_last = [index_char, index_need_last];    
    else    
        str_label = 1;    
        data_all_last = [data_numshuju2, data_shuju];    
        label_all_last = [index_need_last, index_char];    
    end
    
    % 更新变量名
    data_biao_all = data1.Properties.VariableNames;    
    for j = 1:length(label_all_last)    
        data_biao{1, j} = data_biao_all{1, label_all_last(j)};    
    end
end
