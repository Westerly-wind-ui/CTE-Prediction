
% 特征选择与Lasso回归
function [selected_features, print_index_name, selected_features2] = feature_selection(data1, data2, data_biao)
    % 特征选择的个数
    select_feature_num = 8;
    
    % 特征选择
    [B, ~] = lasso(data1(:, 1:end-1), data1(:, end), 'Alpha', 0.01);    
    L_B = (B ~= 0);   
    SL_B = sum(L_B);    
    [~, index_L1] = min(abs(SL_B - select_feature_num));    
    
    feature_need_last = find(L_B(:, index_L1) == 1);    
    data_select = [data1(:, feature_need_last), data1(:, end)];    
    data_select2 = [data2(:, feature_need_last), data2(:, end)];    
    
    % 获取选择的特征名称
    print_index_name = {};    
    for NN = 1:length(feature_need_last)    
        print_index_name{1, NN} = data_biao{1, feature_need_last(NN)};    
    end    
    
    disp('选择特征');    
    disp(print_index_name);
    disp(feature_need_last);
    
    selected_features = data_select;  
    selected_features2 = data_select2;

end