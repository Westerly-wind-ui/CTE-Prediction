clc;clear;close all;	
load('R_09_Jan_2025_16_20_20.mat')	
random_seed=G_out_data.random_seed ;  %界面设置的种子数 	
rng(random_seed)  %固定随机数种子 	
	
data_str="F:\处理中\6-二元热膨胀系数机器学习研究\2024.12-25-AFM投稿\数据\数据\Dataset-binary（数据清洗后）.xlsx";  %读取数据的路径 	
dataO=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO(:,2:end);test_data=table2cell(dataO(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
     end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	

 %% 数值类型数据处理	
if length(index_double)>=1	
    data_numshuju=table2array(data1(:,index_double));	
    index_double1=index_double;	
	
    index_double1_index=1:size(data_numshuju,2);	
    data_NAN=(isnan(data_numshuju));    %找列的缺失值	
    num_NAN_ROW=sum(data_NAN);	
    index_NAN=num_NAN_ROW>round(0.2*size(data1,1));	
    index_double1(index_NAN==1)=[]; index_double1_index(index_NAN==1)=[];	
    data_numshuju1=data_numshuju(:,index_double1_index);	
    data_NAN1=(isnan(data_numshuju1));  %找行的缺失值	
    num_NAN__COL=sum(data_NAN1');	
    index_NAN1=num_NAN__COL>0;	
    index_double2_index=1:size(data_numshuju,1);	
    index_double2_index(index_NAN1==1)=[];	
    data_numshuju2=data_numshuju1(index_double2_index,:);	
    index_need_last=index_double1;	
 else	
    index_need_last=[];	
    data_numshuju2=[];	
end	
%% 文本类型数据处理	
	
data_shuju=[];	
 if length(index_char)>=1	
  for j=1:length(index_char)	
    data_get=table2array(data1(index_double2_index,index_char(j)));	
    data_label=unique(data_get);	
    if j==length(index_char)	
       data_label_str=data_label ;	
    end    	
	
     for NN=1:length(data_label)	
            idx = find(ismember(data_get,data_label{NN,1}));  	
            data_shuju(idx,j)=NN; 	
     end	
  end	
 end	
label_all_last=[index_char,index_need_last];	
[~,label_max]=max(label_all_last);	
 if(label_max==length(label_all_last))	
     str_label=0; %标记输出是否字符类型	
     data_all_last=[data_shuju,data_numshuju2];	
     label_all_last=[index_char,index_need_last];	
 else	
    str_label=1;	
    data_all_last=[data_numshuju2,data_shuju];	
    label_all_last=[index_need_last,index_char];     	
 end	
 data=data_all_last;	
 data_biao_all=data1.Properties.VariableNames;	
 for j=1:length(label_all_last)	
    data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
 end	
	
% 异常值检测	
data=data;	
	
%%  特征处理 特征选择或者降维	
	
 A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=9;   %特征选择的个数	
	
print_index_name=[]; 	
[xiangguan,~]=corr(A_data1,'Type','Pearson'); %Pearson 相关性分析 	
index_name=data_biao1;	
 y_index = index_name; x_index=index_name;  	
 figure(1)	
 HH=heatmap(x_index,y_index,( xiangguan), 'FontSize',12, 'FontName','宋体'); 	
feature_need=abs(xiangguan); feature_need(isnan(feature_need))=0; index_negative=find(xiangguan(end,1:end-1)<0);	
[~,sort_feature]=sort(feature_need(end,1:end-1),'descend'); 	
for NN=1:select_feature_num 	
     print_index_name{1,NN}=index_name{1,sort_feature(NN)}; 	
end 	
disp('选择特征');disp(print_index_name) 	
feature_need_last=sort_feature(1:select_feature_num);    %选择特征的位置 	
data_select=[A_data1(:,feature_need_last),A_data1(:,end)];  %经过特征选择后的数据 	
color_get= G_out_data.color_get;  %热图	
colormap(HH,color_get)	
	
	
	
%% 数据划分	
x_feature_label=data_select(:,1:end-1);    %x特征	
y_feature_label=data_select(:,end);          %y标签	
index_label1=1:(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  % 数据索引	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 测试数据标准化  	
	
%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
attention_label=G_out_data.attention_label;   %注意力机制标签	
attention_head=G_out_data.attention_head;   %注意力机制设置	
	
%% 数据增强部分	
get_mutiple=G_out_data.get_mutiple;  %数据增加倍数	
methodchoose=1; 	
origin_data=[train_x_feature_label_norm,train_y_feature_label_norm;vaild_x_feature_label_norm,vaild_y_feature_label_norm]; 	
	
[SyntheticData,Synthetic_label,origin_data_label]=generate_regressdata(origin_data,methodchoose,get_mutiple);	
% 绘制生成后数据样本图	
figure_data_generate(origin_data,SyntheticData,origin_data_label,Synthetic_label)	
X_new_DATA=[origin_data;SyntheticData];             %生成的X特征数据	
Y_new_DATA=[origin_data_label;Synthetic_label];  %生成的Y标签数据	
	
syn_spilt=round(spilt_ri(1)/(spilt_ri(1)+spilt_ri(2))*length(Y_new_DATA));	
syn_index=randperm(length(Y_new_DATA));	
%以下将生成的数据随机分配到训练集和验证集中	
train_x_feature_label_norm=X_new_DATA(syn_index(1:syn_spilt),1:end-1);	
vaild_x_feature_label_norm=X_new_DATA(syn_index(syn_spilt+1:end),1:end-1);	
train_y_feature_label_norm=X_new_DATA(syn_index(1:syn_spilt),end);	
vaild_y_feature_label_norm=X_new_DATA(syn_index(syn_spilt+1:end),end);	
train_y_feature_label=train_y_feature_label_norm.*y_sig+y_mu;	
vaild_y_feature_label=vaild_y_feature_label_norm.*y_sig+y_mu;	
train_x_feature_label=train_x_feature_label_norm.*x_sig+x_mu;	
vaild_x_feature_label=vaild_x_feature_label_norm.*x_sig+x_mu;	
	
%数据生成输出数据	
train_x_feature_label_aug=(train_x_feature_label_norm.*x_sig)+x_mu;	
vaild_x_feature_label_aug=(vaild_x_feature_label_norm.*x_sig)+x_mu;	
%总体生成数据+原数据保存在以下的 augdata_all 数据里面	
augdata_all=[train_x_feature_label_aug,train_y_feature_label;vaild_x_feature_label_aug,vaild_y_feature_label;test_x_feature_label,test_y_feature_label];	

%% 算法处理块	


disp('MLP回归')	
t1=clock; 	
hidden_size=G_out_data.hidden_size1;    %神经网络隐藏层	
	
[Mdl,fitness,Convergence_curve] = optimize_fitrMLP(train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm,num_pop,num_iter,method_mti);    	
y_train_predict_norm=predict(Mdl,train_x_feature_label_norm);  %训练集预测结果	
y_vaild_predict_norm=predict(Mdl,vaild_x_feature_label_norm);  %验证集预测结果	
y_test_predict_norm=predict(Mdl,test_x_feature_label_norm);  %测试集预测结果
%X_all_predict_norm = predict(Mdl,X_all_norm);
% 创建 X_all_norm 变量
X_all_norm = [train_x_feature_label_norm; vaild_x_feature_label_norm; test_x_feature_label_norm];


%计算所有数据的shap值
shap_values_all = calculate_shap(X_all_norm,Mdl);
shap_interaction_values = compute_shap_interaction_values2(Mdl,X_all_norm);

% shap_values_all = calculate_shap(test_x_feature_label_norm,Mdl);
% shap_interaction_values = compute_shap_interaction_values(Mdl,train_x_feature_label_norm);
% test_x_feature_label_norm_py = py.numpy.array(test_x_feature_label_norm);
% py.numpy.save('test_x_feature_label_norm_py.npy', test_x_feature_label_norm_py);
t2=clock;	
Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));       	
	
	
	
 y_train_predict=y_train_predict_norm*y_sig+y_mu;  %反标准化操作 	
 y_vaild_predict=y_vaild_predict_norm*y_sig+y_mu; 	
 y_test_predict=y_test_predict_norm*y_sig+y_mu; 

 y_train_pred_all_py = py.numpy.array(y_train_predict); 
 py.numpy.save('y_train_pred_all_py.npy', y_train_pred_all_py);
 train_y_feature_label_py = py.numpy.array(train_y_feature_label);
 py.numpy.save('train_y_feature_label_py.npy',train_y_feature_label_py);
 
 y_vaild_pred_all_py = py.numpy.array(y_vaild_predict);
 py.numpy.save('y_vaild_pred_all_py.npy', y_vaild_pred_all_py);
 vaild_y_feature_label_py = py.numpy.array(vaild_y_feature_label);
 py.numpy.save('vaild_y_feature_label_py.npy',vaild_y_feature_label_py);

 y_test_pred_all_py = py.numpy.array(y_test_predict);
 py.numpy.save('y_test_pred_all_py.npy', y_test_pred_all_py);
 test_y_feature_label_py = py.numpy.array(test_y_feature_label);
 py.numpy.save('test_y_feature_label_py.npy',test_y_feature_label_py);
 train_y=train_y_feature_label; disp('***************************************************************************************************************')   	
 train_MAE=sum(abs(y_train_predict-train_y))/length(train_y) ; disp(['训练集平均绝对误差MAE：',num2str(train_MAE)])	
 train_MAPE=sum(abs((y_train_predict-train_y)./train_y))/length(train_y); disp(['训练集平均相对误差MAPE：',num2str(train_MAPE)])	
 train_MSE=(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方误差MSE：',num2str(train_MSE)]) 	
 train_RMSE=sqrt(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方根误差RMSE：',num2str(train_RMSE)]) 	
 train_R2= 1 - (norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp(['训练集R方系数R2：',num2str(train_R2)]) 	
 vaild_y=vaild_y_feature_label;disp('***************************************************************************************************************')	
 vaild_MAE=sum(abs(y_vaild_predict-vaild_y))/length(vaild_y) ; disp(['验证集平均绝对误差MAE：',num2str(vaild_MAE)])	
 vaild_MAPE=sum(abs((y_vaild_predict-vaild_y)./vaild_y))/length(vaild_y); disp(['验证集平均相对误差MAPE：',num2str(vaild_MAPE)])	
 vaild_MSE=(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方误差MSE：',num2str(vaild_MSE)])     	
 vaild_RMSE=sqrt(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方根误差RMSE：',num2str(vaild_RMSE)]) 	
 vaild_R2= 1 - (norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);    disp(['验证集R方系数R2:  ',num2str(vaild_R2)])			
 test_y=test_y_feature_label;disp('***************************************************************************************************************');   	
 test_MAE=sum(abs(y_test_predict-test_y))/length(test_y) ; disp(['测试集平均绝对误差MAE：',num2str(test_MAE)])        	
 test_MAPE=sum(abs((y_test_predict-test_y)./test_y))/length(test_y); disp(['测试集平均相对误差MAPE：',num2str(test_MAPE)])	
 test_MSE=(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方误差MSE：',num2str(test_MSE)]) 	
 test_RMSE=sqrt(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方根误差RMSE：',num2str(test_RMSE)]) 	
 test_R2= 1 - (norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp(['测试集R方系数R2：',num2str(test_R2)]) 	
 disp(['算法运行时间Time: ',num2str(Time)])	
 disp("已保存shap值，请调用python画图")
 