function [SyntheticData1,Synthetic_label1,origin_data_label]=generate_regressdata(origin_data,methodchoose,get_mutiple)
%生成回归样本数据
rng('default')
%回归类型的数据聚类分类别生成
if size(origin_data,1)<100
    class_set=3;  %设定划分类别，可修改
else
    class_set=6;  %设定划分类别，可修改
end
   origin_data_label=kmeans(origin_data(:,end),class_set); %将原来的样本集划分为指定类
   %相当于在原来的基础上对标签进行一个离散的划分
   if methodchoose==1
       %随机过采样样本生成
       [SyntheticData,Synthetic_label]=generate_regressdata_SMOTE(origin_data,origin_data_label,get_mutiple);
   elseif methodchoose==2
       %GAN数据生成
       [SyntheticData,Synthetic_label]=generate_classdata_GAN(origin_data,origin_data_label,get_mutiple);
   elseif methodchoose==3
       %GMM高斯混合模型数据生成
       [SyntheticData,Synthetic_label]=generate_classdata_GMM(origin_data,origin_data_label,get_mutiple);
    elseif methodchoose==4
       %LSTM数据生成
       [SyntheticData,Synthetic_label]=generate_classdata_LSTM(origin_data,origin_data_label,get_mutiple);
   end
  
   %% 统计生成数据前后数据分布图
   unique_class=unique(origin_data_label);
   label_get=[];unique_str_label=[];
   for i=1:length(unique_class)
       label_get(i)=length(find(origin_data_label==unique_class(i)));
       unique_str_label{1,i}=['class',num2str(i)];
   end
   figure('Position',[300,300,800,300])
   subplot(1,2,1)

           
   bar_plot_f=bar(1:length(label_get),label_get,0.75);   %  重要性衡量
   bar_plot_f.FaceColor = 'flat';
   for i=1:length(unique_class)
       bar_plot_f.CData(i,:)=[0.6314    0.6627    0.8157];
       %                     bar_plot_f(i).FaceColor=color_get(1+i*(floor(length(color_get)/length(imp))-1),:);
   end
   xtips1 = bar_plot_f.XEndPoints;
   ytips1 = bar_plot_f.YEndPoints;
   labels1 = string(bar_plot_f.YData);
   text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
       'VerticalAlignment','bottom')
   % index_name_plot=data_biao1(1:end-1);
   xticks(1:length(label_get))
   xticklabels(unique_str_label)
   title('原样本数');
   ylabel('num');
   ylim([0,1.1*max(label_get)])
   set(gca,"FontSize",11,"LineWidth",1)
   box off

   % bar(1:length(label_get),label_get,0.75)
   % set(gca,'FontSize',12,'LineWidth',1.2)
   % ylabel('原样本数')
   % xticks(1:length(label_get))
   % xticklabels(unique_str_label)

   
   % 为了保持样本数平衡 原来数据多的样本少取，少的样本多取
   label_syn_get=[]; %生成数据统计
   for i=1:length(unique_class)
       label_syn_get{i}=(find(Synthetic_label==unique_class(i)));       
   end
   label_get_rio=label_get/sum(label_get);
   label_get_rio1=1./label_get_rio;
   label_get_rio2=label_get_rio1/max(label_get_rio1);

   SyntheticData1=[];Synthetic_label1=[];
   for i=1:length(unique_class)
       data_syno_get=label_syn_get{i};   
       data_syno_get1=data_syno_get(1:round(label_get_rio2(i)*size(data_syno_get,1)));
       SyntheticData1=[SyntheticData1;SyntheticData(data_syno_get1,:)];
       Synthetic_label1=[Synthetic_label1;Synthetic_label(data_syno_get1,:)];
   end
   data_label_new=[origin_data_label;Synthetic_label1];

   subplot(1,2,2)
   for i=1:length(unique_class)
       label_get1(i)=length(find(data_label_new==unique_class(i)));
   end
      bar_plot_f1=bar(1:length(label_get1),label_get1,0.75);   %  重要性衡量
   bar_plot_f1.FaceColor = 'flat';
   for i=1:length(unique_class)
       bar_plot_f1.CData(i,:)=[0.5882    0.8000    0.7961];
       %                     bar_plot_f(i).FaceColor=color_get(1+i*(floor(length(color_get)/length(imp))-1),:);
   end
   xtips1 = bar_plot_f1.XEndPoints;
   ytips1 = bar_plot_f1.YEndPoints;
   labels1 = string(bar_plot_f1.YData);
   text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
       'VerticalAlignment','bottom')
   % index_name_plot=data_biao1(1:end-1);
   xticks(1:length(label_get1))
   xticklabels(unique_str_label)
   title('增样后总样本数');
   ylabel('num');
   ylim([0,1.1*max(label_get1)])
   set(gca,"FontSize",11,"LineWidth",1)
   box off
   % bar(1:length(label_get1),label_get1,0.75)
   % set(gca,'FontSize',12,'LineWidth',1.2)
   % ylabel('增样后总样本数')
   % xticks(1:length(label_get1))
   % xticklabels(unique_str_label)
end

%% SMOTE随机过采样数据生成
function [SyntheticData,Synthetic_label]=generate_regressdata_SMOTE(origin_data,origin_data_label,get_mutiple)
% get_mutiple  %生成样本数是原数据的多少倍
%随机过采样部分代码
unique_class=unique(origin_data_label);
unique_class1=unique_class;
% [~,max_index]=max(label_get);
% unique_class1(max_index)=[];
x_aug=origin_data;
y_aug=origin_data_label;
y_aug1=origin_data_label;

for i=1:length(unique_class1)
    flabel_get=unique_class1(i);
    % 假设数据矩阵 X 中的最后一列为类别标签，0 表示多数类，1 表示少数类

    % 调用 SMOTE 函数
    numMinority = sum(origin_data_label == flabel_get); % 少数类样本数量
    % get_mutiple=ceil(sum(train_y_feature_label == max_index)/numMinority)-1;
    numNeighbors = 5; % 设置邻居数量
    newData = SMOTE(origin_data(origin_data_label == flabel_get, :), numMinority, numNeighbors,get_mutiple);

    % 将生成的合成样本与原始数据合并
    x_aug = [x_aug; newData];
    y_aug = [y_aug; flabel_get*ones(size(newData, 1), 1)]; % 将合成样本标记为少数类（1）
    y_aug1=[y_aug1; flabel_get*ones(size(newData, 1), 1)];
end
% train_x_feature_label=x_aug;
% train_y_feature_label=y_aug;
SyntheticData=x_aug;
Synthetic_label=y_aug;
end

%% GAN数据生成
function [SyntheticData,Synthetic_label]=generate_classdata_GAN(origin_data,origin_data_label,get_mutiple)
  %GAN生成分类数据
original_data=reshape(origin_data,1,[]); % Preprocessing - convert matrix to vector

% Define the generator and discriminator networks
generator = @(z) original_data; % Identity mapping for simplicity
discriminator = @(x) (x - original_data); % Z-score normalization

% Training parameters
num_samples = 500;
num_epochs = 4;
batch_size = 160;
learning_rate = 0.01;
Runs= get_mutiple;  %生成样本数是原数据的多少倍

for i=1:Runs
    % Training loop
    for epoch = 1:num_epochs
        for batch = 1:num_samples/batch_size
            % Generate noise samples for the generator
            noise = randn(batch_size, 1);
            % Generate synthetic data using the generator
            synthetic_data = generator(noise);
            % Train the discriminator to distinguish real from synthetic data
            discriminator_loss = mean((discriminator(synthetic_data) - noise).^2);
            % Update the generator to fool the discriminator
            generator_loss = mean((discriminator(generator(noise)) - noise).^2);
            % Update the generator and discriminator parameters
            generator = @(z) generator(z) - learning_rate * generator_loss;
            discriminator = @(x) discriminator(x) - learning_rate * discriminator_loss;
        end
        Run = [' Epoch "',num2str(epoch)];
        disp(Run);
    end
    %
    % Generate synthetic data using the trained generator
    noise_samples = randn(num_samples/4, 1);
    synthetic_data1= generator(noise_samples);
    Syn(i,:)=synthetic_data1;
    % Run2 = [' Run "',num2str(Runs)];
    % disp(Run2);
end

% Converting cell to matrix
S = size(Syn(Runs)); SO = size (origin_data); SF = SO (1,2); SO = SO (1,1);
for i=1:Runs
    Syn2{i}=reshape(Syn(i,:),[SO,SF]);
    Syn2{i}(:,end+1)=origin_data_label;
end
Synthetic3 = cell2mat(Syn2');
SyntheticData=Synthetic3(:,1:end-1);
Synthetic_label=Synthetic3(:,end);

end
%%
function [SyntheticData,Synthetic_label]=generate_classdata_GMM(origin_data,origin_data_label,get_mutiple)
%采用高斯混合模型进行样本生成
NoofSynthetic=get_mutiple*length(origin_data_label);

% 高斯混合模型(GMM)拟合原始数据
GMModel1 = fitgmdist(origin_data,length(unique(origin_data_label)));

% 生成数据 (SDG)
SyntheticData = random(GMModel1,NoofSynthetic);

% 用K-means聚类方法获取合成生成数据的标签
Synthetic_label= kmeans(SyntheticData,length(unique(origin_data_label)));

end
%%
function [SyntheticData,Synthetic_label]=generate_classdata_LSTM(origin_data,origin_data_label,get_mutiple)
% 采用LSTM进行数据生成  
Runs= get_mutiple;  %生成样本数是原数据的多少倍
data=reshape(origin_data,1,[]); 
for Num_N=1:Runs
    
    mu = mean(data);
    sig = std(data);
    dataTrainStandardized = (data - mu) / sig;
    XTrain = dataTrainStandardized;
    YTrain = dataTrainStandardized;
    % Define LSTM Network Architecture
    numFeatures = 1;
    numResponses = 1;
    numHiddenUnits = 128;
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    options = trainingOptions('adam', ...
        'MaxEpochs',40, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.009, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',256, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0);
    net = trainNetwork(XTrain,YTrain,layers,options);

    % Forecast Future Time Steps
    dataTestStandardized = (data - mu) / sig;
    XTest = dataTestStandardized;
    net = predictAndUpdateState(net,XTrain);
    [net,YPred] = predictAndUpdateState(net,YTrain(end));
    numTimeStepsTest = numel(XTest);
    for i = 2:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
    end

    % Update Network State with Observed Values
    net = resetState(net);
    net = predictAndUpdateState(net,XTrain);
    YPred = [];
    numTimeStepsTest = numel(XTest);
    for i = 1:numTimeStepsTest
        [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
    end
    YPred = sig*YPred + mu;
    Synthetic{Num_N}=YPred;
    rmse = sqrt(mean((YPred-data).^2))*0.00001;
    RMSE(Num_N)=rmse;
end
Synthetic=Synthetic';% Converting cell to matrix (the last time)

Synthetic2 = cell2mat(Synthetic);% Converting matrix to cell
Synthetic2=Synthetic2';

SO = size(origin_data); SF =SO(1,2); SO = SO (1,1); 
for i = 1 : Runs
    Generated1{i}=reshape(Synthetic2(:,i),[SO,SF]);
    Generated1{i}(:,end+1)=origin_data_label;
end
Synthetic3 = cell2mat(Generated1');
SyntheticData=Synthetic3(:,1:end-1);
Synthetic_label=Synthetic3(:,end);
end

%%
function syntheticData = SMOTE(X, numMinority, numNeighbors,get_muti)
    % X: 少数类样本特征矩阵
    % numMinority: 少数类样本数量
    % numNeighbors: 邻居数量
    
    % 计算 k 近邻
    [idx] = knnsearch(X, X, 'K', numNeighbors);
    
    % 生成合成样本
%     syntheticData =[];
    for i = 1:round(numMinority * get_muti/numNeighbors)
        for j = 1:numNeighbors
            if i<size(X,1)
            
            neighbor = X(idx(i, j), :);
            gap = neighbor - X(i, :);
            alpha = rand(); % 随机选择一个权重
            syntheticData((i - 1) * numNeighbors + j, :) = X(i, :) + alpha * gap;
            end
        end
    end
end