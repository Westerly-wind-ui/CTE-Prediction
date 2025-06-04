function PlotProbability1(Predict,True_test,Lower,Upper,llimit,rlimit,Name,IColor,RColor,TColor,title_str,beta)
   %% 区间估计
    figure('Position',[100,100,1000,600]); %概率绘图
    transparence = [0.5;0.5;0.7;0.7]; %图窗透明度
    x = 1:length(Predict);
    h = gca;
    FColor=[1,1,1]; 
    hp = patch([x(1),x(end),x(end),x(1)],...
    [min([min(Lower),min(True_test)]),min([min(Lower),min(True_test)]),max([max(Upper),max(True_test)]),max([max(Upper),max(True_test)])],FColor,'FaceVertexAlphaData',transparence,'FaceAlpha',"interp");
    uistack(hp,'bottom');
    hold on
    % n = [0.25;0.3;0.35;0.4;0.45;0.5;0.6]; %区间透明度
    for j = 1:size(Lower,2)
        window(j)=fill([x,fliplr(x)],[Lower(:,j)',fliplr(Upper(:,j)')],IColor,'FaceAlpha',0.2+(1-beta(j)));%,'DisplayName',[num2str(100*beta(j)),'%置信区间']
        window(j).EdgeColor = 'none';
        hold on
        plot(Upper(:,j),'Marker',"none","LineStyle","none","Tag",'none',"Visible","off");
        hold on
        plot(Lower(:,j),'Marker',"none","LineStyle","none","Tag",'none',"Visible","off");
        hold on
    end
    h1=plot(True_test,'*','MarkerSize',4,'Color',RColor,'DisplayName','真实值');
    hold on
    h2=plot(Predict,'Color',TColor,'LineWidth',1.5,'DisplayName','预测值');
    hold on
    xlim([llimit rlimit]);
    str1=[];
   for n=1:length(beta)
       str1{1,n}=[num2str(100*beta(n)),'%置信区间'];
     
   end
   str1{1,length(beta)+1}='真实值';
   str1{1,length(beta)+2}='预测值';
   % str1{1,2+3*(length(beta))}='真实值';str1{1,3+3*(length(beta))}='';str1{1,4+3*(length(beta))}='';str1{1,5+3*(length(beta))}='预测值';
    % ylim([min(min(Lower)) max(max(Upper))]);
    title(title_str)
    xlabel('样本点',"FontSize",12,"FontWeight","bold");
    ylabel('预测值',"FontSize",12,"FontWeight","bold");
    legend('Location','bestoutside')
    % legend((str1),...
    % 'Location','bestoutside');
   legend([window,h1,h2],str1,'Location','bestoutside')
    % legend('','95%置信区间',"","",'90%置信区间',"","",'85%置信区间',"","",...
    % '80%置信区间',"","",'75%置信区间',"","",'70%置信区间',"","",...
    % '60%置信区间',"","",'真实值',strcat(Name,'预测值'),...
    % 'Location','bestoutside');
    grid on
    box off

end

