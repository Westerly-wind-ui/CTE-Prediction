function [value_result]=interval_valuate1(Lower,Upper,Real,eta,beta)
   %评估区间预测的效果指标 
   %PICP 指标 真实值落在上下边界的比例 这个指标越大越好
   %mean_PICP 指标 不同置信度条件下的平均值
   [PICP,mean_PICP] = PICP_FUN(Lower,Upper,Real);
   value_result.PICP=PICP;
   value_result.mean_PICP=mean_PICP;

   %PINAW 指标 误差带的狭窄程度 这个指标越小越好|狭窄的PI区间被认为比宽的PI区间有更多的信息
   %mean_PINAW 指标 不同置信度条件下的平均值  
   [PINAW,mean_PINAW] = PINAW_FUN(Lower,Upper,Real);
   value_result.PINAW=PINAW;
   value_result.mean_PINAW=mean_PINAW;
  
   %CWC 综合考虑覆盖率和狭窄程度，越小越好，一些智能优化算法也拿这个作为目标函数
   %Eta是惩罚函数的系数，越大的话代表没达到置信水平的惩罚越大
   [CWC,mean_CWC] = CWC_FUN(PINAW,PICP,eta,beta);
   value_result.CWC=CWC;
   value_result.mean_CWC=mean_CWC;

   %MPICD 考虑误差带的中间与真实值越小越好
   [MPICD,mean_MPICD] = MPICDF(Lower,Upper,Real);
   value_result.MPICD=MPICD;
   value_result.mean_MPICD=mean_MPICD;

   %AIS区间分位数 综合考虑覆盖率和区间宽度,越小越好的
   [AIS,mean_AIS] = AIS_FUN(Lower,Upper,Real,beta);
   value_result.AIS=AIS;
   value_result.mean_AIS=mean_AIS;

end

function [PICP,mean_PICP] = PICP_FUN(Lower,Upper,Real)
    temp = zeros(size(Lower,2),1);
    for i = 1:size(Lower,2)
        for j = 1:length(Real)
            if Lower(j,i)<=Real(j)&&Upper(j,i)>=Real(j)
                temp(i,:) = temp(i,:)+1;
                count_picp(:,i) = temp(i,:);
            end
        end  
    end
    PICP = count_picp/length(Real);
    mean_PICP = mean(PICP);
end

function [PINAW,mean_PINAW] = PINAW_FUN(Lower,Upper,Real)
    PINAW = sum(Upper-Lower)/(length(Real)*(max(Real)-min(Real)));
    mean_PINAW = mean(PINAW);
end

function [CWC,mean_CWC] = CWC_FUN(PINAW,PICP,Eta,Beta)
   % CWC
    PINAW = PINAW';
    PICP = PICP';
    for i = 1:numel(PINAW)
        if PICP(i) < Beta(i)
            Gamma(i) = 1;
        else 
            Gamma(i) = 0;
        end
    end
    for m = 1:numel(PINAW)
        CWC(m) = PINAW(m)*(1+Gamma(m)*exp(-Eta*(PICP(m)-Beta(m))));
    end
    mean_CWC = mean(CWC);
end

function [MPICD,mean_MPICD] = MPICDF(Lower,Upper,Real)
    MPICD = sum(abs(Upper+Lower-2*Real))/2/(length(Real));
    mean_MPICD = mean(MPICD);
end

function [AIS,mean_AIS] = AIS_FUN(Lower,Upper,Real,beta)
    S=[];
    for i = 1:size(Lower,2)
        for j = 1:length(Real)
            theta=Upper(j,i)-Lower(j,i); %预测的宽度
            betai=1-beta(i);  %第i个置信区间

            if Lower(j,i)<=Real(j)&&Upper(j,i)>=Real(j)
               S(i,j)=-2*betai*theta;
            elseif Lower(j,i)>Real(j)
               S(i,j)=-2*betai*theta-4*(Lower(j,i)-Real(j)); 
            elseif Upper(j,i)<Real(j)
               S(i,j)=-2*betai*theta-4*(Real(j)-Upper(j,i));
            end
        end  
    end
    AIS=abs(mean(S'));   
    mean_AIS = mean(AIS);

end