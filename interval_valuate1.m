function [value_result]=interval_valuate1(Lower,Upper,Real,eta,beta)
   %��������Ԥ���Ч��ָ�� 
   %PICP ָ�� ��ʵֵ�������±߽�ı��� ���ָ��Խ��Խ��
   %mean_PICP ָ�� ��ͬ���Ŷ������µ�ƽ��ֵ
   [PICP,mean_PICP] = PICP_FUN(Lower,Upper,Real);
   value_result.PICP=PICP;
   value_result.mean_PICP=mean_PICP;

   %PINAW ָ�� ��������խ�̶� ���ָ��ԽСԽ��|��խ��PI���䱻��Ϊ�ȿ��PI�����и������Ϣ
   %mean_PINAW ָ�� ��ͬ���Ŷ������µ�ƽ��ֵ  
   [PINAW,mean_PINAW] = PINAW_FUN(Lower,Upper,Real);
   value_result.PINAW=PINAW;
   value_result.mean_PINAW=mean_PINAW;
  
   %CWC �ۺϿ��Ǹ����ʺ���խ�̶ȣ�ԽСԽ�ã�һЩ�����Ż��㷨Ҳ�������ΪĿ�꺯��
   %Eta�ǳͷ�������ϵ����Խ��Ļ�����û�ﵽ����ˮƽ�ĳͷ�Խ��
   [CWC,mean_CWC] = CWC_FUN(PINAW,PICP,eta,beta);
   value_result.CWC=CWC;
   value_result.mean_CWC=mean_CWC;

   %MPICD �����������м�����ʵֵԽСԽ��
   [MPICD,mean_MPICD] = MPICDF(Lower,Upper,Real);
   value_result.MPICD=MPICD;
   value_result.mean_MPICD=mean_MPICD;

   %AIS�����λ�� �ۺϿ��Ǹ����ʺ�������,ԽСԽ�õ�
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
            theta=Upper(j,i)-Lower(j,i); %Ԥ��Ŀ��
            betai=1-beta(i);  %��i����������

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