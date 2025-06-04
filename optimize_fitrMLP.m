function [Mdl,fMin,Convergence_curve]  = optimize_fitrMLP(train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm,num_pop,num_iter,method_mti)  
%  global train_x_feature_label_norm train_y_feature_label_norm vaild_x_feature_label_norm vaild_y_feature_label_norm
%  Mdl= fitrensemble(train_x_feature_label_norm,train_y_feature_label_norm,"NumLearningCycles",50,"LearnRate",1);  
     pop=num_pop;
     M=num_iter;
     LB=[1,2,2,0.001];
     UB=[3,25,25,0.1];
     nvars=length(LB);
     fit_fitrensemble1=@fit_fitrMLP;
    if strcmp( method_mti,'SSA��ȸ�����㷨')==1
     [fMin,Mdl,Convergence_curve,pos]=SSA_SSA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);      
   elseif strcmp( method_mti,'DBO�����Ż��㷨')==1
     [fMin,Mdl,Convergence_curve,pos]=DBO_DBO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);      
    elseif strcmp( method_mti,'SCA�������Ż��㷨')==1 
    [fMin,Mdl,Convergence_curve,pos]=SCA_SCA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm); 
   elseif strcmp( method_mti,'SAģ���˻��㷨')==1 
      [fMin,Mdl,Convergence_curve,pos]=SA_SA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);       
   elseif strcmp( method_mti,'PSO����Ⱥ�㷨')==1 
      [fMin,Mdl,Convergence_curve,pos]=PSO_PSO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);         
   elseif strcmp( method_mti,'POA�����Ż��㷨')==1 
      [fMin,Mdl,Convergence_curve,pos]=POA_POA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);        
   elseif strcmp( method_mti,'GWO�����Ż��㷨')==1
      [fMin,Mdl,Convergence_curve,pos]=GWO_GWO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );  
   elseif strcmp( method_mti,'IGWO�Ľ������Ż��㷨')==1
     [fMin,Mdl,Convergence_curve,pos]=IGWO_IGWO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );      
   elseif strcmp( method_mti,'AVOA����ͺӥ�Ż��㷨')==1
     [fMin,Mdl,Convergence_curve,pos]=AVOA_AVOV(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );      
  elseif strcmp( method_mti,'CSA��ɫ���Ż��㷨')==1  
     [fMin,Mdl,Convergence_curve,pos]=CSA_CAS(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);        
 elseif strcmp( method_mti,'GTO�������Ż��㷨')==1  
     [fMin,Mdl,Convergence_curve,pos]=GTO_GTO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);       
 elseif strcmp( method_mti,'NGO������ӥ�Ż��㷨')==1  
     [fMin,Mdl,Convergence_curve,pos]=NGO_NGO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);        
 elseif strcmp( method_mti,'SO���Ż��㷨')==1  
     [fMin,Mdl,Convergence_curve,pos]=SO_SO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);       
  elseif strcmp( method_mti,'WSO�����Ż��㷨')==1    
     [fMin,Mdl,Convergence_curve,pos]=WSO_WSO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            
  elseif strcmp( method_mti,'CGO���粩���Ż��㷨')==1    
     [fMin,Mdl,Convergence_curve,pos]=CGO_CGO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            
  elseif strcmp( method_mti,'INFO��Ȩ������ֵ�Ż��㷨')==1    
     [fMin,Mdl,Convergence_curve,pos]=INFO_INFO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            
  elseif strcmp( method_mti,'COA����Ż��㷨')==1    
     [fMin,Mdl,Convergence_curve,pos]=COA_COA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            
  elseif strcmp( method_mti,'RIME˪���Ż��㷨')==1    
     [fMin,Mdl,Convergence_curve,pos]=RIME_RIME(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            
  elseif strcmp( method_mti,'KOA�������Ż��㷨')==1    
     [fMin,Mdl,Convergence_curve,pos]=KOA_KOA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            
 elseif strcmp( method_mti,'RUN��������Ż��㷨')==1
    [fMin,Mdl,Convergence_curve,pos]=RUN_RUN(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'IVY�������Ż��㷨')==1
    [fMin,Mdl,Convergence_curve,pos]=IVY_IVY(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'HOAͽ���Ż��㷨')==1
    [fMin,Mdl,Convergence_curve,pos]=HOA_HOA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'BKA�ڳ���Ż��㷨')==1
    [fMin,Mdl,Convergence_curve,pos]=BKA_BKA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'COA��Ϻ�Ż��㷨')==1
    [fMin,Mdl,Convergence_curve,pos]=COA_COA1(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'BSLO��ˮ���Ż��㷨')==1
    [fMin,Mdl,Convergence_curve,pos]=BSLO_BSLO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'GHOAӳ��͸�˹���߸Ľ�HOA')==1
    [fMin,Mdl,Convergence_curve,pos]=GHOA_GHOA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'GSSAӳ��������ҸĽ�SSA')==1
    [fMin,Mdl,Convergence_curve,pos]=GSSA_GSSA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'GBKAӳ��Ľ�BKA')==1
    [fMin,Mdl,Convergence_curve,pos]=GBKA_GBKA(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'GRIMEӳ��ͷ���ѧϰ�Ľ�RIME')==1
    [fMin,Mdl,Convergence_curve,pos]=GRIME_GRIME(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
elseif strcmp( method_mti,'GDPSO�ƽ������ҸĽ�PSO')==1
    [fMin,Mdl,Convergence_curve,pos]=GDPSO_GDPSO(pop,M,LB,UB,nvars,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
       
    end
    if(pos(1)<=1)
        hid_size=round(pos(2));
    else
        hid_size=[round(pos(2)),round(pos(3))];  
    end
     figure
     plot(Convergence_curve,'--p','LineWidth',1.2,'Color',[160,123,194]./255) 
     xticks([1:length(Convergence_curve)])
     title('optimize process')
     xlabel('iter')
     ylabel('fitness')
     grid on
   set(gca,"FontName","Times New Roman","FontSize",12,"LineWidth",1.2)
   box off 
 disp([method_mti, '�Ż� fitrnet:   ',"LayerSizes:",num2str(hid_size),'   Lambda: ',num2str((pos(4)))]) 
end
function [fitness_value,Mdl]=fit_fitrMLP(pop,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
%    global train_x_feature_label_norm train_y_feature_label_norm vaild_x_feature_label_norm vaild_y_feature_label_norm
if(pop(1)<=2)
    Mdl=fitrnet(train_x_feature_label_norm,train_y_feature_label_norm,"LayerSizes",round(pop(2)),"Lambda",(pop(4))); %��Ԫ�ͼ����,'KernelScale',pop(1),'Epsilon',pop(2));  
else
   Mdl=fitrnet(train_x_feature_label_norm,train_y_feature_label_norm,"LayerSizes",round([pop(2),pop(3)]),"Lambda",(pop(4))); %��Ԫ�ͼ����,'KernelScale',pop(1),'Epsilon',pop(2));  
% elseif (pop(1)==3)
%   Mdl=fitrnet(train_x_feature_label_norm,train_y_feature_label_norm,"LayerSizes",[pop(2),pop(3),pop(4)]); %��Ԫ�ͼ����,'KernelScale',pop(1),'Epsilon',pop(2));  
end
    P_vaild_y_feature_label_norm=predict(Mdl,vaild_x_feature_label_norm);
    fitness_value=sum(abs(P_vaild_y_feature_label_norm-vaild_y_feature_label_norm))/length(vaild_y_feature_label_norm);
end

function [fMin,Mdl_best,Convergence_curve,bestX]=SSA_SSA(pop,M,LB,UB,nvars,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)       
 P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size       
   %������ռ������Ⱥ��0.2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pNum = round( pop *  P_percent );    % The population size of the producers    
%����������ȡ��
dim=nvars;
lb= LB.*ones( 1,dim );    % Lower limit/bounds/     a vector    Լ������
ub= UB.*ones( 1,dim );    % Upper limit/bounds/     a vector  Լ������
%Initialization
for i = 1 : pop    
    x( i, : ) = lb + (ub - lb) .* rand( 1, dim );   %�����ʼ��n����Ⱥ
    x(i, : ) = Bounds( x(i, : ),lb, ub);
    [fit( i ),model{1,i}] = fobj( x( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm ) ;               %��������Ⱥ�����Ӧ������������Сֵ��,ԽС������Ӧ��Խ��      
end
% disp(1111)
% �����ҵ���Сֵ��Ӧ����ȸȺ
pFit = fit;                      
pX = x;                            % The individual's best position corresponding to the pFit
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX denotes the global optimum position corresponding to fMin
Mdl_best=model{1,bestI};
% disp(333)
 % Start updating the solutions.
for t = 1 : M         
  [ ~, sortIndex ] = sort( pFit );% Sort.
     
  [fmax,B]=max( pFit );
   worse= x(B,:);         %�ҵ����ĸ���      

   r2=rand(1);      %���������    �о�û��ɶ��ѧ���ݣ����������
   %�����˼������0.8������ԭ����Ⱥ��һ��С��1��������Ⱥ������ֵ��С��
   %�����˼������0.2������ԭ����Ⱥ��һ��С��1��������Ⱥ������ֵ+1
   %�仯����Ⱥ����ֵ����Ҫ������Լ������
   %��ǰpNum��Ӧ����õĽ��б仯 ���������߽��б仯���ɼ�������������õ�
if(r2<0.8)
    for i = 1 : pNum                                                        % Equation (3)
         r1=rand(1);
        x( sortIndex( i ), : ) = pX( sortIndex( i ), : )*exp(-(i)/(r1*M)); %����Ⱥ����Ӧ���������� 
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );  %����Ⱥ������Լ����Χ��
%         [fit( sortIndex( i ) ),model{1,sortIndex( i )}]
         [fit( sortIndex( i ) ),model{1,sortIndex( i )}]= fobj( x( sortIndex( i ), : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );   
    end
  else
  for i = 1 : pNum            
  x( sortIndex( i ), : ) = pX( sortIndex( i ), : )+randn(1)*ones(1,dim);
  x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
   [fit( sortIndex( i ) ),model{1,sortIndex( i )}] = fobj( x( sortIndex( i ), : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );       
  end      
end
 %�Ѿ����仯����õ���Ⱥ��¼����  
 [ ~, bestII ] = min( fit );      
  bestXX = x( bestII, : );            
%   disp(2222)
 %������������ 
   for i = ( pNum + 1 ) : pop                     % Equation (4)
         A=floor(rand(1,dim)*2)*2-1;           %����1��-1�������
         
          if( i>(pop/2))
           %���i>��Ⱥ��һ�룬�����������Ӧ�ȿ����һ�Σ�������Щ���е���Ⱥ�����ڰ���
           x( sortIndex(i ), : )=randn(1)*exp((worse-pX( sortIndex( i ), : ))/(i)^2);  
           %��Ӧ�Ȳ��õģ����������ȸ����һ������1������������չ
          else
        x( sortIndex( i ), : )=bestXX+(abs(( pX( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);  
           %������Ӧ�ȳ�ȥ����������֮��������Ⱥ��ǰ��ε�,ȥ���������ߵ�ʳ���ǰ�������Ⱥ�Ļ�����
           %�ٽ��б仯һ�Σ���ԭ���Ļ����ϼ�һЩֵ���߼�һЩֵ
         end  
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );  %���º���Ⱥ�������ڱ�����Χ
         [fit( sortIndex( i ) ),model{1,sortIndex( i )}]= fobj( x( sortIndex( i ), : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );                    %���¹������¼�����Ӧ��
   end
   %��ȫ����Ⱥ���ҿ�����ʶ��Σ�յ���ȸ
  c=randperm(numel(sortIndex));
   b=sortIndex(c(1:round(0.2*length(c))));
  for j =  1  : length(b)      % Equation (5)
    if( pFit( sortIndex( b(j) ) )>(fMin) )
         %�����Ӧ�ȱ��ʼ��С��Ӧ�Ȳ�Ļ�������ԭ���������Ⱥ������һ����ֵ
        x( sortIndex( b(j) ), : )=bestX+(randn(1,dim)).*(abs(( pX( sortIndex( b(j) ), : ) -bestX)));
        
        else
        %�����Ӧ�ȴﵽ��ʼ��С����Ӧ��ֵ������ԭ���������Ⱥ������������Сһ����
        x( sortIndex( b(j) ), : ) =pX( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pX( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);

          end
        x( sortIndex(b(j) ), : ) = Bounds( x( sortIndex(b(j) ), : ), lb, ub );
        
       [fit( sortIndex(b(j) ) ),model{1,sortIndex( b(j) )}] = fobj( x( sortIndex( b(j) ), : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
 end
    for i = 1 : pop 
        %����ĸ���Ⱥ��Ӧ�Ⱥ��ˣ��Ͱѱ仯���滻��ԭ������Ⱥ
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end
        
        if( pFit( i ) < fMin )   %����ֵ�Լ�����ֵλ�ÿ��Ƿ�仯
           fMin= pFit( i );
            bestX = pX( i, : );
            Mdl_best=model{1,i};
         
            
        end
    end
    Convergence_curve(t)=fMin;
end
end
%%
function [fMin , Mdl_best, Convergence_curve,bestX ] = DBO_DBO(pop, M,c,d,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm  )
        
   P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size       


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pNum = round( pop *  P_percent );    % The population size of the producers   


lb= c.*ones( 1,dim );    % Lower limit/bounds/     a vector
ub= d.*ones( 1,dim );    % Upper limit/bounds/     a vector
%Initialization
for i = 1 : pop
    
    x( i, : ) = lb + (ub - lb) .* rand( 1, dim );  
    x(i, : ) = Bounds( x(i, : ),lb, ub);
    [fit( i ),model{1,i}]= fobj( x( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )  ;                       
end

pFit = fit;                       
pX = x; 
 XX=pX;    
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX denotes the global optimum position corresponding to fMin
Mdl_best=model{1,bestI};
 % Start updating the solutions.
for t = 1 : M    
       
        [fmax,B]=max(fit);
        worse= x(B,:);   
       r2=rand(1);
 
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1 : pNum    
        if(r2<0.9)
            r1=rand(1);
          a=rand(1,1);
          if (a>0.1)
           a=1;
          else
           a=-1;
          end
    x( i , : ) =  pX(  i , :)+0.3*abs(pX(i , : )-worse)+a*0.1*(XX( i , :)); % Equation (1)
       else
            
           aaa= randperm(180,1);
           if ( aaa==0 ||aaa==90 ||aaa==180 )
            x(  i , : ) = pX(  i , :);   
           end
         theta= aaa*pi/180;   
       
       x(  i , : ) = pX(  i , :)+tan(theta).*abs(pX(i , : )-XX( i , :));    % Equation (2)      

        end
      
        x(  i , : ) = Bounds( x(i , : ), lb, ub );    
       [fit( i ),model{1,i}] = fobj( x( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm ) ;    
    end 
 [ fMMin, bestII ] = min( fit );      % fMin denotes the current optimum fitness value
  bestXX = x( bestII, : );             % bestXX denotes the current optimum position 
Mdl_best=model{1,bestII};
 R=1-t/M;                           %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Xnew1 = bestXX.*(1-R); 
     Xnew2 =bestXX.*(1+R);                    %%% Equation (3)
   Xnew1= Bounds( Xnew1, lb, ub );
   Xnew2 = Bounds( Xnew2, lb, ub );
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Xnew11 = bestX.*(1-R); 
     Xnew22 =bestX.*(1+R);                     %%% Equation (5)
   Xnew11= Bounds( Xnew11, lb, ub );
    Xnew22 = Bounds( Xnew22, lb, ub );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    for i = ( pNum + 1 ) : round( pop *  0.4 )                  % Equation (4)
     x( i, : )=bestXX+((rand(1,dim)).*(pX( i , : )-Xnew1)+(rand(1,dim)).*(pX( i , : )-Xnew2));
   x(i, : ) = Bounds( x(i, : ), Xnew1, Xnew2 );
  [fit( i ),model{1,i}] = fobj( x( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm ) ;    
   end
   
  for i = round( pop *  0.4 )+1: round( pop *  0.6 )                  % Equation (6)

   
        x( i, : )=pX( i , : )+((randn(1)).*(pX( i , : )-Xnew11)+((rand(1,dim)).*(pX( i , : )-Xnew22)));
       x(i, : ) = Bounds( x(i, : ),lb, ub);
       [fit( i ),model{1,i}]= fobj( x( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm ) ;    
  
  end
  
  for j = round( pop *  0.6 )+1 : pop                 % Equation (7)
       x( j,: )=bestX+randn(1,dim).*((abs(( pX(j,:  )-bestXX)))+(abs(( pX(j,:  )-bestX))))./2;
      x(j, : ) = Bounds( x(j, : ), lb, ub );
      [fit( j ),model{1,j}] = fobj( x( j, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm ) ;    
  end
   % Update the individual's best fitness vlaue and the global best fitness value
     XX=pX;
    for i = 1 : pop 
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end
        
        if( pFit( i ) < fMin )
           % fMin= pFit( i );
           fMin= pFit( i );
            bestX = pX( i, : );
            Mdl_best=model{1,i};
          %  a(i)=fMin;
            
        end
    end
  
     Convergence_curve(t)=fMin;
  
     
end
end
%%
function [Destination_fitness,Mdl_best,Convergence_curve,Destination_position]=SCA_SCA(N,Max_iteration,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)         
% [Destination_fitness,Mdl_best,Convergence_curve]=SCA_SCA(N,Max_iteration,lb,ub,dim,fit_fitrensemble1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);         
% display('SCA is optimizing your problem');

%Initialize the set of random solutions
% X=initialization(N,dim,ub,lb);
for i = 1 : N
    
    X( i, : ) = Bounds(lb + (ub - lb) .* rand( 1, dim ) ,lb,ub ) ;    
  
end

Destination_position=zeros(1,dim);
Destination_fitness=inf;

Convergence_curve=zeros(1,Max_iteration);
Objective_values = zeros(1,size(X,1));

% Calculate the fitness of the first set and find the best one
for i=1:size(X,1)
%     =fobj(X(i,:));
    [Objective_values(1,i),model{1,i}]= fobj( X( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
    if i==1
        Destination_position=X(i,:);
        Destination_fitness=Objective_values(1,i);
    elseif Objective_values(1,i)<Destination_fitness
        Destination_position=X(i,:);
        Destination_fitness=Objective_values(1,i);
    end
    
    All_objective_values(1,i)=Objective_values(1,i);
end
[~,index_choose]=min(Objective_values);
Mdl_best=model{1,index_choose};
%Main loop
t=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness
while t<=Max_iteration
    
    % Eq. (3.4)
    a = 2;
    Max_iteration = Max_iteration;
    r1=a-t*((a)/Max_iteration); % r1 decreases linearly from a to 0
    
    % Update the position of solutions with respect to destination
    for i=1:size(X,1) % in i-th solution
        for j=1:size(X,2) % in j-th dimension
            
            % Update r2, r3, and r4 for Eq. (3.3)
            r2=(2*pi)*rand();
            r3=2*rand;
            r4=rand();
            
            % Eq. (3.3)
            if r4<0.5
                % Eq. (3.1)
                X(i,j)= X(i,j)+(r1*sin(r2)*abs(r3*Destination_position(j)-X(i,j)));
                X(i,:)= Bounds(X(i,:),lb,ub );  
               
            else
                % Eq. (3.2)
                X(i,j)= X(i,j)+(r1*cos(r2)*abs(r3*Destination_position(j)-X(i,j)));
                X(i,:)= Bounds(X(i,:),lb,ub );  
            end
            
        end
    end
    
    for i=1:size(X,1)
         
        % Check if solutions go outside the search spaceand bring them back
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        X(i,:)= Bounds(X(i,:),lb,ub );  
        
        % Calculate the objective values
         [Objective_values(1,i),model{1,i}]= fobj( X( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
        
        % Update the destination if there is a better solution
        if Objective_values(1,i)<Destination_fitness
            Destination_position=X(i,:);
            Destination_fitness=Objective_values(1,i);
             Mdl_best=model{1,i};
        end
    end
    
    Convergence_curve(t)=Destination_fitness;
    
    % Display the iteration and best optimum obtained so far
%     if mod(t,50)==0
%         display(['At iteration ', num2str(t), ' the optimum is ', num2str(Destination_fitness)]);
%     end
    
    % Increase the iteration counter
    t=t+1;
end
end
%%
function[Best_score,Mdl_best,POA_curve,Best_pos]=POA_POA(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

lowerbound=ones(1,dimension).*(lowerbound);                              % Lower limit for variables
upperbound=ones(1,dimension).*(upperbound);                              % Upper limit for variables

% INITIALIZATION
for i=1:dimension
    X(:,i) = lowerbound(i)+rand(SearchAgents,1).*(upperbound(i) - lowerbound(i));                          % Initial population
end

for i =1:SearchAgents
%     L=X(i,:);
     [fit(i),model{1,i}]= fobj( X( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%     fit(i)=fobj(L);
end
%
[~,index_nedd]=min(fit);
Mdl_best=model{1,index_nedd};
for t=1:Max_iterations
    % update the best condidate solution
    [best , location]=min(fit);
    if t==1
        Xbest=X(location,:);                                           % Optimal location
        fbest=best;                                           % The optimization objective function
    elseif best<fbest
        fbest=best;
        Xbest=X(location,:);
    end
    
    % UPDATE location of food
    
    X_FOOD=[];
    k=randperm(SearchAgents,1);
    X_FOOD=X(k,:);
    F_FOOD=fit(k);
    
    %%
    for i=1:SearchAgents
        
        % PHASE 1: Moving towards prey (exploration phase)
        I=round(1+rand(1,1));
        if fit(i)> F_FOOD
            X_new=X(i,:)+ rand(1,1).*(X_FOOD-I.* X(i,:)); %Eq(4)
        else
            X_new=X(i,:)+ rand(1,1).*(X(i,:)-1.*X_FOOD); %Eq(4)
        end
        X_new= max(X_new,lowerbound);X_new = min(X_new,upperbound);
        
        % Updating X_i using (5)
          [f_new,model_new]= fobj(X_new,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%         f_new = fobj(X_new);
        if f_new <= fit (i)
            X(i,:) = X_new;
            fit (i)=f_new;
            Mdl_best=model_new;
        end
        % END PHASE 1: Moving towards prey (exploration phase)
        
        % PHASE 2: Winging on the water surface (exploitation phase)
        X_new=X(i,:)+0.2*(1-t/Max_iterations).*(2*rand(1,dimension)-1).*X(i,:);% Eq(6)
        X_new= max(X_new,lowerbound);X_new = min(X_new,upperbound);
        
        % Updating X_i using (7)
%         f_new = fobj(X_new);
%         if f_new <= fit (i)
%             X(i,:) = X_new;
%             fit (i)=f_new;
%         end
      % Updating X_i using (7)
        [f_new,model_new]= fobj(X_new,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%         f_new = fobj(X_new);
        if f_new <= fit (i)
            X(i,:) = X_new;
            fit (i)=f_new;
            Mdl_best=model_new;
        end
        % END PHASE 2: Winging on the water surface (exploitation phase)
    end

    best_so_far(t)=fbest;
    average(t) = mean (fit);
    
end
Best_score=fbest;
Best_pos=Xbest;
POA_curve=best_so_far;
end
%%
function [Alpha_score,Mdl_best,Convergence_curve,Alpha_pos]=GWO_GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )
 
% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iter);

l=0;% Loop counter

% Main loop
while l<Max_iter
    for i=1:size(Positions,1)  
        
       % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        
        % Calculate objective function for each search agent
%           fitness=fobj(Positions(i,:));
         [fitness1(i),model{1,i}]= fobj(Positions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
         fitness=fitness1(i);
%         Mdl_best=model{1,1};
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score 
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score 
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
        end

    end
    
    
    a=2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    l=l+1;    
    Convergence_curve(l)=Alpha_score;
end
[~,indx_best]=min(fitness1);
Mdl_best= model{1,indx_best};
%   [fitness1(i),model{1,i}]
end
%%
% [Alpha_score,Mdl_best,Convergence_curve,Alpha_pos]=GWO_GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )
function [Alpha_score,Mdl_best,Convergence_curve,Alpha_pos]=IGWO_IGWO(N,Max_iter,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

lu = [lb .* ones(1, dim); ub .* ones(1, dim)];

% Initialize alpha, beta, and delta positions
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

% Initialize the positions of wolves
Positions=initialization(N,dim,ub,lb);
Positions = Bounds (Positions, lb, ub);

% Calculate objective function for each wolf
for i=1:size(Positions,1)
%     Fit(i) = fobj(Positions(i,:));
    [Fit(i),model{1,i}]= fobj(Positions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
end

% Personal best fitness and position obtained by each wolf
pBestScore = Fit;
pBest = Positions;

neighbor = zeros(N,N);
Convergence_curve=zeros(1,Max_iter);
iter = 0;% Loop counter

%% Main loop
while iter < Max_iter
    for i=1:size(Positions,1)
        fitness = Fit(i);
        
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
        end
    end
    
    %% Calculate the candiadate position Xi-GWO
    a=2-iter*((2)/Max_iter); % a decreases linearly from 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)
            
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a;                                    % Equation (3.3)
            C1=2*r2;                                        % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j));    % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha;                     % Equation (3.6)-part 1
            
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a;                                    % Equation (3.3)
            C2=2*r2;                                        % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j));      % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta;                       % Equation (3.6)-part 2
            
            r1=rand();
            r2=rand();
            
            A3=2*a*r1-a;                                    % Equation (3.3)
            C3=2*r2;                                        % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j));    % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta;                     % Equation (3.5)-part 3
            
            X_GWO(i,j)=(X1+X2+X3)/3;                        % Equation (3.7)
            
        end
        X_GWO(i,:) = Bounds(X_GWO(i,:), lb, ub);
         [Fit_GWO(i),model{1,i}]= fobj(X_GWO(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%         Fit_GWO(i) = fobj(X_GWO(i,:));
    end
    
    % Calculate the candiadate position Xi-DLH
    radius = pdist2(Positions, X_GWO, 'euclidean');         % Equation (10)
    dist_Position = squareform(pdist(Positions));
    r1 = randperm(N,N);
    
    for t=1:N
        neighbor(t,:) = (dist_Position(t,:)<=radius(t,t));
        [~,Idx] = find(neighbor(t,:)==1);                   % Equation (11)             
        random_Idx_neighbor = randi(size(Idx,2),1,dim);
        
        for d=1:dim
            X_DLH(t,d) = Positions(t,d) + rand .*(Positions(Idx(random_Idx_neighbor(d)),d)...
                - Positions(r1(t),d));                      % Equation (12)
        end
        X_DLH(t,:) = Bounds(X_DLH(t,:), lb, ub);
         [  Fit_DLH(t) ,model{1,t}]= fobj(X_DLH(t,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%         Fit_DLH(t) = fobj(X_DLH(t,:));
    end
    
    % Selection  
    tmp = Fit_GWO < Fit_DLH;                                % Equation (13)
    tmp_rep = repmat(tmp',1,dim);
    
    tmpFit = tmp .* Fit_GWO + (1-tmp) .* Fit_DLH;
    tmpPositions = tmp_rep .* X_GWO + (1-tmp_rep) .* X_DLH;
    
    % Updating
    tmp = pBestScore <= tmpFit;                             % Equation (13)
    tmp_rep = repmat(tmp',1,dim);
    
    pBestScore = tmp .* pBestScore + (1-tmp) .* tmpFit;
    pBest = tmp_rep .* pBest + (1-tmp_rep) .* tmpPositions;
    
    Fit = pBestScore;
    Positions = pBest;
    
    %
    iter = iter+1;
    neighbor = zeros(N,N);
    Convergence_curve(iter) = Alpha_score;  
end
[~,best_index]=min(Fit_DLH(t));
Mdl_best=model{1,best_index};
end

%%
function [Best_vulture1_F,Mdl_best,convergence_curve,Best_vulture1_X]=AVOA_AVOV(pop_size,max_iter,lower_bound,upper_bound,variables_no,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

    % initialize Best_vulture1, Best_vulture2
    Best_vulture1_X=zeros(1,variables_no);
    Best_vulture1_F=inf;
    Best_vulture2_X=zeros(1,variables_no);
    Best_vulture2_F=inf;

    %Initialize the first random population of vultures
    X=initialization(pop_size,variables_no,upper_bound,lower_bound);

    %  Controlling parameter
    p1=0.6;
    p2=0.4;
    p3=0.6;
    alpha=0.8;
    betha=0.2;
    gamma=2.5;

    %%Main loop
    current_iter=0; % Loop counter

    while current_iter < max_iter
        for i=1:size(X,1)
            % Calculate the fitness of the population
            current_vulture_X = X(i,:);
               [current_vulture_F,Mdl_best1]= fobj(current_vulture_X,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%             current_vulture_F=fobj(current_vulture_X);
                if i==1
                   Mdl_best=Mdl_best1;
                end
            % Update the first best two vultures if needed
            if current_vulture_F<Best_vulture1_F
                Best_vulture1_F=current_vulture_F; % Update the first best bulture
                Best_vulture1_X=current_vulture_X;
                Mdl_best=Mdl_best1;
            end
            if current_vulture_F>Best_vulture1_F && current_vulture_F<Best_vulture2_F
                Best_vulture2_F=current_vulture_F; % Update the second best bulture
                Best_vulture2_X=current_vulture_X;
            end
        end

        a=unifrnd(-2,2,1,1)*((sin((pi/2)*(current_iter/max_iter))^gamma)+cos((pi/2)*(current_iter/max_iter))-1);
        P1=(2*rand+1)*(1-(current_iter/max_iter))+a;

        % Update the location
        for i=1:size(X,1)
            current_vulture_X = X(i,:);  % pick the current vulture back to the population
            F=P1*(2*rand()-1);  

            random_vulture_X=random_select(Best_vulture1_X,Best_vulture2_X,alpha,betha);
            
            if abs(F) >= 1 % Exploration:
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound);
            elseif abs(F) < 1 % Exploitation:
                current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound);
            end

            X(i,:) = current_vulture_X; % place the current vulture back into the population
             X(i,:)  = Bounds(X(i,:) , lower_bound, upper_bound);
        end

        current_iter=current_iter+1;
        convergence_curve(current_iter)=Best_vulture1_F;

       

%         fprintf("In Iteration %d, best estimation of the global optimum is %4.4f \n ", current_iter,Best_vulture1_F );
    end

end

%
function [current_vulture_X] = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, ...
                                                                      random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound)

% phase 1
    if  abs(F)<0.5
        if rand<p2
            A=Best_vulture1_X-((Best_vulture1_X.*current_vulture_X)./(Best_vulture1_X-current_vulture_X.^2))*F;
            B=Best_vulture2_X-((Best_vulture2_X.*current_vulture_X)./(Best_vulture2_X-current_vulture_X.^2))*F;
            current_vulture_X=(A+B)/2;
        else
            current_vulture_X=random_vulture_X-abs(random_vulture_X-current_vulture_X)*F.*levyFlight(variables_no);
        end
    end
    % phase 2
    if  abs(F)>=0.5
        if rand<p3
            current_vulture_X=(abs((2*rand)*random_vulture_X-current_vulture_X))*(F+rand)-(random_vulture_X-current_vulture_X);
        else
            s1=random_vulture_X.* (rand()*current_vulture_X/(2*pi)).*cos(current_vulture_X);
            s2=random_vulture_X.* (rand()*current_vulture_X/(2*pi)).*sin(current_vulture_X);
            current_vulture_X=random_vulture_X-(s1+s2);
        end
    end
end
%
function [current_vulture_X] = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)

    if rand<p1
        current_vulture_X=random_vulture_X-(abs((2*rand)*random_vulture_X-current_vulture_X))*F;
    else
        current_vulture_X=(random_vulture_X-(F)+rand()*((upper_bound-lower_bound)*rand+lower_bound));
    end
    
end
%
function [random_vulture_X]=random_select(Best_vulture1_X,Best_vulture2_X,alpha,betha)

    probabilities=[alpha, betha ];
    
    if (rouletteWheelSelection( probabilities ) == 1)
            random_vulture_X=Best_vulture1_X;
    else
            random_vulture_X=Best_vulture2_X;
    end

end
%
function [ o ]=levyFlight(d)
  
    beta=3/2;

    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u=randn(1,d)*sigma;
    v=randn(1,d);
    step=u./abs(v).^(1/beta);

    o=step;

end
%
% function [ X ] = boundaryCheck(X, lb, ub)
% 
%     for i=1:size(X,1)
%             FU=X(i,:)>ub;
%             FL=X(i,:)<lb;
%             X(i,:)=(X(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
%     end
% end
function [index] = rouletteWheelSelection(x)
    index=find(rand() <= cumsum(x) ,1,'first');
end
%%
% [Best_vulture1_F,Mdl_best,convergence_curve,Best_vulture1_X]=AVOA_AVOV(pop_size,max_iter,lower_bound,upper_bound,variables_no,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
function [fmin0,Mdl_best,cg_curve,gPosition]=CSA_CAS(searchAgents,iteMax,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

%%%%* 1
if size(ub,2)==1
    ub=ones(1,dim)*ub;
    lb=ones(1,dim)*lb;
end

% Convergence curve
cg_curve=zeros(1,iteMax);

% Initial population

chameleonPositions=initialization(searchAgents,dim,ub,lb);% Generation of initial solutions 

% Evaluate the fitness of the initial population

fit=zeros(searchAgents,1);


for i=1:searchAgents
     [fit(i,1),Model{1,i}]= fobj(chameleonPositions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%      fit(i,1)=fobj(chameleonPositions(i,:));
end

% Initalize the parameters of CSA
fitness=fit; % Initial fitness of the random positions
 
[fmin0,index]=min(fit);

chameleonBestPosition = chameleonPositions; % Best position initialization
gPosition = chameleonPositions(index,:); % initial global position
Mdl_best=Model{1,index};

v=0.1*chameleonBestPosition;% initial velocity
 
v0=0.0*v;

% Start CSA 
% Main parameters of CSA
rho=1.0;
p1=2.0;  
p2=2.0;  
c1=2.0; 
c2=1.80;  
gamma=2.0; 
alpha = 4.0;  
beta=3.0; 
 

 % Start CSA
for t=1:iteMax
a = 2590*(1-exp(-log(t))); 
omega=(1-(t/iteMax))^(rho*sqrt(t/iteMax)) ; 
p1 = 2* exp(-2*(t/iteMax)^2);  % 
p2 = 2/(1+exp((-t+iteMax/2)/100)) ;
        
mu= gamma*exp(-(alpha*t/iteMax)^beta) ;

ch=ceil(searchAgents*rand(1,searchAgents));
% Update the position of CSA (Exploration)
for i=1:searchAgents  
             if rand>=0.1
                  chameleonPositions(i,:)= chameleonPositions(i,:)+ p1*(chameleonBestPosition(ch(i),:)-chameleonPositions(i,:))*rand()+... 
                     + p2*(gPosition -chameleonPositions(i,:))*rand();
             else 
                 for j=1:dim
                   chameleonPositions(i,j)=   gPosition(j)+mu*((ub(j)-lb(j))*rand+lb(j))*sign(rand-0.50) ;
                 end 
              end   
end       
 % Rotation of the chameleons - Update the position of CSA (Exploitation)
 
 %  % Chameleon velocity updates and find a food source
     for i=1:searchAgents
               
        v(i,:)= omega*v(i,:)+ p1*(chameleonBestPosition(i,:)-chameleonPositions(i,:))*rand +.... 
               + p2*(gPosition-chameleonPositions(i,:))*rand;        

         chameleonPositions(i,:)=chameleonPositions(i,:)+(v(i,:).^2 - v0(i,:).^2)/(2*a);
     end
    
  v0=v;
  
 % handling boundary violations
 for i=1:searchAgents
     if chameleonPositions(i,:)<lb
        chameleonPositions(i,:)=lb;
     elseif chameleonPositions(i,:)>ub
            chameleonPositions(i,:)=ub;
     end
 end
 
 % Relocation of chameleon positions (Randomization) 
for i=1:searchAgents
    
    ub_=sign(chameleonPositions(i,:)-ub)>0;   
    lb_=sign(chameleonPositions(i,:)-lb)<0;
       
    chameleonPositions(i,:)=(chameleonPositions(i,:).*(~xor(lb_,ub_)))+ub.*ub_+lb.*lb_;  %%%%%*2
 
  [fit(i,1),Model{1,i}]= fobj(chameleonPositions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
      
      if fit(i)<fitness(i)
                 chameleonBestPosition(i,:) = chameleonPositions(i,:); % Update the best positions  
                 fitness(i)=fit(i); % Update the fitness
                 model2{1,i}=Model{1,i};
      end
 end


% Evaluate the new positions

[fmin,index]=min(fitness); % finding out the best positions  

% Updating gPosition and best fitness
if fmin < fmin0
    gPosition = chameleonBestPosition(index,:); % Update the global best positions
    Mdl_best=model2{1,index};
    fmin0 = fmin;
end

% Visualize the results
   cg_curve(t)=fmin0; % Best found value until iteration t

end
% ngPosition=find(fitness== min(fitness)); 
% g_best=chameleonBestPosition(ngPosition(1),:);  % Solutin of the problem
% fmin0 =fobj(g_best);
end
%%
% [fmin0,Mdl_best,cg_curve,gPosition]=CSA_CAS(searchAgents,iteMax,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
function [Silverback_Score,Mdl_best,convergence_curve,Silverback]=GTO_GTO(pop_size,max_iter,lower_bound,upper_bound,variables_no,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

% initialize Silverback
Silverback=[];
Silverback_Score=inf;

%Initialize the first random population of Gorilla
X=initialization(pop_size,variables_no,upper_bound,lower_bound);


convergence_curve=zeros(max_iter,1);

for i=1:pop_size   
%     Pop_Fit(i)=fobj(X(i,:));%#ok
     [Pop_Fit(i),Model{1,i}]= fobj(X(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
      Mdl_best=Model{1,1};
     if Pop_Fit(i)<Silverback_Score 
            Silverback_Score=Pop_Fit(i); 
            Silverback=X(i,:);
            Mdl_best=Model{1,i};
    end
end


GX=X(:,:);
lb=ones(1,variables_no).*lower_bound; 
ub=ones(1,variables_no).*upper_bound; 

%  Controlling parameter

p=0.03;
Beta=3;
w=0.8;

%%Main loop
for It=1:max_iter 
    
    a=(cos(2*rand)+1)*(1-It/max_iter);
    C=a*(2*rand-1); 

% Exploration:

    for i=1:pop_size
        if rand<p    
            GX(i,:) =(ub-lb)*rand+lb;
        else  
            if rand>=0.5
                Z = unifrnd(-a,a,1,variables_no);
                H=Z.*X(i,:);   
                GX(i,:)=(rand-a)*X(randi([1,pop_size]),:)+C.*H; 
            else   
                GX(i,:)=X(i,:)-C.*(C*(X(i,:)- GX(randi([1,pop_size]),:))+rand*(X(i,:)-GX(randi([1,pop_size]),:))); %ok ok 
               
            end
             
        end
        GX(i,:) = Bounds(GX(i,:), lower_bound, upper_bound);
    end       
       

    
    % Group formation operation 
    for i=1:pop_size
         [New_Fit,Model1]= fobj(GX(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%          New_Fit= fobj(GX(i,:));          
         if New_Fit<Pop_Fit(i)
            Pop_Fit(i)=New_Fit;
            X(i,:)=GX(i,:);
            Mdl_best=Model1;
         end
         if New_Fit<Silverback_Score 
            Silverback_Score=New_Fit; 
            Silverback=GX(i,:);
         end
    end
    
% Exploitation:  
    for i=1:pop_size
       if a>=w  
            g=2^C;
            delta= (abs(mean(GX)).^g).^(1/g);
            GX(i,:)=C*delta.*(X(i,:)-Silverback)+X(i,:); 
       else
           
           if rand>=0.5
              h=randn(1,variables_no);
           else
              h=randn(1,1);
           end
           r1=rand; 
           GX(i,:)= Silverback-(Silverback*(2*r1-1)-X(i,:)*(2*r1-1)).*(Beta*h); 
           
       end
       GX(i,:) = Bounds(GX(i,:), lower_bound, upper_bound);
    end
   
%     GX = Bounds(GX, lower_bound, upper_bound);
    
    % Group formation operation    
    for i=1:pop_size
          [New_Fit,Model1]= fobj(GX(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
         if New_Fit<Pop_Fit(i)
            Pop_Fit(i)=New_Fit;
            X(i,:)=GX(i,:);
            Mdl_best=Model1;
         end
         if New_Fit<Silverback_Score 
            Silverback_Score=New_Fit; 
            Silverback=GX(i,:);
         end
    end
             
convergence_curve(It)=Silverback_Score;
% fprintf("In Iteration %d, best estimation of the global optimum is %4.4f \n ", It,Silverback_Score );
         
end 
end
%%
function [Score,Mdl_best,NGO_curve,Best_pos]=NGO_NGO(Search_Agents,Max_iterations,Lowerbound,Upperbound,dimensions,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
% tic
% disp('PLEASE WAIT, The program is running.')
Lowerbound=ones(1,dimensions).*(Lowerbound);                              % Lower limit for variables
Upperbound=ones(1,dimensions).*(Upperbound);                              % Upper limit for variables

X=[];
X_new=[];
fit=[];
fit_new=[];
NGO_curve=zeros(1,Max_iterations);

%
for i=1:dimensions
    X(:,i) = Lowerbound(i)+rand(Search_Agents,1).*(Upperbound(i) -Lowerbound(i));              % Initial population
end
for i =1:Search_Agents
%     L=X(i,:);
    [fit(i),Model{1,i}]= fobj(X(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%     fit(i)=fobj(L);                    % Fitness evaluation (Explained at the top of the page. )
end

for t=1:Max_iterations  % algorithm iteration    
    %  update: BEST proposed solution
    [best , blocation]=min(fit);
    
    if t==1
        xbest=X(blocation,:);                                           % Optimal location
        fbest=best;  
        % The optimization objective function
        Mdl_best=Model{1,blocation};
    elseif best<fbest
        fbest=best;
        xbest=X(blocation,:);
        Mdl_best=Model{1,blocation};
    end
    
    
    % UPDATE Northern goshawks based on PHASE1 and PHASE2
    
    for i=1:Search_Agents
        % Phase 1: Exploration
        I=round(1+rand);
        k=randperm(Search_Agents,1);
        P=X(k,:); % Eq. (3)
        F_P=fit(k);
        
        if fit(i)> F_P
            X_new(i,:)=X(i,:)+rand(1,dimensions) .* (P-I.*X(i,:)); % Eq. (4)
        else
            X_new(i,:)=X(i,:)+rand(1,dimensions) .* (X(i,:)-P); % Eq. (4)
        end
        X_new(i,:) = max(X_new(i,:),Lowerbound);X_new(i,:) = min(X_new(i,:),Upperbound);
        
        % update position based on Eq (5)
%         L=X_new(i,:);
%         fit_new(i)=fobj(L);
         [fit_new(i),Model1{1,i}]= fobj(X_new(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
        if(fit_new(i)<fit(i))
            X(i,:) = X_new(i,:);
            fit(i) = fit_new(i);
            Model{1,i}=Model1{1,i};
        end
        % END PHASE 1
        
        % PHASE 2 Exploitation
        R=0.02*(1-t/Max_iterations);% Eq.(6)
        X_new(i,:)= X(i,:)+ (-R+2*R*rand(1,dimensions)).*X(i,:);% Eq.(7)
        
        X_new(i,:) = max(X_new(i,:),Lowerbound);X_new(i,:) = min(X_new(i,:),Upperbound);
        
        % update position based on Eq (8)
         [fit_new(i),Model1{1,i}]= fobj(X_new(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
        if(fit_new(i)<fit(i))
            X(i,:) = X_new(i,:);
            fit(i) = fit_new(i);
            Model{1,i}=Model1{1,i};
        end
        % END PHASE 2
        
    end% end for i=1:N
    
    %
    % SAVE BEST SCORE
    best_so_far(t)=fbest; % save best solution so far
    average(t) = mean (fit);
    Score=fbest;
    Best_pos=xbest;
    NGO_curve(t)=Score;
end
%
end
%%
function [fval,Mdl_best,gbest_t,Xfood] = SO_SO(N,T, lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

%initial 
vec_flag=[1,-1];
Threshold=0.25;
Thresold2= 0.6;
C1=0.5;
C2=.05;
C3=2;
if length(lb)<2
X=lb+rand(N,dim)*(ub-lb);
else 
X=repmat(lb,N,1)+rand(N,dim).*repmat((ub-lb),N,1);
end
for i=1:N
     [fitness(i),Model1{1,i}]= fobj(X(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%  fitness(i)=fobj(X(i,:));   
end
[GYbest, gbest] = min(fitness);
Xfood = X(gbest,:);
Mdl_best=Model1{1,gbest};
%Diving the swarm into two equal groups males and females
Nm=round(N/2);%eq.(2&3)
Nf=N-Nm;
Xm=X(1:Nm,:);
Xf=X(Nm+1:N,:);
fitness_m=fitness(1:Nm);
fitness_f=fitness(Nm+1:N);
[fitnessBest_m, gbest1] = min(fitness_m);
Xbest_m = Xm(gbest1,:);
[fitnessBest_f, gbest2] = min(fitness_f);
Xbest_f = Xf(gbest2,:);
for t = 1:T
    Temp=exp(-((t)/T));  %eq.(4)
  Q=C1*exp(((t-T)/(T)));%eq.(5)
    if Q>1        Q=1;    end
    % Exploration Phase (no Food)
if Q<Threshold
    for i=1:Nm
        for j=1:1:dim
            rand_leader_index = floor(Nm*rand()+1);
            X_randm = Xm(rand_leader_index, :);
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            Am=exp(-fitness_m(rand_leader_index)/(fitness_m(i)+eps));%eq.(7)
            Xnewm(i,j)=X_randm(j)+Flag*C2*Am*((ub(1)-lb(1))*rand+lb(1));%eq.(6)
        end
    end
    for i=1:Nf
        for j=1:1:dim
            rand_leader_index = floor(Nf*rand()+1);
            X_randf = Xf(rand_leader_index, :);
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            Af=exp(-fitness_f(rand_leader_index)/(fitness_f(i)+eps));%eq.(9)
            Xnewf(i,j)=X_randf(j)+Flag*C2*Af*((ub(1)-lb(1))*rand+lb(1));%eq.(8)
        end
    end
else %Exploitation Phase (Food Exists)
    if Temp>Thresold2  %hot
        for i=1:Nm
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            for j=1:1:dim
                Xnewm(i,j)=Xfood(j)+C3*Flag*Temp*rand*(Xfood(j)-Xm(i,j));%eq.(10)
            end
        end
        for i=1:Nf
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            for j=1:1:dim
                Xnewf(i,j)=Xfood(j)+Flag*C3*Temp*rand*(Xfood(j)-Xf(i,j));%eq.(10)
            end
        end
    else %cold
        if rand>0.6 %fight
            for i=1:Nm
                for j=1:1:dim
                    FM=exp(-(fitnessBest_f)/(fitness_m(i)+eps));%eq.(13)
                    Xnewm(i,j)=Xm(i,j) +C3*FM*rand*(Q*Xbest_f(j)-Xm(i,j));%eq.(11)
                    
                end
            end
            for i=1:Nf
                for j=1:1:dim
                    FF=exp(-(fitnessBest_m)/(fitness_f(i)+eps));%eq.(14)
                    Xnewf(i,j)=Xf(i,j)+C3*FF*rand*(Q*Xbest_m(j)-Xf(i,j));%eq.(12)
                end
            end
        else%mating
            for i=1:Nf
                for j=1:1:dim
                    Mm=exp(-fitness_f(i)/(fitness_m(i)+eps));%eq.(17)
                    Xnewm(i,j)=Xm(i,j) +C3*rand*Mm*(Q*Xf(i,j)-Xm(i,j));%eq.(15
                end
            end
            for i=1:Nf
                for j=1:1:dim
                    Mf=exp(-fitness_m(i)/(fitness_f(i)+eps));%eq.(18)
                    Xnewf(i,j)=Xf(i,j) +C3*rand*Mf*(Q*Xm(i,j)-Xf(i,j));%eq.(16)
                end
            end
            flag_index = floor(2*rand()+1);
            egg=vec_flag(flag_index);
            if egg==1
                [GYworst, gworst] = max(fitness_m);
                Xnewm(gworst,:)=lb+rand*(ub-lb);%eq.(19)
                [GYworst, gworst] = max(fitness_f);
                Xnewf(gworst,:)=lb+rand*(ub-lb);%eq.(20)
            end
        end
    end
end
    for j=1:Nm
         Flag4ub=Xnewm(j,:)>ub;
         Flag4lb=Xnewm(j,:)<lb;
        Xnewm(j,:)=(Xnewm(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        [y,Model2]= fobj( Xnewm(j,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%         y = fobj(Xnewm(j,:));
        if y<fitness_m(j)
            fitness_m(j)=y;
            Model1{1,j}=Model2;
            Xm(j,:)= Xnewm(j,:);
          
        end
    end
    
    [Ybest1,gbest1] = min(fitness_m);
    Mdl_best=Model1{1,gbest1};
    for j=1:Nf
         Flag4ub=Xnewf(j,:)>ub;
         Flag4lb=Xnewf(j,:)<lb;
        Xnewf(j,:)=(Xnewf(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
               [y,Model2]= fobj( Xnewf(j,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%         y = fobj(Xnewm(j,:));
        if y<fitness_f(j)
            fitness_f(j)=y;
            Xf(j,:)= Xnewf(j,:);
            Model1{1,j}=Model2;
        end
    end
    
    [Ybest2,gbest2] = min(fitness_f);
%     disp(gbest2)
%     disp(Model1{1,gbest2});
    Mdl_best=Model1{1,gbest2};

    if Ybest1<fitnessBest_m
        Xbest_m = Xm(gbest1,:);
        fitnessBest_m=Ybest1;
    end
    if Ybest2<fitnessBest_f
        Xbest_f = Xf(gbest2,:);
        fitnessBest_f=Ybest2;
        
    end
    if Ybest1<Ybest2
        gbest_t(t)=min(Ybest1);
    else
        gbest_t(t)=min(Ybest2);
        
    end
    if fitnessBest_m<fitnessBest_f
        GYbest=fitnessBest_m;
        Xfood=Xbest_m;
    else
        GYbest=fitnessBest_f;
        Xfood=Xbest_f;
    end
    
end
fval = GYbest;
end
%%
%% ��������Ⱥ�Ż��㷨
% fval,Mdl_best,gbest_t,Xfood
function [gBestScore,Mdl_best,cg_curve,gBest]=PSO_PSO(N,Max_iteration,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

%PSO Infotmation
% if(max(size(ub)) == 1)
    UB=ub;
    LB=lb;
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
% end

Vmax=6;
noP=N;
wMax=0.9;
wMin=0.6;
c1=2;
c2=2;

% Initializations
iter=Max_iteration;
vel=zeros(noP,dim);
pBestScore=zeros(noP);
pBest=zeros(noP,dim);
gBest=zeros(1,dim);
cg_curve=zeros(1,iter);
vel=zeros(N,dim);
pos=zeros(N,dim);

%Initialization
for i=1:size(pos,1) 
    for j=1:size(pos,2) 
        pos(i,j)=(ub(j)-lb(j))*rand()+lb(j);
        vel(i,j)=0.3*rand();
    end
end

for i=1:noP
    pBestScore(i)=inf;
end

% Initialize gBestScore for a minimization problem
 gBestScore=inf;
     
 [~,Mdl]=fobj(pos(1,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
  Mdl_best=Mdl; 
for l=1:iter 
    
    % Return back the particles that go beyond the boundaries of the search
    % space
     Flag4ub=pos(i,:)>ub;
     Flag4lb=pos(i,:)<lb;
     pos(i,:)=(pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
%      disp(pos)
    for i=1:size(pos,1)     
        %Calculate objective function for each particle
        [fitness,Mdl]=fobj(pos(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);

        if(pBestScore(i)>fitness)
            pBestScore(i)=fitness;
            pBest(i,:)=pos(i,:);
            Mdl_best=Mdl;
        end
        if(gBestScore>fitness)
            gBestScore=fitness;
            gBest=pos(i,:);
        end
    end

    %Update the W of PSO
    w=wMax-l*((wMax-wMin)/iter);
    %Update the Velocity and Position of particles
    for i=1:size(pos,1)
        for j=1:size(pos,2)       
            vel(i,j)=w*vel(i,j)+c1*rand()*(pBest(i,j)-pos(i,j))+c2*rand()*(gBest(j)-pos(i,j));
            
            if(vel(i,j)>Vmax)
                vel(i,j)=Vmax;
            end
            if(vel(i,j)<-Vmax)
                vel(i,j)=-Vmax;
            end            
            pos(i,j)=pos(i,j)+vel(i,j);
            if (pos(i,j)>UB(j))
                pos(i,j)=UB(j);
            elseif (pos(i,j)<LB(j))
                pos(i,j)=LB(j);
            end
        end
    end
    cg_curve(l)=gBestScore;
end

end
%% ģ���˻�
%[gBestScore,Mdl_best,cg_curve,gBest]=PSO_PSO(N,Max_iteration,lb,ub,dim,fobj,data_O,data_Z)
function [Best_score,Mdl_best,curve,Best_pos]=SA_SA(N,Mmax,l,u,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
%function [x0,f0]=sim_anl(f,x0,l,u,Mmax,TolFun)
% ����: 
%        fobj = ��Ӧ�Ⱥ���
%        x0 = ������Ⱥ
%        l = ��Ⱥ�±߽�
%        u = ��Ⱥ�ϱ߽�
%        Mmax = ����¶�
%        TolFun = �Ż��仯���̶�
%
% ���: 
%        x0 = ����Ż������Ⱥ
%        f0 = ����Ż������Ⱥ����Ӧ��ֵ
TolFun = 10E-10;%ģ���˻����̶�
x0 = (u-l).*rand(1,dim)+l;%�����ʼ��ģ���˻�;
% f = fobj;%��Ӧ�Ⱥ���
x=x0;
% fx=feval(f,x);%������Ӧ��ֵ
[fx,Mdl]=fobj(x0,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
Mdl_best=Mdl;
f0=fx;
count = 1;%���ڼ�¼�������߱��
%ģ���˻���Ҫ����
for m=1:Mmax
    T=m/Mmax; %�¶�
    mu=10^(T*1000);  
    %For each temperature we take 100 test points to simulate reach termal
    for k=0:N
        dx=mu_inv(2*rand(1,dim)-1,mu).*(u-l);
        if(isnan(dx))
            dx=0;
        end
%         disp(dx)
        x1=x+dx;
        %�߽紦���ֹԽ��
        x1=(x1 < l).*l+(l <= x1).*(x1 <= u).*x1+(u < x1).*u;
%         disp(x1)
        %���㵱ǰλ����Ӧ��ֵ����Ӧ��ֵƫ��
        [fx1,Mdl]=fobj(x1,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);df=fx1-fx;
        % ���df<0����ܸý⣬�������0 ������Metropolis׼������ж��Ƿ����       
        if (df < 0 || rand < exp(-T*df/(abs(fx)+eps)/TolFun))==1
            x=x1;fx=fx1;
            Mdl_best=Mdl;
        end        
        %�жϵ�ǰ���Ƿ���ţ����������.       
        if fx1 < f0 ==1
            x0=x1;f0=fx1;
        end 
    end
     curve(count) = f0;
     count = count+1;
end
Best_pos = x0;
Best_score = f0;
end

function x=mu_inv(y,mu)
%ģ���˻������λ��ƫ��
x=(((1+mu).^abs(y)-1)/mu).*sign(y);
end
%%
function [fmin0,Mdl_best,ccurve,gbest]=WSO_WSO(whiteSharks,itemax,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)  
% Convergence curve
ccurve=zeros(1,itemax);

%%% Show the convergence curve
%     figure (1);
%     set(gcf,'color','w');
%     hold on
%     xlabel('Iteration','interpreter','latex','FontName','Times','fontsize',10)
%     ylabel('fitness value','interpreter','latex','FontName','Times','fontsize',10); 
%     grid;

% Start the WSO  Algorithm
% Generation of initial solutions
WSO_Positions=initialization(whiteSharks,dim,ub,lb);% Initial population

% initial velocity
v=0.0*WSO_Positions; 

% Evaluate the fitness of the initial population
fit=zeros(whiteSharks,1);

for i=1:whiteSharks
    [fit(i,1),Model1{1,i}]= fobj(WSO_Positions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%      fit(i,1)=fobj(WSO_Positions(i,:));
end

% Initalize the parameters of WSO
fitness=fit; % Initial fitness of the random positions of the WSO
 model2=Model1;
[fmin0,index]=min(fit);
Mdl_best=Model1{1,index};

wbest = WSO_Positions; % Best position initialization
gbest = WSO_Positions(index,:); % initial global position

% WSO Parameters
    fmax=0.75; %  Maximum frequency of the wavy motion
    fmin=0.07; %  Minimum frequency of the wavy motion   
    tau=4.11;  
       
    mu=2/abs(2-tau-sqrt(tau^2-4*tau));

    pmin=0.5;
    pmax=1.5;
    a0=6.250;  
    a1=100;
    a2=0.0005;
  % Start the iterative process of WSO 
for ite=1:itemax

    mv=1/(a0+exp((itemax/2.0-ite)/a1)); 
    s_s=abs((1-exp(-a2*ite/itemax))) ;
 
    p1=pmax+(pmax-pmin)*exp(-(4*ite/itemax)^2);
    p2=pmin+(pmax-pmin)*exp(-(4*ite/itemax)^2);
    
 % Update the speed of the white sharks in water  
     nu=floor((whiteSharks).*rand(1,whiteSharks))+1;

     for i=1:size(WSO_Positions,1)
           rmin=1; rmax=3.0;
          rr=rmin+rand()*(rmax-rmin);
          wr=abs(((2*rand()) - (1*rand()+rand()))/rr);       
          v(i,:)=  mu*v(i,:) +  wr *(wbest(nu(i),:)-WSO_Positions(i,:));
           %% or                

%          v(i,:)=  mu*(v(i,:)+ p1*(gbest-WSO_Positions(i,:))*rand+.... 
%                    + p2*(wbest(nu(i),:)-WSO_Positions(i,:))*rand);          
     end
 
 % Update the white shark position
     for i=1:size(WSO_Positions,1)
       
        f =fmin+(fmax-fmin)/(fmax+fmin);
         
        a=sign(WSO_Positions(i,:)-ub)>0;
        b=sign(WSO_Positions(i,:)-lb)<0;
         
        wo=xor(a,b);

        % locate the prey based on its sensing (sound, waves)
            if rand<mv
                WSO_Positions(i,:)=  WSO_Positions(i,:).*(~wo) + (ub.*a+lb.*b); % random allocation  
            else   
                WSO_Positions(i,:) = WSO_Positions(i,:)+ v(i,:)/f;  % based on the wavy motion
            end
    end 
    
    % Update the position of white sharks consides_sng fishing school 
for i=1:size(WSO_Positions,1)
        for j=1:size(WSO_Positions,2)
            if rand<s_s      
                
             Dist=abs(rand*(gbest(j)-1*WSO_Positions(i,j)));
             
                if(i==1)
                    WSO_Positions(i,j)=gbest(j)+rand*Dist*sign(rand-0.5);
                else    
                    WSO_Pos(i,j)= gbest(j)+rand*Dist*sign(rand-0.5);
                    WSO_Positions(i,j)=(WSO_Pos(i,j)+WSO_Positions(i-1,j))/2*rand;
                end   
            end
         
        end       
end
%     

% Update global, best and new positions
 
    for i=1:whiteSharks 
        % Handling boundary violations
           if WSO_Positions(i,:)>=lb & WSO_Positions(i,:)<=ub%         
            % Find the fitness
             [fit(i,1),Model1{1,i}]= fobj(WSO_Positions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
%               fit(i)=fobj(WSO_Positions(i,:));    
              
             % Evaluate the fitness
            if fit(i)<fitness(i)
                 wbest(i,:) = WSO_Positions(i,:); % Update the best positions
                 fitness(i)=fit(i);   % Update the fitness
                 model2{1,i}=Model1{1,i};
            end
            
            % Finding out the best positions
            if (fitness(i)<fmin0)
               fmin0=fitness(i);
               gbest = wbest(index,:); % Update the global best positions
               Mdl_best=model2{1,index};
            end 
            
        end
    end

% Obtain the results
%   outmsg = ['Iteration# ', num2str(ite) , '  Fitness= ' , num2str(fmin0)];
%   disp(outmsg);
% 
%  ccurve(ite)=fmin0; % Best found value until iteration ite
% 
%  if ite>2
%         line([ite-1 ite], [ccurve(ite-1) ccurve(ite)],'Color','b'); 
%         title({'Convergence characteristic curve'},'interpreter','latex','FontName','Times','fontsize',12);
%         xlabel('Iteration');
%         ylabel('Best score obtained so far');
%         drawnow 
%  end 
  
end 
end
%%
function [fMin , Mdl_best, Convergence_curve,bestX ] = CGO_CGO(pop, M,c,d,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )
LB = c.*ones(1,dim) ;  %Ѱ������
UB = d.*ones(1,dim) ;  % Ѱ������
% M = 1000 ; % ����������
% pop = 25 ; % ��Ⱥ��ģ

%% ��ʼ��λ�ú���Ӧ��
for i=1:pop   
    x(i,:)=LB + (UB-LB).*rand(1, dim);
    [fit( i ),model{1,i}]= fobj( x( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )  ;  
end

%% ��ʼ����
for Iter=1:M
    for i=1:pop
        %% ���µ�ǰ���Ž�
        [fMin,bestI]=min(fit);
        bestX=x(bestI,:);
        Mdl_best=model{1,bestI};
        %% ��ѡ�����
        % Random Numbers
        I=randi([1,2],1,12); % Beta and Gamma
        Ir=randi([0,1],1,5);
        % Random Groups
        RandGroupNumber=randperm(pop,1);
        RandGroup=randperm(pop,RandGroupNumber);
        % Mean of Random Group
        MeanGroup=mean(x(RandGroup,:)).*(length(RandGroup)~=1)...
            +x(RandGroup(1,1),:)*(length(RandGroup)==1);   
        % New Seeds
        Alfa(1,:)=rand(1,dim);
        Alfa(2,:)= 2*rand(1,dim)-1;
        Alfa(3,:)= (Ir(1)*rand(1,dim)+1);
        Alfa(4,:)= (Ir(2)*rand(1,dim)+(~Ir(2)));   
        ii=randi([1,4],1,3);
        SelectedAlfa=Alfa(ii,:);

        NewSeed(1,:)=x(i,:)+SelectedAlfa(1,:).*(I(1)*bestX-I(2)*MeanGroup);
        NewSeed(2,:)=bestX+SelectedAlfa(2,:).*(I(3)*MeanGroup-I(4)*x(i,:));
        NewSeed(3,:)=MeanGroup+SelectedAlfa(3,:).*(I(5)*bestX-I(6)*x(i,:));
        NewSeed(4,:)=unifrnd(LB,UB);
        for j=1:4
            % Checking/Updating the boundary limits for Seeds
            NewSeed(j,:)=bound_CGO(NewSeed(j,:),UB,LB);
            % Evaluating New Solutions
            [Fun_evalNew(j),Mdl_f]=fobj(NewSeed( j, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
            x(size(x,1)+1,:)=Fun_evalNew(j);
            model{1,size(x,1)+1}=Mdl_f;
        end
        % x=[x; NewSeed];       
        fit=[fit, Fun_evalNew];
    end
    [fit, SortOrder]=sort(fit);
    x=x(SortOrder,:);
    [fMin,bestI]=min(fit);   
    bestX=x(bestI,:);
    Mdl_best=model{1,bestI};
    x=x(1:pop,:);
    fit=fit(1:pop);
    % Store Best Cost Ever Found
    Convergence_curve(Iter)=fMin;
    % Show Iteration Information
    % disp(['Iteration ' num2str(Iter) ': Best Cost = ' num2str(Conv_History(Iter))]);
end
function x=bound_CGO(x,UB,LB)
x(x>UB)=UB(x>UB); x(x<LB)=LB(x<LB);
end
end
%%  INFO/INFO
function [Best_Cost,Mdl_best,Convergence_curve,Best_X]=INFO_INFO(nP,MaxIt,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )
%% Initialization
Cost=zeros(nP,1);
M=zeros(nP,1);

X=initialization(nP,dim,ub,lb);

for i=1:nP
    [Cost(i),model{1,i}] = fobj(X(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
    M(i)=Cost(i);
end
Mdl_new1=model;
[~, ind]=sort(Cost);
Best_X = X(ind(1),:);
Best_Cost = Cost(ind(1));
Mdl_best=model{1,ind(1)};
Worst_Cost = Cost(ind(end));
Worst_X = X(ind(end),:);

I=randi([2 3]);
Better_X=X(ind(I),:);
Better_Cost=Cost(ind(I));

%% Main Loop of INFO
for it=1:MaxIt
    alpha=2*exp(-4*(it/MaxIt));                                                           % Eqs. (5.1) & % Eq. (9.1)

    M_Best=Best_Cost;
    M_Better=Better_Cost;
    M_Worst=Worst_Cost;

    for i=1:nP

        % Updating rule stage
        del=2*rand*alpha-alpha;                                                           % Eq. (5)
        sigm=2*rand*alpha-alpha;                                                          % Eq. (9)

        % Select three random solution
        A1=randperm(nP);
        % A1(A1==i)=[];
        a=A1(1);b=A1(2);c=A1(3);

        e=1e-25;
        epsi=e*rand;

        omg = max([M(a) M(b) M(c)]);
        MM = [(M(a)-M(b)) (M(a)-M(c)) (M(b)-M(c))];

        W(1) = cos(MM(1)+pi)*exp(-(MM(1))/omg);                                           % Eq. (4.2)
        W(2) = cos(MM(2)+pi)*exp(-(MM(2))/omg);                                           % Eq. (4.3)
        W(3)= cos(MM(3)+pi)*exp(-(MM(3))/omg);                                            % Eq. (4.4)
        Wt = sum(W);

        WM1 = del.*(W(1).*(X(a,:)-X(b,:))+W(2).*(X(a,:)-X(c,:))+ ...                      % Eq. (4.1)
            W(3).*(X(b,:)-X(c,:)))/(Wt+1)+epsi;

        omg = max([M_Best M_Better M_Worst]);
        MM = [(M_Best-M_Better) (M_Best-M_Better) (M_Better-M_Worst)];

        W(1) = cos(MM(1)+pi)*exp(-MM(1)/omg);                                             % Eq. (4.7)
        W(2) = cos(MM(2)+pi)*exp(-MM(2)/omg);                                             % Eq. (4.8)
        W(3) = cos(MM(3)+pi)*exp(-MM(3)/omg);                                             % Eq. (4.9)
        Wt = sum(W);

        WM2 = del.*(W(1).*(Best_X-Better_X)+W(2).*(Best_X-Worst_X)+ ...                   % Eq. (4.6)
            W(3).*(Better_X-Worst_X))/(Wt+1)+epsi;

        % Determine MeanRule
        r = unifrnd(0.1,0.5);
        MeanRule = r.*WM1+(1-r).*WM2;                                                     % Eq. (4)

        if rand<0.5
            z1 = X(i,:)+sigm.*(rand.*MeanRule)+randn.*(Best_X-X(a,:))/(M_Best-M(a)+1);
            z2 = Best_X+sigm.*(rand.*MeanRule)+randn.*(X(a,:)-X(b,:))/(M(a)-M(b)+1);
        else                                                                              % Eq. (8)
            z1 = X(a,:)+sigm.*(rand.*MeanRule)+randn.*(X(b,:)-X(c,:))/(M(b)-M(c)+1);
            z2 = Better_X+sigm.*(rand.*MeanRule)+randn.*(X(a,:)-X(b,:))/(M(a)-M(b)+1);
        end

        % Vector combining stage
        u=zeros(1,dim);
        for j=1:dim
            mu = 0.05*randn;
            if rand <0.5
                if rand<0.5
                    u(j) = z1(j) + mu*abs(z1(j)-z2(j));                                   % Eq. (10.1)
                else
                    u(j) = z2(j) + mu*abs(z1(j)-z2(j));                                   % Eq. (10.2)
                end
            else
                u(j) = X(i,j);                                                            % Eq. (10.3)
            end
        end

        % Local search stage
        if rand<0.5
            L=rand<0.5;v1=(1-L)*2*(rand)+L;v2=rand.*L+(1-L);                              % Eqs. (11.5) & % Eq. (11.6)
            Xavg=(X(a,:)+X(b,:)+X(c,:))/3;                                                % Eq. (11.4)
            phi=rand;
            Xrnd = phi.*(Xavg)+(1-phi)*(phi.*Better_X+(1-phi).*Best_X);                   % Eq. (11.3)
            Randn = L.*randn(1,dim)+(1-L).*randn;
            if rand<0.5
                u = Best_X + Randn.*(MeanRule+randn.*(Best_X-X(a,:)));                    % Eq. (11.1)
            else
                u = Xrnd + Randn.*(MeanRule+randn.*(v1*Best_X-v2*Xrnd));                  % Eq. (11.2)
            end

        end

        % Check if new solution go outside the search space and bring them back
        New_X= BC(u,lb,ub);
        [New_Cost,Mdl_new] = fobj(New_X,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );

        if New_Cost<Cost(i)
            X(i,:)=New_X;
            Cost(i)=New_Cost;
            M(i)=Cost(i);
            Mdl_new1{1,i}=Mdl_new;
            if Cost(i)<Best_Cost
                Best_X=X(i,:);
                Best_Cost = Cost(i);
                Mdl_best=Mdl_new1{1,i};
            end
        end
    end

    % Determine the worst solution
    [~, ind]=sort(Cost);
    Worst_X=X(ind(end),:);
    Worst_Cost=Cost(ind(end));
    % Determine the better solution
    I=randi([2 3]);
    Better_X=X(ind(I),:);
    Better_Cost=Cost(ind(I));

    % Update Convergence_curve
    Convergence_curve(it)=Best_Cost;
    % 
    % Show Iteration Information
    % disp(['Iteration ' num2str(it) ',: Best Cost = ' num2str(Best_Cost)]);
end
    function X = BC(X,lb,ub)
    Flag4ub=X>ub;
    Flag4lb=X<lb;
    X=(X.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    end
function X=initialization(nP,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    X=rand(nP,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for in=1:dim
        ub_i=ub(in);
        lb_i=lb(in);
        X(:,in)=rand(nP,1).*(ub_i-lb_i)+lb_i;
    end
end
end
end

%%  COA�����Ż��㷨
function[Best_score, Mdl_best,COA_curve,Best_pos]=COA_COA(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

lowerbound=ones(1,dimension).*(lowerbound);                              % Lower limit for variables
upperbound=ones(1,dimension).*(upperbound);                              % Upper limit for variables

% INITIALIZATION
for i=1:dimension
    X(:,i) = lowerbound(i)+rand(SearchAgents,1).*(upperbound(i) - lowerbound(i));                          % Initial population
end

for i =1:SearchAgents
    L=X(i,:);
    % fit(i)=fobj(L);
    [fit( i ),model{1,i}]= fobj( X( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )  ;  

end
model_m=model;
%

for t=1:Max_iterations
    % update the best condidate solution
    [best , location]=min(fit);
    if t==1
        Xbest=X(location,:);  
        Mdl_best=model{1,location};                           % Optimal location
        fbest=best;                                           % The optimization objective function
    elseif best<fbest
        fbest=best;
        Xbest=X(location,:);
        Mdl_best=model{1,location};
    end



    %
    for i=1:SearchAgents/2

        % Phase1: Hunting and attacking strategy on iguana (Exploration Phase)
        iguana=Xbest;
        I=round(1+rand(1,1));

        X_P1(i,:)=X(i,:)+rand(1,1) .* (iguana-I.*X(i,:)); % Eq. (4)
        X_P1(i,:) = max(X_P1(i,:),lowerbound);X_P1(i,:) = min(X_P1(i,:),upperbound);

        % update position based on Eq (7)
        L=X_P1(i,:);
       [F_P1( i ),model_m{1,i}]= fobj( X_P1(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )  ;  

        % F_P1(i)=fobj(L);
        if(F_P1(i)<fit(i))
            X(i,:) = X_P1(i,:);
            fit(i) = F_P1(i);
            model{1,i}=model_m{1,i};
        end

    end
    %%

    % for i=1+SearchAgents/2 :SearchAgents
    % 
    %     iguana=lowerbound+rand.*(upperbound-lowerbound); %Eq(5)
    %     L=iguana;
    %     F_HL=fobj(L,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
    %     I=round(1+rand(1,1));
    % 
    %     if fit(i)> F_HL
    %         X_P1(i,:)=X(i,:)+rand(1,1) .* (iguana-I.*X(i,:)); % Eq. (6)
    %     else
    %         X_P1(i,:)=X(i,:)+rand(1,1) .* (X(i,:)-iguana); % Eq. (6)
    %     end
    %     X_P1(i,:) = max(X_P1(i,:),lowerbound);X_P1(i,:) = min(X_P1(i,:),upperbound);
    % 
    %     % update position based on Eq (7)
    %     L=X_P1(i,:);
    % 
    %     F_P1(i)=fobj(L,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
    %     if(F_P1(i)<fit(i))
    %         X(i,:) = X_P1(i,:);
    %         fit(i) = F_P1(i);
    %     end
    % end
    % END Phase1: Hunting and attacking strategy on iguana (Exploration Phase)

    %

    % Phase2: The process of escaping from predators (Exploitation Phase)
    for i=1:SearchAgents
        LO_LOCAL=lowerbound/t;% Eq(9)
        HI_LOCAL=upperbound/t;% Eq(10)

        X_P2(i,:)=X(i,:)+(1-2*rand).* (LO_LOCAL+rand(1,1) .* (HI_LOCAL-LO_LOCAL)); % Eq. (8)
        X_P2(i,:) = max(X_P2(i,:),LO_LOCAL);X_P2(i,:) = min(X_P2(i,:),HI_LOCAL);

        % update position based on Eq (11)
        L=X_P2(i,:);
        [F_P2( i ),model_m{1,i}]= fobj( X_P2(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )  ;  
        % F_P2(i)=fobj(L);
        if(F_P2(i)<fit(i))
            X(i,:) = X_P2(i,:);
            model{1,i}=model_m{1,i};
            fit(i) = F_P2(i);
        end

    end
    % END Phase2: The process of escaping from predators (Exploitation Phase)


    best_so_far(t)=fbest;
    average(t) = mean (fit);

end
Best_score=fbest;
Best_pos=Xbest;
COA_curve=best_so_far;
end

%% rime
function [Best_rime_rate,Mdl_best,Convergence_curve,Best_rime]=RIME_RIME(N,Max_iter,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
% disp('RIME is now tackling your problem')

% initialize position
Best_rime=zeros(1,dim);
Best_rime_rate=inf;%change this to -inf for maximization problems
Rimepop=initialization(N,dim,ub,lb);%Initialize the set of random solutions
Lb=lb.*ones(1,dim);% lower boundary 
Ub=ub.*ones(1,dim);% upper boundary
it=1;%Number of iterations
Convergence_curve=zeros(1,Max_iter);
Rime_rates=zeros(1,N);%Initialize the fitness value
newRime_rates=zeros(1,N);
W = 5;%Soft-rime parameters, discussed in subsection 4.3.1 of the paper
%Calculate the fitness value of the initial position

for i=1:N
    [Rime_rates(1,i),model{1,i}]=fobj(Rimepop(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );%Calculate the fitness value for each search agent
    %Make greedy selections
    if Rime_rates(1,i)<Best_rime_rate
        Best_rime_rate=Rime_rates(1,i);
        Best_rime=Rimepop(i,:);
        Mdl_best=model{1,i};
    end
end
% Main loop
while it <= Max_iter
    RimeFactor = (rand-0.5)*2*cos((pi*it/(Max_iter/10)))*(1-round(it*W/Max_iter)/W);%Parameters of Eq.(3),(4),(5)
    E =sqrt(it/Max_iter);%Eq.(6)
    newRimepop = Rimepop;%Recording new populations
    normalized_rime_rates=normr(Rime_rates);%Parameters of Eq.(7)
    for i=1:N
        for j=1:dim
            %Soft-rime search strategy
            r1=rand();
            if r1< E
                newRimepop(i,j)=Best_rime(1,j)+RimeFactor*((Ub(j)-Lb(j))*rand+Lb(j));%Eq.(3)
            end
            %Hard-rime puncture mechanism
            r2=rand();
            if r2<normalized_rime_rates(i)
                newRimepop(i,j)=Best_rime(1,j);%Eq.(7)
            end
        end
    end
    for i=1:N
        %Boundary absorption
        Flag4ub=newRimepop(i,:)>ub;
        Flag4lb=newRimepop(i,:)<lb;
        newRimepop(i,:)=(newRimepop(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        [newRime_rates(1,i),modelm{1,i}]=fobj(newRimepop(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
        %Positive greedy selection mechanism
        if newRime_rates(1,i)<Rime_rates(1,i)
            Rime_rates(1,i) = newRime_rates(1,i);
            Rimepop(i,:) = newRimepop(i,:);
            if newRime_rates(1,i)< Best_rime_rate
               Best_rime_rate=Rime_rates(1,i);
               Best_rime=Rimepop(i,:);
               Mdl_best=modelm{1,i};
            end
        end
    end
    Convergence_curve(it)=Best_rime_rate;
    it=it+1;
end

function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for in=1:dim
        ub_i=ub(in);
        lb_i=lb(in);
        Positions(:,in)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
end

%%
% The Kepler Optimization Algorithm
function [Sun_Score,Mdl_best,Convergence_curve,Sun_Pos]=KOA_KOA(SearchAgents_no,Tmax,ub,lb,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

%%%%-------------------Definitions--------------------------%%
%
Sun_Pos=zeros(1,dim); % A vector to include the best-so-far Solution, representing the Sun
Sun_Score=inf; % A Scalar variable to include the best-so-far score
Convergence_curve=zeros(1,Tmax);

%%-------------------Controlling parameters--------------------------%%
%
Tc=3;
M0=0.1;
lambda=15;
% Step 1: Initialization process
%---------------Initialization----------------------%%
 % Orbital Eccentricity (e)   
orbital=rand(1,SearchAgents_no); %% Eq.(4)
 % Orbital Period (T) 
T=abs(randn(1,SearchAgents_no)); %% Eq.(5)
Positions=initialization(SearchAgents_no,dim,ub,lb); % Initialize the positions of planets
t=0; %% Function evaluation counter 
%
%---------------------Evaluation-----------------------%%
for i=1:SearchAgents_no
    %=
    % PL_Fit(i)=feval(fhd, Positions(i,:)',fobj);
    [PL_Fit( i ),model{1,i}]= fobj(  Positions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )  ;  
    % Update the best-so-far solution
    if PL_Fit(i)<Sun_Score % Change this to > for maximization problem
       Sun_Score=PL_Fit(i); % Update the best-so-far score
       Sun_Pos=Positions(i,:); % Update te best-so-far solution
       Mdl_best=model{1,i};
    end
end

while t<Tmax %% Termination condition  
 [Order] = sort(PL_Fit);  % Sorting the fitness values of the solutions in current population
 % The worst Fitness value at function evaluation t
 worstFitness = Order(SearchAgents_no); %% Eq.(11)
 M=M0*(exp(-lambda*(t/Tmax))); %% Eq. (12)
 % Computer R that represents the Euclidian distance between the best-so-far solution and the ith solution
 for i=1:SearchAgents_no
    R(i)=0;
    for j=1:dim
       R(i)=R(i)+(Sun_Pos(j)-Positions(i,j))^2; %% Eq.(7)
    end
    R(i)=sqrt(R(i));
 end
 % The mass of the Sun and object i at time t is computed as follows:
 for i=1:SearchAgents_no
    sum=0;
    for k=1:SearchAgents_no
        sum=sum+(PL_Fit(k)-worstFitness);
    end
    MS(i)=rand*(Sun_Score-worstFitness)/(sum); %% Eq.(8)
    m(i)=(PL_Fit(i)-worstFitness)/(sum); %% Eq.(9)
 end
 
 % Step 2: Defining the gravitational force (F)
 % Computing the attraction force of the Sun and the ith planet according to the universal law of gravitation:
 for i=1:SearchAgents_no
    Rnorm(i)=(R(i)-min(R))/(max(R)-min(R)); %% The normalized R (Eq.(24))
    MSnorm(i)=(MS(i)-min(MS))/(max(MS)-min(MS)); %% The normalized MS
    Mnorm(i)=(m(i)-min(m))/(max(m)-min(m)); %% The normalized m
    Fg(i)=orbital(i)*M*((MSnorm(i)*Mnorm(i))/(Rnorm(i)*Rnorm(i)+eps))+(rand); %% Eq.(6)
 end
 % a1 represents the semimajor axis of the elliptical orbit of object i at time t,
 for i=1:SearchAgents_no
    a1(i)=rand*(T(i)^2*(M*(MS(i)+m(i))/(4*pi*pi)))^(1/3); %% Eq.(23)
 end
 
 for i=1:SearchAgents_no
    % a2 is a cyclic controlling parameter that is decreasing gradually from -1 to ?2
    a2=-1+-1*(rem(t,Tmax/Tc)/(Tmax/Tc)); %% Eq.(29)
    % ? is a linearly decreasing factor from 1 to ?2
    n=(a2-1)*rand+1; %% Eq.(28)
    a=randi(SearchAgents_no);  %% An index of a solution selected at random
    b=randi(SearchAgents_no); %% An index of a solution selected at random
    rd=rand(1,dim); %% A vector generated according to the normal distribution
    r=rand; %% r1 is a random number in [0,1]
    % A randomly-assigned binary vector 
    U1=rd<r; %% Eq.(21)  
    O_P=Positions(i,:); %% Storing the current position of the ith solution
    % Step 6: Updating distance with the Sun
    if rand<rand
         % h is an adaptive factor for controlling the distance between the Sun and the current planet at time t
         h=(1/(exp(n.*randn))); %% Eq.(27)
         % An verage vector based on three solutions: the Current solution, best-so-far solution, and randomly-selected solution
         Xm=(Positions(b,:)+Sun_Pos+Positions(i,:))/3.0; 
         Positions(i,:)=Positions(i,:).*U1+(Xm+h.*(Xm-Positions(a,:))).*(1-U1); %% Eq.(26)
    else
        
        % Step 3: Calculating an object�1�7 velocity
        % A flag to opposite or leave the search direction of the current planet
         if rand<0.5 %% Eq.(18)
           f=1;
         else
           f=-1;
         end
         L=(M*(MS(i)+m(i))*abs((2/(R(i)+eps))-(1/(a1(i)+eps))))^(0.5); %% Eq.(15)
         U=rd>rand(1,dim); %% A binary vector
         if Rnorm(i)<0.5 %% Eq.(13)
            M=(rand.*(1-r)+r); %% Eq.(16)
            l=L*M*U; %% Eq.(14)
            Mv=(rand*(1-rd)+rd); %% Eq.(20)
            l1=L.*Mv.*(1-U);%% Eq.(19)
            V(i,:)=l.*(2*rand*Positions(i,:)-Positions(a,:))+l1.*(Positions(b,:)-Positions(a,:))+(1-Rnorm(i))*f*U1.*rand(1,dim).*(ub-lb); %% Eq.(13a)
         else
            U2=rand>rand; %% Eq. (22) 
            V(i,:)=rand.*L.*(Positions(a,:)-Positions(i,:))+(1-Rnorm(i))*f*U2*rand(1,dim).*(rand*ub-lb);  %% Eq.(13b)
         end %% End IF
         
         % Step 4: Escaping from the local optimum
         % Update the flag f to opposite or leave the search direction of the current planet  
         if rand<0.5 %% Eq.(18)
            f=1;
         else
            f=-1;
         end
         % Step 5
         Positions(i,:)=((Positions(i,:)+V(i,:).*f)+(Fg(i)+abs(randn))*U.*(Sun_Pos-Positions(i,:))); %% Eq.(25)
    end %% End If
    % Return the search agents that exceed the search space's bounds
    if rand<rand
       for j=1:size(Positions,2)
          if  Positions(i,j)>ub(j)
              Positions(i,j)=lb(j)+rand*(ub(j)-lb(j));
          elseif  Positions(i,j)<lb(j)
              Positions(i,j)=lb(j)+rand*(ub(j)-lb(j));
          end % End If
       end   % End For
    else
       Positions(i,:) = min(max(Positions(i,:),lb),ub);
    end % End If
    % 
    % Calculate objective function for each search agent
    % PL_Fit1=feval(fhd, Positions(i,:)',fobj);
    [PL_Fit1,model_get]= fobj( Positions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm )  ;  

 %  Step 7: Elitism, Eq.(30)
    if PL_Fit1<PL_Fit(i) % Change this to > for maximization problem
       PL_Fit(i)=PL_Fit1; % 
       model{1,i}=Mdl_best;
       % Update the best-so-far solution
       if PL_Fit(i)<Sun_Score % Change this to > for maximization problem
           Sun_Score=PL_Fit(i); % Update the best-so-far score
           Sun_Pos=Positions(i,:); % Update te best-so-far solution
           Mdl_best=model{1,i};
       end
    else
       Positions(i,:)=O_P;
    end %% End IF
    t=t+1; %% Increment the current function evaluation
    if t>Tmax %% Checking the termination condition
        break;
    end %% End IF
    Convergence_curve(t)=Sun_Score; %% Set the best-so-far fitness value at function evaluation t in the convergence curve 
 end %% End for i
end %% End while 


function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= length(ub); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
     Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for in=1:dim
        ub_i=ub(in);
        lb_i=lb(in);
         Positions(:,in)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;      
    end
end
end

end% End Function

%%
%%
function [fMin,Mdl_best,Convergence_curve,bestX]=GSSA_GSSA(pop,M,LB,UB,nvars,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size
%������ռ������Ⱥ��0.2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pNum = round( pop *  P_percent );    % The population size of the producers
%����������ȡ��
dim=nvars;
lb= LB.*ones( 1,dim );    % Lower limit/bounds/     a vector    Լ������
ub= UB.*ones( 1,dim );    % Upper limit/bounds/     a vector  Լ������
% �Ľ���ʼӳ��ѡȡ
label=4;  %ӳ�� % 1-6 �ֱ��Ӧ tent��chebyshev��Singer��Logistic��Sine, Circle
x = yinshe( pop, dim,label,ub,lb);
x = min(UB,x);        % bound violating to upper bound
x = max(LB,x);        % bound violating to lower bound
%Initialization
for i = 1 : pop
    % x( i, : ) = lb + (ub - lb) .* rand( 1, dim );   %�����ʼ��n����Ⱥ
    [fit( i ),model{1,i}] = fobj( x( i, : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm ) ;               %��������Ⱥ�����Ӧ������������Сֵ��,ԽС������Ӧ��Խ��
end
% disp(1111)
% �����ҵ���Сֵ��Ӧ����ȸȺ
pFit = fit;
pX = x;                            % The individual's best position corresponding to the pFit
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX denotes the global optimum position corresponding to fMin
Mdl_best=model{1,bestI};
% disp(333)
% Start updating the solutions.
for t = 1 : M
    [ ~, sortIndex ] = sort( pFit );% Sort.

    [fmax,B]=max( pFit );
    worse= x(B,:);         %�ҵ����ĸ���

    r2=rand(1);      %���������    �о�û��ɶ��ѧ���ݣ����������
    %�����˼������0.8������ԭ����Ⱥ��һ��С��1��������Ⱥ������ֵ��С��
    %�����˼������0.2������ԭ����Ⱥ��һ��С��1��������Ⱥ������ֵ+1
    %�仯����Ⱥ����ֵ����Ҫ������Լ������
    %��ǰpNum��Ӧ����õĽ��б仯 ���������߽��б仯���ɼ�������������õ�
    if(r2<0.8)
        for i = 1 : pNum                                                        % Equation (3)
            r1=rand(1);
            x( sortIndex( i ), : ) = pX( sortIndex( i ), : )*exp(-(i)/(r1*M)); %����Ⱥ����Ӧ����������
            x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );  %����Ⱥ������Լ����Χ��
            %         [fit( sortIndex( i ) ),model{1,sortIndex( i )}]
            [fit( sortIndex( i ) ),model{1,sortIndex( i )}]= fobj( x( sortIndex( i ), : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
        end
    else
        for i = 1 : pNum
            x( sortIndex( i ), : ) = pX( sortIndex( i ), : )+randn(1)*ones(1,dim);
            x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
            [fit( sortIndex( i ) ),model{1,sortIndex( i )}] = fobj( x( sortIndex( i ), : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
        end
    end
    %�Ѿ����仯����õ���Ⱥ��¼����
    [ ~, bestII ] = min( fit );
    bestXX = x( bestII, : );
    %   disp(2222)
    %������������
    for i = ( pNum + 1 ) : pop                     % Equation (4)
        A=floor(rand(1,dim)*2)*2-1;           %����1��-1�������

        if( i>(pop/2))
            %���i>��Ⱥ��һ�룬�����������Ӧ�ȿ����һ�Σ�������Щ���е���Ⱥ�����ڰ���
            x( sortIndex(i ), : )=randn(1)*exp((worse-pX( sortIndex( i ), : ))/(i)^2);
            %��Ӧ�Ȳ��õģ����������ȸ����һ������1������������չ
        else
            x( sortIndex( i ), : )=bestXX+(abs(( pX( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);
            %������Ӧ�ȳ�ȥ����������֮��������Ⱥ��ǰ��ε�,ȥ���������ߵ�ʳ���ǰ�������Ⱥ�Ļ�����
            %�ٽ��б仯һ�Σ���ԭ���Ļ����ϼ�һЩֵ���߼�һЩֵ
        end
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );  %���º���Ⱥ�������ڱ�����Χ
        [fit( sortIndex( i ) ),model{1,sortIndex( i )}]= fobj( x( sortIndex( i ), : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );                    %���¹������¼�����Ӧ��
    end
    %��ȫ����Ⱥ���ҿ�����ʶ��Σ�յ���ȸ
    c=randperm(numel(sortIndex));
    b=sortIndex(c(1:round(0.2*length(c))));
    for j =  1  : length(b)      % Equation (5)
        if( pFit( sortIndex( b(j) ) )>(fMin) )
            %�����Ӧ�ȱ��ʼ��С��Ӧ�Ȳ�Ļ�������ԭ���������Ⱥ������һ����ֵ
            x( sortIndex( b(j) ), : )=bestX+(randn(1,dim)).*(abs(( pX( sortIndex( b(j) ), : ) -bestX)));

        else
            %�����Ӧ�ȴﵽ��ʼ��С����Ӧ��ֵ������ԭ���������Ⱥ������������Сһ����
            x( sortIndex( b(j) ), : ) =pX( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pX( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);

        end
        x( sortIndex(b(j) ), : ) = Bounds( x( sortIndex(b(j) ), : ), lb, ub );

        [fit( sortIndex(b(j) ) ),model{1,sortIndex( b(j) )}] = fobj( x( sortIndex( b(j) ), : ),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
    end
    for i = 1 : pop
        %����ĸ���Ⱥ��Ӧ�Ⱥ��ˣ��Ͱѱ仯���滻��ԭ������Ⱥ
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end

        if( pFit( i ) < fMin )   %����ֵ�Լ�����ֵλ�ÿ��Ƿ�仯
            fMin= pFit( i );
            bestX = pX( i, : );
            Mdl_best=model{1,i};


        end
    end

     % �ƽ����Ҳ��ԸĽ�
    gold=double((sqrt(5)-1)/2);      % golden proportion coefficient, around 0.618
    x1=-pi+(1-gold)*(2*pi);
    x2=-pi+gold*(2*pi);
    for i=1:size(pX,1)
        r=rand;
        r1=(2*pi)*r;
        r2=r*pi;
        for vv = 1:dim % in j-th dimension
            NEW_pos(i,vv) = pX(i,vv)*abs(sin(r1)) - r2*sin(r1)*abs(x1*bestX(1,vv)-x2*pX(1,vv));   %�ƽ����Ҽ��㹫ʽ
        end
        Flag4ub=NEW_pos(i,:)>ub;
        Flag4lb=NEW_pos(i,:)<lb;
        NEW_pos(i,:)=(NEW_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        % disp(NEW_pos(i,:))
        [fit( i ),model{1,i}] = fobj(NEW_pos(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );
    end
    for i = 1 : pop
        %����ĸ���Ⱥ��Ӧ�Ⱥ��ˣ��Ͱѱ仯���滻��ԭ������Ⱥ
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end

        if( pFit( i ) < fMin )   %����ֵ�Լ�����ֵλ�ÿ��Ƿ�仯
            fMin= pFit( i );
            bestX = pX( i, : );
            Mdl_best=model{1,i};


        end
    end
    Convergence_curve(t)=fMin;
end
end
%%

function [Destination_fitness,Mdl_best,Convergence_curve,Destination_position]=IVY_IVY(N,Max_iteration,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
%���������Ż��㷨  IVY
% fobj = @(x) fobj(x);

VarMin = lb;       %Variables Lower Bound
VarMax =ub ;       %Variables Upper Bound

% IVYA Parameters

MaxIt = Max_iteration;        % Maximum Number of Iterations
nPop = N;                     % Population Size
VarSize = [1 dim];            % Decision Variables Matrix Size

% Initialization

% Empty Plant Structure
empty_plant.Position = [];
empty_plant.Cost = [];
empty_plant.GV= [];
pop = repmat(empty_plant, nPop, 1);    % Initial Population Array

for i = 1:numel(pop)

    % Initialize Position
    % Eq.(1)
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    % Eq.(6)-upper condition
    pop(i).GV=(pop(i).Position /(VarMax-VarMin));

    % Evaluation
    [pop(i).Cost,pop(i).Mdl] = fobj(pop(i).Position,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);

end

% Initialize Best Cost History
BestCosts = zeros(MaxIt, 1);

% Ivy Main Loop

for it = 1:MaxIt

    % Get Best and Worst Cost Values
    Costs = [pop.Cost];
    % Best and worst costs
    BestCost = min(Costs);
    WorstCost = max(Costs);

    % Initialize new Population
    newpop = [];

    for i = 1:numel(pop)

        ii=i+1;
        if i==numel(pop)
            ii=1;
        end

        newsol = empty_plant;

        beta_1=1+(rand/2); % beta value in line 8 of "Algorithm 1 (Fig.2 in paper)"

        if  pop(i).Cost<beta_1*pop(1).Cost
            % Eq.(5)-(6)
            newsol.Position =pop(i).Position +abs(randn(VarSize)) .*( pop(ii).Position - pop(i).Position )+ randn(VarSize) .*(pop(i).GV);
        else
            % Eq.(7)
            newsol.Position =pop(1).Position .*(rand+randn(VarSize).*(pop(i).GV));
        end

        % Eq.(3)
        pop(i).GV=(pop(i).GV).*((rand^2)*(randn(1,dim)));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        newsol.Position = max(newsol.Position, VarMin);
        newsol.Position = min(newsol.Position, VarMax);
        % Eq.(8)
        newsol.GV=(newsol.Position /(VarMax-VarMin));

        % Evaluate new population
        % newsol.Cost = fobj(newsol.Position);
        [newsol.Cost ,newsol.Mdl] = fobj(newsol.Position,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);

        newpop = [newpop
            newsol];

    end

    % Merge Populations
    pop = [pop
        newpop];

    % Sort Population
    [~, SortOrder]=sort([pop.Cost]);
    pop = pop(SortOrder);

    % Competitive Exclusion (Delete Extra Members)
    if numel(pop)>nPop
        pop = pop(1:nPop);
    end

    % Store Best Solution Ever Found
    BestSol = pop(1);
    BestCosts(it) = BestSol.Cost;
    Mdl_best=BestSol.Mdl;
    Convergence_curve(it) =pop(1).Cost;

end


% Results
Destination_fitness=pop(1).Cost;
Destination_position=pop(1).Position;

end

%%
function [Leeches_best_score,Mdl_best,Convergence_curve,Leeches_best_pos]=BSLO_BSLO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
%_________________________________________________________________________________
%  Blood-Sucking Leech Optimizer                                                                                                                                     %   ��ע΢�Ź��ںţ��Ż��㷨������ø�����ѡ���Ʒ�����������ƴ���
%  Developed in MATLAB R2023b
%
%  programming: Jianfu Bai
%
%  e-Mail: Jianfu.Bai@UGent.be, magd.abdelwahab@ugent.be
%  Soete Laboratory, Department of Electrical Energy, Metals, Mechanical Constructions, and Systems,
%  Faculty of Engineering and Architecture, Ghent University, Belgium
%
%  paper: Jianfu Bai, H. Nguyen-Xuan, Elena Atroshchenko, Gregor Kosec, Lihua Wang, Magd Abdel Wahab, Blood-sucking leech optimizer[J]. Advances in Engineering Software, 2024, 195: 103696.
%  doi.org/10.1016/j.advengsoft.2024.103696
%____________________________________________________________________________________
% ��ˮ���Ż��㷨
% initialize best Leeches
Leeches_best_pos=zeros(1,dim);
Leeches_best_score=inf;
% Initialize the positions of search agents
Leeches_Positions=initialization(SearchAgents_no,dim,ub,lb);
Convergence_curve=zeros(1,Max_iter);
Temp_best_fitness=zeros(1,Max_iter);
fitness=zeros(1,SearchAgents_no);
% Initialize parameters
t=0;m=0.8;a=0.97;b=0.001;t1=20;t2=20;
% Main loop
while t<Max_iter
    N1=floor((m+(1-m)*(t/Max_iter)^2)*SearchAgents_no);
    %calculate fitness values
    for i=1:size(Leeches_Positions,1)
        % boundary checking
        Flag4ub=Leeches_Positions(i,:)>ub;
        Flag4lb=Leeches_Positions(i,:)<lb;
        Leeches_Positions(i,:)=(Leeches_Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        % Calculate objective function for each search agent
        [fitness(i),Mdl_all{1,i}]=fobj(Leeches_Positions(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
        % Update best Leeches
        if fitness(i)<=Leeches_best_score
            Leeches_best_score=fitness(i);
            Leeches_best_pos=Leeches_Positions(i,:);
            Mdl_best=Mdl_all{1,i};
        end
    end
    Prey_Position=Leeches_best_pos;
    % Re-tracking strategy
    Temp_best_fitness(t+1)=Leeches_best_score;
    if t>t1
        if Temp_best_fitness(t+1)==Temp_best_fitness(t+1-t2)
            for i=1:size(Leeches_Positions,1)
                if fitness(i)==Leeches_best_score
                    Leeches_Positions(i,:)=rand(1,dim).*(ub-lb)+lb;
                end
            end
        end
    end
    if rand()<0.5
        s=8-1*(-(t/Max_iter)^2+1);
    else
        s=8-7*(-(t/Max_iter)^2+1);
    end
    beta=-0.5*(t/Max_iter)^6+(t/Max_iter)^4+1.5;
    LV=0.5*levy(SearchAgents_no,dim,beta);
    % Generate random integers
    minValue = 1;  % minimum integer value
    maxValue = floor(SearchAgents_no*(1+t/Max_iter)); % maximum integer value
    k2 = randi([minValue, maxValue], SearchAgents_no, dim);
    k = randi([minValue, dim], SearchAgents_no, dim);
    for i=1:N1
        for j=1:size(Leeches_Positions,2)
            r1=2*rand()-1; % r1 is a random number in [0,1]
            r2=2*rand()-1;
            r3=2*rand()-1;
            PD=s*(1-(t/Max_iter))*r1;
            if abs(PD)>=1
                % Exploration of directional leeches
                b=0.001;
                W1=(1-t/Max_iter)*b*LV(i,j);
                L1=r2*abs(Prey_Position(j)-Leeches_Positions(i,j))*PD*(1-k2(i,j)/SearchAgents_no);
                L2=abs(Prey_Position(j)-Leeches_Positions(i,k(i,j)))*PD*(1-(r2^2)*(k2(i,j)/SearchAgents_no));
                if rand()<a
                    if abs(Prey_Position(j))>abs(Leeches_Positions(i,j))
                        Leeches_Positions(i,j)=Leeches_Positions(i,j)+W1*Leeches_Positions(i,j)-L1;
                    else
                        Leeches_Positions(i,j)=Leeches_Positions(i,j)+W1*Leeches_Positions(i,j)+L1;
                    end
                else
                    if abs(Prey_Position(j))>abs(Leeches_Positions(i,j))
                        Leeches_Positions(i,j)=Leeches_Positions(i,j)+W1*Leeches_Positions(i,k(i,j))-L2;
                    else
                        Leeches_Positions(i,j)=Leeches_Positions(i,j)+W1*Leeches_Positions(i,k(i,j))+L2;
                    end
                end
            else
                % Exploitation of directional leeches
                if t>=0.1*Max_iter
                    b=0.00001;
                end
                W1=(1-t/Max_iter)*b*LV(i,j);
                L3=abs(Prey_Position(j)-Leeches_Positions(i,j))*PD*(1-r3*k2(i,j)/SearchAgents_no);
                L4=abs(Prey_Position(j)-Leeches_Positions(i,k(i,j)))*PD*(1-r3*k2(i,j)/SearchAgents_no);
                if rand()<a
                    if abs(Prey_Position(j))>abs(Leeches_Positions(i,j))
                        Leeches_Positions(i,j)=Prey_Position(j)+W1*Prey_Position(j)-L3;
                    else
                        Leeches_Positions(i,j)=Prey_Position(j)+W1*Prey_Position(j)+L3;
                    end
                else
                    if abs(Prey_Position(j))>abs(Leeches_Positions(i,j))
                        Leeches_Positions(i,j)=Prey_Position(j)+W1*Prey_Position(j)-L4;
                    else
                        Leeches_Positions(i,j)=Prey_Position(j)+W1*Prey_Position(j)+L4;
                    end
                end
            end
        end
    end
    %Search strategy of directionless Leeches
    for i=(N1+1):size(Leeches_Positions,1)
        for j=1:size(Leeches_Positions,2)
            if min(lb)>=0
                LV(i,j)=abs(LV(i,j));
            end
            if rand()>0.5
                Leeches_Positions(i,j)=(t/Max_iter)*LV(i,j)*Leeches_Positions(i,j)*abs(Prey_Position(j)-Leeches_Positions(i,j));
            else
                Leeches_Positions(i,j)=(t/Max_iter)*LV(i,j)*Prey_Position(j)*abs(Prey_Position(j)-Leeches_Positions(i,j));
            end
        end
    end
    t=t+1;
    Convergence_curve(t)=Leeches_best_score;
end

%
    function X=initialization(SearchAgents_no,dim,ub,lb)

        Boundary_no= size(ub,2); % numnber of boundaries

        % If the boundaries of all variables are equal
        if Boundary_no==1
            X=rand(SearchAgents_no,dim).*(ub-lb)+lb;
        end

        % If each variable has a different lb and ub
        if Boundary_no>1
            for i=1:dim
                ub_i=ub(i);
                lb_i=lb(i);
                X(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
            end
        end
    end

    function [z] = levy(n,m,beta)

        num = gamma(1+beta)*sin(pi*beta/2); % used for Numerator

        den = gamma((1+beta)/2)*beta*2^((beta-1)/2); % used for Denominator

        sigma_u = (num/den)^(1/beta);% Standard deviation

        u = random('Normal',0,sigma_u,n,m);

        v = random('Normal',0,1,n,m);

        z =u./(abs(v).^(1/beta));


    end
end

%%
function [gBestScore,Mdl_best,cg_curve,gBest]=GDPSO_GDPSO(N,iter,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)

%PSO Infotmation
Vmax=ones(1,dim).*(ub-lb).*0.15;           %�ٶ����ֵ
noP=N;
w=0.8;
c1=1.5;
c2=1.5;

% Initializations
vel=zeros(noP,dim);
pBestScore=zeros(noP,1);
pBest=zeros(noP,dim);
gBest=zeros(1,dim);
cg_curve=zeros(1,iter);

% Random initialization for agents.

pos = repmat(lb,N,1)+rand(N,dim).* repmat((ub-lb),N,1);
NEW_pos = zeros(N,dim);
for i=1:noP
    pBestScore(i,1)=inf;
end

% Initialize gBestScore for a minimization problem
gBestScore=inf;


% Return back the particles that go befobjond the boundaries of the search space
for i=1:size(pos,1)
    Flag4ub=pos(i,:)>ub;
    Flag4lb=pos(i,:)<lb;
    pos(i,:)=(pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
end

for i=1:size(pos,1)
    %Calculate objective function for each particle
    [fitness,mdl_new]= fobj(pos(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
    if(pBestScore(i)>fitness)
        pBestScore(i)=fitness;
        pBest(i,:)=pos(i,:);
        Mdl_best=mdl_new;
    end
    if(gBestScore>fitness)
        gBestScore=fitness;
        gBest=pos(i,:);
        Mdl_best=mdl_new;
    end
end


for l=1:iter


    %Update the W of PSO

    %Update the Velocitfobj and Position of particles
    for i=1:size(pos,1)
        for j=1:size(pos,2)
            vel(i,j)=w*vel(i,j)+c1*rand()*(pBest(i,j)-pos(i,j))+c2*rand()*(gBest(j)-pos(i,j));

            if(vel(i,j)>Vmax(j))
                vel(i,j)=Vmax(j);
            end
            if(vel(i,j)<-Vmax(j))
                vel(i,j)=-Vmax(j);
            end
            pos(i,j)=pos(i,j)+vel(i,j);
        end
    end

    % Return back the particles that go befobjond the boundaries of the search space
    for i=1:size(pos,1)
        Flag4ub=pos(i,:)>ub;
        Flag4lb=pos(i,:)<lb;
        pos(i,:)=(pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    end

    for i=1:size(pos,1)
        %Calculate objective function for each particle
        % fitness= fobj(pos(i,:));
          [fitness,mdl_new]= fobj(pos(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
  
        if(pBestScore(i)>fitness)
            pBestScore(i)=fitness;

            pBest(i,:)=pos(i,:);
            Mdl_best=mdl_new;
        end
        if(gBestScore>fitness)
            gBestScore=fitness;

            gBest=pos(i,:);
            Mdl_best=mdl_new;
        end
    end
    %
    % �ƽ����Ҳ��ԸĽ�
    gold=double((sqrt(5)-1)/2);      % golden proportion coefficient, around 0.618
    x1=-pi+(1-gold)*(2*pi);
    x2=-pi+gold*(2*pi);
    for i=1:size(NEW_pos,1)
        r=rand;
        r1=(2*pi)*r;
        r2=r*pi;
        for vv = 1:dim % in j-th dimension
            NEW_pos(i,vv) = pos(i,vv)*abs(sin(r1)) - r2*sin(r1)*abs(x1*gBest(1,vv)-x2*pos(1,vv));   %�ƽ����Ҽ��㹫ʽ
        end
        Flag4ub=NEW_pos(i,:)>ub;
        Flag4lb=NEW_pos(i,:)<lb;
        NEW_pos(i,:)=(NEW_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        % fitness= fobj(NEW_pos(i,:));
        [fitness,mdl_new]= fobj(NEW_pos(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
  
        if(pBestScore(i)>fitness)
            pBestScore(i)=fitness;
            pos(i,:) = NEW_pos(i,:);
            pBest(i,:)=pos(i,:);
            Mdl_best=mdl_new;
        end
        if(gBestScore>fitness)
            gBestScore=fitness;
            pos(i,:) = NEW_pos(i,:);
            gBest=pos(i,:);
            Mdl_best=mdl_new;
        end

    end
    cg_curve(l)=gBestScore;

end

end

%%
function [Best_rime_rate,Mdl_best,Convergence_curve,Best_rime]=GRIME_GRIME(N,Max_iter,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
% disp('RIME is now tackling your problem')
% ӳ��Ľ�+����ѧϰ�Ľ�HOA
% initialize position
Best_rime=zeros(1,dim);
Best_rime_rate=inf;%change this to -inf for maximization problems

% Rimepop=initialization(N,dim,ub,lb);%Initialize the set of random solutions
% �Ľ���ʼӳ��Ľ�
label=5;  %ӳ�� % 1-6 �ֱ��Ӧ tent��chebyshev��Singer��Logistic��Sine, Circle
Rimepop = yinshe( N, dim,label,ub,lb);
Rimepop = min(ub,Rimepop);        % bound violating to upper bound
Rimepop = max(lb,Rimepop);        % bound violating to lower bound

Lb=lb.*ones(1,dim);% lower boundary 
Ub=ub.*ones(1,dim);% upper boundary
it=1;%Number of iterations
Convergence_curve=zeros(1,Max_iter);
Rime_rates=zeros(1,N);%Initialize the fitness value
newRime_rates=zeros(1,N);
W = 5;%Soft-rime parameters, discussed in subsection 4.3.1 of the paper
%Calculate the fitness value of the initial position
for i=1:N
    [Rime_rates(1,i),Mdl_all{1,i}]=fobj(Rimepop(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);%Calculate the fitness value for each search agent
    %Make greedy selections
    if Rime_rates(1,i)<Best_rime_rate
        Best_rime_rate=Rime_rates(1,i);
        Best_rime=Rimepop(i,:);
        Mdl_best=Mdl_all{1,i};
    end
end
% Main loop
while it <= Max_iter
    RimeFactor = (rand-0.5)*2*cos((pi*it/(Max_iter/10)))*(1-round(it*W/Max_iter)/W);%Parameters of Eq.(3),(4),(5)
    E =sqrt(it/Max_iter);%Eq.(6)
    newRimepop = Rimepop;%Recording new populations
    normalized_rime_rates=normr(Rime_rates);%Parameters of Eq.(7)
    for i=1:N
        for j=1:dim
            %Soft-rime search strategy
            r1=rand();
            if r1< E
                newRimepop(i,j)=Best_rime(1,j)+RimeFactor*((Ub(j)-Lb(j))*rand+Lb(j));%Eq.(3)
            end
            %Hard-rime puncture mechanism
            r2=rand();
            if r2<normalized_rime_rates(i)
                newRimepop(i,j)=Best_rime(1,j);%Eq.(7)
            end
        end
    end
    for i=1:N
        %Boundary absorption
        Flag4ub=newRimepop(i,:)>ub;
        Flag4lb=newRimepop(i,:)<lb;
        newRimepop(i,:)=(newRimepop(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        % newRime_rates(1,i)=fobj(newRimepop(i,:));
        [newRime_rates(1,i),Mdl_new{1,i}]=fobj(newRimepop(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);%Calculate the fitness value for each search agent

        %Positive greedy selection mechanism
        if newRime_rates(1,i)<Rime_rates(1,i)
            Rime_rates(1,i) = newRime_rates(1,i);
            Rimepop(i,:) = newRimepop(i,:);
            Mdl_all{1,i}=Mdl_new{1,i};
            if newRime_rates(1,i)< Best_rime_rate
               Best_rime_rate=Rime_rates(1,i);
               Best_rime=Rimepop(i,:);
               Mdl_best=Mdl_all{1,i};

            end
        end
    end
    %����ѧϰ����
    %%LOBL strategy
    for i=1:N
        k=(1+(it/Max_iter)^0.5)^10;
        newRimepop(i,:) = (ub+lb)/2+(ub+lb)/(2*k)-Rimepop(i,:)/k;
        Flag4ub=newRimepop(i,:)>ub;
        Flag4lb=newRimepop(i,:)<lb;
        newRimepop(i,:)=(newRimepop(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        % newRime_rates(1,i)=fobj(newRimepop(i,:));
       [newRime_rates(1,i),Mdl_new{1,i}]=fobj(newRimepop(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);%Calculate the fitness value for each search agent
        
        if newRime_rates(1,i)<Rime_rates(1,i)
            Rime_rates(1,i) = newRime_rates(1,i);
            Rimepop(i,:) = newRimepop(i,:);
            Mdl_all{1,i}=Mdl_new{1,i};
            if newRime_rates(1,i)< Best_rime_rate
               Best_rime_rate=Rime_rates(1,i);
               Best_rime=Rimepop(i,:);
               Mdl_best=Mdl_all{1,i};

            end
        end
    end

  
    Convergence_curve(it)=Best_rime_rate;
    it=it+1;
end
end

%%

function [Best_Fitness_BKA,Mdl_best,Convergence_curve,Best_Pos_BKA]=BKA_BKA(pop,T,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
% ----------------Initialize the locations of Blue Sheep------------------%

p=0.9;r=rand;
XPos=initialization(pop,dim,ub,lb);% Initial population
for i =1:pop
    [XFit(i),Mdl_all{1,i}]=fobj(XPos(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
end
Convergence_curve=zeros(1,T);

% -------------------Start iteration------------------------------------%

for t=1:T
    [~,sorted_indexes]=sort(XFit);
    XLeader_Pos=XPos(sorted_indexes(1),:);
    XLeader_Fit = XFit(sorted_indexes(1));
    Mdl_best=Mdl_all{1,sorted_indexes(1)};
    % -------------------Attacking behavior-------------------%

    for i=1:pop

        n=0.05*exp(-2*(t/T)^2);
        if p<r
            XPosNew(i,:)=XPos(i,:)+n.*(1+sin(r))*XPos(i,:);
        else
            XPosNew(i,:)= XPos(i,:).*(n*(2*rand(1,dim)-1)+1);
        end
        XPosNew(i,:) = max(XPosNew(i,:),lb);XPosNew(i,:) = min(XPosNew(i,:),ub);%%Boundary checking
        % ------------ Select the optimal fitness value--------------%

        % XFit_New(i)=fobj(XPosNew(i,:));
        [XFit_New(i),Mdl_new{1,i}]=fobj(XPosNew(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);

        if(XFit_New(i)<XFit(i))
            XPos(i,:) = XPosNew(i,:);
            XFit(i) = XFit_New(i);
            Mdl_all{1,i}=Mdl_new{1,i};
        end
        % -------------------Migration behavior-------------------%

        m=2*sin(r+pi/2);
        s = randi([1,pop],1);
        r_XFitness=XFit(s);
        ori_value = rand(1,dim);cauchy_value = tan((ori_value-0.5)*pi);
        if XFit(i)< r_XFitness
            XPosNew(i,:)=XPos(i,:)+cauchy_value(:,dim).* (XPos(i,:)-XLeader_Pos);
        else
            XPosNew(i,:)=XPos(i,:)+cauchy_value(:,dim).* (XLeader_Pos-m.*XPos(i,:));
        end
        XPosNew(i,:) = max(XPosNew(i,:),lb);XPosNew(i,:) = min(XPosNew(i,:),ub); %%Boundary checking
        % --------------  Select the optimal fitness value---------%

        % XFit_New(i)=fobj(XPosNew(i,:));
        [XFit_New(i),Mdl_new{1,i}]=fobj(XPosNew(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);

        if(XFit_New(i)<XFit(i))
            XPos(i,:) = XPosNew(i,:);
            XFit(i) = XFit_New(i);
            Mdl_all{1,i}=Mdl_new{1,i};
        end
    end
    % -------Update the optimal Black-winged Kite----------%

    if(XFit<XLeader_Fit)
        Best_Fitness_BKA=XFit(i);
        Best_Pos_BKA=XPos(i,:);
        Mdl_best=Mdl_all{1,i};
    else
        Best_Fitness_BKA=XLeader_Fit;
        Best_Pos_BKA=XLeader_Pos;
    end
    Convergence_curve(t)=Best_Fitness_BKA;
end


% This function initialize the first population of search agents
    function Positions=initialization(SearchAgents_no,dim,ub,lb)

        Boundary_no= size(ub,2); % numnber of boundaries

        % If the boundaries of all variables are equal and user enter a signle
        % number for both ub and lb
        if Boundary_no==1
            Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
        end

        % If each variable has a different lb and ub
        if Boundary_no>1
            for i=1:dim
                ub_i=ub(i);
                lb_i=lb(i);
                Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
            end
        end
    end

end

%%
function [Best_Fitness_BKA,Mdl_best,Convergence_curve,Best_Pos_BKA]=GBKA_GBKA(pop,T,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
% ----------------Initialize the locations of Blue Sheep------------------%

p=0.9;r=rand;
% �Ľ���ʼӳ��Ľ�
label=2;  %ӳ�� % 1-6 �ֱ��Ӧ tent��chebyshev��Singer��Logistic��Sine, Circle
XPos = yinshe( pop, dim,label,ub,lb);
XPos = min(ub,XPos);        % bound violating to upper bound
XPos = max(lb,XPos);        % bound violating to lower bound
for i =1:pop
    [XFit(i),Mdl_all{1,i}]=fobj(XPos(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
end
Convergence_curve=zeros(1,T);

% -------------------Start iteration------------------------------------%

for t=1:T
    [~,sorted_indexes]=sort(XFit);
    XLeader_Pos=XPos(sorted_indexes(1),:);
    XLeader_Fit = XFit(sorted_indexes(1));
    Mdl_best=Mdl_all{1,sorted_indexes(1)};
    % -------------------Attacking behavior-------------------%

    for i=1:pop

        n=0.05*exp(-2*(t/T)^2);
        if p<r
            XPosNew(i,:)=XPos(i,:)+n.*(1+sin(r))*XPos(i,:);
        else
            XPosNew(i,:)= XPos(i,:).*(n*(2*rand(1,dim)-1)+1);
        end
        XPosNew(i,:) = max(XPosNew(i,:),lb);XPosNew(i,:) = min(XPosNew(i,:),ub);%%Boundary checking
        % ------------ Select the optimal fitness value--------------%

        % XFit_New(i)=fobj(XPosNew(i,:));
        [XFit_New(i),Mdl_new{1,i}]=fobj(XPosNew(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);

        if(XFit_New(i)<XFit(i))
            XPos(i,:) = XPosNew(i,:);
            XFit(i) = XFit_New(i);
            Mdl_all{1,i}=Mdl_new{1,i};
        end
        % -------------------Migration behavior-------------------%

        m=2*sin(r+pi/2);
        s = randi([1,pop],1);
        r_XFitness=XFit(s);
        ori_value = rand(1,dim);cauchy_value = tan((ori_value-0.5)*pi);
        if XFit(i)< r_XFitness
            XPosNew(i,:)=XPos(i,:)+cauchy_value(:,dim).* (XPos(i,:)-XLeader_Pos);
        else
            XPosNew(i,:)=XPos(i,:)+cauchy_value(:,dim).* (XLeader_Pos-m.*XPos(i,:));
        end
        XPosNew(i,:) = max(XPosNew(i,:),lb);XPosNew(i,:) = min(XPosNew(i,:),ub); %%Boundary checking
        % --------------  Select the optimal fitness value---------%

        % XFit_New(i)=fobj(XPosNew(i,:));
        [XFit_New(i),Mdl_new{1,i}]=fobj(XPosNew(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);

        if(XFit_New(i)<XFit(i))
            XPos(i,:) = XPosNew(i,:);
            XFit(i) = XFit_New(i);
            Mdl_all{1,i}=Mdl_new{1,i};
        end
    end
    % -------Update the optimal Black-winged Kite----------%

    if(XFit<XLeader_Fit)
        Best_Fitness_BKA=XFit(i);
        Best_Pos_BKA=XPos(i,:);
        Mdl_best=Mdl_all{1,i};
    else
        Best_Fitness_BKA=XLeader_Fit;
        Best_Pos_BKA=XLeader_Pos;
    end
    Convergence_curve(t)=Best_Fitness_BKA;
end


% This function initialize the first population of search agents
    function Positions=initialization(SearchAgents_no,dim,ub,lb)

        Boundary_no= size(ub,2); % numnber of boundaries

        % If the boundaries of all variables are equal and user enter a signle
        % number for both ub and lb
        if Boundary_no==1
            Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
        end

        % If each variable has a different lb and ub
        if Boundary_no>1
            for i=1:dim
                ub_i=ub(i);
                lb_i=lb(i);
                Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
            end
        end
    end

end

%%
function [best_fun,Mdl_best,cuve_f,best_position]  =COA_COA1(N,T,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
%  Crayfish Optimization Algorithm(COA)
%
%  Source codes demo version 1.0                                                                      
%                                                                                                     
%  The 11th Gen Intel(R) Core(TM) i7-11700 processor with the primary frequency of 2.50GHz, 16GB memory, and the operating system of 64-bit windows 11 using matlab2021a.                                                                
%                                                                                                     
%  Author and programmer: Heming Jia,Raohonghua,Wen changsheng,Seyedali Mirjalili                                                                          
%         e-Mail: jiaheminglucky99@126.com;rao12138@163.com                                                           
%                                   
%                                                                                                                                                   
%_______________________________________________________________________________________________
% You can simply define your cost function in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% T = maximum number of iterations
% N = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single numbers
%% Define Parameters
cuve_f=zeros(1,T); 
X=initialization(N,dim,ub,lb); %Initialize population
global_Cov = zeros(1,T);
Best_fitness = inf;
best_position = zeros(1,dim);
fitness_f = zeros(1,N);
for i=1:N
   [fitness_f(i),Mdl_all{1,i}] =  fobj(X(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm); %Calculate the fitness value of the function
   if fitness_f(i)<Best_fitness
       Best_fitness = fitness_f(i);
       best_position = X(i,:);
       Mdl_best=Mdl_all{1,i};
   end
end
global_position = best_position; 
global_fitness = Best_fitness;
cuve_f(1)=Best_fitness;
t=1; 
while(t<=T)
    C = 2-(t/T); %Eq.(7)
    temp = rand*15+20; %Eq.(3)
    xf = (best_position+global_position)/2; % Eq.(5)
    Xfood = best_position;
    for i = 1:N
        if temp>30
            %% summer resort stage
            if rand<0.5
                Xnew(i,:) = X(i,:)+C*rand(1,dim).*(xf-X(i,:)); %Eq.(6)
            else
            %% competition stage
                for j = 1:dim
                    z = round(rand*(N-1))+1;  %Eq.(9)
                    Xnew(i,j) = X(i,j)-X(z,j)+xf(j);  %Eq.(8)
                end
            end
        else
            %% foraging stage
               % [fitness_f(i),Mdl_all{1,i}] =  fobj(X(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm); %Calculate the fitness value of the function
 
            P = 3*rand*fitness_f(i)/fobj(Xfood,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm); %Eq.(4)
            if P>2   % The food is too big
                 Xfood = exp(-1/P).*Xfood;   %Eq.(12)
                for j = 1:dim
                    Xnew(i,j) = X(i,j)+cos(2*pi*rand)*Xfood(j)*p_obj(temp)-sin(2*pi*rand)*Xfood(j)*p_obj(temp); %Eq.(13)
                end
            else
                Xnew(i,:) = (X(i,:)-Xfood)*p_obj(temp)+p_obj(temp).*rand(1,dim).*X(i,:); %Eq.(14)
            end
        end
    end
    %% boundary conditions
    for i=1:N
        for j =1:dim
            if length(ub)==1
                Xnew(i,j) = min(ub,Xnew(i,j));
                Xnew(i,j) = max(lb,Xnew(i,j));
            else
                Xnew(i,j) = min(ub(j),Xnew(i,j));
                Xnew(i,j) = max(lb(j),Xnew(i,j));
            end
        end
    end
   
    global_position = Xnew(1,:);
    [global_fitness,gloabl_mdl] = fobj(global_position,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
 
    for i =1:N
         %% Obtain the optimal solution for the updated population
        [new_fitness,new_mdl ]= fobj(Xnew(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
        
        if new_fitness<global_fitness
                 global_fitness = new_fitness;
                 global_position = Xnew(i,:);
                 gloabl_mdl=new_mdl;
                 
        end
        %% Update the population to a new location
        if new_fitness<fitness_f(i)
             fitness_f(i) = new_fitness;
             X(i,:) = Xnew(i,:);
             Mdl_all{1,i}=new_mdl;
             if fitness_f(i)<Best_fitness
                 Best_fitness=fitness_f(i);
                 best_position = X(i,:);
                 Mdl_best=Mdl_all{1,i};
             end
        end
    end
    global_Cov(t) = global_fitness;
    cuve_f(t) = Best_fitness;
    t=t+1;
    if mod(t,50)==0
      % disp("COA"+"iter"+num2str(t)+": "+Best_fitness); 
   end
end
 best_fun = Best_fitness;

function y = p_obj(x)   %Eq.(4)
    y = 0.2*(1/(sqrt(2*pi)*3))*exp(-(x-25).^2/(2*3.^2));
end

function X=initialization(N,Dim,UB,LB)

B_no= size(UB,2); % numnber of boundaries

if B_no==1
    X=rand(N,Dim).*(UB-LB)+LB;
end

% If each variable has a different lb and ub
if B_no>1
    for i=1:Dim
        Ub_i=UB(i);
        Lb_i=LB(i);
        X(:,i)=rand(N,1).*(Ub_i-Lb_i)+Lb_i;
    end
end
end

end

%%

function [fHike,Mdl_best,fiteration,fPosition] = HOA_HOA(hiker, MaxIter,LB,UB,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
% Authors: Sunday Oladejo, Stephen Ekwe, and Mirjalili Seyedali                                                  %
% Version: v2 - TOBLER's HIKING FUNCTION (THF) APPROACH                                     %
% Last Update: 2023-02-09      
% Developed in MATLAB R2023a based on the following paper:
%
%  The Hiking Optimization Algorithm: A novel human-based metaheuristic approach
%  https://doi.org/10.1016/j.knosys.2024.111880
%ͽ�������Ż��㷨
% Problem Parameters

% prob = fobj;                  % objective function
nVar = dim;                     % dimension of problem
lb = LB;                        % lower bound
ub = UB;                        % upper bound
nPop = hiker;                   % no. of hikers
MaxIt = MaxIter-1;                % max iteration

% Pre-allocate
fit = zeros(nPop,1);                    % vectorize variable to store fitness value
Best.iteration = zeros(MaxIt+1,1);      % vectorize variable store fitness value per iteration

% Start Tobler Hiking Function Optimizer
Pop = repmat(lb,nPop,1) + repmat((ub-lb),nPop,1).*rand(nPop,nVar); % generate initial position of hiker

% Evaluate fitness
for q = 1:nPop
    [fit(q),Mdl_all{1,q}] = fobj(Pop(q,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            % evaluating initial fitness 
end
Best.iteration(1) = min(fit);           % stores initial fitness value at 0th iteration


% Main Loop
for i = 1:MaxIt
        [~,ind] = min(fit);             % obtain initial fitness
        Xbest = Pop(ind,:);             % obtain golbal best position of initial fitness
        Mdl_best=Mdl_all{1,ind};
%         Lead_Hiker(i) = ind;
%         [~,isd] = max(fit); 
%         Sweaper_Hiker(i) = isd;
    for j = 1:nPop
        
        Xini = (Pop(j,:));              % obtain initial position of jth hiker
    
        theta = randi([0 50],1,1);      % randomize elevation angle of hiker        
        s = tan(theta);                 % compute slope
        SF = randi([1 2],1,1);          % sweep factor generate either 1 or 2 randomly 
        
        Vel = 6.*exp(-3.5.*abs(s+0.05)); % Compute walking velocity based on Tobler's Hiking Function
        newVel =  Vel + rand(1,nVar).*(Xbest - SF.*Xini) ; % determine new position of hiker
        
        
        newPop = Pop(j,:) + newVel;     % Update position of hiker 
        
        %

        
        newPop = min(ub,newPop);        % bound violating to upper bound
        newPop = max(lb,newPop);        % bound violating to lower bound
        [fnew,Mdlnew] = fobj(newPop,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            % evaluating initial fitness 
      
        % fnew = fobj(newPop);            % re-evaluate fitness
        if (fnew < fit(j))              % apply greedy selection strategy
            Pop(j,:) = newPop;          % store best position
            fit(j) = fnew;              % store new fitness
            Mdl_all{1,j}=Mdlnew;

        end
    end
    
     Best.iteration(i+1) = min(fit); % store best fitness per iteration
%     disp(['Iteration ' num2str(i+1) ': Best Cost = ' num2str(Best.iteration(i+1))]);
    
    
%     Best.iteration(i+1) = min(fit);     % store best fitness per iteration
%     % Show Iteration Information
%     disp(['Iteration ' num2str(i) ': Best Hike = ' num2str(min(fit))]);    
end
% THFO Solution: Store global best fitness and position
[Best.Hike,idx] = min(fit); 
Best.Position = Pop(idx,:);
Mdl_best=Mdl_all{1,idx};

fPosition=Best.Position;
fHike=Best.Hike;
fiteration=Best.iteration';
end

%%
function [fHike,Mdl_best,fiteration,fPosition] = GHOA_GHOA(hiker, MaxIter,LB,UB,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
% Authors: Sunday Oladejo, Stephen Ekwe, and Mirjalili Seyedali                                                  %
% Version: v2 - TOBLER's HIKING FUNCTION (THF) APPROACH                                     %
% Last Update: 2023-02-09      
% Developed in MATLAB R2023a based on the following paper:
%
%  The Hiking Optimization Algorithm: A novel human-based metaheuristic approach
%  https://doi.org/10.1016/j.knosys.2024.111880
%ͽ�������Ż��㷨
% Problem Parameters

% prob = fobj;                  % objective function
nVar = dim;                     % dimension of problem
lb = LB;                        % lower bound
ub = UB;                        % upper bound
nPop = hiker;                   % no. of hikers
MaxIt = MaxIter-1;                % max iteration

% Pre-allocate
fit = zeros(nPop,1);                    % vectorize variable to store fitness value
Best.iteration = zeros(MaxIt+1,1);      % vectorize variable store fitness value per iteration

% Start Tobler Hiking Function Optimizer
% Pop = repmat(lb,nPop,1) + repmat((ub-lb),nPop,1).*rand(nPop,nVar); % generate initial position of hiker
label=3;  %ӳ�� % 1-6 �ֱ��Ӧ tent��chebyshev��Singer��Logistic��Sine, Circle
Pop = yinshe( nPop, dim,label,ub,lb);
Pop = min(ub,Pop);        % bound violating to upper bound
Pop = max(lb,Pop);        % bound violating to lower bound
% Evaluate fitness
for q = 1:nPop
    [fit(q),Mdl_all{1,q}] = fobj(Pop(q,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            % evaluating initial fitness 
end
Best.iteration(1) = min(fit);           % stores initial fitness value at 0th iteration


% Main Loop
for i = 1:MaxIt
    [~,ind] = min(fit);             % obtain initial fitness
    Xbest = Pop(ind,:);             % obtain golbal best position of initial fitness
    Mdl_best=Mdl_all{1,ind};
    %         Lead_Hiker(i) = ind;
    %         [~,isd] = max(fit);
    %         Sweaper_Hiker(i) = isd;
    for j = 1:nPop

        if rand > 0.3  %�����ø��� ������ڲ���ԭ�����Ը��£�С�ڵľ����µĲ���

            Xini = (Pop(j,:));              % obtain initial position of jth hiker

            theta = randi([0 50],1,1);      % randomize elevation angle of hiker
            s = tan(theta);                 % compute slope
            SF = randi([1 2],1,1);          % sweep factor generate either 1 or 2 randomly

            Vel = 6.*exp(-3.5.*abs(s+0.05)); % Compute walking velocity based on Tobler's Hiking Function
            newVel =  Vel + rand(1,nVar).*(Xbest - SF.*Xini) ; % determine new position of hiker


            newPop = Pop(j,:) + newVel;     % Update position of hiker
        else
            %****��˹������߲��Դ���**
            sigma = 0.7;  %��˹�������
            %          g1 = Gaussian(Pop(j,:),Xbest,sigma);
            g1 = 1/(sqrt(2*pi)*sigma).*exp(-(Pop(j,:)-Xbest).^2/(2*sigma^2));
            gstepsize=g1.*(Pop(j,:)-Xbest);
            newPop =Pop(j,:)+gstepsize.*randn(size(Pop(j,:)));%��������߻�������Ӹ�˹

        end
        %


        newPop = min(ub,newPop);        % bound violating to upper bound
        newPop = max(lb,newPop);        % bound violating to lower bound
        [fnew,Mdlnew] = fobj(newPop,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);            % evaluating initial fitness

        % fnew = fobj(newPop);            % re-evaluate fitness
        if (fnew < fit(j))              % apply greedy selection strategy
            Pop(j,:) = newPop;          % store best position
            fit(j) = fnew;              % store new fitness
            Mdl_all{1,j}=Mdlnew;

        end
    end

    Best.iteration(i+1) = min(fit); % store best fitness per iteration
    %     disp(['Iteration ' num2str(i+1) ': Best Cost = ' num2str(Best.iteration(i+1))]);


    %     Best.iteration(i+1) = min(fit);     % store best fitness per iteration
    %     % Show Iteration Information
    %     disp(['Iteration ' num2str(i) ': Best Hike = ' num2str(min(fit))]);
end
% THFO Solution: Store global best fitness and position
[Best.Hike,idx] = min(fit);
Best.Position = Pop(idx,:);
Mdl_best=Mdl_all{1,idx};

fPosition=Best.Position;
fHike=Best.Hike;
fiteration=Best.iteration';
end

%%
function [Best_Cost,Mdl_best,Convergence_curve,Best_X]=RUN_RUN(nP,MaxIt,lb,ub,dim,fobj,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm)
%---------------------------------------------------------------------------------------------------------------------------

% RUNge Kutta optimizer (RUN)
% RUN Beyond the Metaphor: An Efficient Optimization Algorithm Based on Runge Kutta Method
% Codes of RUN:http://imanahmadianfar.com/codes/
% Website of RUN:http://www.aliasgharheidari.com/RUN.html

% Iman Ahmadianfar, Ali asghar Heidari, Amir H. Gandomi , Xuefeng  Chu, and Huiling Chen  
%---------------------------------------------------------------------------------------------------------------------------

% ��������Ż��㷨
Cost=zeros(nP,1);                % Record the Fitness of all Solutions
X=initializationRUN(nP,dim,ub,lb);  % Initialize the set of random solutions   
Xnew2=zeros(1,dim);

Convergence_curve = zeros(1,MaxIt);

for i=1:nP
    [Cost(i),model{1,i}] = fobj(X(i,:),train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm );      % Calculate the Value of Objective Function
end

[Best_Cost,ind] = min(Cost);     % Determine the Best Solution
Best_X = X(ind,:);
Mdl_best=model{1,ind};
Convergence_curve(1) = Best_Cost;

% Main Loop of RUN 
it=1;%Number of iterations
while it<MaxIt
    it=it+1;
    f=20.*exp(-(12.*(it/MaxIt))); % (Eq.17.6) 
    Xavg = mean(X);               % Determine the Average of Solutions
    SF=2.*(0.5-rand(1,nP)).*f;    % Determine the Adaptive Factor (Eq.17.5)
    
    for i=1:nP
            [~,ind_l] = min(Cost);
            lBest = X(ind_l,:);   
            
            [A,B,C]=RndX(nP,i);   % Determine Three Random Indices of Solutions
            [~,ind1] = min(Cost([A B C]));
            
            % Determine Delta X (Eqs. 11.1 to 11.3)
            gama = rand.*(X(i,:)-rand(1,dim).*(ub-lb)).*exp(-4*it/MaxIt);  
            Stp=rand(1,dim).*((Best_X-rand.*Xavg)+gama);
            DelX = 2*rand(1,dim).*(abs(Stp));
            
            % Determine Xb and Xw for using in Runge Kutta method
            if Cost(i)<Cost(ind1)                
                Xb = X(i,:);
                Xw = X(ind1,:);
            else
                Xb = X(ind1,:);
                Xw = X(i,:);
            end

            SM = RungeKutta(Xb,Xw,DelX);   % Search Mechanism (SM) of RUN based on Runge Kutta Method
                        
            L=rand(1,dim)<0.5;
            Xc = L.*X(i,:)+(1-L).*X(A,:);  % (Eq. 17.3)
            Xm = L.*Best_X+(1-L).*lBest;   % (Eq. 17.4)
              
            vec=[1,-1];
            flag = floor(2*rand(1,dim)+1);
            r=vec(flag);                   % An Interger number 
            
            g = 2*rand;
            mu = 0.5+.1*randn(1,dim);
            
            % Determine New Solution Based on Runge Kutta Method (Eq.18) 
            if rand<0.5
                Xnew = (Xc+r.*SF(i).*g.*Xc) + SF(i).*(SM) + mu.*(Xm-Xc);
            else
                Xnew = (Xm+r.*SF(i).*g.*Xm) + SF(i).*(SM)+ mu.*(X(A,:)-X(B,:));
            end  
            
        % Check if solutions go outside the search space and bring them back
        FU=Xnew>ub;FL=Xnew<lb;Xnew=(Xnew.*(~(FU+FL)))+ub.*FU+lb.*FL; 
        [CostNew,model_best1]=fobj(Xnew,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
        
        if CostNew<Cost(i)
            X(i,:)=Xnew;
            Cost(i)=CostNew;
            model{1,i}=model_best1;
        end
% Enhanced solution quality (ESQ)  (Eq. 19)      
        if rand<0.5
            EXP=exp(-5*rand*it/MaxIt);
            r = floor(Unifrnd(-1,2,1,1));

            u=2*rand(1,dim); 
            w=Unifrnd(0,2,1,dim).*EXP;               %(Eq.19-1)
            
            [A,B,C]=RndX(nP,i);
            Xavg=(X(A,:)+X(B,:)+X(C,:))/3;           %(Eq.19-2)         
            
            beta=rand(1,dim);
            Xnew1 = beta.*(Best_X)+(1-beta).*(Xavg); %(Eq.19-3)
            
            for j=1:dim
                if w(j)<1 
                    Xnew2(j) = Xnew1(j)+r*w(j)*abs((Xnew1(j)-Xavg(j))+randn);
                else
                    Xnew2(j) = (Xnew1(j)-Xavg(j))+r*w(j)*abs((u(j).*Xnew1(j)-Xavg(j))+randn);
                end
            end
            
            FU=Xnew2>ub;FL=Xnew2<lb;Xnew2=(Xnew2.*(~(FU+FL)))+ub.*FU+lb.*FL;
            % CostNew=fobj(Xnew2);            
              [CostNew,model_best1]=fobj(Xnew2,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
              
            if CostNew<Cost(i)
                X(i,:)=Xnew2;
                Cost(i)=CostNew;
                 model{1,i}=model_best1;
            else
                if rand<w(randi(dim)) 
                    SM = RungeKutta(X(i,:),Xnew2,DelX);
                    Xnew = (Xnew2-rand.*Xnew2)+ SF(i)*(SM+(2*rand(1,dim).*Best_X-Xnew2));  % (Eq. 20)
                    
                    FU=Xnew>ub;FL=Xnew<lb;Xnew=(Xnew.*(~(FU+FL)))+ub.*FU+lb.*FL;
                    % CostNew=fobj(Xnew);
                 [CostNew,model_best1]=fobj(Xnew,train_x_feature_label_norm, train_y_feature_label_norm, vaild_x_feature_label_norm, vaild_y_feature_label_norm);
                             
                    if CostNew<Cost(i)
                        X(i,:)=Xnew;
                        Cost(i)=CostNew;
                        model{1,i}=model_best1;
                    end
                end
            end
        end
% End of ESQ         
% Determine the Best Solution
        if Cost(i)<Best_Cost
            Best_X=X(i,:);
            Best_Cost=Cost(i);
            Mdl_best=model{1,i};
        end

    end
% Save Best Solution at each iteration    
Convergence_curve(it) = Best_Cost;
% disp(['it : ' num2str(it) ', Best Cost = ' num2str(Convergence_curve(it) )]);

end



% A funtion to determine a random number 
%with uniform distribution (unifrnd function in Matlab) 
function z=Unifrnd(a,b,c,dim)
a2 = a/2;
b2 = b/2;
mu = a2+b2;
sig = b2-a2;
z = mu + sig .* (2*rand(c,dim)-1);
end

% A function to determine thress random indices of solutions
function [A,B,C]=RndX(nP,i)
Qi=randperm(nP);Qi(Qi==i)=[];
if length(Qi)<2
    A=Qi(1);B=Qi(1);C=Qi(1);
else
    A=Qi(1);B=Qi(2);C=Qi(3);
end
end


function SM=RungeKutta(XB,XW,DelX)

dim=size(XB,2);
C=randi([1 2])*(1-rand);
r1=rand(1,dim);
r2=rand(1,dim);

K1 = 0.5*(rand*XW-C.*XB);
K2 = 0.5*(rand*(XW+r2.*K1.*DelX/2)-(C*XB+r1.*K1.*DelX/2));
K3 = 0.5*(rand*(XW+r2.*K2.*DelX/2)-(C*XB+r1.*K2.*DelX/2));
K4 = 0.5*(rand*(XW+r2.*K3.*DelX)-(C*XB+r1.*K3.*DelX));

XRK = (K1+2.*K2+2.*K3+K4);
SM=1/6*XRK;
end


function Positions=initializationRUN(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ����ӳ������ѡ��label1-6�ֱ�Ϊtent��chebyshev
% tent��chebyshev��Singer��Logistic��Sine, Circle
function result = yinshe( N, dim,label,ub,lb)
if label==1
        % tent ӳ��
        tent=1.2;  %tent����ϵ��
        Tent=rand(N,dim);
        for i=1:N
            for j=2:dim
                if Tent(i,j-1)<tent
                    Tent(i,j)=Tent(i,j-1)/tent;
                elseif Tent(i,j-1)>=tent
                    Tent(i,j)=(1-Tent(i,j-1))/(1-tent);
                end
            end
        end
        result = lb+Tent.*(ub-lb);
elseif label==2
        %chebyshev ӳ��
        chebyshev=2;
        Chebyshev=rand(N,dim);
        for i=1:N
            for j=2:dim
                Chebyshev(i,j)=cos(chebyshev.*acos(Chebyshev(i,j-1)));
            end
        end
        result = lb+(Chebyshev+1)/2.*(ub-lb);
        
elseif label==3
        %singer ӳ��
        u=1;
        singer=rand(N,dim);
          for i=1:N
            for j=2:dim
        singer(i,j)=u*(7.86*singer(i,j-1)-23.31*singer(i,j-1).^2+28.75*singer(i,j-1).^3-13.302875*singer(i,j-1).^4);
            end
          end
          result = lb+singer.*(ub-lb);

elseif label==4
        %Logistic ӳ��
        miu=2;  %����ϵ��
        Logistic=rand(N,dim);
        for i=1:N
            for j=2:dim
                Logistic(i,j)=miu.* Logistic(i,j-1).*(1-Logistic(i,j-1));
            end
        end

       result = lb+Logistic.*(ub-lb);
elseif label==5
                %Sine ӳ��
        sine=2;
        Sine=rand(N,dim);
        for i=1:N
            for j=2:dim
                Sine(i,j)=(4/sine)*sin(pi*Sine(i,j-1));
            end
        end
         result = lb+Sine.*(ub-lb);

elseif label==6

        % Circle ӳ��
        a = 0.5; b=0.6;
        Circle=rand(N,dim);
        for i=1:N
            for j=2:dim
                Circle(i,j)=mod(Circle(i,j-1)+a-b/(2*pi)*sin(2*pi*Circle(i,j-1)),1);
            end
        end
        result = lb+Circle.*(ub-lb);
       
else
    %��ӳ��
   result =lb+rand(N,dim).*(ub-lb);
end
end
%%
function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
%%
% Application of simple limits/bounds
function s = Bounds( s, Lb, Ub)
  % Apply the lower bound vector
  temp = s;
  I = temp < Lb;
  temp(I) = Lb(I);
  
  % Apply the upper bound vector 
  J = temp > Ub;
  temp(J) = Ub(J);
  % Update this new move 
  s = temp;
  s(1:3)=round(s(1:3));
end