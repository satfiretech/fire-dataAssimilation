function wt = solverStep(F, Ht, Hb, Q1, Bo, N, R, Rb, yb, xsfvector,j,x,t,tHorizon)
%[length(find(isnan(F)==1)) length(find(isnan(Ht)==1)) length(find(isnan(Hb)==1)) length(find(isnan(Q1)==1)) length(find(isnan(Bo)==1)) length(find(isnan(N)==1)) length(find(isnan(R)==1)) length(find(isnan(Rb)==1)) length(find(isnan(yb)==1)) length(find(isnan(xsfvector)==1)) length(find(isnan(j)==1)) length(find(isnan(x)==1)) length(find(isnan(t)==1))]
                

%Step 2 and Step 3: 
%w is unknown

%if j==1
    Q = [Q1 Q1 Q1 Q1 Q1 Q1];% Q1 Q1];%*0+1;
%else
%    Q = [Q1 Q1 Q1 Q1 Q1 Q1]*0;;
%end

%ovariance = Ht(:,1).'*Bo*Ht(:,1) + R(1);
%ovariance = R(1);

%if ovariance==0
%    ovariance = eps;
%end

%ovariance
%xsfvector(1:size(Ht,1))

%init = inv(ovariance)*(yb(1) - Ht(:,1).'*xsfvector(1:size(Ht,1)));



% wt = 30*[rand rand rand rand rand rand]; %It includes w0 %Initialization
%wt = 30*rand(size(yb)).'; %[Pixels x Day]
%wt = init(1)*ones(size(yb)).';
%wt = init(1)*randn(size(yb)).';

%----
%ovariance = R;
%ovariance(ovariance==0) = eps;
%----

% for i = 1:size(Ht,2) %NBRE of DAYS
%     wt(:,i) = inv(ovariance(i))*(yb(i) - Ht(:,i).'*xsfvector(1+(i-1)*size(Ht,1):i*size(Ht,1)));
% end

%wt = 0*randn(size(yb)).'; %Or expected value of the scaled residual: a^j
wt = 0*randn(length(yb)/(N+1),N+1); %Or expected value of the scaled residual: a^j


%wt = eps*rand(size(yb)).';
%wt = randn(size(yb)).';


%y = yb.';  %[1 x Day]
%y = yb.';  %[2 x Day]
y = reshape(yb,length(yb)/(N+1),N+1);


%iv = transformedRepresenterCoefficients(F,Ht,Q, Bo,N, R,wt);



if j==1
    %ivo = sqrt(inv(Rb))*(yb-Hb*xsfvector);
%     ivo = (yb-Hb*xsfvector);
    %for k = 1:size(yb,2)
    %    ivo = (yb(:,k)-Hb*xsfvector(:,k));
    %end
    
    %for i = 1:size(Ht,2) %NBRE of DAYS
    for i = 1:N+1 %NBRE of DAYS    
        %hnl(:,i) = observationFunction(xsfvector(1+(i-1)*size(Ht,1):i*size(Ht,1)),y(1,i),t(i));
        hnl(:,i) = [observationFunction(xsfvector(1+(i-1)*size(Ht,1):i*size(Ht,1)),y(1,i),t(i));
        thermalsunriseFunction(xsfvector(1+(i-1)*size(Ht,1):i*size(Ht,1)),y(2,i),t(i),tHorizon(i))];
                               
        %c_predicted = thermalsunriseFunction(x_,c_perturbed,t,tHorizon);
        %Htc(:,:,i) = thermalsunriseFunction(xsf(:,:,i),c(:,:,i),t(:,:,i),tHorizon(:,:,i)); %[6 x Pixels x Day]

        
    end
    %hnlb = hnl.';
    hnlb = reshape(hnl,2*(N+1),1); %(2.Day) x 1
    ivo = (yb-hnlb);
    
else
    for i = 0+1:N+1
        %hnl(:,i) = observationFunction(x(:,i,j-1),y(:,i),t(i));
        hnl(:,i) = [observationFunction(x(:,i,j-1),y(1,i),t(i));
          thermalsunriseFunction(x(:,i,j-1),y(2,i),t(i),tHorizon(i))];  
        %hnl(:,i) = observationFunction(x(:,i,j-1),y(:,i),reshape(t(:,:,i),size(yb,2),1));
    end
    %hnlb = hnl.';
    hnlb = reshape(hnl,2*(N+1),1);
    %yb - hnlb - Hb(:,:,j-1)*deltaxpb(:,:,j)
    %sqrt(inv(Rb))*(yb - hnlb - Hb(:,:,j-1)*deltaxpvector)
    %ivo = sqrt(inv(Rb))*(yb - hnlb - Hb*xsfvector)
    ivo = (yb - hnlb - Hb*xsfvector);
    %sqrt(inv(Rb))*(yb-Hb*xsfvector)
    %ZERO = length(find(isnan(hnlb(:))==1))
end


% for i = 1:size(Ht,2) %NBRE of DAYS
%     wt(:,i) = inv(ovariance(i))*(ivo(i));
% end

%FIRST = [length(find(isnan(wt(:))==1)) length(find(isnan(ivo(:))==1))]
%FIRST = [xsfvector(1:6).' length(find(isnan(wt(:))==1)) ivo(1) wt]


%wb = preconjugategradient_v3(F,Ht,Q, Bo,N, R, Rb, wt,ivo);

wb = flexpreconjugategradient_v4(F,Ht,Q, Bo,N, R, Rb, wt,ivo);
%wt = wb.';
wt = reshape(wb,length(wb)/(N+1),N+1);

%SECOND = [length(find(isnan(wt(:))==1)) wt]





