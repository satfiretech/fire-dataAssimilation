function [parameters AB meanvector]= blue(data,distribution)

%[parameters AB]= blue(data,distribution)
%Parameter(1) = Location
%Parameter(2) = Scale

%AB = Coefficients

%distribution:
%0: Gumbel distribution
%1: Normal distribution
%2: Logistic distribution


%'pdf',pdf,'cdf',cdf,'numberofobservations',numberOfObservations) 

%rand('state',sum(100*clock));
%rand('state',0);

%CM = cumsum(1/observation*ones(1,observation));
%rand('state',min(find(CM>rand)));


observation = 2000;

data = data(:);
data = sort(data); %Sort data in ascending order


U = zeros(observation,length(data)); %Number of observation = observation, number of variables = length(data)
for i = 1:observation
    U(i,:) = rand(1,length(data)); %unifrnd(0,1,1,length(data));%rand(1,length(data));
    
    %U(i,:) = U(i,:)*1/length(data) +  [0:length(data)-1]/length(data);
    
end

% for i = 1:length(data)
%     U(:,i) = rand(observation,1); %unifrnd(0,1,1,length(data));%rand(1,length(data));
% end


% for i = 1:observation
%     for j = 1:length(data)
%         U(i,j) = rand;
%     end
% end

%U = rand(observation,length(data));


if distribution == 0 %Gumbel distribution (extreme value distribution)
    %Generating standard Gumbel variates (location = 0, scale = 1): Quantile
    %function
    gumbelQuantile = @(u) log(-log(u)); %log(-log(1 - U))

    v = ones(observation,length(data)) - U;
    generatedData = gumbelQuantile(v);
    %generatedData = evrnd(0,1,size(v,1),size(v,2));
    
elseif distribution ==1   %Gaussian distribution    
    gaussianQuantile = @(u) sqrt(2)* erfinv(u); %sqrt(2)*erfinv(2*U - 1)    %Gaussian Quantile function or probit function          
    
    v = 2*U - ones(observation,length(data));
    generatedData = gaussianQuantile(v);
    %generatedData = randn(observation,length(data));
    %generatedData = generatedData - ((mean(generatedData))'*ones(1,observation))';
    
elseif distribution == 2
    %Generating standard Logistic variates (location = 0, scale = 1): Quantile
    %function
    logisticQuantile = @(u) log(u); %log(U/(1-U))

    v = U./(ones(observation,length(data)) - U);
    generatedData = logisticQuantile(v);

else
    disp('Distributions are 0:Gumbel, 1:Gaussian, 2:Logistic');
end

%%
sortGeneratedData = sort(generatedData,2); %Sort the data generated (order statistic) %Ordered in ascending order(and possibly censored) 


% if distribution == 1
% figure
% hist(sortGeneratedData(:,1))
% figure
% hist(sortGeneratedData(:,2))
% figure
% hist(sortGeneratedData(:,3))
% end

meanvector = (mean(sortGeneratedData,1))'; %Mean vector
H = [ones(length(meanvector),1) meanvector];
C = cov(sortGeneratedData);
%C = diag(diag(C));

parameters = inv(H.'*inv(C)*H)*H.'*inv(C)*data; %The first parameter: location parameter, and the second parameter: scale parameter


AB = inv(H.'*inv(C)*H)*H.'*inv(C);

A = inv(H.'*inv(C)*H);

%BLI estimate
parameters(1) = parameters(1) - parameters(2)/(1+A(2,2))*A(1,2);
parameters(2) = parameters(2)/(1+A(2,2));


