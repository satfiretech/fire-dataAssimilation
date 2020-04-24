%Calculated error using LogLikelihood (the best is MLE), but I must also
%calculate Least Square to prove that BLUE is the best.

%Generate variates, compute BLUE (3 ways with 1 way using the file blue) and MLE

% % % % % % % % % % % % % % % % % % % randn('state',0)
% % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % DD = randn(1000,3);
% % % % % % % % % % % % % % % % % % % %DD = (DD - mean(DD,2)*[1 1 1])./(std(DD,[],2)*[1 1 1]); %Standardized
% % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % %LL = DD;
% % % % % % % % % % % % % % % % % % % DD = sort(DD,2); %Ordered in ascending order(and possibly censored)
% % % % % % % % % % % % % % % % % % % %DD = LL; %Ordered in ascending order(and possibly censored)
% % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % %Data I have
% % % % % % % % % % % % % % % % % % % %------------
% % % % % % % % % % % % % % % % % % % % SS = sqrt(3)*randn(3,1);%(DD(1,:))'*sqrt(2);
% % % % % % % % % % % % % % % % % % % % %EE = SS; %((SS - mean(SS))/std(SS)*std(XX) + mean(XX));
% % % % % % % % % % % % % % % % % % % % EE = sort(SS);
% % % % % % % % % % % % % % % % % % % %===========s
% % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % %Approximate the quantile
% % % % % % % % % % % % % % % % % % % %--------------
% % % % % % % % % % % % % % % % % % % XX = (mean(DD,1))'; %Mean vector %(SS - mean(SS))/std(SS);
% % % % % % % % % % % % % % % % % % % %XX = sort(XX);
% % % % % % % % % % % % % % % % % % % %===============
% % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % ZZ = [ones(length(XX),1) XX];
% % % % % % % % % % % % % % % % % % % TT = [mean(EE);std(EE)];
% % % % % % % % % % % % % % % % % % % CC = cov(DD,1);
% % % % % % % % % % % % % % % % % % % %CC = eye(3);
% % % % % % % % % % % % % % % % % % % HH = inv(ZZ'*inv(CC)*ZZ)*ZZ'*inv(CC);
% % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % [inv(ZZ'*inv(CC)*ZZ)*ZZ'*inv(CC)*EE TT]




% % % % % % % % % % % % % % % randn('state',0)
% % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % DD = randn(1000,3);
% % % % % % % % % % % % % % % DD = sort(DD,2); %Ordered in ascending order(and possibly censored)
% % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % %Data I have
% % % % % % % % % % % % % % % %------------
% % % % % % % % % % % % % % % SS = sqrt(3)*randn(3,1);%(DD(1,:))'*sqrt(2);
% % % % % % % % % % % % % % % EE = sort(SS);
% % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % %Approximate the quantile
% % % % % % % % % % % % % % % %--------------
% % % % % % % % % % % % % % % XX = (mean(DD,1))'; %Mean vector %(SS - mean(SS))/std(SS);
% % % % % % % % % % % % % % % %===============
% % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % ZZ = [ones(length(XX),1) XX];
% % % % % % % % % % % % % % % TT = [mean(EE);std(EE)];
% % % % % % % % % % % % % % % CC = cov(DD);
% % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % inv(ZZ'*inv(CC)*ZZ)*ZZ'*inv(CC)*EE




clear all
clc
close all

% randn('state',0)
% rand('state',0)

%Generation of Gumbel, Normal and Logistic variates

% Y = log([119
% 138
% 146
% 151
% 27.5
% 69
% 150
% 8.6
% 51.5
% 89
% 109
% 6
% 74
% 118
% 141
% 18
% 33.5
% 144
% 17.8
% 153
% 153.1
% 153.2
% 50.5
% 74.0000]).'; %Mann_1967

%Y = [1 5];% 8];
Y =  normrnd(10,2,1,96); %log([0.22 0.5 0.88 1 1.32 1.54 1.76 2.50 3]); %normrnd(10,2,1,96); %1:20; %evrnd(10,2,1,96); %normrnd(10,2,1,96); %evrnd(10,2,1,96);%randn(1,96); %log([0.22 0.5 0.88 1 1.32 1.54 1.76 2.50 3]); %(Balakrishnan_1991)%1:20; %20*randn(1,96);
Y = Y';
Y = sort(Y);

[mean(Y) std(Y)]

XU = rand(length(Y),1);

%Generating Gumbel variate
%-------------------------
XG = log(-log(ones(size(XU)) - XU));
OG = sort(XG);

initialScale = sqrt(6*var(OG,1))/pi; %Initialize scale beta
initialLocation = mean(OG) + initialScale*0.5772156649015328606; %mean(KeepStResidual) - SKEW_GUMBEL * initialScale*0.5772156649015328606; %CHANGE THIS IN OTHER FILES

p0 = [initialLocation;initialScale];



gumbelPDF = @(x,mu,bt) exp((x-mu)/bt).*exp(-exp((x-mu)/bt))/bt;
gumbelCDF = @(x,mu,bt) (1 - exp(-exp((x-mu)/bt)));

p = mle(OG,'pdf',gumbelPDF,'cdf',gumbelCDF,'start',p0); %,'cdf',@(p) normalCDF(p),'start',p0);
%p = evfit(OG);
% p = gevfit(KeepStResidual)
% mu = 0; %p(3);
% bt = sqrt(6*1)/pi; %p(2);

LG = mean(OG); %p(1);
SG = std(OG); %p(2);

D = [ones(length(OG),1) (OG - LG)/SG];
C = eye(length(OG))*pi^2/6;

PG = inv(D.'*inv(C)*D)*D.'*inv(C)*Y

initialScale = sqrt(6*var(Y,1))/pi; %Initialize scale beta
initialLocation = mean(Y) + initialScale*0.5772156649015328606; %mean(KeepStResidual) - SKEW_GUMBEL * initialScale*0.5772156649015328606; %CHANGE THIS IN OTHER FILES

p0 = [initialLocation;initialScale];
%p0 = [100;100];
%p0 = blue(Y,1);


gumbelPDF = @(x,mu,bt) exp((x-mu)/bt).*exp(-exp((x-mu)/bt))/bt;
gumbelCDF = @(x,mu,bt) (1 - exp(-exp((x-mu)/bt)));

MLEG = mle(Y,'pdf',gumbelPDF,'cdf',gumbelCDF,'start',p0) %,'cdf',@(p) normalCDF(p),'start',p0);
LOG_LIKELIHOOD_GUMBEL = sum(log(exp((Y-MLEG(1))/MLEG(2)).*exp(-exp((Y-MLEG(1))/MLEG(2)))/MLEG(2)))


DG = log(-log(ones(1000,length(Y)) - rand(1000,length(Y))));
SG = sort(DG,2); %Ordered in ascending order(and possibly censored)
MG = (mean(SG,1))'; %Mean vector %(SS - mean(SS))/std(SS);
%MG = (Y - mean(Y))/std(Y);
ZG = [ones(length(MG),1) MG];
CG = cov(SG);
%CG = eye(length(OG))*pi^2/6;

NGBLUE = inv(ZG'*inv(CG)*ZG)*ZG'*inv(CG)*Y


%Generating Gaussian variate
%---------------------------
XN = randn(length(Y),1);
ON = sort(XN);


initialLocation = mean(ON);%mean(KeepStResidual);
initialScale = sqrt(var(ON,1)); %Scale is standard deviation
p0 = [initialLocation;initialScale];
%p0 = [rand;rand];

normalPDF = @(x,mu,sigma) 1/sqrt(2*pi*sigma^2)*exp(-(x-mu).^2/(2*sigma^2));
normalCDF = @(x,mu,sigma) 1/2*(1+erf((x-mu)/(sigma*sqrt(2))));

p = mle(ON,'pdf',normalPDF,'cdf',normalCDF,'start',p0); %,'cdf',@(p) normalCDF(p),'start',p0);


LN = mean(ON); %p(1); %mean(ON);
SN = std(ON); %p(2); %std(ON);

D = [ones(length(ON),1) (ON - LN)/SN];
C = eye(length(ON))*1;

PN = inv(D.'*inv(C)*D)*D.'*inv(C)*Y

initialLocation = mean(Y);%mean(KeepStResidual);
initialScale = sqrt(var(Y,1)); %Scale is standard deviation
p0 = [initialLocation;initialScale];

normalPDF = @(x,mu,sigma) 1/sqrt(2*pi*sigma^2)*exp(-(x-mu).^2/(2*sigma^2));
normalCDF = @(x,mu,sigma) 1/2*(1+erf((x-mu)/(sigma*sqrt(2))));
MLEN = mle(Y,'pdf',normalPDF,'cdf',normalCDF,'start',p0)
LOG_LIKELIHOOD_NORMAL = sum(log(1/sqrt(2*pi*(MLEN(2))^2)*exp(-(Y-MLEN(1)).^2/(2*(MLEN(2))^2))))



DN = randn(1000,length(Y));
SN = sort(DN,2); %Ordered in ascending order(and possibly censored)
MN = (mean(SN,1))'; %Mean vector %(SS - mean(SS))/std(SS);
ZN = [ones(length(MN),1) MN];
CN = cov(SN);

NNBLUE = inv(ZN'*inv(CN)*ZN)*ZN'*inv(CN)*Y




%Generating Logistic variate
%---------------------------
XL = log(XU./(1-XU));
%XL(:,1) = XL;
OL = sort(XL);



initialLocation = mean(OL);%mean(KeepStResidual);
initialScale = sqrt(3*var(OL,1))/pi; %Initialize scale s
p0 = [initialLocation;initialScale];

logisticPDF = @(x,mu,s) 1/(4*s)*(sech((x-mu)/(2*s))).^2;
logisticCDF = @(x,mu,s) 1/2 + 1/2* tanh((x-mu)/(2*s));

p = mle(OL,'pdf',logisticPDF,'cdf',logisticCDF,'start',p0); %,'cdf',@(p) normalCDF(p),'start',p0);
% p = mle(KeepStResidual,'pdf',logisticPDF,'start',p0); %,'cdf',@(p) normalCDF(p),'start',p0);


%PARAM_LOGISTIC_BEFORE = [p(1)]



LL = mean(OL);%p(1);
SL = std(OL);%p(2);

D = [ones(length(OL),1) (OL - LL)/SL];
C = eye(length(OL));%*pi^2/6;

PL = inv(D.'*inv(C)*D)*D.'*inv(C)*Y

initialLocation = mean(Y);%mean(KeepStResidual);
initialScale = sqrt(3*var(Y,1))/pi; %Initialize scale s
p0 = [initialLocation;initialScale];

logisticPDF = @(x,mu,s) 1/(4*s)*(sech((x-mu)/(2*s))).^2;
logisticCDF = @(x,mu,s) 1/2 + 1/2* tanh((x-mu)/(2*s));

MLEL = mle(Y,'pdf',logisticPDF,'cdf',logisticCDF,'start',p0)
LOG_LIKELIHOOD_LOGISTIC = sum(log(1/(4*MLEL(2))*(sech((Y-MLEL(1))/(2*MLEL(2)))).^2))

AL = rand(1000,length(Y));
DL = log(AL./(1-AL));
SL = sort(DL,2); %Ordered in ascending order(and possibly censored)
ML = (mean(SL,1))'; %Mean vector %(SS - mean(SS))/std(SS);
ZL = [ones(length(ML),1) ML];
CL = cov(SL);

NLBLUE = inv(ZL'*inv(CL)*ZL)*ZL'*inv(CL)*Y


keepPG = [];
keepPN = [];
keepPL = [];

keepAB_G = zeros(2,length(Y));
keepAB_N = zeros(2,length(Y));
keepAB_L = zeros(2,length(Y));


keepM_G = zeros(2,1);
keepM_N = zeros(2,1);
keepM_L = zeros(2,1);



%rand('state',0);
rand('twister',5489);
%rand('seed',0);
%format short eng
for i = 1:50000
 i   
    [pG AB_G M_G]= blue(Y,0);
    keepPG = [keepPG pG];
    %keepM_G = keepM_G + M_G;
    %keepAB_G(:,:,i) = AB_G;
    keepAB_G = keepAB_G + AB_G;
    LOG_LIKELIHOOD_GUMBEL = sum(log(exp((Y-pG(1))/pG(2)).*exp(-exp((Y-pG(1))/pG(2)))/pG(2)));
    
%     [i sum((mean(keepPG,2) - [4.61398;0.56889]).^2)]
    
%     if sum((mean(keepPG,2) - [4.61398;0.56889]).^2) < 1e-8
%         break;
%     end

    %keepCoeff_G = keepCoeff_G + AB_G;  
    %gumbelCoefficients = AB_G;

    %save gumbelCoefficients gumbelCoefficients

    [pN AB_N M_N]= blue(Y,1);
    keepPN = [keepPN pN];
    %keepM_N = keepM_N + M_N;
    %keepAB_N(:,:,i) = AB_N;
    keepAB_N = keepAB_N + AB_N;
    LOG_LIKELIHOOD_NORMAL = sum(log(1/sqrt(2*pi*(pN(2))^2)*exp(-(Y-pN(1)).^2/(2*(pN(2))^2))));


    %keepCoeff_N = keepCoeff_N + AB_N;
    %normalCoefficients = AB_N;

    %save normalCoefficients normalCoefficients


    [pL AB_L M_L]= blue(Y,2);
    keepPL = [keepPL pL];
    %keepM_L = keepM_L + M_L;%Mean vector 
    %keepAB_L(:,:,i) = AB_L;
    keepAB_L = keepAB_L + AB_L; %Coefficients
    LOG_LIKELIHOOD_LOGISTIC = sum(log(1/(4*pL(2))*(sech((Y-pL(1))/(2*pL(2)))).^2));

    %keepCoeff_L = keepCoeff_L + AB_L;
    %logisticCoefficients = AB_L;

    %save logisticCoefficients logisticCoefficients

    LOG_LIKELIHOOD_LOGISTIC = sum(log(1/sqrt(2*pi*(std(Y))^2)*exp(-(Y-mean(Y)).^2/(2*(std(Y))^2))));
end
%format

%mean(keepPN,2)
%mean(keepPG,2)

gumbelCoefficients = keepAB_G/50000;
save gumbelCoefficients gumbelCoefficients

normalCoefficients = keepAB_N/50000;
save normalCoefficients normalCoefficients

logisticCoefficients = keepAB_L/50000;
save logisticCoefficients logisticCoefficients

%keepM_N/10000