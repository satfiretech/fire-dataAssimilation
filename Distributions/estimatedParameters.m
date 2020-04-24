function p = estimatedParameters(Y,distribution)

%Load the coefficients to estimate location and scale parameters for
%Gumbel, Gaussian and Logistic and estimate the location and scale
%parameters given the data. In this file we have considered the case the 
%data are always of length 96.

%addpath('C:\MATLAB71\work\FireDetection2\Distributions2\Coefficients3');
%addpath('C:\Users\JOSINE\WORK1\work\FireDetection2\Distributions2\Coefficients3');
%addpath('C:\Users\JOSINE\MODEL_CODE_SMALL\UJ_PREDICTION1\Distributions2\Coefficients3');

%parameters = estimatedParameters(data)

%parameters 1: Location parameter
%parameters 2: Scale parameter


%First file

%Generate variates, compute BLUE (1 way using the file blue) and MLE
%BLUE coefficients were found using state=0 on uniform random number 

%clear all
%clc

% randn('state',0)
% rand('state',0)

%Generation of Gumbel, Normal and Logistic variates

%Y =  evrnd(0,10,1,96); %normrnd(10,2,1,96); %log([0.22 0.5 0.88 1 1.32 1.54 1.76 2.50 3]); %normrnd(10,2,1,96); %1:20; %evrnd(10,2,1,96); %normrnd(10,2,1,96); %evrnd(10,2,1,96);%randn(1,96); %log([0.22 0.5 0.88 1 1.32 1.54 1.76 2.50 3]); %(Balakrishnan_1991)%1:20; %20*randn(1,96);
%Y = Y';
%Residual data


Y = sort(Y(:)); %sort data


if distribution==0 %Gumbel
%Generating Gumbel variate
%-------------------------
%MLE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % initialScale = sqrt(6*var(Y,1))/pi; %Initialize scale beta
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % initialLocation = mean(Y) + initialScale*0.5772156649015328606; %mean(KeepStResidual) - SKEW_GUMBEL * initialScale*0.5772156649015328606; %CHANGE THIS IN OTHER FILES
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % p0 = [initialLocation;initialScale];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % gumbelPDF = @(x,mu,bt) exp((x-mu)/bt).*exp(-exp((x-mu)/bt))/bt;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % gumbelCDF = @(x,mu,bt) (1 - exp(-exp((x-mu)/bt)));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % MLEG = mle(Y,'pdf',gumbelPDF,'cdf',gumbelCDF,'start',p0) %,'cdf',@(p) normalCDF(p),'start',p0);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % LOG_LIKELIHOOD_GUMBEL = sum(log(exp((Y-MLEG(1))/MLEG(2)).*exp(-exp((Y-MLEG(1))/MLEG(2)))/MLEG(2)))

%BLUE
load gumbelCoefficients;

pG = gumbelCoefficients*Y;
LOG_LIKELIHOOD_GUMBEL = sum(log(exp((Y-pG(1))/pG(2)).*exp(-exp((Y-pG(1))/pG(2)))/pG(2)));
p = pG;

elseif distribution==1 %Gaussian
%Generating Gaussian variate
%---------------------------

%MLE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % initialLocation = mean(Y);%mean(KeepStResidual);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % initialScale = sqrt(var(Y,1)); %Scale is standard deviation
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % p0 = [initialLocation;initialScale];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % normalPDF = @(x,mu,sigma) 1/sqrt(2*pi*sigma^2)*exp(-(x-mu).^2/(2*sigma^2));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % normalCDF = @(x,mu,sigma) 1/2*(1+erf((x-mu)/(sigma*sqrt(2))));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % MLEN = mle(Y,'pdf',normalPDF,'cdf',normalCDF,'start',p0)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % LOG_LIKELIHOOD_NORMAL = sum(log(1/sqrt(2*pi*(MLEN(2))^2)*exp(-(Y-MLEN(1)).^2/(2*(MLEN(2))^2))))


%BLUE
load normalCoefficients;

pN = normalCoefficients*Y;
LOG_LIKELIHOOD_NORMAL = sum(log(1/sqrt(2*pi*(pN(2))^2)*exp(-(Y-pN(1)).^2/(2*(pN(2))^2))));


%MLE: use of sample variance and sample mean
sNV = [mean(Y);std(Y)];
LOG_LIKELIHOOD_NORMAL = sum(log(1/sqrt(2*pi*(std(Y))^2)*exp(-(Y-mean(Y)).^2/(2*(std(Y))^2))));
p = pN;

elseif distribution==2 %Logistic
%Generating Logistic variate
%---------------------------

%MLE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % initialLocation = mean(Y);%mean(KeepStResidual);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % initialScale = sqrt(3*var(Y,1))/pi; %Initialize scale s
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % p0 = [initialLocation;initialScale];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % logisticPDF = @(x,mu,s) 1/(4*s)*(sech((x-mu)/(2*s))).^2;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % logisticCDF = @(x,mu,s) 1/2 + 1/2* tanh((x-mu)/(2*s));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % MLEL = mle(Y,'pdf',logisticPDF,'cdf',logisticCDF,'start',p0)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % LOG_LIKELIHOOD_LOGISTIC = sum(log(1/(4*MLEL(2))*(sech((Y-MLEL(1))/(2*MLEL(2)))).^2))


%BLUE
load logisticCoefficients;

pL = logisticCoefficients*Y;
LOG_LIKELIHOOD_LOGISTIC = sum(log(1/(4*pL(2))*(sech((Y-pL(1))/(2*pL(2)))).^2));
p = pL;
else
    disp('0:Gumbel, 1:Gaussian, 2:Logistic')
end




