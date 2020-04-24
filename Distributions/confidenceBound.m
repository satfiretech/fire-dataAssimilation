clc
%Calculate confidence bound of a fit (e.g. fitting a curve on data (e.g. fitting a time series))

%Linear in coefficients.

%Search confidence bound
%Evaluating the Goodness of Fit
%The Least squares Fitting Method 

xx = (1:10)';        %Data for Predictor?
%AA = randn(1,10); %Observation

% AA = [-0.4326
%    -1.6656
%     0.1253
%     0.2877
%    -1.1465
%     1.1909
%     1.1892
%    -0.0376
%     0.3273
%     0.1746];

% p1 = 0.1433; p2 = -0.7868; y3 = p1*xx + p2;  %The fit found for ax+b (linear polynomial) 
% p1 = 0.3567; p2 = 0.5373; y2 = p1*xx + p2;   %Up bound
% p1 = -0.07011; p2 = -2.111; y1 = p1*xx + p2; %Lower bound 
% %Bound found using curve fitting tool.
% figure;plot(xx,AA,'r*');hold on;plot(xx,y1,'g');plot(xx,y3,'b');plot(xx,y2,'y');hold off

%MM = ((sum((AA - y3).^2))/8) %MSE

XX = [xx ones(10,1)]; %Design matrix
BB = XX\AA %Coefficients found
ys = XX*BB; %Prediction (estimate)

%Calculate 95% Upper Bound
%0.5/2 and 1-0.5/2
UP_BOUND = diag(sqrt(inv(XX'*XX)*sum((AA-ys).^2)/8))*tinv(0.975,8) + BB

%Calculate 95% Lower Bound
LOWER_BOUND = diag(sqrt(inv(XX'*XX)*sum((AA-ys).^2)/8))*tinv(1-0.975,8) + BB


%RMSE
RMSE = sqrt((sum((AA - ys).^2))/8)


%SSE
SSE = sum((AA - ys).^2) 

%Residual Degree of freedom
DF = length(AA) - 2; %2 coefficients

%R-square
RS = sum((ys-mean(AA)).^2)/sum((AA-mean(AA)).^2)  %= SSR/SST or Same as 1 - SSE/sum((AA-mean(AA)).^2)=1 - SSE/SST



%Degrees of Freedom Adjusted R-square
ARS = 1 - (SSE*(length(AA)-1))/(sum((AA-mean(AA)).^2)*(DF))
