function regressionParameters = quantilefunctionFit(data,Location,Scale)


regressionParameters = [];


data = (data - Location)/Scale;

[cdfF,valuex] = ecdf(data);
FAR = 1 - cdfF(valuex>0);
ThresholdCoefficient = valuex(valuex>0);

FAR(1:end) = FAR(end:-1:1);
ThresholdCoefficient(1:end) = ThresholdCoefficient(end:-1:1);

ThresholdCoefficient(FAR==0) = []; %Remove FAR = 0 from the data
FAR(FAR==0) = [];


%MaxIter = 100
options = optimset('Display','off','TolX',[],'TolFun',[],'MaxIter',[],'MaxFunEvals',[],'FunValCheck','on');%,'MaxFunEvals',200);  %,'TolX',1e-20); MaxIter and/or MaxFunEvals

%p0 = [-0.9469 -1.072 1.212];%r = -0.9469*log(x)  - 1.072*x + 1.212; %GUMBEL
%p0 = [-0.2046 -2.165 0.5854];%r = -0.2046*log(x)  - 2.165*x + 0.5854; %GUMBEL right
%p0 = [-0.3788 -1.694 0.5801];%r = -0.3788*log(x)  - 1.694*x + 0.5801; %GAUSS right
p0 = [-0.3769 -1.705 0.5859];%r = -0.3788*log(x)  - 1.694*x + 0.5801; %GAUSS right
%Use Nelder & Mead's simplex method for error minimization but another
%method can be used. Also the function fit in MATLAB can be used and with
%this the following method can be error minimization can be implemented:
%Linear least square, least absolute residual and bisquare with algorithm:
%-Levenberg-Marquardt or Gauss-Newton or Trust-region. Check MATLAB: Fit
%options(Title:Fit options, Section:Context-Sensitive Help, Product:Curve Fitting Toolbox) 
[p,FVAL,EXITFLAG,OUTPUT] = fminsearch(@(p) derror(ThresholdCoefficient,FAR,p),p0,options); %Nelder & Mead's simplex method for error minimisation

regressionParameters = p(:);

% %figure
% %hold on
% p
% 
% %plot(valuex(valuex>=0),1 - cdfF(valuex>=0),'r*')
% plot(1 - cdfF(valuex>0),valuex(valuex>0),'g--')
% title('Standardized data CDF')
% xlabel('P_{FAR}')
% ylabel('Threshold')
% %hold off

