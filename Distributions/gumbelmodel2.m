%This file is used to model the (1-Pfa)-quantile standard Negative skewed Gumbel (or
%threshold coefficient) for the standard Negative-Skewed Gumbel by ignoring the dependency on
%number of samples used in modelling.

clear all
clc

close all
%clc

x = 0.001:0.001:0.999; %Probability 
T = log(-log(1-x)); 
plot(x(T>=0),T(T>=0))


x = 0.000001:0.000001:exp(-1); %0.00001:0.0001:0.8647-0.0001; %x used in fitting (FA probability) T>=0
%x = exp(-2):0.0001:1; %x used in fitting (FA probability)

%x = 0.000001:0.000001:0.1;
%y = 1 - log(-log(1 - x)); %y used in fitting     (Threshold coefficient)
y = log(-log(x));
%z = 1.8333 - 3*log(1-x) + 1.5*log(1-x).^2 + 1/3* log(1-x).^3; 





w = -((1 + log(x)) + (1 + log(x)).^2/2 + (1 + log(x)).^3/3 + (1 + log(x)).^4/4 + (1 + log(x)).^5/5 + (1 + log(x)).^6/6 + (1 + log(x)).^7/7 + (1 + log(x)).^8/8 + (1 + log(x)).^9/9 + (1 + log(x)).^10/10 + (1 + log(x)).^11/11 + ...
    (1 + log(x)).^12/12 + (1 + log(x)).^13/13 + (1 + log(x)).^14/14 + (1 + log(x)).^15/15 + (1 + log(x)).^16/16 + (1 + log(x)).^17/17 + (1 + log(x)).^18/18 + (1 + log(x)).^19/19 + (1 + log(x)).^20/20 + (1 + log(x)).^21/21 + ... 
    (1 + log(x)).^22/22 + (1 + log(x)).^23/23 + (1 + log(x)).^24/24 + (1 + log(x)).^25/25 + (1 + log(x)).^26/26 + (1 + log(x)).^27/27 + (1 + log(x)).^28/28 + (1 + log(x)).^29/29 + (1 + log(x)).^30/30 + (1 + log(x)).^31/31 + ...
    (1 + log(x)).^32/32 + (1 + log(x)).^33/33 + (1 + log(x)).^34/34 + (1 + log(x)).^35/35 + (1 + log(x)).^36/36 + (1 + log(x)).^37/37 + (1 + log(x)).^38/38 + (1 + log(x)).^39/39 + (1 + log(x)).^40/40 + (1 + log(x)).^41/41 + ...
    (1 + log(x)).^42/42 + (1 + log(x)).^43/43 + (1 + log(x)).^44/44 + (1 + log(x)).^45/45 + (1 + log(x)).^46/46 + (1 + log(x)).^47/47 + (1 + log(x)).^48/48 + (1 + log(x)).^49/49 + (1 + log(x)).^50/50 + (1 + log(x)).^51/51 + ...
    (1 + log(x)).^52/52 + (1 + log(x)).^53/53 + (1 + log(x)).^54/54 + (1 + log(x)).^55/55 + (1 + log(x)).^56/56 + (1 + log(x)).^57/57 + (1 + log(x)).^58/58 + (1 + log(x)).^59/59 + (1 + log(x)).^60/60 + (1 + log(x)).^61/61);
%exp(-2)<X<1
mean((w - y).^2)

figure
plot(x,y,'b')
hold on
%plot(x,z,'r')
%plot(x,w,'g')

%r = -0.1981*log(x)  - 2.215*x + 0.6074;  %VALID FOR 0<X<EXP(-1) for T>=0: step:e-4
%r = -0.2041*log(x)  - 2.169*x + 0.5871;  %VALID FOR 0<X<EXP(-1) for T>=0: step:e-5
r = -0.2046*log(x)  - 2.165*x + 0.5854;  %VALID FOR 0<X<EXP(-1) for T>=0: step:e-6

plot(x,r,'r')

mean((r - y).^2)

hold off



% v1 = (1 + log(1-x));% + (1 + log(1-x)).^2/2;
% v2 = -log(x);
% figure
% plot(x,v1)
% hold on
% plot(x,v2,'r')
% hold off