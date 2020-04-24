%pfa = 1 - F(theta);
%pfa = 2*F(theta)*(1 - F(theta))
close all
x = 0.0001:0.0001:0.5;
y = -log(2./(1 + sqrt(1 - 2*x))-1);
z = -log(2./(1 - sqrt(1 - 2*x))-1); %THIS IS WRONG, at T<=0, we have only one variate (reduction to monovariate, no more bivariate as in case of T>0).
plot(x,y)
xlabel('False alarm')
ylabel('Threshold coefficient')
hold on
plot(x,z,'r')

% close all
% x = 0:0.1:5;
% F = 1./(1 + exp(-x));
% G = 2*F.*(1-F);
% 
% y = -5:0.1:0;
% F = 1./(1 + exp(-y));
% H = 1 - F;
% 
% plot(x,G)
% hold on
% plot(y,H,'r')
% xlabel('Threshold')
% ylabel('FA')
% hold off


close all
x = -5:0.1:5;
y = (x>=0).*x;
F = 1./(1 + exp(-x));
G = 1./(1 + exp(-y));

H = 2*G.*(1-F);

plot(x,H)
xlabel('Threshold')
ylabel('FA')

y = -log(2./(1 + sqrt(1 - 2*x))-1);


close all
Pf = 0:0.001:0.5;
Pf_d1 = (1 - sqrt(1-2*Pf))/2;
Pf_d2 = Pf./(1 + sqrt(1-2*Pf));
plot(Pf,Pf_d1,Pf,Pf_d2)

